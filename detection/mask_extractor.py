from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np

import io, argparse, time
import requests
from PIL import Image
import cv2

# -----------------------------------------------------------------------------#
#  Public API
# -----------------------------------------------------------------------------#


def extract_mask(mode: str, **kwargs) -> List[dict]:
    mode = mode.lower().strip()
    if mode == "color":
        image = kwargs["image"]
        color_map = kwargs["color_map"]
        tolerance = kwargs.get("tolerance", 10)
        return _extract_by_color(image, color_map, tolerance)

    if mode == "json":
        fid = kwargs["fid"]
        detection_dir = kwargs.get("detection_dir", "detection")
        score_threshold = kwargs.get("score_threshold", 0.5)
        return _extract_by_json(fid, detection_dir, score_threshold)

    if mode == "server":
        image = kwargs["image"]
        return _extract_by_server(image)
        # raise NotImplementedError(
        #     "mode='server' is a stub – plug in your own implementation."
        # )

    raise ValueError(f"Unsupported mode '{mode}'.  Choose from 'color'/'json'.")


def _extract_by_server(image: np.ndarray) -> List[dict]:
    """
    将本地 numpy 图像发给 /infer，按 server 返回的 detections 解码出 mask。
    返回的每个元素包含:
      {
        "label": <top-1 类别名>,
        "score": <top-1 置信度>,
        "box":   [x1,y1,x2,y2],
        "mask":  <uint8 二值掩码 0/1>
      }
    默认连接 http://127.0.0.1:8000，可通过环境变量 OWL_SAM_SERVER 覆盖。
    """
    # 1) 规范化输入为 RGB uint8
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxW, HxWx3 or HxWx4")

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    # 2) PNG 编码并按 client_demo 协议 POST 到 /infer
    server = os.environ.get("OWL_SAM_SERVER", "http://127.0.0.1:8000").rstrip("/")
    url = server + "/infer"

    buf = BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("client.png", buf.getvalue(), "application/octet-stream")}
    r = requests.post(url, files=files, timeout=600)
    r.raise_for_status()
    resp = r.json()

    # 3) 解析 detections，解码 mask（base64 PNG -> 二值 uint8）
    detections = resp.get("detections", [])
    masks: List[dict] = []

    for det in detections:
        b64 = det.get("mask")
        if not b64:
            continue

        mask_arr = _decode_base64_mask(b64)
        if mask_arr.ndim == 3:
            # RGBA/彩色 PNG 的情况，取第一通道
            mask_arr = mask_arr[..., 0]
        bin_mask = (mask_arr > 127).astype(np.uint8)

        # 取 top-1 类别显示
        label = "unknown"
        score = 0.0
        cls_list = det.get("detection", [])
        if cls_list:
            top = max(cls_list, key=lambda x: x.get("score", 0.0))
            label = str(top.get("label", "unknown"))
            score = float(top.get("score", 0.0))

        masks.append(
            {
                "label": label,
                "score": score,
                "box": det.get("box", None),
                "mask": bin_mask,
            }
        )

    return masks



# -----------------------------------------------------------------------------#
#  Internal helpers – COLOR MODE
# -----------------------------------------------------------------------------#


def _extract_by_color(
    image: np.ndarray,
    color_map: Dict[str, Tuple[int, int, int]],
    tolerance: int = 10,
) -> List[dict]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    # Ensure 0-255 uint8 range
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    img_i16 = image.astype(np.int16)  # avoid uint8 wrap-around
    masks: List[dict] = []

    for label, rgb in color_map.items():
        target = np.asarray(rgb, dtype=np.int16)
        if target.shape != (3,):
            raise ValueError(f"RGB for label '{label}' must be length-3 tuple")

        diff = np.abs(img_i16 - target)        # H×W×3
        within = (diff <= tolerance).all(axis=2)  # H×W bool

        if within.any():  # only record non-empty results
            masks.append(
                {
                    "label": label,
                    "score": 1.0,
                    "mask": within.astype(np.uint8),  # 0 / 1
                }
            )

    return masks


# -----------------------------------------------------------------------------#
#  Internal helpers – JSON MODE
# -----------------------------------------------------------------------------#


def _extract_by_json(
    fid: int,
    detection_dir: str = "detection",
    score_threshold: float = 0.2,
) -> List[dict]:
    """
    从 <detection_dir>/det_XXXXXX.json 读取检测结果并解码 mask。
    • 若文件不存在 -> FileNotFoundError
    • 若找不到任何检测 / 所有检测score都低于阈值 -> 返回 []
    """
    fname = os.path.join(detection_dir, f"detection_{fid:06d}_final.json")
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Detection file '{fname}' not found")

    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)

    dets = data.get("detections", [])
    if not dets:                     # 没有任何检测
        return []

    # ───────────────── 根据score阈值过滤检测 ──────────────────
    chosen = [d for d in dets if d.get("score", 0.0) >= score_threshold]
    
    if not chosen:                   # 所有检测score都低于阈值
        return []

    # ───────────────── 解码 mask ──────────────────
    masks: List[dict] = []
    for det in chosen:
        b64 = det.get("mask")
        if not b64:
            continue                 # 跳过空 / 错误字段

        mask_arr = _decode_base64_mask(b64)
        if mask_arr.ndim == 3:       # RGBA / RGB → 单通道
            mask_arr = mask_arr[..., 0]

        bin_mask = (mask_arr > 0).astype(np.uint8)  # 保证 0/1

        masks.append(
            {
                "id": det.get("object_id"),
                "label": det.get("label", "unknown"),
                "score": float(det.get("score", 1.0)),
                "mask": bin_mask,
            }
        )

    return masks                     # 可能为空，但不会再抛错



def _decode_base64_mask(b64str: str) -> np.ndarray:
    raw = base64.b64decode(b64str, validate=True)

    # Try Pillow first (lighter import)
    try:
        from PIL import Image

        img = Image.open(BytesIO(raw))
        return np.array(img)
    except (ImportError, ModuleNotFoundError):
        pass

    # Fallback to OpenCV
    try:
        import cv2

        nparr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("cv2.imdecode() failed")
        return img
    except ImportError as e:
        raise RuntimeError(
            "Pillow or OpenCV (cv2) is required to decode base-64 masks"
        ) from e

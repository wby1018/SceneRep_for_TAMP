# server.py
import os, io, json, base64, time, gc, uuid, threading, argparse
from typing import List, Dict, Any, Tuple
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # 禁止一次性吃满
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")  # 只用 35% 显存，可按需 0.25~0.5
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform") # 减少碎片/更友好与其他框架共存

# ---- JAX / OWL-ViT ----
import jax
from scipy.special import expit as sigmoid
from skimage import transform as skt
from scenic.projects.owl_vit import configs
from scenic.projects.owl_vit import models

# ---- MobileSAM ----
import torch
from mobile_sam import SamPredictor, sam_model_registry

# ---- FastAPI ----
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# 提速
torch.backends.cudnn.benchmark = True

# ================== 配置 ==================
OBJECTS = ['milkbox', 'cola', 'cup', 'apple', 'pot', 'flowerpot']

# 仅 OWL 的保留阈值（不做 NMS/TopK）
SCORE_THRESHOLD = 0.02

# MobileSAM
MOBILE_SAM_CKPT = "weights/mobile_sam.pt"
MOBILE_SAM_TYPE = "vit_t"

# SAM 批量与显存策略 / 稳定性
SAM_MULTIMASK_OUTPUT = False
SAM_INIT_BATCH_SIZE  = 64
USE_FP16_FOR_SAM     = False   # ⚠ 默认关闭：更稳
USE_AUTOMIXED_PREC   = False   # ⚠ 默认关闭：更稳
CPU_FALLBACK_ON_OOM  = True

# 预热图片（可被命令行覆盖）
_WARMUP_IMAGE_PATH = os.environ.get("WARMUP_IMAGE", "").strip() or None

# ================== 全局模型状态 ==================
MODEL_LOCK = threading.Lock()

_cfg = None
_mdl = None
_variables = None
_jitted = None
_tokenized = None
_owl_input_size = None

_predictor = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ready = False

# ================== 工具函数 ==================
def mask_to_base64_png(mask_bool: np.ndarray) -> str:
    from PIL import Image
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    img = Image.fromarray(mask_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def xywh_norm_to_xyxy_image_coords(
    box_xywh_norm: np.ndarray,
    orig_h: int, orig_w: int, model_input_size: int
) -> Tuple[int, int, int, int]:
    S = float(model_input_size)
    size = float(max(orig_h, orig_w))
    scale_back = size / S

    cx_s, cy_s, w_s, h_s = box_xywh_norm * S
    cx = cx_s * scale_back
    cy = cy_s * scale_back
    ww = w_s * scale_back
    hh = h_s * scale_back

    x1 = int(round(cx - ww / 2.0))
    y1 = int(round(cy - hh / 2.0))
    x2 = int(round(cx + ww / 2.0))
    y2 = int(round(cy + hh / 2.0))

    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(0, min(x2, orig_w - 1))
    y2 = max(0, min(y2, orig_h - 1))

    if x2 <= x1: x2 = min(orig_w - 1, x1 + 1)
    if y2 <= y1: y2 = min(orig_h - 1, y1 + 1)
    return x1, y1, x2, y2

@torch.no_grad()
def init_mobilesam(device: torch.device):
    sam = sam_model_registry[MOBILE_SAM_TYPE](checkpoint=MOBILE_SAM_CKPT)
    # 建议默认 FP32 更稳；需要省显存可将 USE_FP16_FOR_SAM=True
    if device.type == "cuda" and USE_FP16_FOR_SAM:
        sam = sam.to(device=device, dtype=torch.float16)
    else:
        sam = sam.to(device=device, dtype=torch.float32)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor

@torch.no_grad()
def sam_segment_boxes_adaptive(
    predictor: SamPredictor,
    image_rgb_u8: np.ndarray,
    boxes_xyxy: List[List[int]],
    *,
    multimask_output: bool = SAM_MULTIMASK_OUTPUT,
    init_batch_size: int = SAM_INIT_BATCH_SIZE,
    cpu_fallback: bool = CPU_FALLBACK_ON_OOM
) -> Tuple[List[np.ndarray], List[float]]:
    """自适应批量分割，必要时回退 CPU；始终返回与原图同尺寸(H,W)的 bool 掩码。"""
    image_rgb_u8 = np.ascontiguousarray(image_rgb_u8)
    H, W = image_rgb_u8.shape[:2]
    device = next(predictor.model.parameters()).device

    def _postprocess_to_hw(m: torch.Tensor) -> torch.Tensor:
        if m.shape[-2:] == (H, W):
            return m
        try:
            input_size = getattr(predictor, "input_size",
                                 getattr(predictor.transform, "target_size", (H, W)))
            return predictor.postprocess_masks(m, input_size, (H, W))
        except Exception:
            return torch.nn.functional.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)

    def _predict_on_device(dev: torch.device, batch_boxes: torch.Tensor):
        autocast_enabled = (dev.type == "cuda") and USE_AUTOMIXED_PREC
        with torch.autocast(device_type=dev.type, enabled=autocast_enabled):
            m, s, _ = predictor.predict_torch(
                point_coords=None, point_labels=None,
                boxes=batch_boxes, multimask_output=multimask_output
            )
        m = _postprocess_to_hw(m)
        if multimask_output:
            best = torch.argmax(s, dim=1)
            m = m[torch.arange(m.shape[0]), best]
            s = s.max(dim=1).values
        else:
            m = m[:, 0]
            s = s[:, 0] if s.ndim == 2 else s.squeeze()
        return m, s

    predictor.set_image(image_rgb_u8)
    masks_out: List[np.ndarray] = []
    confs_out: List[float] = []

    if hasattr(predictor, "predict_torch"):
        dev = device
        boxes_t = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=dev)
        boxes_trans = predictor.transform.apply_boxes_torch(boxes_t, (H, W))

        bs = max(1, int(init_batch_size))
        i = 0
        while i < boxes_trans.shape[0]:
            j = min(i + bs, boxes_trans.shape[0])
            batch = boxes_trans[i:j]
            try:
                m, s = _predict_on_device(dev, batch)
                m_cpu = (m.detach().to("cpu").float() > 0.5).to(torch.bool).numpy()
                s_cpu = s.detach().to("cpu").float().numpy().tolist()
                for k in range(m_cpu.shape[0]):
                    masks_out.append(m_cpu[k].copy())
                    confs_out.append(float(s_cpu[k]))
                i = j
                del m, s, m_cpu
                if dev.type == "cuda":
                    torch.cuda.empty_cache(); gc.collect()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if bs > 1:
                        bs = max(1, bs // 2)
                        if dev.type == "cuda":
                            torch.cuda.empty_cache(); gc.collect()
                        continue
                    else:
                        if cpu_fallback:
                            print("[WARN] CUDA OOM @ batch=1 → fallback to CPU")
                            predictor.model.to("cpu")
                            predictor.set_image(image_rgb_u8)
                            dev_cpu = torch.device("cpu")
                            boxes_cpu = predictor.transform.apply_boxes_torch(
                                boxes_t.to(dev_cpu), (H, W)
                            )
                            for k in range(i, boxes_cpu.shape[0]):
                                bm = boxes_cpu[k:k+1]
                                m2, s2 = _predict_on_device(dev_cpu, bm)
                                m2 = (m2.to("cpu").float() > 0.5).numpy()[0]
                                s2 = float(s2.to("cpu").numpy().reshape(-1)[0])
                                masks_out.append(m2.copy())
                                confs_out.append(s2)
                            return masks_out, confs_out
                        else:
                            raise
                else:
                    raise
    else:
        for box in boxes_xyxy:
            m, s, _ = predictor.predict(
                box=np.array(box, dtype=np.float32)[None, :],
                multimask_output=multimask_output
            )
            if m.shape[-2:] != (H, W):
                m = torch.from_numpy(m).unsqueeze(0)
                m = torch.nn.functional.interpolate(m.float(), size=(H, W), mode="bilinear", align_corners=False)
                m = m.squeeze(0).numpy()
            if multimask_output:
                idx = int(np.argmax(s))
                masks_out.append((m[idx] > 0.5).astype(bool).copy())
                confs_out.append(float(s[idx]))
            else:
                masks_out.append((m[0] > 0.5).astype(bool).copy())
                confs_out.append(float(np.squeeze(s)))
    return masks_out, confs_out

def owl_detect(
    image_rgb_u8: np.ndarray,
    module, variables, jitted_apply,
    tokenized_queries: np.ndarray,
    input_size: int,
    score_threshold: float
) -> List[Dict[str, Any]]:
    img = image_rgb_u8.astype(np.float32) / 255.0
    h, w, _ = img.shape
    size = max(h, w)
    pad_img = np.pad(img, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)
    input_img = skt.resize(pad_img, (input_size, input_size), anti_aliasing=True)

    preds = jitted_apply(
        variables,
        input_img[None, ...],
        tokenized_queries[None, ...],
        train=False
    )
    preds = jax.tree_util.tree_map(lambda x: np.array(x[0]), preds)
    boxes_xywh = preds["pred_boxes"]
    logits = preds["pred_logits"][..., :len(OBJECTS)]

    detections: List[Dict[str, Any]] = []
    for i, box_wh in enumerate(boxes_xywh):
        cls_scores = sigmoid(logits[i])
        max_score = float(np.max(cls_scores))
        if max_score < score_threshold:
            continue

        det_list = [{"label": OBJECTS[j], "score": float(cls_scores[j])}
                    for j in range(len(OBJECTS))]
        det_list.sort(key=lambda d: d["score"], reverse=True)

        x1, y1, x2, y2 = xywh_norm_to_xyxy_image_coords(
            box_xywh_norm=box_wh,
            orig_h=h, orig_w=w,
            model_input_size=input_size
        )
        detections.append({"detection": det_list, "box": [int(x1), int(y1), int(x2), int(y2)]})
    return detections

def build_owl():
    cfg = configs.owl_v2_clip_b16.get_config(init_mode='canonical_checkpoint')
    mdl = models.TextZeroShotDetectionModule(
        body_configs=cfg.model.body,
        objectness_head_configs=cfg.model.objectness_head,
        normalize=cfg.model.normalize,
        box_bias=cfg.model.box_bias
    )
    variables = mdl.load_variables(cfg.init_from.checkpoint_path)
    jitted = jax.jit(mdl.apply, static_argnames=('train',))
    tokenized = np.array([mdl.tokenize(q, cfg.dataset_configs.max_query_length) for q in OBJECTS])
    tokenized = np.pad(tokenized, pad_width=((0, 100 - len(OBJECTS)), (0, 0)), constant_values=0)
    return cfg, mdl, variables, jitted, tokenized

def process_array_to_detections(
    img_u8: np.ndarray,
    predictor: SamPredictor,
    mdl, variables, jitted, tokenized, owl_input_size: int
) -> Dict[str, Any]:
    """对 numpy RGB uint8 图像跑完整 pipeline，返回 JSON detections（包含 mask 的 base64 PNG）。"""
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8]*3, axis=-1)
    if img_u8.shape[-1] == 4:
        img_u8 = img_u8[..., :3]
    img_u8 = img_u8.astype(np.uint8)

    # 1) OWL
    candidates = owl_detect(
        img_u8, mdl, variables, jitted, tokenized,
        input_size=owl_input_size, score_threshold=SCORE_THRESHOLD
    )

    results = {"detections": []}
    if len(candidates) == 0:
        return results

    # 2) SAM
    boxes = [c["box"] for c in candidates]
    masks_list, _ = sam_segment_boxes_adaptive(
        predictor, img_u8, boxes,
        multimask_output=SAM_MULTIMASK_OUTPUT,
        init_batch_size=SAM_INIT_BATCH_SIZE,
        cpu_fallback=CPU_FALLBACK_ON_OOM
    )

    # 3) JSON（仅返回 det，不返回叠加 PNG）
    for cand, mask in zip(candidates, masks_list):
        results["detections"].append({
            "detection": cand["detection"],
            "box": cand["box"],
            "mask": mask_to_base64_png(mask)
        })
    return results

# ================== FastAPI 应用（启动即加载 & 预热） ==================
def bootstrap_models():
    global _cfg, _mdl, _variables, _jitted, _tokenized, _owl_input_size, _predictor, _device, _ready
    _cfg, _mdl, _variables, _jitted, _tokenized = build_owl()
    _owl_input_size = int(_cfg.dataset_configs.input_size)
    _predictor = init_mobilesam(_device)
    _ready = True
    print(f"[READY] OWL input size={_owl_input_size}, SAM device={_device}")

def _warmup_once(image_path: str):
    try:
        if not (image_path and os.path.isfile(image_path)):
            print("[WARMUP] skip (no valid image)")
            return
        print(f"[WARMUP] running on: {image_path}")
        t0 = time.time()
        img = np.array(Image.open(image_path).convert("RGB"))
        with MODEL_LOCK:
            _ = process_array_to_detections(img, _predictor, _mdl, _variables, _jitted, _tokenized, _owl_input_size)
        t1 = time.time()
        print(f"[WARMUP] done in {t1 - t0:.3f}s")
    except Exception as e:
        print(f"[WARMUP] failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_models()
    # 预热：可选
    if _WARMUP_IMAGE_PATH:
        _warmup_once(_WARMUP_IMAGE_PATH)
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

APP = FastAPI(title="OWL+MobileSAM Server", version="1.0", lifespan=lifespan)

# ================== 路由 ==================
@APP.get("/health")
def health():
    return {
        "status": "ok" if _ready else "loading",
        "device": str(_device),
        "owl_input_size": _owl_input_size if _ready else None,
        "warmup_image": _WARMUP_IMAGE_PATH,
    }

@APP.post("/infer")
async def infer(file: UploadFile = File(...)):
    if not _ready:
        return JSONResponse({"error": "model not ready"}, status_code=503)

    t0 = time.time()
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    img = np.array(image)

    # 串行锁定，避免 SamPredictor.set_image 冲突
    with MODEL_LOCK:
        results = process_array_to_detections(
            img, _predictor, _mdl, _variables, _jitted, _tokenized, _owl_input_size
        )

    t1 = time.time()
    payload = {
        "job_id": time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8],
        "time_sec": round(t1 - t0, 3),
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "detections": results["detections"],
    }
    return JSONResponse(payload)

# ================== 作为脚本启动 ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--warmup_image", type=str, default="test.png", help="path to a test image for startup warmup")
    args = parser.parse_args()

    # 覆盖全局的预热图片路径（若提供）
    if args.warmup_image:
        _WARMUP_IMAGE_PATH = args.warmup_image

    uvicorn.run(APP, host=args.host, port=args.port, log_level="info")

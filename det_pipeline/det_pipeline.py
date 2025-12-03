# owl_sam_pipeline.py
import os, io, json, base64, time, gc
import numpy as np
from typing import List, Dict, Any, Tuple

# ---- JAX / OWL-ViT ----
import jax
from scipy.special import expit as sigmoid
from skimage import io as skio
from skimage import transform as skt
from scenic.projects.owl_vit import configs
from scenic.projects.owl_vit import models

# ---- MobileSAM ----
import torch
from mobile_sam import SamPredictor, sam_model_registry

# 提速
torch.backends.cudnn.benchmark = True

# ================== 配置 ==================
DATASET_NAME   = "/media/wby/2AB9-4188/data_move"   # 你的数据集根目录
INPUT_DIR_NAME = "rgb"                               # 输入图片子目录
OUTPUT_DIR_TAG = "detection_masks_owl_sam"           # 输出子目录名
OBJECTS        = ['milkbox', 'cola', 'cup', 'apple', 'pot', 'flowerpot']

# 检测阈值（每个候选框里取各类得分的最大值，只要最大值 >= SCORE_THRESHOLD 就保留该框）
SCORE_THRESHOLD = 0.02

# MobileSAM 权重与类型
MOBILE_SAM_CKPT = "weights/mobile_sam.pt"
MOBILE_SAM_TYPE = "vit_t"  # 'vit_t' for MobileSAM-tiny

# SAM 批量与显存策略
SAM_MULTIMASK_OUTPUT = False   # False 更省显存更快
SAM_INIT_BATCH_SIZE  = 64      # 初始批大小，自适应降到 1
USE_FP16_FOR_SAM     = False   # ⚠ 置为 False 更稳；如需省显存可改 True
USE_AUTOCast_INFER   = False   # ⚠ 为 True 时在 CUDA 上 autocast；默认关闭以避免精度问题
CPU_FALLBACK_ON_OOM  = True    # batch=1 仍 OOM 时改用 CPU
EMPTY_CACHE_EVERY_CHUNK = True # 每个小批次后清缓存
# ==========================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def mask_to_base64_png(mask_bool: np.ndarray) -> str:
    """
    将布尔 / 0-1 掩码( H x W )编码为PNG并Base64返回；输出与原图同尺寸。
    """
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
    """
    将 OWL-ViT 输出的 [cx, cy, w, h] (相对 model_input_size 的归一化坐标) 
    映射回“原图坐标系”的 [x1, y1, x2, y2]（含对 pad->resize 的逆映射与裁剪）。
    """
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

# ---------------- MobileSAM 初始化/分割 ----------------
@torch.no_grad()
def init_mobilesam(device: torch.device):
    sam = sam_model_registry[MOBILE_SAM_TYPE](checkpoint=MOBILE_SAM_CKPT)
    # **建议默认 FP32 更稳**
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
    """
    自适应批量分割：每个批次如果 OOM 就减半重试；若最终 batch=1 仍 OOM 且允许，则切 CPU。
    返回：
      masks_bool_list: List[np.ndarray(H,W)]
      confs_list:      List[float]
    """
    # 确保图像是连续内存，避免部分版本下的奇怪行为
    image_rgb_u8 = np.ascontiguousarray(image_rgb_u8)

    H, W = image_rgb_u8.shape[:2]
    device = next(predictor.model.parameters()).device

    def _predict_on_device(dev: torch.device, batch_boxes: torch.Tensor):
        # 关闭 autocast（默认），防止 half 精度导致的奇异现象；如需，可把 USE_AUTOCast_INFER 设为 True
        autocast_enabled = (dev.type == "cuda") and USE_AUTOCast_INFER
        with torch.autocast(device_type=dev.type, enabled=autocast_enabled):
            m, s, _ = predictor.predict_torch(
                point_coords=None, point_labels=None,
                boxes=batch_boxes, multimask_output=multimask_output
            )
        # 有些版本的 MobileSAM 返回的 m 不是 (H,W)，需要 postprocess 回原图尺寸
        if m.shape[-2:] != (H, W):
            # 兼容 official SAM 的后处理接口
            try:
                m = predictor.postprocess_masks(m, predictor.input_size, (H, W))
            except Exception:
                # 保底：双线性插值回原图大小
                m = torch.nn.functional.interpolate(
                    m, size=(H, W), mode="bilinear", align_corners=False
                )

        if multimask_output:
            best = torch.argmax(s, dim=1)          # (B,)
            m = m[torch.arange(m.shape[0]), best]  # (B,H,W)
            s = s.max(dim=1).values                # (B,)
        else:
            m = m[:, 0]                             # (B,H,W)
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
                masks_cpu = m.detach().to("cpu").float()  # 转 float 再阈值
                # 阈值转换为 bool（有时 half->bool 直接 cast 会不稳定）
                masks_cpu = (masks_cpu > 0.5).to(torch.bool).numpy()
                confs_cpu = s.detach().to("cpu").float().numpy().tolist()
                for k in range(masks_cpu.shape[0]):
                    # **拷贝**，避免未来视图被意外覆盖
                    masks_out.append(masks_cpu[k].copy())
                    confs_out.append(float(confs_cpu[k]))
                i = j
                del m, s, masks_cpu
                if EMPTY_CACHE_EVERY_CHUNK and dev.type == "cuda":
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
                            print("[WARN] CUDA OOM at batch=1, falling back to CPU for remaining boxes.")
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
        # 回退：无 predict_torch 接口，逐框（仍只 set_image 一次）
        for box in boxes_xyxy:
            m, s, _ = predictor.predict(
                box=np.array(box, dtype=np.float32)[None, :],
                multimask_output=multimask_output
            )
            if m.shape[-2:] != (H, W):
                # numpy 路径一般已是原图大小；做一次健壮性处理
                from skimage.transform import resize as _resize
                m = _resize(m, (m.shape[0], 1, H, W), order=1, mode="reflect", anti_aliasing=True)
            if multimask_output:
                idx = int(np.argmax(s))
                masks_out.append((m[idx] > 0.5).astype(bool).copy())
                confs_out.append(float(s[idx]))
            else:
                masks_out.append((m[0] > 0.5).astype(bool).copy())
                confs_out.append(float(np.squeeze(s)))
    return masks_out, confs_out

# ---------------- OWL 检测 ----------------
def owl_detect(
    image_rgb_u8: np.ndarray,
    module, variables, jitted_apply,
    tokenized_queries: np.ndarray,
    input_size: int,
    score_threshold: float
) -> List[Dict[str, Any]]:
    """
    对单张图做 OWL-ViT 检测，返回每个候选的:
      {
        "box": [x1,y1,x2,y2],       # 映射回原图
        "detection": [{"label": str, "score": float}, ...]  # 各类得分，按降序
      }
    """
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
    boxes_xywh = preds["pred_boxes"]                          # [N,4]
    logits = preds["pred_logits"][..., :len(OBJECTS)]         # [N,C]

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

# ---------------- 主流程 ----------------
def process_image(
    file_path: str,
    predictor: SamPredictor,
    mdl, variables, jitted, tokenized, owl_input_size: int
) -> Dict[str, Any]:
    """
    对单张图片跑完整 pipeline，返回与示例一致的 JSON dict:
      {"detections": [ { "detection": [...], "box": [...], "mask": "..." }, ... ]}
    """
    img_u8 = skio.imread(file_path)
    if img_u8.ndim == 2:  # 灰度 -> RGB
        img_u8 = np.stack([img_u8]*3, axis=-1)
    if img_u8.shape[-1] == 4:  # 丢 alpha
        img_u8 = img_u8[..., :3]
    img_u8 = img_u8.astype(np.uint8)

    start = time.time()

    # 1) OWL 检测（仅保留 detector 自身的 score 阈值，不做额外筛选）
    candidates = owl_detect(
        img_u8, mdl, variables, jitted, tokenized,
        input_size=owl_input_size,
        score_threshold=SCORE_THRESHOLD
    )

    results = {"detections": []}
    if len(candidates) == 0:
        print(f"[OK] {os.path.basename(file_path)} -> 0 masks  (0.00s)")
        return results

    # 2) SAM 批量分割（自适应；不做任何 NMS/Top-K/面积过滤）
    boxes = [c["box"] for c in candidates]
    masks_list, confs = sam_segment_boxes_adaptive(
        predictor, img_u8, boxes,
        multimask_output=SAM_MULTIMASK_OUTPUT,
        init_batch_size=SAM_INIT_BATCH_SIZE,
        cpu_fallback=CPU_FALLBACK_ON_OOM
    )

    # 3) 组装 JSON
    for cand, mask in zip(candidates, masks_list):
        results["detections"].append({
            "detection": cand["detection"],
            "box": cand["box"],
            "mask": mask_to_base64_png(mask)
        })

    # 清理
    del masks_list, confs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    dur = time.time() - start
    print(f"[OK] {os.path.basename(file_path)} -> {len(results['detections'])} masks  ({dur:.2f}s)")
    return results

def main():
    input_folder = os.path.join(DATASET_NAME, INPUT_DIR_NAME)
    output_folder = os.path.join(DATASET_NAME, OUTPUT_DIR_TAG)
    ensure_dir(output_folder)

    # ---- 初始化 OWL-ViT ----
    cfg, mdl, variables, jitted, tokenized = build_owl()
    owl_input_size = int(cfg.dataset_configs.input_size)

    # ---- 初始化 MobileSAM ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = init_mobilesam(device)

    # 遍历图片
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    for fname in files:
        base, _ = os.path.splitext(fname)
        index = base.split("_")[-1] if "_" in base else base

        img_path = os.path.join(input_folder, fname)
        result = process_image(
            img_path, predictor, mdl, variables, jitted, tokenized,
            owl_input_size=owl_input_size
        )

        json_save_path = os.path.join(output_folder, f"detection_{index}.json")
        with open(json_save_path, "w") as f:
            json.dump(result, f, indent=2)

        # 每张图后做一次清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()

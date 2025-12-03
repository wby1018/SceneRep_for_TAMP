import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.restoration import inpaint_biharmonic

def inpaint_color_pyramid(color: np.ndarray, hole_mask: np.ndarray, scale: float = 0.1,
                          radius: int = 9, method: str = 'telea') -> np.ndarray:
    """
    先缩小图像 → inpaint → 双线性放大 → 填回原图
    参数:
        color     - 输入RGB图 (H,W,3) uint8
        hole_mask - bool型洞mask (True表示洞)
        scale     - 缩小比例 (0~1)，越小速度越快
        radius    - inpaint半径（在缩小后的图上）
        method    - 'telea' 或 'ns'
    """
    h, w = color.shape[:2]

    # 1. 缩小图像和mask
    small_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    color_small = cv2.resize(color, small_size, interpolation=cv2.INTER_NEAREST)
    mask_small = cv2.resize(hole_mask.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST)

    # 2. inpaint 小图
    inpaint_flag = cv2.INPAINT_TELEA if method.lower() == 'telea' else cv2.INPAINT_NS
    color_small_filled = cv2.inpaint(color_small, mask_small * 255, float(radius), inpaint_flag)

    # 3. 双线性放大回原图尺寸
    color_up = cv2.resize(color_small_filled, (w, h), interpolation=cv2.INTER_LINEAR)

    # 4. 用放大结果填回洞的位置
    result = color.copy()
    result[hole_mask] = color_up[hole_mask]

    return result

def inpaint_depth_biharmonic(depth: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    bad = hole_mask | ~np.isfinite(d) | (d <= 0)
    if not np.any(bad):
        return d

    valid = ~bad
    v = d[valid]
    lo, hi = (np.min(v), np.max(v)) if v.size > 0 else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0

    # 归一化到 [0,1]
    dn = np.zeros_like(d, np.float32)
    dn[valid] = (d[valid]-lo)/(hi-lo)

    # skimage 内部解方程（无 python for）
    repaired = inpaint_biharmonic(dn, bad)
    out = lo + repaired*(hi-lo)
    # 保留原有效像素
    out[valid] = d[valid]
    return out.astype(np.float32)

def inpaint_depth_fgs(depth: np.ndarray, hole_mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    bad = hole_mask | ~np.isfinite(d) | (d <= 0)
    if not np.any(bad):
        return d
    _, (iy, ix) = ndi.distance_transform_edt(bad, return_indices=True)
    filled = d.copy(); filled[bad] = d[iy[bad], ix[bad]]

    if not hasattr(cv2, "ximgproc"):
        return filled  # 无 ximgproc 时直接返回最近邻填充

    # FGS：lambda 较小保细节，sigma 较小贴边
    guide = rgb.astype(np.uint8)
    fgs = cv2.ximgproc.createFastGlobalSmootherFilter(guide, lambda_=12.0, sigma_color=1.5)
    smoothed = fgs.filter(filled)
    smoothed[~bad] = d[~bad]
    return smoothed.astype(np.float32)

def inpaint_depth_nn_jbf(
    depth: np.ndarray,
    hole_mask: np.ndarray,
    rgb: np.ndarray=None,
    jbf_d: int=9,
    jbf_sigma_color: float=12.0,
    jbf_sigma_space: float=7.0,
    gf_radius: int=5,
    gf_eps: float=1e-3,
) -> np.ndarray:
    d = depth.astype(np.float32)
    bad = hole_mask | ~np.isfinite(d) | (d <= 0)
    if not np.any(bad):
        return d

    valid = ~bad
    if not np.any(valid):
        return np.zeros_like(d, np.float32)

    # 最近邻“必然填满”（向量化）
    _, (iy, ix) = ndi.distance_transform_edt(bad, return_indices=True)
    filled = d.copy()
    filled[bad] = d[iy[bad], ix[bad]]

    # 保边细化（全部在C/C++实现里跑）
    out = filled
    if rgb is not None:
        if hasattr(cv2, "ximgproc"):
            try:
                out = cv2.ximgproc.jointBilateralFilter(rgb, out, d=jbf_d,
                                                        sigmaColor=jbf_sigma_color,
                                                        sigmaSpace=jbf_sigma_space)
            except Exception:
                # guided filter（需要 ximgproc.guidedFilter）
                g = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
                out = cv2.ximgproc.guidedFilter(g, out, gf_radius, gf_eps)
        else:
            # 无ximgproc时，退化为对深度自身做双边
            out = cv2.bilateralFilter(out, d=jbf_d,
                                      sigmaColor=jbf_sigma_color,
                                      sigmaSpace=jbf_sigma_space)
    else:
        out = cv2.bilateralFilter(out, d=jbf_d,
                                  sigmaColor=jbf_sigma_color,
                                  sigmaSpace=jbf_sigma_space)

    # 保留原始有效像素
    out[valid] = d[valid]
    return out.astype(np.float32)

def inpaint_depth_pyramid(depth: np.ndarray, hole_mask: np.ndarray, scale: float = 0.1,
                          radius: int = 9, method: str = 'telea') -> np.ndarray:
    """
    先缩小 depth → inpaint → 双线性放大 → 填回原图
    depth: float32 (米或毫米)，NaN 或 <=0 为洞
    """
    h, w = depth.shape
    depth_f32 = depth.astype(np.float32)

    # 保存原始有效值范围
    valid_mask = ~hole_mask
    if np.any(valid_mask):
        min_val = np.nanmin(depth_f32[valid_mask])
        max_val = np.nanmax(depth_f32[valid_mask])
    else:
        min_val, max_val = 0.0, 1.0

    # 转成 8-bit 用于 inpaint
    norm_depth = np.zeros_like(depth_f32, dtype=np.uint8)
    if max_val > min_val:
        norm_depth[valid_mask] = np.clip(
            255 * (depth_f32[valid_mask] - min_val) / (max_val - min_val),
            0, 255
        ).astype(np.uint8)

    # 1. 缩小 depth 和 mask
    small_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    depth_small = cv2.resize(norm_depth, small_size, interpolation=cv2.INTER_NEAREST)
    mask_small = cv2.resize(hole_mask.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST)

    # 2. inpaint 小图
    inpaint_flag = cv2.INPAINT_TELEA if method.lower() == 'telea' else cv2.INPAINT_NS
    depth_small_filled = cv2.inpaint(depth_small, mask_small * 255, float(radius), inpaint_flag)

    # 3. 双线性放大回原图
    depth_up = cv2.resize(depth_small_filled, (w, h), interpolation=cv2.INTER_LINEAR)

    # 4. 反归一化回 float32
    depth_up_f32 = min_val + (depth_up.astype(np.float32) / 255.0) * (max_val - min_val)

    # 5. 用填充结果更新原 depth
    result = depth_f32.copy()
    result[hole_mask] = depth_up_f32[hole_mask]

    return result.astype(np.float32)


def inpaint_depth_fast(depth, hole_mask, rgb=None, mode="nn_jbf"):
    # 1) 仍然可以保留你的下采样/上采样金字塔以提速（都是 OpenCV 底层实现）
    h, w = depth.shape
    scale = 1
    small = cv2.resize(depth.astype(np.float32), (int(w*scale), int(h*scale)),
                       interpolation=cv2.INTER_NEAREST)
    mask_small = cv2.resize(hole_mask.astype(np.uint8), (small.shape[1], small.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    rgb_small = None if rgb is None else cv2.resize(rgb, (small.shape[1], small.shape[0]),
                                                    interpolation=cv2.INTER_LINEAR)

    # 2) 小图修补（任选其一，均无 python for）
    if mode == "biharmonic":
        small_filled = inpaint_depth_biharmonic(small, mask_small)
    elif mode == "fgs" and rgb is not None:
        small_filled = inpaint_depth_fgs(small, mask_small, rgb_small)
    else:
        small_filled = inpaint_depth_nn_jbf(small, mask_small, rgb=rgb_small)

    # 3) 放大回原分辨率并仅更新洞区
    up = cv2.resize(small_filled, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = depth.astype(np.float32).copy()
    out[hole_mask] = up[hole_mask]
    return out


def inpaint_background(
    depth_bg: np.ndarray,
    color_bg: np.ndarray,
    none_mask,
    hand_mask
) -> tuple[np.ndarray, np.ndarray]:
    """
    深度：优先 Fast Bilateral Solver（有置信度的边缘保持传播）
         失败则 Guided Filter 归一化方案
         再失败则 Domain Transform 做兜底
    颜色：OpenCV inpaint
    """
    depth_bg[hand_mask > 0] = 0
    hole_mask = ((np.isnan(depth_bg)) | (depth_bg <= 0)) & (~none_mask)
    color_filled = inpaint_color_pyramid(color_bg, hole_mask)
    depth_fill = inpaint_depth_fast(depth_bg, hole_mask, color_filled, mode="nn_jbf")
    
    
    return depth_fill.astype(np.float32), color_filled

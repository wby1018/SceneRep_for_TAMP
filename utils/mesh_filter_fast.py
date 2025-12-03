"""
百分位裁剪盒 Mesh 过滤
沿 x/y/z 三轴分别去掉最远和最近的少量点，只保留盒内顶点；同步更新 F/N/C。
"""

import numpy as np
from typing import Tuple

EMPTY = (
    np.empty((0, 3), np.float32),
    np.empty((0, 3), np.int32),
    np.empty((0, 3), np.float32),
    np.empty((0, 3), np.uint8),
)

def filter_mesh_by_percentile_box(
    V: np.ndarray, F: np.ndarray, N: np.ndarray, C: np.ndarray, trim_ratio
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    基于百分位包围盒过滤 Mesh：
      对 V 在 x/y/z 三个维度分别计算下/上百分位阈值（按 trim_ratio），
      仅保留位于该盒内的顶点，丢弃盒外顶点；同步更新 F/N/C 并重映射面片索引。

    Args:
        V: 顶点坐标 (Nv, 3), float32
        F: 三角面索引 (Nf, 3), int32
        N: 顶点法向量 (Nv, 3), float32
        C: 顶点颜色 (Nv, 3), uint8
        trim_ratio: (0~0.49) 每个维度两端裁剪的比例，如 0.02 表示裁掉最小/最大各 2%

    Returns:
        (V_new, F_new, N_new, C_new)
          - V_new: 保留后的顶点
          - F_new: 仅由保留顶点组成的三角面（索引已重映射）
          - N_new: 对应保留顶点的法向
          - C_new: 对应保留顶点的颜色
        若过滤后无有效顶点或无有效三角面，返回全空数组（形状 (0,3)）。
    """
    # 统一 dtype（以防调用方传入其他类型）
    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.int32)
    N = np.asarray(N, dtype=np.float32)
    C = np.asarray(C, dtype=np.uint8)

    if V.size == 0:
        return EMPTY

    # 约束 trim_ratio 合理范围
    trim_ratio = float(trim_ratio)
    if trim_ratio < 0.0:
        trim_ratio = 0.0
    if trim_ratio >= 0.5:
        trim_ratio = 0.49  # 避免把所有点都裁掉

    # 计算每个轴上的下/上百分位阈值
    # 如果顶点非常少，np.percentile 仍可用；当 trim_ratio=0 时等价于不裁掉两端
    p_lo = 100.0 * trim_ratio
    p_hi = 100.0 * (1.0 - trim_ratio)

    lo = np.percentile(V, p_lo, axis=0, interpolation='linear')
    hi = np.percentile(V, p_hi, axis=0, interpolation='linear')

    mid = (hi+lo)/2
    d = (hi-lo)/2
    lo = mid-d*(1+trim_ratio+0.1)
    hi = mid+d*(1+trim_ratio+0.1)
    

    # 盒内点掩码（含边界）
    mask = (V[:, 0] >= lo[0]) & (V[:, 0] <= hi[0]) & \
           (V[:, 1] >= lo[1]) & (V[:, 1] <= hi[1]) & \
           (V[:, 2] >= lo[2]) & (V[:, 2] <= hi[2])

    if not np.any(mask):
        return EMPTY

    # 旧 → 新 索引映射：被保留的顶点按原顺序压紧
    old_to_new = -np.ones(V.shape[0], dtype=np.int64)
    old_to_new[mask] = np.arange(mask.sum(), dtype=np.int64)

    # 过滤并重映射 F：仅保留三个顶点都在 mask 内的三角面
    if F.size == 0:
        # 没有面，直接裁顶点即可
        V_new = V[mask]
        N_new = N[mask] if N.shape[0] == V.shape[0] else np.empty((V_new.shape[0], 3), np.float32)
        C_new = C[mask] if C.shape[0] == V.shape[0] else np.empty((V_new.shape[0], 3), np.uint8)
        return V_new, np.empty((0, 3), np.int32), N_new, C_new

    # 面有效性：三个顶点索引都在保留集合内（映射值非 -1）
    valid_faces_mask = (old_to_new[F] >= 0).all(axis=1)
    if not np.any(valid_faces_mask):
        # 虽然顶点保留了一些，但没有完整三角面留下；此时可按需要选择返回仅顶点或全空。
        # 为避免后续渲染/算法对“无面”的处理复杂，这里返回全空。
        return EMPTY

    F_kept = F[valid_faces_mask]
    F_new = old_to_new[F_kept].astype(np.int32, copy=False)

    # 压紧 V/N/C
    V_new = V[mask]
    N_new = N[mask] if N.shape[0] == V.shape[0] else np.empty((V_new.shape[0], 3), np.float32)
    C_new = C[mask] if C.shape[0] == V.shape[0] else np.empty((V_new.shape[0], 3), np.uint8)

    return V_new, F_new, N_new, C_new


# 可选：将“快速过滤入口”只保留这一种方法，兼容你的旧调用点
def filter_mesh_fast(
    V: np.ndarray, F: np.ndarray, N: np.ndarray, C: np.ndarray, trim_ratio
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    快速 Mesh 过滤（仅百分位包围盒方法）
    """
    return filter_mesh_by_percentile_box(V, F, N, C, trim_ratio=trim_ratio)
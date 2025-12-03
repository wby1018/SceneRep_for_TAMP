from gettext import find
import numpy as np
from utils.utils import find_object_by_id
import open3d as o3d
from utils.mesh_filter_fast import filter_mesh_fast

# def gravity_simulation(obj_id_in_ee, objects, bg_tsdf):
#     obj_in_ee = find_object_by_id(obj_id_in_ee, objects)
#     V_obj,F,N,C = obj_in_ee.tsdf.get_mesh()
#     V_land,F,N,C = None
#     if obj_in_ee.parent_obj_id is not None:
#         landing_target = find_object_by_id(obj_in_ee.parent_obj_id, objects)
#         V_land,F,N,C = landing_target.tsdf.get_mesh()
#     else:
#         V_land,F,N,C = bg_tsdf.get_mesh()
#         # 在 XY 平面上取 obj_in_ee 的包围盒
#     min_x, max_x = float(np.min(V_obj[:, 0])), float(np.max(V_obj[:, 0]))
#     min_y, max_y = float(np.min(V_obj[:, 1])), float(np.max(V_obj[:, 1]))

#     # 在降落地点中选取 XY 落在该包围盒范围内的顶点
#     xs, ys, zs = V_land[:, 0], V_land[:, 1], V_land[:, 2]
#     in_x = (xs >= min_x) & (xs <= max_x)
#     in_y = (ys >= min_y) & (ys <= max_y)
#     mask = in_x & in_y
#     if np.any(mask):
#         target_height = float(np.max(zs[mask]))
#     else:
#         # 若没有命中点，退化为降落地点整体最高点
#         target_height = float(np.max(zs))
#     return

def vertical_hit_height_o3d(scene, device, origin_xyz):
    """
    用 Open3D RaycastingScene 在 (cx,cy,z_start) 处向 -Z 发射射线，返回命中高度或 None
    """

    # 一条垂直向下的射线
    origin = np.asarray(origin_xyz, dtype=np.float32).reshape(1, 3)
    direction = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)  # 向下
    rays = np.concatenate([origin, direction], axis=1)  # (1,6): ox,oy,oz, dx,dy,dz
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32, device=device)

    ans = scene.cast_rays(rays_t)  # dict: t_hit, prim_id, geometry_ids, ...
    t_hit = ans["t_hit"].numpy()[0]  # 距离；inf 表示未命中
    if not np.isfinite(t_hit):
        return None

    # 命中点 z = oz + t * dz；此处 dz = -1
    oz = origin[0, 2]
    hit_z = float(oz + t_hit * (-1.0))
    return hit_z


def gravity_simulation(obj_id_in_ee, objects, bg_tsdf):
    obj_in_ee = find_object_by_id(obj_id_in_ee, objects)
    obj_in_ee.to_be_repaint = True
    T_obj = obj_in_ee.pose_cur @ np.linalg.inv(obj_in_ee.pose_init)
    # print(obj_in_ee.pose_init)
    # print(obj_in_ee.pose_cur)
    V_obj, F_obj, N_obj, C_obj = obj_in_ee.tsdf.get_mesh()
    # print(f"current_height{np.min(V_obj[:, 2])}")
    # print(f"current_top{np.max(V_obj[:, 2])}")
    V_obj, F_obj, N_obj, C_obj = filter_mesh_fast(V_obj, F_obj, N_obj, C_obj, trim_ratio=0.05)
    V_array = np.array(V_obj) if isinstance(V_obj, list) else V_obj
    V_homogeneous = np.hstack([V_array, np.ones((len(V_array), 1))])
    V_obj = (T_obj @ V_homogeneous.T).T[:, :3]
    # print(f"current_height{np.min(V_obj[:, 2])}")
    # print(f"current_top{np.max(V_obj[:, 2])}")

    if obj_in_ee.parent_obj_id is not None:
        landing_target = find_object_by_id(obj_in_ee.parent_obj_id, objects)
        T_land = landing_target.pose_cur @ np.linalg.inv(landing_target.pose_init)
        V_land, F_land, _, _ = landing_target.tsdf.get_mesh()
        V_land, F_land, _, _ = filter_mesh_fast(V_land, F_land, _, _,trim_ratio=0.05)
        V_array = np.array(V_land) if isinstance(V_land, list) else V_land
        V_homogeneous = np.hstack([V_array, np.ones((len(V_array), 1))])
        V_land = (T_land @ V_homogeneous.T).T[:, :3]
    else:
        V_land, F_land, _, _ = bg_tsdf.get_mesh()

    if V_obj is None or V_obj.size == 0 or V_land is None or V_land.size == 0 or F_land.size == 0:
        return

    if V_land.size == 0 or F_land.size == 0:
        return None

    # 构建 t::TriangleMesh（一次构建可复用；若频繁调用，建议把 scene 缓存起来）
    device = o3d.core.Device("CPU:0")
    mesh_t = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(V_land.astype(np.float32), device=device),
        o3d.core.Tensor(F_land.astype(np.int64),   device=device)
    )
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    # 射线起点：以物体 XY 质心、Z 从物体最高点稍上方开始
    cx = float(np.mean(V_obj[:, 0]))
    cy = float(np.mean(V_obj[:, 1]))
    z_start = float(np.max(V_obj[:, 2]) + 1e-3)

    # 在中心附近圆周上再取八个点，连同中心共九个点，取命中高度的中位数
    range_x = float(np.max(V_obj[:, 0]) - np.min(V_obj[:, 0]))
    range_y = float(np.max(V_obj[:, 1]) - np.min(V_obj[:, 1]))
    radius = 0.2 * min(range_x, range_y) + 1e-6
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    sample_xy = [(cx, cy)] + [(cx + radius * float(np.cos(a)), cy + radius * float(np.sin(a))) for a in angles]

    hit_heights = []
    for sx, sy in sample_xy:
        hz = vertical_hit_height_o3d(scene, device, (sx, sy, z_start))
        print(f"heights:{hz}")
        if hz is not None:
            hit_heights.append(float(hz))

    offset = 0.01
    if len(hit_heights) > 0:
        target_height = float(np.median(hit_heights))+offset
    else:
        print("failed to find landing height at parent object, retry falling to bg")
        
        # 回退到背景TSDF寻找着陆点
        try:
            V_bg, F_bg, _, _ = bg_tsdf.get_mesh()
            if V_bg is not None and V_bg.size > 0 and F_bg.size > 0:
                # 在背景上寻找着陆高度
                hit_z_bg = vertical_hit_height_o3d(V_bg, F_bg, (cx, cy, z_start))
                if hit_z_bg is not None:
                    target_height = hit_z_bg
                    print(f"found landing height on background: {target_height}")
                else:
                    print("failed to find landing height on background as well")
                    return
            else:
                print("background TSDF mesh is empty")
                return
        except Exception as e:
            print(f"error in falling to background: {e}")
            return

    # 根据 target_height 更新物体落地姿态
    current_height = float(np.min(V_obj[:, 2]))
    print(f"current_height{current_height}")
    # print(f"current_top{np.max(V_obj[:, 2])}")
    print(f"target_height{target_height}")
    if current_height-target_height > 0 and current_height-target_height < 0.05:
        print("drop object to target height")
        obj_in_ee.pose_cur[2, 3] = obj_in_ee.pose_cur[2, 3] - (current_height-target_height)
        if obj_in_ee.parent_obj_id is not None:
            parent_obj = find_object_by_id(obj_in_ee.parent_obj_id, objects)
            if parent_obj is not None:
                parent_obj.child_objs[obj_in_ee.id] = np.linalg.inv(parent_obj.pose_cur) @ obj_in_ee.pose_cur
        if obj_in_ee.child_objs is not None:
            for related_id, T in obj_in_ee.child_objs.items():
                child_obj = find_object_by_id(related_id, objects)
                child_obj.pose_cur = obj_in_ee.pose_cur @ T
                child_obj.to_be_repaint = True
    else:
        print("target height higher than current height or too low")
        return
    return

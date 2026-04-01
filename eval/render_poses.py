#!/usr/bin/env python3
import os
import cv2
import numpy as np
import base64
import io
import json
from scipy.spatial.transform import Rotation
from eval_all import PoseEvaluator  # Important: Reuse existing parser logic

class PoseRenderer(PoseEvaluator):
    def __init__(self, dataset_dir):
        # Prevent initialization of global_csv_path in super() if not needed, 
        # but calling super().__init__() is safest.
        super().__init__(dataset_dir)
        
    def render(self, object_id, object_name, output_base_dir):
        # Update IDs based on user input
        self.object_id = object_id
        self.object_name = object_name
        
        self.eval_file = os.path.join(self.dataset_dir, "eval", f"object_{object_id}.txt")
        self.foundation_file = os.path.join(self.dataset_dir, "eval_foundationpose_comp", f"object_{object_id}.txt")
        self.bundle_sdf_file = os.path.join(self.dataset_dir, "eval_bundlesdf_comp", f"object_{object_id}.txt")
        self.midfusion_file = os.path.join(self.dataset_dir, "eval_midfusion", f"object_{object_id}.txt")
        self.tsdfpp_file = os.path.join(self.dataset_dir, "eval_tsdfpp_comp", f"object_{object_id}.txt")
        
        # Load all pose lists
        self.estimated_poses, self.evaluation_segments = self.read_estimated_poses()
        self.foundation_poses = self.read_foundation_poses()
        self.bundle_sdf_poses = self.read_bundle_sdf_poses()
        self.midfusion_poses = self.read_midfusion_poses()
        self.tsdfpp_poses = self.read_tsdfpp_poses()
        
        # Output dirs setup
        methods_colors = {
            "gt": (0, 255, 0),             # Green
            "ours": (0, 0, 255),           # Red
            "foundationpose": (255, 0, 0), # Blue
            "bundlesdf": (0, 255, 255),    # Yellow
            "midfusion": (255, 0, 255),    # Magenta
            "tsdfpp": (255, 128, 0),       # Orange
        }
        
        for k in methods_colors.keys():
            os.makedirs(os.path.join(output_base_dir, k), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, "combined"), exist_ok=True)

        # Camera Intrinsics
        fx, fy = 554.3827, 554.3827
        cx, cy = 320.5, 240.5
        
        def project_points(points_3d):
            """ Project 3D points [N, 3] to 2D pixels [N, 2] """
            Z = points_3d[:, 2]
            valid = Z > 0
            points_3d = points_3d[valid]
            Z = Z[valid]
            u = (points_3d[:, 0] * fx / Z) + cx
            v = (points_3d[:, 1] * fy / Z) + cy
            return np.stack([u, v], axis=1).astype(np.int32)

        def draw_points(img, points_2d, color, radius=1):
            out = img.copy()
            for p in points_2d:
                if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
                    cv2.circle(out, (p[0], p[1]), radius, color, -1)
            return out
            
        def add_label(img, text, color=(255, 255, 255)):
            out = img.copy()
            cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return out

        def filter_points_percentile(points_3d, lower_p=2.0, upper_p=98.0):
            if points_3d is None or len(points_3d) < 10:
                return points_3d
            # Filter based on percentile distances along each axis to robustly remove outliers
            p_low = np.percentile(points_3d, lower_p, axis=0)
            p_high = np.percentile(points_3d, upper_p, axis=0)
            
            mask = (
                (points_3d[:, 0] >= p_low[0]) & (points_3d[:, 0] <= p_high[0]) &
                (points_3d[:, 1] >= p_low[1]) & (points_3d[:, 1] <= p_high[1]) &
                (points_3d[:, 2] >= p_low[2]) & (points_3d[:, 2] <= p_high[2])
            )
            return points_3d[mask]

        if not self.has_offset:
            # Note: evaluate() checks this but hasn't run. We use default offset as eval_all does for fast test.
            self.has_offset = True

        for i, segment in enumerate(self.evaluation_segments):
            print(f"Renderer: segment {i+1}/{len(self.evaluation_segments)}")
            if len(segment) == 0: continue
            
            first_frame_idx = segment[0]
            # Compute transforming matrix from the first frame 
            self.compute_transformation_matrix(first_frame_idx)
            obj_points_ours = self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None
            
            # Save the transformations since subsequent compute_transformation will overwrite them
            mocap_obj_trans = self.obj_transformation.copy() if hasattr(self, 'obj_transformation') else np.eye(4)
            mocap_cam_trans = self.camera_transformation.copy() if hasattr(self, 'camera_transformation') else np.eye(4)
            
            # --- Initialize FoundationPose ---
            if first_frame_idx in self.foundation_poses:
                self.compute_transformation_matrix_foundation_pose(first_frame_idx)
                obj_points_fp = self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None
            else:
                obj_points_fp = None
                
            # --- Initialize BundleSDF ---
            if first_frame_idx in self.bundle_sdf_poses:
                self.compute_transformation_matrix_bundle_sdf(first_frame_idx)
                obj_points_bs = self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None
            else:
                obj_points_bs = None
                
            # --- Initialize MidFusion ---
            if first_frame_idx in self.midfusion_poses:
                self.compute_transformation_matrix_midfusion(first_frame_idx)
                obj_points_mid = self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None
            else:
                obj_points_mid = None
                
            # --- Initialize TSDF++ ---
            if first_frame_idx in self.tsdfpp_poses:
                self.compute_transformation_matrix_tsdfpp(first_frame_idx)
                obj_points_tsdfpp = self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None
            else:
                obj_points_tsdfpp = None
            
            if obj_points_ours is None:
                print("Failed to get object point cloud from first frame, skipping segment.")
                continue
            # Apply Point Cloud filtering to base points
            if obj_points_ours is not None: obj_points_ours = filter_points_percentile(obj_points_ours)
            if obj_points_fp is not None: obj_points_fp = filter_points_percentile(obj_points_fp)
            if obj_points_bs is not None: obj_points_bs = filter_points_percentile(obj_points_bs)
            if obj_points_mid is not None: obj_points_mid = filter_points_percentile(obj_points_mid)
            if obj_points_tsdfpp is not None: obj_points_tsdfpp = filter_points_percentile(obj_points_tsdfpp)
            
            nearest_idx = None
            
            for frame_idx in segment:
                rgb_path = os.path.join(self.dataset_dir, "rgb", f"rgb_{frame_idx:06d}.png")
                if not os.path.exists(rgb_path):
                    continue
                
                rgb_img = cv2.imread(rgb_path)
                
                # Fetch Mocap nearest idx
                nearest_idx, _ = self.find_nearest_mocap_idx(
                    self.estimated_poses[frame_idx]['timestamp'], nearest_idx
                )
                if nearest_idx is None:
                    continue
                
                # --- GT Pose Mapping --- 
                mocap_obj_pose, mocap_cam_pose = self.extract_mocap_pose(nearest_idx)
                gt_obj_pose = self.mocap_robot @ mocap_obj_pose @ mocap_obj_trans
                gt_cam_pose = self.mocap_robot @ mocap_cam_pose @ mocap_cam_trans
                est_cam_pose = self.camera_poses.get(frame_idx, np.eye(4))
                
                gt_obj_pose_cam = np.linalg.inv(gt_cam_pose) @ gt_obj_pose
                points_gt = self.transform_points(obj_points_ours, gt_obj_pose_cam)
                
                img_gt = draw_points(rgb_img, project_points(points_gt), methods_colors["gt"])
                cv2.imwrite(os.path.join(output_base_dir, "gt", f"frame_{frame_idx:06d}.png"), img_gt)
                
                # --- Ours Pose Mapping --- 
                img_ours = rgb_img.copy()
                if frame_idx in self.estimated_poses and obj_points_ours is not None:
                    our_pose = self.estimated_poses[frame_idx]['transform']
                    our_pose_cam = np.linalg.inv(est_cam_pose) @ our_pose
                    points_ours = self.transform_points(obj_points_ours, our_pose_cam)
                    img_ours = draw_points(rgb_img, project_points(points_ours), methods_colors["ours"])
                cv2.imwrite(os.path.join(output_base_dir, "ours", f"frame_{frame_idx:06d}.png"), img_ours)
                
                # --- FoundationPose ---
                img_fp = rgb_img.copy()
                if frame_idx in self.foundation_poses and obj_points_fp is not None:
                    fp_pose = self.foundation_poses[frame_idx]['transform']
                    # Evaluator logic: est_pose = est_cam_pose @ fp_pose -> cam space logic: est_pose_cam = fp_pose
                    fp_pose_cam = fp_pose
                    points_fp = self.transform_points(obj_points_fp, fp_pose_cam)
                    img_fp = draw_points(rgb_img, project_points(points_fp), methods_colors["foundationpose"])
                cv2.imwrite(os.path.join(output_base_dir, "foundationpose", f"frame_{frame_idx:06d}.png"), img_fp)
                
                # --- BundleSDF ---
                img_bs = rgb_img.copy()
                if frame_idx in self.bundle_sdf_poses and obj_points_bs is not None:
                    bs_pose = self.bundle_sdf_poses[frame_idx]['transform']
                    bs_pose_cam = bs_pose
                    points_bs = self.transform_points(obj_points_bs, bs_pose_cam)
                    img_bs = draw_points(rgb_img, project_points(points_bs), methods_colors["bundlesdf"])
                cv2.imwrite(os.path.join(output_base_dir, "bundlesdf", f"frame_{frame_idx:06d}.png"), img_bs)
                
                # --- MidFusion ---
                img_mid = rgb_img.copy()
                if frame_idx in self.midfusion_poses and obj_points_mid is not None:
                    mid_pose = self.midfusion_poses[frame_idx]['transform']
                    mid_pose_cam = np.linalg.inv(est_cam_pose) @ mid_pose
                    points_mid = self.transform_points(obj_points_mid, mid_pose_cam)
                    img_mid = draw_points(rgb_img, project_points(points_mid), methods_colors["midfusion"])
                cv2.imwrite(os.path.join(output_base_dir, "midfusion", f"frame_{frame_idx:06d}.png"), img_mid)
                
                # --- TSDF++ ---
                img_tsdfpp = rgb_img.copy()
                if frame_idx in self.tsdfpp_poses and obj_points_tsdfpp is not None:
                    tsdfpp_pose = self.tsdfpp_poses[frame_idx]['transform']
                    tsdfpp_pose_cam = np.linalg.inv(est_cam_pose) @ tsdfpp_pose
                    points_tsdfpp = self.transform_points(obj_points_tsdfpp, tsdfpp_pose_cam)
                    img_tsdfpp = draw_points(rgb_img, project_points(points_tsdfpp), methods_colors["tsdfpp"])
                cv2.imwrite(os.path.join(output_base_dir, "tsdfpp", f"frame_{frame_idx:06d}.png"), img_tsdfpp)
                
                # --- Combined Plot ---
                h, w = rgb_img.shape[:2]
                combined_canvas = np.zeros((h*2, w*3, 3), dtype=np.uint8)
                
                combined_canvas[0:h, 0:w] = add_label(img_gt, "GT", methods_colors['gt'])
                combined_canvas[0:h, w:w*2] = add_label(img_ours, "Ours", methods_colors['ours'])
                combined_canvas[0:h, w*2:w*3] = add_label(img_fp, "FoundationPose", methods_colors['foundationpose'])
                
                combined_canvas[h:h*2, 0:w] = add_label(img_bs, "BundleSDF", methods_colors['bundlesdf'])
                combined_canvas[h:h*2, w:w*2] = add_label(img_mid, "MidFusion", methods_colors['midfusion'])
                combined_canvas[h:h*2, w*2:w*3] = add_label(img_tsdfpp, "TSDF++", methods_colors['tsdfpp'])
                
                cv2.imwrite(os.path.join(output_base_dir, "combined", f"frame_{frame_idx:06d}.png"), combined_canvas)
                print(f"Rendered frame {frame_idx:06d}")

    def render_dual(self, object_id, object_name, output_base_dir,
                    downsample_n=40000, alpha=0.55):
        """
        Per-frame dual-image rendering:
          - dual_a/: GT (green) + Ours (red) + FoundationPose (blue) overlaid on RGB
          - dual_b/: BundleSDF (cyan) + MidFusion (magenta) + TSDF++ (orange) overlaid on RGB

        Points are randomly downsampled to `downsample_n` and drawn with opacity `alpha`
        via alpha-blending so the underlying RGB texture remains visible.
        """
        self.object_id = object_id
        self.object_name = object_name

        self.eval_file        = os.path.join(self.dataset_dir, "eval",                    f"object_{object_id}.txt")
        self.foundation_file  = os.path.join(self.dataset_dir, "eval_foundationpose_comp", f"object_{object_id}.txt")
        self.bundle_sdf_file  = os.path.join(self.dataset_dir, "eval_bundlesdf_comp",      f"object_{object_id}.txt")
        self.midfusion_file   = os.path.join(self.dataset_dir, "eval_midfusion",           f"object_{object_id}.txt")
        self.tsdfpp_file      = os.path.join(self.dataset_dir, "eval_tsdfpp_comp",         f"object_{object_id}.txt")

        self.estimated_poses, self.evaluation_segments = self.read_estimated_poses()
        self.foundation_poses  = self.read_foundation_poses()
        self.bundle_sdf_poses  = self.read_bundle_sdf_poses()
        self.midfusion_poses   = self.read_midfusion_poses()
        self.tsdfpp_poses      = self.read_tsdfpp_poses()

        dir_a = os.path.join(output_base_dir, "dual_a")   # GT / Ours / FP
        dir_b = os.path.join(output_base_dir, "dual_b")   # BS / Mid / TSDF++
        os.makedirs(dir_a, exist_ok=True)
        os.makedirs(dir_b, exist_ok=True)

        fx, fy = 554.3827, 554.3827
        cx_k, cy_k = 320.5, 240.5

        # ---- helpers -------------------------------------------------------
        def project_pts(pts3d):
            Z     = pts3d[:, 2]
            valid = Z > 0
            pts3d = pts3d[valid]; Z = Z[valid]
            if len(pts3d) == 0:
                return np.empty((0, 2), dtype=np.int32)
            u = (pts3d[:, 0] * fx / Z) + cx_k
            v = (pts3d[:, 1] * fy / Z) + cy_k
            return np.stack([u, v], axis=1).astype(np.int32)

        def downsample(pts2d, n):
            if pts2d is None or len(pts2d) <= n:
                return pts2d
            idx = np.random.choice(len(pts2d), n, replace=False)
            return pts2d[idx]

        def blend_points(base, pts2d, color, radius=1, alpha=0.55):
            """Draw filled circles on a transparent layer, then alpha-blend."""
            if pts2d is None or len(pts2d) == 0:
                return base
            h, w = base.shape[:2]
            layer = base.copy()
            for p in pts2d:
                if 0 <= p[0] < w and 0 <= p[1] < h:
                    cv2.circle(layer, (int(p[0]), int(p[1])), radius, color, -1)
            return cv2.addWeighted(layer, alpha, base, 1 - alpha, 0)

        def filter_pct(pts, lo=2.0, hi=98.0):
            if pts is None or len(pts) < 10:
                return pts
            pl = np.percentile(pts, lo, axis=0)
            ph = np.percentile(pts, hi, axis=0)
            mask = ((pts[:,0]>=pl[0])&(pts[:,0]<=ph[0])&
                    (pts[:,1]>=pl[1])&(pts[:,1]<=ph[1])&
                    (pts[:,2]>=pl[2])&(pts[:,2]<=ph[2]))
            return pts[mask]

        def stamp_legend(img, entries, pos=(8, 8)):
            """Semi-transparent legend box."""
            lh, lw = 34, 280
            bx, by = pos
            bh = len(entries) * lh + 10
            overlay = img.copy()
            cv2.rectangle(overlay, (bx, by), (bx + lw, by + bh), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            for i, (txt, col) in enumerate(entries):
                cy_l = by + 8 + i * lh
                cv2.circle(img, (bx + 14, cy_l + 9), 9, col, -1)
                cv2.putText(img, txt, (bx + 30, cy_l + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
            return img

        def draw_contour_dashed(img, pts2d, color, thickness=3, dash_len=4, gap_len=2):
            """
            Draw a thick dashed outline of the projected point cloud:
            1. Render pts2d as dots onto a blank mask
            2. Dilate the mask to merge nearby dots into a solid region
            3. Find the outer contour with cv2.findContours
            4. Walk along the contour and draw dashes by pixel distance
            """
            if pts2d is None or len(pts2d) < 3:
                return img
            h, w = img.shape[:2]
            # Build binary mask from projected dots
            mask = np.zeros((h, w), dtype=np.uint8)
            for p in pts2d:
                if 0 <= p[0] < w and 0 <= p[1] < h:
                    cv2.circle(mask, (int(p[0]), int(p[1])), 2, 255, -1)
            # Dilate to bridge gaps between points, then erode back one ring
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
            mask = cv2.dilate(mask, kernel_dilate)
            mask = cv2.erode(mask, kernel_erode)
            # Find outer contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                return img
            cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
            # Walk along contour drawing dashes by pixel distance
            result = img.copy()
            draw_flag = True
            budget = float(dash_len)
            accumulated = 0.0
            n = len(cnt)
            for i in range(n):
                p1 = cnt[i]
                p2 = cnt[(i + 1) % n]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                seg_len = np.hypot(dx, dy)
                if seg_len < 1e-6:
                    continue
                ux, uy = dx / seg_len, dy / seg_len
                seg_t = 0.0
                while seg_t < seg_len:
                    remaining = budget - accumulated
                    step = min(remaining, seg_len - seg_t)
                    if draw_flag:
                        sp = (int(p1[0] + ux * seg_t),          int(p1[1] + uy * seg_t))
                        ep = (int(p1[0] + ux * (seg_t + step)),  int(p1[1] + uy * (seg_t + step)))
                        cv2.line(result, sp, ep, color, thickness, cv2.LINE_AA)
                    accumulated += step
                    seg_t   += step
                    if accumulated >= budget:
                        draw_flag  = not draw_flag
                        budget     = gap_len if not draw_flag else dash_len
                        accumulated = 0.0
            return result


        if not self.has_offset:
            self.has_offset = True

        for seg_i, segment in enumerate(self.evaluation_segments):
            print(f"Dual Renderer: segment {seg_i+1}/{len(self.evaluation_segments)}")
            if len(segment) == 0:
                continue

            first = segment[0]

            # ── initialise per-segment point clouds ──────────────────────
            self.compute_transformation_matrix(first)
            opc_ours = filter_pct(self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None)
            if opc_ours is None:
                print("No Ours point cloud, skipping."); continue

            mocap_obj_trans = self.obj_transformation.copy() if hasattr(self, 'obj_transformation') else np.eye(4)
            mocap_cam_trans = self.camera_transformation.copy() if hasattr(self, 'camera_transformation') else np.eye(4)

            opc_fp = None
            if first in self.foundation_poses:
                self.compute_transformation_matrix_foundation_pose(first)
                opc_fp = filter_pct(self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None)

            opc_bs = None
            if first in self.bundle_sdf_poses:
                self.compute_transformation_matrix_bundle_sdf(first)
                opc_bs = filter_pct(self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None)

            opc_mid = None
            if first in self.midfusion_poses:
                self.compute_transformation_matrix_midfusion(first)
                opc_mid = filter_pct(self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None)

            opc_tsdfpp = None
            if first in self.tsdfpp_poses:
                self.compute_transformation_matrix_tsdfpp(first)
                opc_tsdfpp = filter_pct(self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None)

            nearest_idx = None

            for frame_idx in segment:
                rgb_path = os.path.join(self.dataset_dir, "rgb", f"rgb_{frame_idx:06d}.png")
                if not os.path.exists(rgb_path):
                    continue
                rgb = cv2.imread(rgb_path)

                nearest_idx, _ = self.find_nearest_mocap_idx(
                    self.estimated_poses[frame_idx]['timestamp'], nearest_idx)
                if nearest_idx is None:
                    continue

                est_cam = self.camera_poses.get(frame_idx, np.eye(4))

                # ── GT pose ──────────────────────────────────────────────
                mocap_obj, mocap_cam = self.extract_mocap_pose(nearest_idx)
                gt_obj_pose = self.mocap_robot @ mocap_obj @ mocap_obj_trans
                gt_cam_pose = self.mocap_robot @ mocap_cam @ mocap_cam_trans
                gt_pose_cam = np.linalg.inv(gt_cam_pose) @ gt_obj_pose

                # ── project all methods ───────────────────────────────────
                def proj(opc, pose_cam):
                    if opc is None: return None
                    return downsample(project_pts(self.transform_points(opc, pose_cam)), downsample_n)

                pts_gt    = proj(opc_ours, gt_pose_cam)

                pts_ours  = None
                if frame_idx in self.estimated_poses and opc_ours is not None:
                    p = np.linalg.inv(est_cam) @ self.estimated_poses[frame_idx]['transform']
                    pts_ours = proj(opc_ours, p)

                pts_fp    = None
                if frame_idx in self.foundation_poses and opc_fp is not None:
                    pts_fp = proj(opc_fp, self.foundation_poses[frame_idx]['transform'])

                pts_bs    = None
                if frame_idx in self.bundle_sdf_poses and opc_bs is not None:
                    pts_bs = proj(opc_bs, self.bundle_sdf_poses[frame_idx]['transform'])

                pts_mid   = None
                if frame_idx in self.midfusion_poses and opc_mid is not None:
                    p = np.linalg.inv(est_cam) @ self.midfusion_poses[frame_idx]['transform']
                    pts_mid = proj(opc_mid, p)

                pts_tsdfpp = None
                if frame_idx in self.tsdfpp_poses and opc_tsdfpp is not None:
                    p = np.linalg.inv(est_cam) @ self.tsdfpp_poses[frame_idx]['transform']
                    pts_tsdfpp = proj(opc_tsdfpp, p)

                # ── Image A: GT + Ours + FoundationPose ──────────────────
                img_a = rgb.copy()
                # GT: filled dots
                img_a = blend_points(img_a, pts_gt,   (0,   255,   0), alpha=alpha)
                # FoundationPose: light dots + thick dashed contour
                img_a = blend_points(img_a, pts_fp,   (255,   0,   0), alpha=alpha * 0.6)
                img_a = draw_contour_dashed(img_a, pts_fp,   (255,   0,   0), thickness=1)
                # Ours: light dots + thick dashed contour
                img_a = blend_points(img_a, pts_ours, (0,     0, 255), alpha=alpha * 0.6)
                img_a = draw_contour_dashed(img_a, pts_ours, (0,     0, 255), thickness=1)
                img_a = stamp_legend(img_a, [
                    ("GT",            (0,   255,   0)),
                    ("Ours",          (0,     0, 255)),
                    ("FoundationPose",(255,   0,   0)),
                ])
                cv2.imwrite(os.path.join(dir_a, f"frame_{frame_idx:06d}.png"), img_a)

                # ── Image B: BundleSDF + MidFusion + TSDF++ ──────────────
                img_b = rgb.copy()
                img_b = blend_points(img_b, pts_bs,     (255, 255,   0), alpha=alpha)  # cyan
                img_b = blend_points(img_b, pts_mid,    (255,   0, 255), alpha=alpha)  # magenta
                img_b = blend_points(img_b, pts_tsdfpp, (0,   128, 255), alpha=alpha)  # orange
                img_b = stamp_legend(img_b, [
                    ("BundleSDF", (255, 255,   0)),
                    ("MidFusion", (255,   0, 255)),
                    ("TSDF++",    (0,   128, 255)),
                ])
                cv2.imwrite(os.path.join(dir_b, f"frame_{frame_idx:06d}.png"), img_b)

                print(f"Dual rendered frame {frame_idx:06d}")

    def render_all_overlay(self, object_id, object_name, output_base_dir, alpha=0.55):
        """
        Single-image overlay: all 6 methods on one RGB frame.
          - GT:            semi-transparent filled point cloud (green)
          - Ours, FP, BS, MidFusion, TSDF++:  dashed contour outline only (no fill)
        Output saved to output_base_dir/all_overlay/
        """
        self.object_id = object_id
        self.object_name = object_name

        self.eval_file        = os.path.join(self.dataset_dir, "eval",                    f"object_{object_id}.txt")
        self.foundation_file  = os.path.join(self.dataset_dir, "eval_foundationpose_comp", f"object_{object_id}.txt")
        self.bundle_sdf_file  = os.path.join(self.dataset_dir, "eval_bundlesdf_comp",      f"object_{object_id}.txt")
        self.midfusion_file   = os.path.join(self.dataset_dir, "eval_midfusion",           f"object_{object_id}.txt")
        self.tsdfpp_file      = os.path.join(self.dataset_dir, "eval_tsdfpp_comp",         f"object_{object_id}.txt")

        self.estimated_poses, self.evaluation_segments = self.read_estimated_poses()
        self.foundation_poses  = self.read_foundation_poses()
        self.bundle_sdf_poses  = self.read_bundle_sdf_poses()
        self.midfusion_poses   = self.read_midfusion_poses()
        self.tsdfpp_poses      = self.read_tsdfpp_poses()

        out_dir = os.path.join(output_base_dir, "all_overlay")
        os.makedirs(out_dir, exist_ok=True)

        fx, fy = 554.3827, 554.3827
        cx_k, cy_k = 320.5, 240.5

        # ── helpers ────────────────────────────────────────────────────────
        def project_pts(pts3d):
            Z = pts3d[:, 2]; valid = Z > 0
            pts3d = pts3d[valid]; Z = Z[valid]
            if len(pts3d) == 0:
                return np.empty((0, 2), dtype=np.int32)
            u = (pts3d[:, 0] * fx / Z) + cx_k
            v = (pts3d[:, 1] * fy / Z) + cy_k
            return np.stack([u, v], axis=1).astype(np.int32)

        def blend_points(base, pts2d, color, radius=1, alpha=0.55):
            if pts2d is None or len(pts2d) == 0: return base
            h, w = base.shape[:2]
            layer = base.copy()
            for p in pts2d:
                if 0 <= p[0] < w and 0 <= p[1] < h:
                    cv2.circle(layer, (int(p[0]), int(p[1])), radius, color, -1)
            return cv2.addWeighted(layer, alpha, base, 1 - alpha, 0)

        def draw_contour_dashed(img, pts2d, color, thickness=2, dash_len=4, gap_len=4):
            if pts2d is None or len(pts2d) < 3: return img
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for p in pts2d:
                if 0 <= p[0] < w and 0 <= p[1] < h:
                    cv2.circle(mask, (int(p[0]), int(p[1])), 2, 255, -1)
            k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.dilate(mask, k_d)
            mask = cv2.erode(mask, k_e)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours: return img
            cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
            result = img.copy()
            draw_flag, budget, accumulated = True, float(dash_len), 0.0
            for i in range(len(cnt)):
                p1, p2 = cnt[i], cnt[(i + 1) % len(cnt)]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                seg_len = np.hypot(dx, dy)
                if seg_len < 1e-6: continue
                ux, uy = dx / seg_len, dy / seg_len
                seg_t = 0.0
                while seg_t < seg_len:
                    step = min(budget - accumulated, seg_len - seg_t)
                    if draw_flag:
                        sp = (int(p1[0] + ux * seg_t),          int(p1[1] + uy * seg_t))
                        ep = (int(p1[0] + ux * (seg_t + step)), int(p1[1] + uy * (seg_t + step)))
                        cv2.line(result, sp, ep, color, thickness, cv2.LINE_AA)
                    accumulated += step; seg_t += step
                    if accumulated >= budget:
                        draw_flag = not draw_flag
                        budget = gap_len if not draw_flag else dash_len
                        accumulated = 0.0
            return result

        def filter_pct(pts, lo=2.0, hi=98.0):
            if pts is None or len(pts) < 10: return pts
            pl = np.percentile(pts, lo, axis=0); ph = np.percentile(pts, hi, axis=0)
            mask = ((pts[:,0]>=pl[0])&(pts[:,0]<=ph[0])&
                    (pts[:,1]>=pl[1])&(pts[:,1]<=ph[1])&
                    (pts[:,2]>=pl[2])&(pts[:,2]<=ph[2]))
            return pts[mask]

        def blend_mask_region(img, pts2d, color, alpha=0.45):
            """Fill the point-cloud region (after dilate+erode) with semi-transparent color."""
            if pts2d is None or len(pts2d) < 3: return img
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for p in pts2d:
                if 0 <= p[0] < w and 0 <= p[1] < h:
                    cv2.circle(mask, (int(p[0]), int(p[1])), 2, 255, -1)
            k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.dilate(mask, k_d)
            mask = cv2.erode(mask, k_e)
            layer = img.copy()
            layer[mask > 0] = color
            return cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)

        def stamp_legend(img, entries, pos=(8, 8)):
            lh, lw = 34, 310; bx, by = pos; bh = len(entries) * lh + 10
            overlay = img.copy()
            cv2.rectangle(overlay, (bx, by), (bx + lw, by + bh), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            for i, (txt, col) in enumerate(entries):
                cy_l = by + 8 + i * lh
                cv2.circle(img, (bx + 14, cy_l + 9), 9, col, -1)
                cv2.putText(img, txt, (bx + 30, cy_l + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
            return img
        # ──────────────────────────────────────────────────────────────────

        if not self.has_offset:
            self.has_offset = True

        for seg_i, segment in enumerate(self.evaluation_segments):
            print(f"AllOverlay: segment {seg_i+1}/{len(self.evaluation_segments)}")
            if len(segment) == 0: continue
            first = segment[0]

            self.compute_transformation_matrix(first)
            opc_ours = filter_pct(self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None)
            if opc_ours is None: print("No Ours point cloud, skipping."); continue
            mocap_obj_trans = self.obj_transformation.copy() if hasattr(self, 'obj_transformation') else np.eye(4)
            mocap_cam_trans = self.camera_transformation.copy() if hasattr(self, 'camera_transformation') else np.eye(4)

            def _init_opc(pose_dict, compute_fn):
                if first not in pose_dict: return None
                compute_fn(first)
                pts = self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None
                return filter_pct(pts)

            opc_fp     = _init_opc(self.foundation_poses,  self.compute_transformation_matrix_foundation_pose)
            opc_bs     = _init_opc(self.bundle_sdf_poses,  self.compute_transformation_matrix_bundle_sdf)
            opc_mid    = _init_opc(self.midfusion_poses,   self.compute_transformation_matrix_midfusion)
            opc_tsdfpp = _init_opc(self.tsdfpp_poses,      self.compute_transformation_matrix_tsdfpp)

            nearest_idx = None
            for frame_idx in segment:
                rgb_path = os.path.join(self.dataset_dir, "rgb", f"rgb_{frame_idx:06d}.png")
                if not os.path.exists(rgb_path): continue
                rgb = cv2.imread(rgb_path)

                nearest_idx, _ = self.find_nearest_mocap_idx(
                    self.estimated_poses[frame_idx]['timestamp'], nearest_idx)
                if nearest_idx is None: continue

                est_cam = self.camera_poses.get(frame_idx, np.eye(4))

                # GT
                mocap_obj, mocap_cam = self.extract_mocap_pose(nearest_idx)
                gt_obj_pose = self.mocap_robot @ mocap_obj @ mocap_obj_trans
                gt_cam_pose = self.mocap_robot @ mocap_cam @ mocap_cam_trans
                gt_pose_cam = np.linalg.inv(gt_cam_pose) @ gt_obj_pose
                pts_gt = project_pts(self.transform_points(opc_ours, gt_pose_cam))

                def _proj_cam(opc, pose): return project_pts(self.transform_points(opc, pose)) if opc is not None else None
                def _inv_cam(raw_pose):   return np.linalg.inv(est_cam) @ raw_pose

                pts_ours = _proj_cam(opc_ours, _inv_cam(self.estimated_poses[frame_idx]['transform'])) \
                    if frame_idx in self.estimated_poses else None
                pts_fp   = _proj_cam(opc_fp,   self.foundation_poses[frame_idx]['transform']) \
                    if frame_idx in self.foundation_poses   and opc_fp     is not None else None
                pts_bs   = _proj_cam(opc_bs,   self.bundle_sdf_poses[frame_idx]['transform']) \
                    if frame_idx in self.bundle_sdf_poses   and opc_bs     is not None else None
                pts_mid  = _proj_cam(opc_mid,  _inv_cam(self.midfusion_poses[frame_idx]['transform'])) \
                    if frame_idx in self.midfusion_poses    and opc_mid    is not None else None
                pts_tsdfpp = _proj_cam(opc_tsdfpp, _inv_cam(self.tsdfpp_poses[frame_idx]['transform'])) \
                    if frame_idx in self.tsdfpp_poses       and opc_tsdfpp is not None else None

                img = rgb.copy()
                # GT: filled mask region (dilate+erode) + dashed contour
                img = blend_mask_region(img, pts_gt, (0, 255, 0), alpha=alpha)
                img = draw_contour_dashed(img, pts_gt,     (0,   255,   0))   # green
                # All others: dashed contour only
                img = draw_contour_dashed(img, pts_ours,   (0,   0,   255))   # red
                img = draw_contour_dashed(img, pts_fp,     (255, 0,   0  ))   # blue
                img = draw_contour_dashed(img, pts_bs,     (255, 255, 0  ))   # cyan
                img = draw_contour_dashed(img, pts_mid,    (255, 0,   255))   # magenta
                img = draw_contour_dashed(img, pts_tsdfpp, (0,   128, 255))   # orange

                img = stamp_legend(img, [
                    ("GT (Motion Capture)", (0,   255,   0)),
                    ("DYNOSR (Ours)",       (0,     0, 255)),
                    ("FoundationPose",      (255,   0,   0)),
                    ("BundleSDF",           (255, 255,   0)),
                    ("MidFusion",           (255,   0, 255)),
                    ("TSDF++",              (0,   128, 255)),
                ])
                cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:06d}.png"), img)
                print(f"AllOverlay rendered frame {frame_idx:06d}")

if __name__ == "__main__":
    dataset_dir = "/media/wby/6d811df4-bde7-479b-ab4c-679222653ea0/dataset_done/tomato_3"
    object_id = 1
    object_name = "tomato"

    output_base_dir = os.path.join(dataset_dir, "render_output")
    os.makedirs(output_base_dir, exist_ok=True)

    print(f"Starting renderer for dataset: {dataset_dir}")
    renderer = PoseRenderer(dataset_dir)
    # renderer.render(object_id, object_name, output_base_dir)
    print(f"Render completed. Images saved to: {output_base_dir}")

    # # Dual-overlay rendering (GT/Ours/FP  vs  BS/Mid/TSDF++)
    # dual_output_dir = os.path.join(dataset_dir, "render_dual")
    # os.makedirs(dual_output_dir, exist_ok=True)
    # print(f"Starting dual renderer ...")
    # renderer.render_dual(object_id, object_name, dual_output_dir,
    #                      downsample_n=400000, alpha=0.55)
    # print(f"Dual render completed. Images saved to: {dual_output_dir}")

    # All-overlay: GT filled + all others dashed contour, single image
    all_output_dir = os.path.join(dataset_dir, "render_all")
    os.makedirs(all_output_dir, exist_ok=True)
    print(f"Starting all-overlay renderer ...")
    renderer.render_all_overlay(object_id, object_name, all_output_dir, alpha=0.55)
    print(f"All-overlay render completed. Images saved to: {all_output_dir}/all_overlay/")





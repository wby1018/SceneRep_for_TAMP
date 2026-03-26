import sys

with open('eval_all.py', 'r') as f:
    content = f.read()

# 1. Add calculate_pose_error
target1 = """    def calculate_adds(self, points1, points2):
        tree = cKDTree(points2)
        dists, _ = tree.query(points1, k=1)
        return np.mean(dists)"""
replacement1 = """    def calculate_adds(self, points1, points2):
        tree = cKDTree(points2)
        dists, _ = tree.query(points1, k=1)
        return np.mean(dists)

    def calculate_pose_error(self, est_pose, gt_pose):
        \"\"\"计算translation error(cm)和rotation error(degree)\"\"\"
        t_err = np.linalg.norm(est_pose[:3, 3] - gt_pose[:3, 3]) * 100.0
        R_est = est_pose[:3, :3]
        R_gt = gt_pose[:3, :3]
        R_err = np.dot(R_est.T, R_gt)
        trace = np.trace(R_err)
        trace = np.clip(trace, -1.0, 3.0)
        theta = np.arccos((trace - 1.0) / 2.0)
        return t_err, np.degrees(theta)"""
content = content.replace(target1, replacement1)

# 2. Update results = {...}
target2 = """        results = {
            'frame_ids': [],
            'add_values': [],
            'adds_values': [],
            'add_correct': 0,
            'adds_correct': 0,
            'total_frames': 0
        }"""
replacement2 = """        results = {
            'frame_ids': [],
            'add_values': [],
            'adds_values': [],
            't_err_values': [],
            'r_err_values': [],
            'add_correct': 0,
            'adds_correct': 0,
            'total_frames': 0
        }"""
content = content.replace(target2, replacement2)

# 3. Update inner loop calculation
target3 = """            # 计算ADD和ADD-S
            add_value = self.calculate_add(points_est, points_mocap)
            adds_value = self.calculate_adds(points_est, points_mocap)
            
            results['add_values'].append(add_value)
            results['adds_values'].append(adds_value)
            results['total_frames'] += 1
            results['frame_ids'].append(frame_idx)
            
            if add_value < self.add_threshold:
                results['add_correct'] += 1
            if adds_value < self.adds_threshold:
                results['adds_correct'] += 1
            
            print(f"帧 {frame_idx}: ADD={add_value:.4f}, ADD-S={adds_value:.4f}")"""
replacement3 = """            # 计算ADD和ADD-S
            add_value = self.calculate_add(points_est, points_mocap)
            adds_value = self.calculate_adds(points_est, points_mocap)
            t_err, r_err = self.calculate_pose_error(est_pose, gt_obj_pose)
            
            results['add_values'].append(add_value)
            results['adds_values'].append(adds_value)
            results['t_err_values'].append(t_err)
            results['r_err_values'].append(r_err)
            results['total_frames'] += 1
            results['frame_ids'].append(frame_idx)
            
            if add_value < self.add_threshold:
                results['add_correct'] += 1
            if adds_value < self.adds_threshold:
                results['adds_correct'] += 1
            
            print(f"帧 {frame_idx}: ADD={add_value:.4f}, ADD-S={adds_value:.4f}, t_err={t_err:.2f}cm, r_err={r_err:.2f}deg")"""
content = content.replace(target3, replacement3)

# 4. _filter_results_by_bad_frames (part 1)
target4_1 = """        new_frame_ids, new_adds, new_adds_s = [], [], []
        for f, a, s in zip(res["frame_ids"], res["add_values"], res["adds_values"]):"""
replacement4_1 = """        new_frame_ids, new_adds, new_adds_s = [], [], []
        new_t_err, new_r_err = [], []
        for f, a, s, t, r in zip(res["frame_ids"], res["add_values"], res["adds_values"], res["t_err_values"], res["r_err_values"]):"""
content = content.replace(target4_1, replacement4_1)

# 4. _filter_results_by_bad_frames (part 2)
target4_2 = """            new_frame_ids.append(f)
            new_adds.append(a)
            new_adds_s.append(s)

        res["frame_ids"] = new_frame_ids
        res["add_values"] = new_adds
        res["adds_values"] = new_adds_s"""
replacement4_2 = """            new_frame_ids.append(f)
            new_adds.append(a)
            new_adds_s.append(s)
            new_t_err.append(t)
            new_r_err.append(r)

        res["frame_ids"] = new_frame_ids
        res["add_values"] = new_adds
        res["adds_values"] = new_adds_s
        res["t_err_values"] = new_t_err
        res["r_err_values"] = new_r_err"""
content = content.replace(target4_2, replacement4_2)

# 5. aggregate_results update
target5 = """        if not all_results:
            row.update({
                "total_frames": 0,
                "add_mean": "",
                "adds_mean": "",
                "add_success_rate": "",
                "adds_success_rate": "",
            })
            # 多阈值列也补空（保证写 CSV 不缺字段）
            for thr in self.add_thresholds:
                row[self._thr_key_mm("add_sr", thr)] = ""
            for thr in self.adds_thresholds:
                row[self._thr_key_mm("adds_sr", thr)] = ""
            return row

        total_frames = sum(r["total_frames"] for r in all_results)

        all_add_values, all_adds_values = [], []
        for r in all_results:
            all_add_values.extend(r["add_values"])
            all_adds_values.extend(r["adds_values"])

        # === 默认阈值成功率（兼容你原列） ===
        add_correct_default = sum(1 for a in all_add_values if a < self.add_threshold)
        adds_correct_default = sum(1 for s in all_adds_values if s < self.adds_threshold)

        row.update({
            "total_frames": int(total_frames),
            "add_mean": float(np.mean(all_add_values)) if total_frames > 0 else "",
            "adds_mean": float(np.mean(all_adds_values)) if total_frames > 0 else "",
            "add_success_rate": float(add_correct_default / total_frames) if total_frames > 0 else "",
            "adds_success_rate": float(adds_correct_default / total_frames) if total_frames > 0 else "",
        })"""

replacement5 = """        if not all_results:
            row.update({
                "total_frames": 0,
                "add_mean": "",
                "adds_mean": "",
                "add_success_rate": "",
                "adds_success_rate": "",
                "t_err_mean": "",
                "r_err_mean": "",
                "pose_5cm_5deg_sr": "",
                "last_t_err": "",
                "last_r_err": "",
            })
            # 多阈值列也补空（保证写 CSV 不缺字段）
            for thr in self.add_thresholds:
                row[self._thr_key_mm("add_sr", thr)] = ""
            for thr in self.adds_thresholds:
                row[self._thr_key_mm("adds_sr", thr)] = ""
            return row

        total_frames = sum(r["total_frames"] for r in all_results)

        all_add_values, all_adds_values = [], []
        all_t_err_values, all_r_err_values = [], []
        last_t_errs, last_r_errs = [], []
        
        for r in all_results:
            all_add_values.extend(r["add_values"])
            all_adds_values.extend(r["adds_values"])
            all_t_err_values.extend(r.get("t_err_values", []))
            all_r_err_values.extend(r.get("r_err_values", []))
            if r.get("t_err_values"):
                last_t_errs.append(r["t_err_values"][-1])
            if r.get("r_err_values"):
                last_r_errs.append(r["r_err_values"][-1])

        # === 默认阈值成功率（兼容你原列） ===
        add_correct_default = sum(1 for a in all_add_values if a < self.add_threshold)
        adds_correct_default = sum(1 for s in all_adds_values if s < self.adds_threshold)
        pose_correct = sum(1 for t, rot in zip(all_t_err_values, all_r_err_values) if t <= 5.0 and rot <= 5.0)

        row.update({
            "total_frames": int(total_frames),
            "add_mean": float(np.mean(all_add_values)) if total_frames > 0 else "",
            "adds_mean": float(np.mean(all_adds_values)) if total_frames > 0 else "",
            "add_success_rate": float(add_correct_default / total_frames) if total_frames > 0 else "",
            "adds_success_rate": float(adds_correct_default / total_frames) if total_frames > 0 else "",
            "t_err_mean": float(np.mean(all_t_err_values)) if total_frames > 0 else "",
            "r_err_mean": float(np.mean(all_r_err_values)) if total_frames > 0 else "",
            "pose_5cm_5deg_sr": float(pose_correct / total_frames) if total_frames > 0 else "",
            "last_t_err": float(np.mean(last_t_errs)) if len(last_t_errs) > 0 else "",
            "last_r_err": float(np.mean(last_r_errs)) if len(last_r_errs) > 0 else "",
        })"""
content = content.replace(target5, replacement5)

# 6. save_results_csv
target6 = """        base_fieldnames = [
            "time",
            "dataset",
            "object_id",
            "object_name",
            "method",
            "num_segments",
            "total_frames",
            "add_mean",
            "adds_mean",
            "add_success_rate",
            "adds_success_rate",
        ]"""
replacement6 = """        base_fieldnames = [
            "time",
            "dataset",
            "object_id",
            "object_name",
            "method",
            "num_segments",
            "total_frames",
            "add_mean",
            "adds_mean",
            "add_success_rate",
            "adds_success_rate",
            "t_err_mean",
            "r_err_mean",
            "pose_5cm_5deg_sr",
            "last_t_err",
            "last_r_err",
        ]"""
content = content.replace(target6, replacement6)

with open('eval_all.py', 'w') as f:
    f.write(content)
print("Patch applied.")

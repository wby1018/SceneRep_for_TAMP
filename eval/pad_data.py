#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
from typing import Dict, Tuple, Optional


def read_pose_map(p: str) -> Tuple[Dict[int, str], Optional[int], Optional[int]]:
    """
    Read poses like:
      idx tx ty tz qx qy qz qw
    Return: {idx: "tx ty tz qx qy qz qw"}, min_idx, max_idx
    """
    pose_map: Dict[int, str] = {}
    min_idx, max_idx = None, None

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 8:
                # skip malformed line
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            pose_str = " ".join(parts[1:8])  # keep 7 numbers as string
            pose_map[idx] = pose_str
            min_idx = idx if min_idx is None else min(min_idx, idx)
            max_idx = idx if max_idx is None else max(max_idx, idx)

    return pose_map, min_idx, max_idx


def read_max_idx(p: str) -> Optional[int]:
    """Read max idx from eval/object_X.txt (take max of first column)."""
    if not os.path.isfile(p):
        return None
    max_idx = None
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            max_idx = idx if max_idx is None else max(max_idx, idx)
    return max_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/media/wby/6d811df4-bde7-479b-ab4c-679222653ea0/dataset_done",
        help="dataset_done root",
    )
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    pattern = os.path.join(root, "*", "eval_foundationpose", "object_*.txt")
    src_files = sorted(glob.glob(pattern))

    if not src_files:
        print(f"[WARN] No files matched: {pattern}")
        return

    for src in src_files:
        seq_dir = os.path.dirname(os.path.dirname(src))  # .../XXX
        base = os.path.basename(src)                     # object_X.txt

        eval_file = os.path.join(seq_dir, "eval", base)
        dst_dir = os.path.join(seq_dir, "eval_foundationpose_comp")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, base)

        pose_map, min_src, max_src = read_pose_map(src)
        if not pose_map or min_src is None:
            print(f"[SKIP] empty or invalid: {src}")
            continue

        max_eval = read_max_idx(eval_file)
        max_target = max_eval if max_eval is not None else max_src

        # Start from the first available idx in eval_bundlesdf (向后 pad)
        start_idx = min_src

        last_pose = None
        wrote = 0
        with open(dst, "w", encoding="utf-8") as f:
            for idx in range(start_idx, max_target + 1):
                if idx in pose_map:
                    last_pose = pose_map[idx]
                else:
                    if last_pose is None:
                        # theoretically shouldn't happen since start_idx = min_src exists
                        continue
                f.write(f"{idx} {last_pose}\n")
                wrote += 1

        print(f"[OK] {src}  ->  {dst}   (start={start_idx}, max={max_target}, lines={wrote})")


if __name__ == "__main__":
    main()

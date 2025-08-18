#!/usr/bin/env python3
# preprocess_borderlands.py

import os, glob, json, argparse
import numpy as np
from PIL import Image
from typing import List

# ----- action schema: W,A,S,D,LMB,Space,R -----
KEY_LABELS = ["W", "A", "S", "D", "LMB", "Space", "R"]  # reference only
CODE_TO_IDX = {87: 0, 65: 1, 83: 2, 68: 3, 1: 4, 32: 5, 82: 6}  # VK -> index

# ----- utils -----
def resize_with_letterbox(img: Image.Image, size: int = 64) -> np.ndarray:
    orig_w, orig_h = img.size
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return np.array(canvas, dtype=np.uint8)

def encode_action(data: dict, action_dim: int) -> np.ndarray:
    """
    One-hot over [W,A,S,D,LMB,Space,R], padded/truncated to action_dim.
    Accepts either 'virtual_key_codes' or 'virtual_keys'.
    """
    base_dim = 7
    vec = np.zeros(base_dim, dtype=np.float32)
    codes = data.get("virtual_key_codes", data.get("virtual_keys", [])) or []
    for c in codes:
        i = CODE_TO_IDX.get(int(c))
        if i is not None:
            vec[i] = 1.0

    if action_dim > base_dim:
        vec = np.concatenate([vec, np.zeros(action_dim - base_dim, dtype=np.float32)], axis=0)
    else:
        vec = vec[:action_dim]
    return vec.astype(np.float16)

def get_episode_dirs(base_dir: str) -> List[str]:
    subs = sorted(
        [os.path.join(base_dir, d) for d in os.listdir(base_dir)
         if os.path.isdir(os.path.join(base_dir, d))]
    )
    return subs if subs else [base_dir]

def index_from_name(path: str, prefix: str, suffix: str) -> int:
    base = os.path.basename(path)
    if not (base.startswith(prefix) and base.endswith(suffix)):
        return -1
    mid = base[len(prefix):-len(suffix)]
    try:
        return int(mid)
    except ValueError:
        return -1

def paired_indices(frame_dir: str, action_dir: str) -> List[int]:
    frame_files = glob.glob(os.path.join(frame_dir, "frame_*.jpg"))
    action_files = glob.glob(os.path.join(action_dir, "action_*.json"))

    f_idx = {index_from_name(p, "frame_", ".jpg") for p in frame_files}
    a_idx = {index_from_name(p, "action_", ".json") for p in action_files}

    common = sorted(i for i in f_idx.intersection(a_idx) if i >= 0)
    missing_frames = sorted(i for i in a_idx - f_idx if i >= 0)
    missing_actions = sorted(i for i in f_idx - a_idx if i >= 0)

    if missing_frames:
        print(f"[warn] {os.path.basename(frame_dir) or frame_dir}: missing frames for indices: "
              f"{missing_frames[:10]}{' ...' if len(missing_frames)>10 else ''}")
    if missing_actions:
        print(f"[warn] {os.path.basename(action_dir) or action_dir}: missing actions for indices: "
              f"{missing_actions[:10]}{' ...' if len(missing_actions)>10 else ''}")

    return common

# ----- main episode processor (simple non-overlapping chunking) -----
def process_episode_simple(
    frame_dir: str,
    action_dir: str,
    output_path: str,
    action_dim: int,
    traj_len: int,
    start_traj_idx: int
) -> int:
    """
    Reads paired (frame_i, action_i) in sorted order.
    Emits non-overlapping chunks of length traj_len as traj_{########}.npz.
    Returns next trajectory index after writing.
    """
    common = paired_indices(frame_dir, action_dir)
    if not common:
        return start_traj_idx

    buffer_imgs, buffer_acts = [], []
    out_idx = start_traj_idx

    for i in common:
        frame_path = os.path.join(frame_dir, f"frame_{i:06d}.jpg")
        action_path = os.path.join(action_dir, f"action_{i:06d}.json")

        with Image.open(frame_path) as img:
            arr = resize_with_letterbox(img.convert("RGB"), 64)
        with open(action_path, "r") as f:
            act = encode_action(json.load(f), action_dim)

        buffer_imgs.append(arr)
        buffer_acts.append(act)

        if len(buffer_imgs) == traj_len:
            imgs = np.stack(buffer_imgs, axis=0)   # (T,64,64,3) uint8
            acts = np.stack(buffer_acts, axis=0)   # (T,action_dim) float16
            yield_name = f"traj_{out_idx:05d}.npz"
            np.savez_compressed(os.path.join(output_path, yield_name),
                                **{"image": imgs, "action": acts})
            out_idx += 1
            buffer_imgs.clear()
            buffer_acts.clear()

    if buffer_imgs:
        print(f"[info] Dropping leftover {len(buffer_imgs)} frames (< traj_len) in {frame_dir}")

    return out_idx

# ----- cli -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Borderlands preprocessor: fixed-size, non-overlapping trajectories."
    )
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Directory with frames (or episodes as subfolders)")
    parser.add_argument("--actions_dir", type=str, required=True,
                        help="Directory with actions (mirrors frames_dir)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save traj_XXXXX.npz files")
    parser.add_argument("--action_dim", type=int, default=7,
                        help="Action vector length (default 7 for W/A/S/D/LMB/Space/R)")
    parser.add_argument("--traj_len", type=int, default=5,
                        help="Frames per trajectory (default 5)")
    args = parser.parse_args()

    if args.action_dim < 7:
        print(f"[warn] action_dim={args.action_dim} < 7; "
              "W/A/S/D/LMB/Space/R will be truncated.")

    os.makedirs(args.output_path, exist_ok=True)

    frame_eps = get_episode_dirs(args.frames_dir)
    action_eps = get_episode_dirs(args.actions_dir)
    assert len(frame_eps) == len(action_eps), "Mismatched episode counts"

    traj_idx = 0
    total_traj = 0
    for f_dir, a_dir in zip(frame_eps, action_eps):
        before = traj_idx
        traj_idx = process_episode_simple(
            f_dir, a_dir, args.output_path, args.action_dim, args.traj_len, traj_idx
        )
        written = traj_idx - before
        total_traj += written
        print(f"[done] {os.path.basename(f_dir) or f_dir}: wrote {written} trajectories")

    print(f"[total] wrote {total_traj} trajectories to {args.output_path}")

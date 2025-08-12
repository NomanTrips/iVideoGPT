# preprocess_borderlands.py

import os
import glob
import json
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# ----------------------------
# Windows VK -> bit positions
# ----------------------------
VK_TO_BIT = {
    87: 0,   # W
    65: 1,   # A
    83: 2,   # S
    68: 3,   # D
    1:  4,   # LMB
    2:  5,   # RMB
    32: 6,   # Space
    16: 7,   # Shift
}
NUM_BITS = 8  # len(VK_TO_BIT)

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

def get_episode_dirs(base_dir: str) -> List[str]:
    entries = sorted(
        [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )
    return entries if entries else [base_dir]

def norm_clip(x: float, max_abs: float) -> float:
    if max_abs <= 0:
        return 0.0
    x = max(-max_abs, min(max_abs, x))
    return x / max_abs

def encode_action(data: dict, max_abs_dx: float, max_abs_dy: float, max_abs_wheel: float) -> np.ndarray:
    # 1) continuous (normalized to roughly [-1,1])
    dx = norm_clip(float(data.get("dx", 0.0)), max_abs_dx)
    dy = norm_clip(float(data.get("dy", 0.0)), max_abs_dy)
    wheel = norm_clip(float(data.get("wheel", 0.0)), max_abs_wheel)

    # 2) multi-hot for VKs
    bits = np.zeros(NUM_BITS, dtype=np.float32)
    vks = data.get("virtual_key_codes", []) or data.get("virtual_keys", []) or []
    for vk in vks:
        pos = VK_TO_BIT.get(int(vk))
        if pos is not None:
            bits[pos] = 1.0

    return np.concatenate([np.array([dx, dy, wheel], dtype=np.float32), bits], axis=0)

def process_episode(
    frame_dir: str,
    action_dir: str,
    diff_threshold: float,
    traj_len: int,
    max_abs_dx: float,
    max_abs_dy: float,
    max_abs_wheel: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    images, actions = [], []

    for frame_file in tqdm(frame_files, desc=f"{os.path.basename(frame_dir) or 'episode'}"):
        base = os.path.basename(frame_file)
        index = base.replace("frame_", "").replace(".jpg", "")
        action_file = os.path.join(action_dir, f"action_{index}.json")
        if not os.path.exists(action_file):
            continue

        with Image.open(frame_file) as img:
            image = resize_with_letterbox(img.convert("RGB"), 64)

        with open(action_file, "r") as f:
            action_data = json.load(f)
        action_vec = encode_action(action_data, max_abs_dx, max_abs_dy, max_abs_wheel)

        images.append(image)
        actions.append(action_vec)

    if len(images) < 2:
        return []

    images = np.stack(images)                 # (N, H, W, 3) uint8
    actions = np.stack(actions).astype(np.float32)  # (N, 11) float32

    # trigger points for building trajs (optional heuristic)
    imgs_int = images.astype(np.int16)
    diffs = np.mean(np.abs(imgs_int[1:] - imgs_int[:-1]), axis=(1, 2, 3))

    trajs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, diff in enumerate(diffs):
        start = i
        end = start + traj_len
        if diff > diff_threshold and end <= images.shape[0]:
            traj_imgs = images[start:end]          # length = traj_len
            traj_actions = actions[start:end]      # same length; model slices last internally
            # Sanity: last action is unused by forward(), that’s ok.
            trajs.append((traj_imgs, traj_actions))

    return trajs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Borderlands dataset")
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--actions_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--diff_threshold", type=float, default=0.0)
    parser.add_argument("--traj_len", type=int, default=20)

    # normalization ranges (tweak to your capture)
    parser.add_argument("--max_abs_dx", type=float, default=50.0, help="Pixels per frame mapped to 1.0")
    parser.add_argument("--max_abs_dy", type=float, default=50.0)
    parser.add_argument("--max_abs_wheel", type=float, default=10.0)

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    frame_episode_dirs = get_episode_dirs(args.frames_dir)
    action_episode_dirs = get_episode_dirs(args.actions_dir)
    assert len(frame_episode_dirs) == len(action_episode_dirs), "Mismatched episode counts"

    traj_idx = 0
    for f_dir, a_dir in zip(frame_episode_dirs, action_episode_dirs):
        trajs = process_episode(
            f_dir,
            a_dir,
            args.diff_threshold,
            args.traj_len,
            args.max_abs_dx,
            args.max_abs_dy,
            args.max_abs_wheel,
        )
        for imgs, acts in trajs:
            save_name = os.path.join(args.output_path, f"traj_{traj_idx:05d}.npz")
            np.savez_compressed(save_name, image=imgs, action=acts)
            traj_idx += 1

    print(f"wrote {traj_idx} trajectories to {args.output_path}")

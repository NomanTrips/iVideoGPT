import os
import glob
import json
import argparse
from typing import List, Tuple

import random

import numpy as np
from PIL import Image
from tqdm import tqdm


def resize_with_letterbox(img: Image.Image, size: int = 64) -> np.ndarray:
    """Resize ``img`` to a square ``size``×``size`` with letterboxing.

    The aspect ratio of ``img`` is preserved and the remaining area is filled
    with black pixels. The returned array has dtype ``uint8``.
    """
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
    """Return a list of episode directories.

    If ``base_dir`` contains subdirectories, each subdirectory is treated
    as an episode. Otherwise, ``base_dir`` itself is considered a single
    episode directory.
    """
    entries = sorted(
        [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )
    if entries:
        return entries
    return [base_dir]


def encode_action(data: dict, action_dim: int) -> np.ndarray:
    """Encode raw action dict into a fixed-length numeric vector.

    The resulting vector contains (W, A, S, D, left mouse button). Input
    ``data`` may specify these controls either by name (e.g. ``"w"``) or by
    their Windows virtual-key codes (e.g. ``87`` for ``W``). Unrecognized
    fields are ignored and missing fields default to ``0``.
    """

    name_to_idx = {
        "w": 0,
        "a": 1,
        "s": 2,
        "d": 3,
        "mouse1": 4,
        "left_click": 4,
    }
    code_to_idx = {87: 0, 65: 1, 83: 2, 68: 3, 1: 4}

    vec = np.zeros(5, dtype=np.float32)
    for key, value in data.items():
        idx = None
        if isinstance(key, str):
            lower = key.lower()
            if lower in name_to_idx:
                idx = name_to_idx[lower]
            elif key.isdigit():
                idx = code_to_idx.get(int(key))
        elif isinstance(key, int):
            idx = code_to_idx.get(key)
        if idx is not None:
            vec[idx] = float(value)

    if action_dim > vec.shape[0]:
        pad = np.zeros(action_dim - vec.shape[0], dtype=np.float32)
        vec = np.concatenate([vec, pad], axis=0)
    else:
        vec = vec[:action_dim]

    return vec


def process_episode(
    frame_dir: str,
    action_dir: str,
    action_dim: int,
    diff_threshold: float,
    traj_len: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return a list of (images, actions) trajectories for an episode.

    Each trajectory contains ``traj_len`` consecutive frames and ``traj_len``
    actions. A trajectory is generated whenever the L1 pixel difference between
    two consecutive frames exceeds ``diff_threshold`` and enough subsequent
    frames remain to form a full trajectory.
    """
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
        action_vec = encode_action(action_data, action_dim).astype(np.float16)

        images.append(image)
        actions.append(action_vec)

    if len(images) < 2:
        return []

    images = np.stack(images)
    actions = np.stack(actions)
    imgs_int = images.astype(np.int16)
    diffs = np.mean(np.abs(imgs_int[1:] - imgs_int[:-1]), axis=(1, 2, 3))

    trajs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, diff in enumerate(diffs):
        start = i
        end = start + traj_len
        if diff > diff_threshold and end <= images.shape[0]:
            traj_imgs = images[start:end]
            traj_actions = actions[start:end]
            trajs.append((traj_imgs, traj_actions))

    return trajs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Borderlands dataset")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing frame images")
    parser.add_argument("--actions_dir", type=str, required=True, help="Directory containing action JSON files")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save processed trajectories")
    parser.add_argument("--diff_threshold", type=float, default=0.0, help="L1 pixel difference threshold")
    parser.add_argument("--action_dim", type=int, default=5, help="Model action dimension")
    parser.add_argument("--traj_len", type=int, default=20, help="Number of frames per trajectory")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    frame_episode_dirs = get_episode_dirs(args.frames_dir)
    action_episode_dirs = get_episode_dirs(args.actions_dir)
    assert len(frame_episode_dirs) == len(action_episode_dirs), "Mismatched episode counts"

    with_press, without_press = [], []
    for f_dir, a_dir in zip(frame_episode_dirs, action_episode_dirs):
        trajs = process_episode(
            f_dir, a_dir, args.action_dim, args.diff_threshold, args.traj_len
        )
        for imgs, acts in trajs:
            if np.any(acts):
                with_press.append((imgs, acts))
            else:
                without_press.append((imgs, acts))

    random.shuffle(without_press)
    keep_without = without_press[: len(with_press)]
    all_trajs = with_press + keep_without
    random.shuffle(all_trajs)

    traj_idx = 0
    for imgs, acts in all_trajs:
        save_name = os.path.join(args.output_path, f"traj_{traj_idx:05d}.npz")
        np.savez_compressed(save_name, **{"image": imgs, "action": acts})
        traj_idx += 1
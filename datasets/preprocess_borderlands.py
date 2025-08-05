import os
import glob
import json
import argparse
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm


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
    """Encode raw action dict into a fixed-length numeric vector."""
    vec = np.array(
        [
            data.get("dx", 0.0),
            data.get("dy", 0.0),
            data.get("wheel", 0.0),
            float(data.get("left_click", 0)),
            float(data.get("right_click", 0)),
        ],
        dtype=np.float32,
    )
    if vec.shape[0] < action_dim:
        pad = np.zeros(action_dim - vec.shape[0], dtype=np.float32)
        vec = np.concatenate([vec, pad], axis=0)
    else:
        vec = vec[:action_dim]
    return vec


def process_episode(frame_dir: str, action_dir: str, action_dim: int, diff_threshold: float) -> tuple:
    """Process a single episode and return images and actions."""
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    images, actions = [], []
    prev_image = None
    for frame_file in tqdm(frame_files, desc=f"{os.path.basename(frame_dir) or 'episode'}"):
        base = os.path.basename(frame_file)
        index = base.replace("frame_", "").replace(".jpg", "")
        action_file = os.path.join(action_dir, f"action_{index}.json")
        if not os.path.exists(action_file):
            continue

        image = np.array(Image.open(frame_file).convert("RGB"))
        with open(action_file, "r") as f:
            action_data = json.load(f)
        action_vec = encode_action(action_data, action_dim)

        if prev_image is None:
            diff = np.inf
        else:
            diff = np.mean(np.abs(image.astype(np.float32) - prev_image.astype(np.float32)))

        if diff > diff_threshold or np.any(action_vec != 0):
            images.append(image)
            actions.append(action_vec)

        prev_image = image

    if images:
        images = np.stack(images)
        actions = np.stack(actions)
    else:
        images = np.empty((0,))
        actions = np.empty((0, action_dim))
    return images, actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Borderlands dataset")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing frame images")
    parser.add_argument("--actions_dir", type=str, required=True, help="Directory containing action JSON files")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save processed episodes")
    parser.add_argument("--diff_threshold", type=float, default=0.0, help="L1 pixel difference threshold")
    parser.add_argument("--action_dim", type=int, default=5, help="Model action dimension")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    frame_episode_dirs = get_episode_dirs(args.frames_dir)
    action_episode_dirs = get_episode_dirs(args.actions_dir)
    assert len(frame_episode_dirs) == len(action_episode_dirs), "Mismatched episode counts"

    for idx, (f_dir, a_dir) in enumerate(zip(frame_episode_dirs, action_episode_dirs)):
        imgs, acts = process_episode(f_dir, a_dir, args.action_dim, args.diff_threshold)
        if imgs.size == 0:
            continue
        save_name = os.path.join(args.output_path, f"episode_{idx:05d}.npz")
        np.savez_compressed(save_name, **{"image": imgs, "action": acts})

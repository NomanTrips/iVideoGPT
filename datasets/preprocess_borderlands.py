# preprocess_borderlands.py

import os
import glob
import json
import argparse
from typing import List, Tuple, Dict, Sequence
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

KEY_LABELS = ["W", "A", "S", "D", "LMB"]

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

def encode_action(data: dict, action_dim: int) -> np.ndarray:
    """
    Encode raw action dict into a fixed-length numeric vector.

    Vector layout (new schema): [W, A, S, D, LMB] as 0/1 bits.
    Input dict should have Windows virtual-key codes under "virtual_key_codes" or "virtual_keys".
    Codes {87, 65, 83, 68, 1} map to [W, A, S, D, LMB]. Unknown codes ignored.
    """
    code_to_idx = {87: 0, 65: 1, 83: 2, 68: 3, 1: 4}
    vec = np.zeros(5, dtype=np.float32)
    codes = data.get("virtual_key_codes", data.get("virtual_keys", []))
    for code in codes or []:
        idx = code_to_idx.get(code)
        if idx is not None:
            vec[idx] = 1.0
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
    stride: int,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Returns a list of (images, actions, key_count) trajectories for an episode.
    Uses L1 frame diff to decide candidate starts; supports stride to reduce overlap.
    key_count = #steps in the trajectory where any of the 5 key bits are active.
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
    diffs = np.mean(np.abs(imgs_int[1:] - imgs_int[:-1]), axis=(1, 2, 3))  # length N-1

    trajs: List[Tuple[np.ndarray, np.ndarray, int]] = []
    i = 0
    N = images.shape[0]
    while i < len(diffs):
        start = i
        end = start + traj_len
        if diffs[i] > diff_threshold and end <= N:
            traj_imgs = images[start:end]
            traj_actions = actions[start:end]
            any_key = np.any(traj_actions[:, :5] != 0, axis=1)
            key_count = int(np.count_nonzero(any_key))
            trajs.append((traj_imgs, traj_actions, key_count))
            i += stride  # jump to reduce overlap
        else:
            i += 1
    return trajs

# ---------- binning / sampling ----------

def parse_count_bins(spec: str, traj_len: int) -> List[Tuple[int, int]]:
    """
    spec example: "0,1-4,5-9,10-14,15-20"
    Returns inclusive (lo, hi) tuples, clamped to [0, traj_len].
    """
    if not spec:
        spec = "0,1-4,5-9,10-14,15-{}".format(traj_len)
    bins = []
    for tok in spec.split(","):
        tok = tok.strip()
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            lo, hi = int(lo), int(hi)
        else:
            lo = hi = int(tok)
        lo = max(0, min(traj_len, lo))
        hi = max(0, min(traj_len, hi))
        if lo > hi:
            lo, hi = hi, lo
        bins.append((lo, hi))
    return bins

def assign_bin(k: int, bins: Sequence[Tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(bins):
        if lo <= k <= hi:
            return idx
    return len(bins) - 1  # fallback to last

def normalize_fracs(fracs: List[float]) -> List[float]:
    s = sum(fracs)
    return [f / s for f in fracs] if s > 0 else fracs

def stratified_select(
    items,                    # List[Tuple[np.ndarray, np.ndarray, int]]
    bins,                     # List[Tuple[int, int]]
    target_fracs,             # List[float]
    max_total,                # int | None
    seed: int,
):
    rng = random.Random(seed)

    # bucket indices by bin
    per_bin = {i: [] for i in range(len(bins))}
    for idx, (_, _, k) in enumerate(items):
        b = assign_bin(k, bins)
        per_bin[b].append(idx)

    before = {b: len(v) for b, v in per_bin.items()}
    total_avail = sum(before.values())
    if max_total is None or max_total <= 0 or max_total > total_avail:
        max_total = total_avail

    # normalize fracs and compute desired per bin
    target_fracs = normalize_fracs(target_fracs)
    desired = [int(np.floor(max_total * f)) for f in target_fracs]
    rem = max_total - sum(desired)  # distribute remainder among nonzero-frac bins
    for i in reversed(range(len(desired))):
        if rem == 0: break
        if target_fracs[i] > 0:
            desired[i] += 1
            rem -= 1

    # primary take
    selected_indices = []
    for b in range(len(bins)):
        rng.shuffle(per_bin[b])
        take = min(desired[b], len(per_bin[b]))
        selected_indices.extend(per_bin[b][:take])

    # only redistribute from bins with frac>0
    leftovers = max_total - len(selected_indices)
    if leftovers > 0:
        for b in reversed(range(len(bins))):
            if leftovers == 0: break
            if target_fracs[b] <= 0:   # <- key fix: don't pull from zero-frac bins
                continue
            pool = per_bin[b]
            already = set(selected_indices)
            avail_extra = [idx for idx in pool if idx not in already]
            if not avail_extra: 
                continue
            take = min(leftovers, len(avail_extra))
            selected_indices.extend(avail_extra[:take])
            leftovers -= take

    # If we still couldn't reach max_total, we just keep fewer. We do NOT backfill from zero-frac bins.
    selected = [(items[i][0], items[i][1]) for i in selected_indices]

    after = {b: 0 for b in range(len(bins))}
    for i in selected_indices:
        b = assign_bin(items[i][2], bins)
        after[b] += 1

    return selected, before, after

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Borderlands dataset with balanced action bins")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing frame images")
    parser.add_argument("--actions_dir", type=str, required=True, help="Directory containing action JSON files")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save processed trajectories")
    parser.add_argument("--diff_threshold", type=float, default=0.0, help="L1 pixel difference threshold")
    parser.add_argument("--action_dim", type=int, default=5, help="Model action dimension")
    parser.add_argument("--traj_len", type=int, default=20, help="Number of frames per trajectory")

    # NEW: overlap control & balancing knobs
    parser.add_argument("--stride", type=int, default=5, help="Advance this many frames after a kept trajectory")
    parser.add_argument("--count_bins", type=str, default=None,
                        help="Comma-separated inclusive ranges over key-step counts, e.g. '0,1-4,5-9,10-14,15-20'. "
                             "Defaults to using traj_len for last bin.")
    parser.add_argument("--target_fracs", type=str, default="0.10,0.20,0.30,0.25,0.15",
                        help="Fractions per bin (same length as bins). Will be normalized.")
    parser.add_argument("--max_total", type=int, default=0,
                        help="Optional cap on total trajectories after sampling (0 = no cap).")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_path, exist_ok=True)

    frame_episode_dirs = get_episode_dirs(args.frames_dir)
    action_episode_dirs = get_episode_dirs(args.actions_dir)
    assert len(frame_episode_dirs) == len(action_episode_dirs), "Mismatched episode counts"

    # collect all trajectories with key_count
    all_items: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for f_dir, a_dir in zip(frame_episode_dirs, action_episode_dirs):
        trajs = process_episode(
            f_dir, a_dir, args.action_dim, args.diff_threshold, args.traj_len, args.stride
        )
        all_items.extend(trajs)

    if not all_items:
        print("No trajectories found.")
        raise SystemExit(0)

    # build bins and parse fracs
    bins = parse_count_bins(args.count_bins, args.traj_len)
    fracs = [float(x.strip()) for x in args.target_fracs.split(",")]
    if len(fracs) != len(bins):
        raise ValueError(f"--target_fracs must have {len(bins)} values (got {len(fracs)})")

    # select
    selected, before, after = stratified_select(
        all_items, bins, fracs, args.max_total if args.max_total > 0 else None, args.seed
    )

    # report
    print("\n=== Bin summary (key-step counts per trajectory) ===")
    for i, (lo, hi) in enumerate(bins):
        print(f"bin {i}: [{lo:2d}-{hi:2d}]  before={before[i]:6d}  after={after[i]:6d}")
    print(f"TOTAL before={sum(before.values())}  after={sum(after.values())}")

    # save
    for traj_idx, (imgs, acts) in enumerate(selected):
        save_name = os.path.join(args.output_path, f"traj_{traj_idx:05d}.npz")
        np.savez_compressed(save_name, **{"image": imgs, "action": acts})

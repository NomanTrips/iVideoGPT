#!/usr/bin/env python3
import os, glob, json, argparse
import numpy as np
from PIL import Image

# -------- 7-bit action schema: [W, A, S, D, LMB, Space, R] --------
VK_TO_BIT = {
    87: 0,  # W
    65: 1,  # A
    83: 2,  # S
    68: 3,  # D
    1:  4,  # Left Mouse Button
    32: 5,  # Space
    82: 6,  # R
}
ACTION_DIM = 7

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

def encode_action_7bit(data: dict) -> np.ndarray:
    """Encode to 7-dim [W,A,S,D,LMB,Space,R] bits."""
    bits = np.zeros(ACTION_DIM, dtype=np.float32)
    vks = data.get("virtual_key_codes") or data.get("virtual_keys") or []
    for vk in vks:
        pos = VK_TO_BIT.get(int(vk))
        if pos is not None:
            bits[pos] = 1.0
    return bits.astype(np.float16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", required=True, help="Folder with frame_*.jpg and action_*.json")
    ap.add_argument("--out_npz", required=True, help="Output .npz path")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seg_len", type=int, default=5, help="Frames per clip (actions will be seg_len-1)")
    args = ap.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.clip_dir, "frame_*.jpg")))
    if len(frame_paths) < args.seg_len:
        raise SystemExit(f"Need at least {args.seg_len} frames, found {len(frame_paths)} in {args.clip_dir}")

    frames = []
    actions = []

    # take first seg_len frames
    for i, fp in enumerate(frame_paths[:args.seg_len]):
        # image
        with Image.open(fp) as img:
            frames.append(resize_with_letterbox(img.convert("RGB"), args.size))

        # action aligned to transition i -> i+1 (skip after last frame)
        if i < args.seg_len - 1:
            base = os.path.basename(fp)
            idx = base.replace("frame_", "").replace(".jpg", "")
            act_path = os.path.join(args.clip_dir, f"action_{idx}.json")
            if os.path.exists(act_path):
                with open(act_path, "r") as f:
                    act = json.load(f)
                actions.append(encode_action_7bit(act))
            else:
                actions.append(np.zeros(ACTION_DIM, dtype=np.float16))

    images_np = np.stack(frames, axis=0).astype(np.uint8)       # (seg_len, H, W, 3)
    actions_np = np.stack(actions, axis=0).astype(np.float16)   # (seg_len-1, 7)

    np.savez_compressed(args.out_npz, image=images_np, action=actions_np)
    print(f"wrote {args.out_npz}  shapes image={images_np.shape} action={actions_np.shape}")

if __name__ == "__main__":
    main()

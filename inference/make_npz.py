#!/usr/bin/env python3
import os, glob, json, argparse
import numpy as np
from PIL import Image

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

def norm_clip(x: float, max_abs: float) -> float:
    if max_abs <= 0: return 0.0
    x = max(-max_abs, min(max_abs, x))
    return x / max_abs

def encode_action(data: dict, max_abs_dx: float, max_abs_dy: float, max_abs_wheel: float) -> np.ndarray:
    dx = norm_clip(float(data.get("dx", 0.0)), max_abs_dx)
    dy = norm_clip(float(data.get("dy", 0.0)), max_abs_dy)
    wheel = norm_clip(float(data.get("wheel", 0.0)), max_abs_wheel)

    bits = np.zeros(NUM_BITS, dtype=np.float32)
    vks = data.get("virtual_key_codes", []) or data.get("virtual_keys", []) or []
    for vk in vks:
        pos = VK_TO_BIT.get(int(vk))
        if pos is not None:
            bits[pos] = 1.0

    return np.concatenate([np.array([dx, dy, wheel], dtype=np.float32), bits], axis=0)  # (11,)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", required=True, help="Folder with frame_*.jpg and action_*.json")
    ap.add_argument("--out_npz", required=True, help="Output .npz path")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seg_len", type=int, default=20, help="Frames per clip (actions will be seg_len-1)")
    ap.add_argument("--max_abs_dx", type=float, default=50.0)
    ap.add_argument("--max_abs_dy", type=float, default=50.0)
    ap.add_argument("--max_abs_wheel", type=float, default=10.0)
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
                actions.append(encode_action(act, args.max_abs_dx, args.max_abs_dy, args.max_abs_wheel))
            else:
                # missing action: fill zeros (11-dim)
                actions.append(np.zeros(11, dtype=np.float32))

    images_np = np.stack(frames, axis=0).astype(np.uint8)     # (20, 64, 64, 3)
    actions_np = np.stack(actions, axis=0).astype(np.float32) # (19, 11)

    np.savez_compressed(args.out_npz, image=images_np, action=actions_np)
    print(f"wrote {args.out_npz}  shapes image={images_np.shape} action={actions_np.shape}")

if __name__ == "__main__":
    main()

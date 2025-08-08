#!/usr/bin/env python3
import numpy as np
from pathlib import Path

root = Path("/home/brian/Desktop/borderlands")   # <-- change me

expected_frames  = 20
expected_actions = 19     # rows; second dim (action_dim) can vary

bad = []          # collect files that don’t match
total = 0

for f in root.rglob("*.npz"):                 # walks sub-dirs too
    total += 1
    try:
        d = np.load(f, allow_pickle=False)
        img_shape = d["image"].shape          # (T, H, W, C)
        act_shape = d["action"].shape         # (T-1, action_dim)
    except Exception as e:                    # corrupt / missing keys
        bad.append((f, str(e), None))
        continue

    if img_shape[0] != expected_frames or act_shape[0] != expected_actions:
        bad.append((f, img_shape, act_shape))

print(f"Scanned {total} .npz files – {len(bad)} mismatch(es).")
for i, (f, img_shape, act_shape) in enumerate(bad[:20]):   # show first 20
    print(f"{i+1:2}. {f}")
    print(f"    image  shape : {img_shape}")
    print(f"    action shape : {act_shape}")

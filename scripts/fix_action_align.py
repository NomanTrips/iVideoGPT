#!/usr/bin/env python3
import re
from pathlib import Path

ACTIONS_DIR = Path("C:/Users/brian/Desktop/world_model_data/run_0005_frames_alignment_fixed/actions")  # change if needed
pattern = re.compile(r"^action_(\d+)\.json$")

def main(dry_run=False):
    files = []
    for p in ACTIONS_DIR.iterdir():
        m = pattern.match(p.name)
        if m:
            files.append((int(m.group(1)), p, m.group(1)))

    if not files:
        print("No matching files found."); return

    # Keep zero-padding width from filenames
    width = max(len(numstr) for _, _, numstr in files)

    # 1) Rename all to unique temp names to avoid collisions
    temps = []
    for idx, src, _ in files:
        tmp = src.with_name(f"__tmpshift__{src.name}")
        temps.append((idx, tmp, src))
    for idx, tmp, src in temps:
        if dry_run:
            print(f"DRY: {src.name} -> {tmp.name}")
        else:
            src.rename(tmp)

    # 2) From temps, shift everything down by 1; delete idx==0
    for idx, tmp, _ in sorted(temps, key=lambda x: x[0]):
        if idx == 0:
            if dry_run:
                print(f"DRY: delete {tmp.name}")
            else:
                tmp.unlink(missing_ok=True)
            continue
        dst_name = f"action_{idx-1:0{width}d}.json"
        dst = ACTIONS_DIR / dst_name
        if dry_run:
            print(f"DRY: {tmp.name} -> {dst_name}")
        else:
            tmp.rename(dst)

if __name__ == "__main__":
    # Set dry_run=True to preview first
    main(dry_run=False)

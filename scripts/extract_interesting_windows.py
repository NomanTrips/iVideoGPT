#!/usr/bin/env python3
# extract_interesting_simple.py
import argparse, json, re, shutil
from pathlib import Path

PAT_FRAME = None  # set after reading --frame-ext
PAT_ACTION = re.compile(r"^action_(\d+)\.json$")

def load_indexed(paths, pat):
    out = {}
    for p in paths:
        m = pat.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Extract non-overlapping 5-frame/5-action episodes when any target VK appears."
    )
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--actions-dir", type=Path, required=True)
    ap.add_argument("--out-frames-dir", type=Path, required=True)
    ap.add_argument("--out-actions-dir", type=Path, required=True)
    ap.add_argument("--window-len", type=int, default=5, help="Episode length (frames/actions).")
    ap.add_argument("--frame-ext", type=str, default="jpg", help="Frame extension (jpg/png).")
    ap.add_argument("--vk", type=int, nargs="+",
                    default=[87, 65, 83, 68, 1, 32, 82],  # W A S D LMB Space R
                    help="Virtual key codes considered interesting.")
    ap.add_argument("--overwrite", action="store_true", help="Clear output dirs first.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions only.")
    args = ap.parse_args()

    W = args.window_len
    if W <= 0:
        raise SystemExit("window-len must be > 0")

    frame_ext = args.frame_ext.lstrip(".")
    global PAT_FRAME
    PAT_FRAME = re.compile(rf"^frame_(\d+)\.{re.escape(frame_ext)}$")

    frames_dir, actions_dir = args.frames_dir, args.actions_dir
    ofr, oac = args.out_frames_dir, args.out_actions_dir

    if not frames_dir.is_dir() or not actions_dir.is_dir():
        raise SystemExit("frames-dir and actions-dir must exist.")

    frame_paths = sorted(frames_dir.glob(f"frame_*.{frame_ext}"))
    action_paths = sorted(actions_dir.glob("action_*.json"))
    if not frame_paths or not action_paths:
        raise SystemExit("No frames or actions found.")

    frames = load_indexed(frame_paths, PAT_FRAME)
    actions = load_indexed(action_paths, PAT_ACTION)
    common = sorted(set(frames.keys()) & set(actions.keys()))
    if not common:
        raise SystemExit("No overlapping indices between frames and actions.")

    # Prepare outputs
    for outdir in (ofr, oac):
        if outdir.exists():
            if args.overwrite:
                for p in outdir.iterdir():
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
            else:
                if any(outdir.iterdir()):
                    raise SystemExit(f"{outdir} not empty. Use --overwrite.")
        else:
            outdir.mkdir(parents=True, exist_ok=True)

    target_vk = set(args.vk)

    # Precompute "interesting" flags per index
    interesting = {}
    for i in common:
        try:
            with open(actions[i], "r", encoding="utf-8") as f:
                data = json.load(f)
            vks = set(data.get("virtual_key_codes", []))
        except Exception:
            vks = set()
        interesting[i] = bool(vks & target_vk)

    min_i, max_i = common[0], common[-1]
    i = min_i
    out_counter = 0
    episodes = 0

    while i + W - 1 <= max_i:
        window = list(range(i, i + W))
        # ensure a full consecutive window exists in both sets
        if all(j in frames and j in actions for j in window):
            hit = any(interesting.get(j, False) for j in window)
            if hit:
                # copy out this episode
                for j in window:
                    srcf = frames[j]
                    srca = actions[j]
                    dstf = ofr / f"frame_{out_counter:06d}.{frame_ext}"
                    dsta = oac / f"action_{out_counter:06d}.json"
                    if args.dry_run:
                        print(f"[DRY] {srcf.name} -> {dstf.name} | {srca.name} -> {dsta.name}")
                    else:
                        shutil.copy2(srcf, dstf)
                        shutil.copy2(srca, dsta)
                    out_counter += 1
                episodes += 1
                # non-overlap: jump to the first index AFTER this window
                i += W
                continue
        # no hit or missing files: advance by 1
        i += 1

    print(f"Episodes written: {episodes} (window={W}, non-overlap).")
    print(f"Total frames/actions copied: {episodes * W} each.")
    print(f"Output -> Frames: {ofr} | Actions: {oac}")

if __name__ == "__main__":
    main()

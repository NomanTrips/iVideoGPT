#!/usr/bin/env python3
# shift_actions.py
import argparse
import re
from pathlib import Path
from typing import List, Tuple

PATTERN = re.compile(r"^action_(\d+)\.json$")


def collect_files(actions_dir: Path) -> List[Tuple[int, Path, str]]:
    files = []
    for p in actions_dir.iterdir():
        m = PATTERN.match(p.name)
        if m:
            files.append((int(m.group(1)), p, m.group(1)))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Shift indices in action_########.json filenames forward or reverse."
    )
    parser.add_argument(
        "actions_dir",
        type=Path,
        help="Path to directory containing action_*.json files",
    )
    parser.add_argument(
        "--direction",
        choices=["forward", "reverse"],
        required=True,
        help="Direction to shift indices: forward = add, reverse = subtract",
    )
    parser.add_argument(
        "--magnitude",
        type=int,
        required=True,
        help="How many steps to shift indices by (>= 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without writing changes",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="If a destination file exists and is NOT one of the files being shifted, overwrite it",
    )

    args = parser.parse_args()
    actions_dir: Path = args.actions_dir
    direction: str = args.direction
    mag: int = args.magnitude
    dry_run: bool = args.dry_run
    allow_overwrite: bool = args.allow_overwrite

    if mag < 0:
        raise SystemExit("magnitude must be >= 0")
    if not actions_dir.is_dir():
        raise SystemExit(f"Not a directory: {actions_dir}")

    files = collect_files(actions_dir)
    if not files:
        print("No matching files found.")
        return

    # Keep original zero-padding width, but expand if needed by new indices
    orig_width = max(len(numstr) for _, _, numstr in files)

    # Compute new index mapping
    mapping = []  # (idx_old, path_tmp, path_src, idx_new)
    negatives_skipped = []
    new_indices = []
    for idx, src, _ in files:
        if direction == "forward":
            new_idx = idx + mag
        else:
            new_idx = idx - mag

        if new_idx < 0:
            negatives_skipped.append(src.name)
            continue
        new_indices.append(new_idx)

    if not new_indices:
        print("Nothing to rename (all targets would be negative).")
        return

    new_width = max(orig_width, max(len(str(n)) for n in new_indices))

    # Build proposed destination paths and detect conflicts with non-batch files
    source_names = {p.name for _, p, _ in files}
    proposed = {}
    for idx, src, _ in files:
        if direction == "forward":
            new_idx = idx + mag
        else:
            new_idx = idx - mag
        if new_idx < 0:
            continue
        dst_name = f"action_{new_idx:0{new_width}d}.json"
        dst = actions_dir / dst_name
        proposed[src] = dst

    conflicts = []
    for src, dst in proposed.items():
        if dst.exists() and dst.name not in source_names:
            conflicts.append(dst.name)

    if conflicts and not allow_overwrite:
        print("Aborting due to conflicts with existing non-batch files:")
        for name in sorted(conflicts):
            print("  ", name)
        print("Re-run with --allow-overwrite to overwrite those files.")
        return

    # Stage 1: rename every source to a unique temp to avoid collisions
    temps: List[Tuple[int, Path, Path]] = []
    for idx, src, _ in files:
        tmp = src.with_name(f"__tmpshift__{src.name}")
        temps.append((idx, tmp, src))

    for _, tmp, src in temps:
        if dry_run:
            print(f"DRY: {src.name} -> {tmp.name}")
        else:
            src.rename(tmp)

    # Stage 2: rename temps to final destinations (skip negatives)
    # Sort by old idx just for consistent output
    for idx, tmp, _ in sorted(temps, key=lambda x: x[0]):
        if direction == "forward":
            new_idx = idx + mag
        else:
            new_idx = idx - mag

        if new_idx < 0:
            if dry_run:
                print(f"DRY: delete {tmp.name} (negative target)")
            else:
                tmp.unlink(missing_ok=True)
            continue

        dst = actions_dir / f"action_{new_idx:0{new_width}d}.json"

        if dst.exists() and dst.name not in source_names:
            if allow_overwrite:
                if dry_run:
                    print(f"DRY: remove existing {dst.name} (overwrite)")
                else:
                    dst.unlink()
            else:
                # Shouldn't happen due to earlier guard, but double-check
                if dry_run:
                    print(f"DRY: SKIP {tmp.name} -> {dst.name} (conflict)")
                else:
                    # try to restore tmp back to original name for safety
                    orig = actions_dir / tmp.name.replace("__tmpshift__", "")
                    tmp.rename(orig)
                continue

        if dry_run:
            print(f"DRY: {tmp.name} -> {dst.name}")
        else:
            tmp.rename(dst)

    if negatives_skipped:
        print(
            f"Skipped {len(negatives_skipped)} file(s) that would go negative in reverse:"
        )
        for n in sorted(negatives_skipped)[:10]:
            print("  ", n)
        if len(negatives_skipped) > 10:
            print("  ...")


if __name__ == "__main__":
    main()

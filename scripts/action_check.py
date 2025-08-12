#!/usr/bin/env python3
import os, glob, argparse, numpy as np
from collections import Counter, defaultdict

def summarize_actions(npz_dir, round_decimals=3, max_suspicious=20, sample=None):
    paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if sample is not None:
        paths = paths[:sample]
    if not paths:
        print("No .npz files found."); return

    total_files = 0
    total_steps = 0
    action_dim = None

    # global stats
    mins = None; maxs = None; sums = None; sqsums = None
    nonzero_counts = None

    # for key-bit rates (assume bits start at index 3 if action_dim>=11; otherwise infer last 8 dims)
    bit_start = None; bit_end = None

    # uniqueness/diversity
    rounded_pattern_counter = Counter()
    file_unique_counts = []
    suspicious_files = []

    shape_mismatch = []
    missing_keys = []

    for p in paths:
        try:
            data = np.load(p)
        except Exception as e:
            print(f"READ ERROR {p}: {e}")
            continue

        if "action" not in data:
            missing_keys.append(p); continue
        A = data["action"]
        if A.ndim != 2:
            shape_mismatch.append((p, A.shape)); continue

        T, D = A.shape
        if action_dim is None:
            action_dim = D
            # Heuristic: 3 continuous + 8 bits if possible
            if D >= 11:
                bit_start, bit_end = 3, 11
            elif D > 3:
                bit_start, bit_end = 3, D
            else:
                bit_start, bit_end = D, D  # no bits

            mins = np.full(D, np.inf, dtype=np.float64)
            maxs = np.full(D, -np.inf, dtype=np.float64)
            sums = np.zeros(D, dtype=np.float64)
            sqsums = np.zeros(D, dtype=np.float64)
            nonzero_counts = np.zeros(D, dtype=np.int64)

        if A.shape[1] != action_dim:
            shape_mismatch.append((p, A.shape)); continue

        total_files += 1
        total_steps += T

        # per-dim stats
        mins = np.minimum(mins, A.min(axis=0))
        maxs = np.maximum(maxs, A.max(axis=0))
        sums += A.sum(axis=0)
        sqsums += (A**2).sum(axis=0)
        nonzero_counts += (A != 0).sum(axis=0)

        # per-file uniqueness (rounded to be tolerant to tiny float noise)
        A_round = np.round(A, round_decimals)
        # Use rows as tuples for hashing
        tuples = list(map(tuple, A_round))
        uniq = len(set(tuples))
        file_unique_counts.append((p, uniq, T))
        if uniq <= max(1, T//10):  # <=10% unique within file is suspicious
            suspicious_files.append((p, uniq, T))

        # global diversity (downsample extreme cases by counting)
        rounded_pattern_counter.update(tuples)

    if total_files == 0:
        print("No valid .npz with 'action' found.")
        return

    means = sums / max(1, total_steps)
    stds = np.sqrt(np.maximum(0.0, sqsums / max(1, total_steps) - means**2))
    nz_rates = nonzero_counts / max(1, total_steps)

    print("\n=== Dataset Action Sanity Summary ===")
    print(f"Files scanned          : {total_files}")
    print(f"Total action steps     : {total_steps}")
    print(f"Action dim             : {action_dim}")
    print(f"Bit indices (heuristic): [{bit_start}, {bit_end})")

    # Continuous channels: assume first three are dx,dy,wheel if dim>=3
    if action_dim >= 1:
        print("\nPer-dimension stats (min / max | mean ± std | nonzero rate):")
        for i in range(action_dim):
            tag = ("dx","dy","wheel")[i] if i < 3 else (f"bit[{i-3}]" if bit_start<=i<bit_end else f"feat[{i}]")
            print(f"  {i:02d} {tag:>7}: {mins[i]: .3f} / {maxs[i]: .3f} | {means[i]: .3f} ± {stds[i]: .3f} | nz={nz_rates[i]*100:5.1f}%")

    # Bit activation summary
    if bit_end > bit_start:
        bit_slice = slice(bit_start, bit_end)
        bit_nz_rates = nz_rates[bit_slice]
        print("\nKey-bit activation rates (percent of steps with bit==1):")
        for j, rate in enumerate(bit_nz_rates):
            print(f"  bit[{j}]: {rate*100:5.2f}%")

        any_key_rate = 100.0 * (rounded_pattern_counter.total() - rounded_pattern_counter.get(tuple([0.0]*action_dim), 0)) / max(1, rounded_pattern_counter.total())
        print(f"\nAny-key-pressed rate   : {any_key_rate:5.2f}% (approx, rounded patterns)")

    # File-level variability
    uniq_counts = np.array([u for _, u, _ in file_unique_counts], dtype=np.int64)
    T_counts = np.array([t for _, _, t in file_unique_counts], dtype=np.int64)
    frac_unique = (uniq_counts / np.maximum(1, T_counts))
    print("\nPer-file uniqueness (rounded rows):")
    print(f"  Median unique/len    : {np.median(frac_unique)*100:5.1f}%")
    print(f"  Files <=10% unique   : {sum(frac_unique<=0.10)}")

    if suspicious_files:
        print("\nExamples of low-variation files:")
        for p, u, t in suspicious_files[:max_suspicious]:
            print(f"  {os.path.basename(p)} : unique={u}/{t}")

    # Global diversity (coarse)
    num_unique_patterns = len(rounded_pattern_counter)
    top_common = rounded_pattern_counter.most_common(10)
    print(f"\nRounded unique action rows (global): {num_unique_patterns}")
    print("Most common patterns (count, first few dims):")
    for (row, c) in top_common:
        preview = ", ".join(map(lambda x: f"{x:.3f}", row[:min(6, len(row))]))
        print(f"  {c:6d} | [{preview}{'...' if len(row)>6 else ''}]")

    # Warnings
    if np.allclose(stds[:min(3, action_dim)], 0.0):
        print("\nWARNING: continuous channels (dx/dy/wheel) look constant.")
    if bit_end > bit_start and np.all(bit_nz_rates < 0.001):
        print("WARNING: key bits are almost never active.")
    if sum(frac_unique<=0.10) > 0.2*len(frac_unique):
        print("WARNING: many files have very low action variation (check alignment).")

    if shape_mismatch:
        print(f"\nNOTE: {len(shape_mismatch)} files had unexpected action shapes (first 5 shown):")
        for p, sh in shape_mismatch[:5]:
            print(f"  {os.path.basename(p)} -> {sh}")
    if missing_keys:
        print(f"NOTE: {len(missing_keys)} files missing 'action' array (first 5 shown):")
        for p in missing_keys[:5]:
            print(f"  {os.path.basename(p)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="Directory with traj_*.npz")
    ap.add_argument("--round_decimals", type=int, default=3, help="Rounding for uniqueness hashing")
    ap.add_argument("--sample", type=int, default=None, help="Optional: limit to first N files")
    args = ap.parse_args()
    summarize_actions(args.npz_dir, round_decimals=args.round_decimals, sample=args.sample)

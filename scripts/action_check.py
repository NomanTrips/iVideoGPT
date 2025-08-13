#!/usr/bin/env python3
import os, glob, argparse, numpy as np
from collections import Counter

KEY_LABELS = ["W", "A", "S", "D", "LMB"]

def parse_thresholds(s: str | None):
    if not s:
        return [1, 3, 5, 10, 20, 50]
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out or [1, 3, 5, 10, 20, 50]

def summarize_actions(npz_dir, round_decimals=3, max_suspicious=20, sample=None, cdf_thresholds=None):
    paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if sample is not None:
        paths = paths[:sample]
    if not paths:
        print("No .npz files found."); return

    total_files = 0
    total_steps = 0
    total_any_key_steps = 0
    action_dim = None

    # global stats
    mins = None; maxs = None; sums = None; sqsums = None
    nonzero_counts = None

    # bit slice and labels
    bit_start = None; bit_end = None
    dim_labels = None

    # uniqueness/diversity
    rounded_pattern_counter = Counter()
    file_unique_counts = []
    suspicious_files = []

    shape_mismatch = []
    missing_keys = []

    # episode-level key counts
    episode_key_counts = []   # number of steps with any key active, per file
    episode_lengths = []      # T per file

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

            # If you are on the new schema, it's 5 bits: W,A,S,D,LMB
            if D == 5:
                bit_start, bit_end = 0, 5
                dim_labels = KEY_LABELS[:]
            else:
                # fallback heuristic: 3 continuous + remainder bits if possible
                if D >= 11:
                    bit_start, bit_end = 3, 11
                elif D > 3:
                    bit_start, bit_end = 3, D
                else:
                    bit_start, bit_end = D, D  # no bits
                # labels for legacy
                dim_labels = []
                for i in range(D):
                    if i == 0: dim_labels.append("dx")
                    elif i == 1: dim_labels.append("dy")
                    elif i == 2: dim_labels.append("wheel")
                    elif bit_start <= i < bit_end:
                        dim_labels.append(f"bit[{i-3}]")
                    else:
                        dim_labels.append(f"feat[{i}]")

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
        tuples = list(map(tuple, A_round))
        uniq = len(set(tuples))
        file_unique_counts.append((p, uniq, T))
        if uniq <= max(1, T//10):  # <=10% unique within file is suspicious
            suspicious_files.append((p, uniq, T))

        # global diversity (downsample extreme cases by counting)
        rounded_pattern_counter.update(tuples)

        # episode key-step count
        if bit_end > bit_start:
            keys = A[:, bit_start:bit_end]
            # treat any nonzero as pressed; robust to float 0/1
            any_key = np.any(keys != 0, axis=1)
            k = int(np.count_nonzero(any_key))
            total_any_key_steps += k
            episode_key_counts.append(k)
            episode_lengths.append(T)
        else:
            # no bits; count is 0
            episode_key_counts.append(0)
            episode_lengths.append(T)

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
    print(f"Bit indices            : [{bit_start}, {bit_end})")
    if action_dim == 5 and bit_start == 0:
        print("Schema                 : 5 key bits [W, A, S, D, LMB]")

    # Per-dimension stats
    if action_dim >= 1:
        print("\nPer-dimension stats (min / max | mean ± std | nonzero rate):")
        for i in range(action_dim):
            tag = dim_labels[i] if dim_labels and i < len(dim_labels) else f"feat[{i}]"
            print(f"  {i:02d} {tag:>7}: {mins[i]: .3f} / {maxs[i]: .3f} | {means[i]: .3f} ± {stds[i]: .3f} | nz={nz_rates[i]*100:5.1f}%")

    # Bit activation summary
    if bit_end > bit_start:
        bit_slice = slice(bit_start, bit_end)
        bit_nz_rates = nz_rates[bit_slice]
        print("\nKey-bit activation rates (percent of steps with bit==1):")
        for j in range(bit_end - bit_start):
            name = KEY_LABELS[j] if (action_dim == 5 and bit_start == 0) else f"bit[{j}]"
            print(f"  {name:>3}: {bit_nz_rates[j]*100:5.2f}%")

        any_key_rate = 100.0 * total_any_key_steps / max(1, total_steps)
        print(f"\nAny-key-pressed rate   : {any_key_rate:5.2f}% (step-level)")

    # File-level variability
    uniq_counts = np.array([u for _, u, _ in file_unique_counts], dtype=np.int64)
    T_counts = np.array([t for _, _, t in file_unique_counts], dtype=np.int64)
    frac_unique = (uniq_counts / np.maximum(1, T_counts))
    print("\nPer-file uniqueness (rounded rows):")
    print(f"  Median unique/len    : {np.median(frac_unique)*100:5.1f}%")
    print(f"  Files <=10% unique   : {int(np.sum(frac_unique<=0.10))}")

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

    # Episode-level key-step distribution
    key_counts = np.array(episode_key_counts, dtype=np.int64)
    lengths = np.array(episode_lengths, dtype=np.int64)
    key_rates = key_counts / np.maximum(1, lengths)

    if bit_end > bit_start:
        print("\nEpisode-level key-step counts (any key pressed in a step):")
        pct = lambda q: np.percentile(key_counts, q)
        print(f"  median steps w/ keys : {int(round(pct(50)))}")
        print(f"  p25 / p75            : {int(round(pct(25)))} / {int(round(pct(75)))}")
        print(f"  p90 / p99            : {int(round(pct(90)))} / {int(round(pct(99)))}")

        # CDF on absolute counts
        thresholds = parse_thresholds(cdf_thresholds)
        print("\nCDF: episodes with < N key-steps")
        for N in thresholds:
            frac = np.mean(key_counts < N) * 100.0
            print(f"   N<{N:>3}: {frac:5.1f}%")

        # Also show rates (normalized by episode length)
        print("\nEpisodes with low key density (fraction of steps with keys):")
        for r in [0.01, 0.05, 0.10, 0.25, 0.50]:
            frac = np.mean(key_rates < r) * 100.0
            print(f"   rate<{r:>4.2f}: {frac:5.1f}%")

    # Notes
    if shape_mismatch:
        print(f"\nNOTE: {len(shape_mismatch)} files had unexpected action shapes (first 5 shown):")
        for p, sh in shape_mismatch[:5]:
            print(f"  {os.path.basename(p)} -> {sh}")
    if missing_keys:
        print(f"NOTE: {len(missing_keys)} files missing 'action' array (first 5 shown):")
        for p in missing_keys[:5]:
            print(f"  {os.path.basename(p)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="Directory with traj_*.npz")
    ap.add_argument("--round_decimals", type=int, default=3, help="Rounding for uniqueness hashing")
    ap.add_argument("--sample", type=int, default=None, help="Optional: limit to first N files")
    ap.add_argument("--cdf_thresholds", type=str, default=None,
                    help="Comma-separated N values for CDF, e.g. '1,5,10,20'")
    args = ap.parse_args()
    summarize_actions(
        args.npz_dir,
        round_decimals=args.round_decimals,
        sample=args.sample,
        cdf_thresholds=args.cdf_thresholds
    )

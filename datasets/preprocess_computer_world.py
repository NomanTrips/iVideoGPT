#!/usr/bin/env python3
"""Preprocess computer task episodes into .npz trajectory files.

This script mirrors ``preprocess_borderlands.py`` but targets a richer action
schema suitable for general computer-interaction demonstrations.  Each action
contains mouse deltas, mouse wheel motion, and a set of pressed virtual key
codes (Windows VK values).  Frames are resized to a configurable resolution and
paired with their corresponding actions to produce fixed-length trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Action schema ----------------------------------------------------------------
# ---------------------------------------------------------------------------
# Default keyboard layout: digits, letters, and common modifiers/punctuation.
# Virtual key code reference: https://cherrytree.at/misc/vk.htm


def _digit_key_schema() -> List[tuple[int, str]]:
    return [(code, chr(code)) for code in range(ord("0"), ord("9") + 1)]


def _letter_key_schema() -> List[tuple[int, str]]:
    return [(code, chr(code)) for code in range(ord("A"), ord("Z") + 1)]


PUNCTUATION_KEYS: Sequence[tuple[int, str]] = (
    (186, ";"),  # VK_OEM_1
    (187, "="),  # VK_OEM_PLUS
    (188, ","),  # VK_OEM_COMMA
    (189, "-"),  # VK_OEM_MINUS
    (190, "."),  # VK_OEM_PERIOD
    (191, "/"),  # VK_OEM_2
    (192, "`"),  # VK_OEM_3
    (219, "["),  # VK_OEM_4
    (220, "\\"),  # VK_OEM_5
    (221, "]"),  # VK_OEM_6
    (222, "'"),  # VK_OEM_7
)

MODIFIER_KEYS: Sequence[tuple[int, str]] = (
    (8, "Backspace"),
    (9, "Tab"),
    (13, "Enter"),
    (16, "Shift"),
    (17, "Ctrl"),
    (18, "Alt"),
    (20, "CapsLock"),
    (27, "Esc"),
    (32, "Space"),
)

NAVIGATION_KEYS: Sequence[tuple[int, str]] = (
    (33, "PageUp"),
    (34, "PageDown"),
    (35, "End"),
    (36, "Home"),
    (37, "Left"),
    (38, "Up"),
    (39, "Right"),
    (40, "Down"),
    (45, "Insert"),
    (46, "Delete"),
)

DEFAULT_KEY_SCHEMA: List[tuple[int, str]] = (
    _digit_key_schema()
    + _letter_key_schema()
    + list(PUNCTUATION_KEYS)
    + list(MODIFIER_KEYS)
    + list(NAVIGATION_KEYS)
)

# Map specific VK variants (e.g. left/right modifier keys) back to canonical
# codes included in DEFAULT_KEY_SCHEMA so they share the same action index.
KEY_ALIASES: Dict[int, int] = {
    160: 16,  # VK_LSHIFT -> Shift
    161: 16,  # VK_RSHIFT -> Shift
    162: 17,  # VK_LCONTROL -> Ctrl
    163: 17,  # VK_RCONTROL -> Ctrl
    164: 18,  # VK_LMENU -> Alt
    165: 18,  # VK_RMENU -> Alt
}


@dataclass
class ActionEncoder:
    """Encodes mouse deltas and keyboard presses into a fixed-size vector."""

    key_schema: Sequence[tuple[int, str]]
    pad_to: int | None = None
    mouse_scale: float = 100.0
    mouse_clip: float | None = 1.0
    wheel_scale: float = 10.0
    wheel_clip: float | None = 1.0

    def __post_init__(self) -> None:
        self._offset = 3  # [dx, dy, wheel] occupy the first three slots
        self._code_to_index: Dict[int, int] = {}
        canonical_order: List[int] = []

        for vk_code, _ in self.key_schema:
            canonical = KEY_ALIASES.get(vk_code, vk_code)
            if canonical not in self._code_to_index:
                canonical_order.append(canonical)
                self._code_to_index[canonical] = self._offset + len(canonical_order) - 1

        for alias, canonical in KEY_ALIASES.items():
            if canonical in self._code_to_index:
                self._code_to_index[alias] = self._code_to_index[canonical]

        self._base_action_dim = self._offset + len(canonical_order)
        if self.pad_to is not None:
            if self.pad_to < self._base_action_dim:
                raise ValueError(
                    f"pad_to={self.pad_to} is smaller than the base action dimension "
                    f"{self._base_action_dim}."
                )
            self.action_dim = self.pad_to
        else:
            self.action_dim = self._base_action_dim

    # ------------------------------------------------------------------
    # Helpers
    def _normalize(self, value: float, scale: float, clip: float | None) -> float:
        if scale == 0:
            scaled = float(value)
        else:
            scaled = float(value) / float(scale)
        if clip is None or clip <= 0:
            return float(scaled)
        return float(np.clip(scaled, -clip, clip))

    # ------------------------------------------------------------------
    def encode(self, data: Dict) -> np.ndarray:
        vec = np.zeros(self.action_dim, dtype=np.float32)

        vec[0] = self._normalize(data.get("dx", 0.0), self.mouse_scale, self.mouse_clip)
        vec[1] = self._normalize(data.get("dy", 0.0), self.mouse_scale, self.mouse_clip)
        vec[2] = self._normalize(data.get("wheel", 0.0), self.wheel_scale, self.wheel_clip)

        codes: Iterable = data.get("virtual_key_codes") or []
        for code in codes:
            try:
                vk_code = int(code)
            except (TypeError, ValueError):
                continue
            idx = self._code_to_index.get(vk_code)
            if idx is not None and idx < self.action_dim:
                vec[idx] = 1.0
        return vec.astype(np.float16)


# ---------------------------------------------------------------------------
# Image utilities ------------------------------------------------------------
# ---------------------------------------------------------------------------

def resize_frame(img: Image.Image, width: int, height: int) -> np.ndarray:
    """Resize an RGB PIL image to ``(width, height)`` using bilinear sampling."""

    resized = img.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Dataset traversal ----------------------------------------------------------
# ---------------------------------------------------------------------------

def get_episode_dirs(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    subs = sorted(
        [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )
    return subs if subs else [base_dir]


def index_from_name(path: str, prefix: str, suffix: str) -> int:
    name = os.path.basename(path)
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return -1
    mid = name[len(prefix) : -len(suffix)]
    try:
        return int(mid)
    except ValueError:
        return -1


def paired_indices(frame_dir: str, action_dir: str) -> List[int]:
    frame_files = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.startswith("frame_") and f.endswith(".jpg")]
    action_files = [
        os.path.join(action_dir, f) for f in os.listdir(action_dir) if f.startswith("action_") and f.endswith(".json")
    ]

    frame_idx = {index_from_name(p, "frame_", ".jpg") for p in frame_files}
    action_idx = {index_from_name(p, "action_", ".json") for p in action_files}

    common = sorted(i for i in frame_idx.intersection(action_idx) if i >= 0)

    missing_frames = sorted(i for i in action_idx - frame_idx if i >= 0)
    missing_actions = sorted(i for i in frame_idx - action_idx if i >= 0)

    if missing_frames:
        preview = ", ".join(str(i) for i in missing_frames[:10])
        print(
            f"[warn] {os.path.basename(frame_dir) or frame_dir}: missing frames for indices: {preview}"
            f"{' ...' if len(missing_frames) > 10 else ''}"
        )
    if missing_actions:
        preview = ", ".join(str(i) for i in missing_actions[:10])
        print(
            f"[warn] {os.path.basename(action_dir) or action_dir}: missing actions for indices: {preview}"
            f"{' ...' if len(missing_actions) > 10 else ''}"
        )

    return common


# ---------------------------------------------------------------------------
# Episode processing ---------------------------------------------------------
# ---------------------------------------------------------------------------

def process_episode(
    frame_dir: str,
    action_dir: str,
    output_path: str,
    encoder: ActionEncoder,
    traj_len: int,
    width: int,
    height: int,
    start_traj_idx: int,
) -> int:
    common = paired_indices(frame_dir, action_dir)
    if not common:
        return start_traj_idx

    buffer_imgs: List[np.ndarray] = []
    buffer_actions: List[np.ndarray] = []
    traj_idx = start_traj_idx

    for idx in common:
        frame_path = os.path.join(frame_dir, f"frame_{idx:06d}.jpg")
        action_path = os.path.join(action_dir, f"action_{idx:06d}.json")

        with Image.open(frame_path) as img:
            frame_arr = resize_frame(img.convert("RGB"), width, height)
        with open(action_path, "r", encoding="utf-8") as f:
            action_arr = encoder.encode(json.load(f))

        buffer_imgs.append(frame_arr)
        buffer_actions.append(action_arr)

        if len(buffer_imgs) == traj_len:
            stacked_imgs = np.stack(buffer_imgs, axis=0)
            stacked_actions = np.stack(buffer_actions, axis=0)
            traj_name = f"traj_{traj_idx:05d}.npz"
            np.savez_compressed(
                os.path.join(output_path, traj_name),
                **{"image": stacked_imgs, "action": stacked_actions},
            )
            traj_idx += 1
            buffer_imgs.clear()
            buffer_actions.clear()

    if buffer_imgs:
        print(
            f"[info] Dropping leftover {len(buffer_imgs)} frames (< traj_len) in "
            f"{os.path.basename(frame_dir) or frame_dir}"
        )

    return traj_idx


# ---------------------------------------------------------------------------
# Command line ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess computer interaction episodes into fixed-length "
            "trajectories saved as .npz files."
        )
    )
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory that stores episode frame folders")
    parser.add_argument("--actions_dir", type=str, required=True, help="Directory that stores episode action folders")
    parser.add_argument("--output_path", type=str, required=True, help="Destination directory for npz trajectories")
    parser.add_argument("--traj_len", type=int, default=16, help="Number of steps per trajectory (default: 16)")
    parser.add_argument("--width", type=int, default=256, help="Output frame width (default: 256)")
    parser.add_argument("--height", type=int, default=192, help="Output frame height (default: 192)")
    parser.add_argument("--mouse_scale", type=float, default=100.0, help="Divisor used to normalize mouse dx/dy")
    parser.add_argument(
        "--mouse_clip",
        type=float,
        default=1.0,
        help="Clip value applied after mouse normalization (<=0 disables)",
    )
    parser.add_argument("--wheel_scale", type=float, default=10.0, help="Divisor used to normalize mouse wheel deltas")
    parser.add_argument(
        "--wheel_clip",
        type=float,
        default=1.0,
        help="Clip value applied after wheel normalization (<=0 disables)",
    )
    parser.add_argument(
        "--pad_action_dim",
        type=int,
        default=None,
        help="Optional dimension to pad/extend the action vector to (must be >= base size)",
    )
    parser.add_argument(
        "--extra_key_codes",
        type=int,
        nargs="*",
        default=None,
        help="Additional virtual-key codes to include in the action vector",
    )
    return parser.parse_args()


def build_key_schema(extra_codes: Iterable[int] | None) -> List[tuple[int, str]]:
    schema = list(DEFAULT_KEY_SCHEMA)
    if extra_codes:
        for code in extra_codes:
            canonical = KEY_ALIASES.get(code, code)
            if canonical not in {existing_code for existing_code, _ in schema}:
                schema.append((canonical, f"VK_{canonical}"))
    return schema


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    key_schema = build_key_schema(args.extra_key_codes)
    encoder = ActionEncoder(
        key_schema=key_schema,
        pad_to=args.pad_action_dim,
        mouse_scale=args.mouse_scale,
        mouse_clip=args.mouse_clip,
        wheel_scale=args.wheel_scale,
        wheel_clip=args.wheel_clip,
    )

    frame_dirs = get_episode_dirs(args.frames_dir)
    action_dirs = get_episode_dirs(args.actions_dir)

    if len(frame_dirs) != len(action_dirs):
        raise RuntimeError(
            f"Mismatched episode directories: {len(frame_dirs)} frames vs {len(action_dirs)} actions"
        )

    traj_idx = 0
    total = 0
    for frame_dir, action_dir in zip(frame_dirs, action_dirs):
        before = traj_idx
        traj_idx = process_episode(
            frame_dir,
            action_dir,
            args.output_path,
            encoder,
            args.traj_len,
            args.width,
            args.height,
            traj_idx,
        )
        written = traj_idx - before
        total += written
        print(f"[done] {os.path.basename(frame_dir) or frame_dir}: wrote {written} trajectories")

    print(
        f"[total] wrote {total} trajectories to {args.output_path}; "
        f"action_dim={encoder.action_dim}, frame_shape=({args.height},{args.width},3)"
    )


if __name__ == "__main__":
    main()

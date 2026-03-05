#!/usr/bin/env python3
"""Visualize recorded robomimic/LIBERO-style HDF5 demos.

Example structure (common):
  /data/demo_0/obs/eye_in_hand_rgb   (T,H,W,3)
  /data/demo_0/obs/agentview_rgb     (T,H,W,3)
  /data/demo_0/actions              (T,7)

Controls:
  q / ESC : quit
  space   : pause/resume
  n       : next frame (when paused)
  p       : prev frame (when paused)
  [ / ]   : slower / faster
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


_H5PY = None


def _h5py():
    global _H5PY
    if _H5PY is None:
        import h5py as _mod

        _H5PY = _mod
    return _H5PY


def _as_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize various image tensor layouts to HxWxC uint8."""
    if img is None:
        raise ValueError("img is None")

    arr = np.asarray(img)

    # Handle CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    # Handle grayscale HxW -> HxWx1
    if arr.ndim == 2:
        arr = arr[:, :, None]

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image ndim={arr.ndim}, shape={arr.shape}")

    # If float [0,1] or [0,255]
    if np.issubdtype(arr.dtype, np.floating):
        maxv = float(np.nanmax(arr)) if arr.size else 0.0
        if maxv <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)

    # If uint16 depth-like, normalize for viewing
    if arr.dtype == np.uint16:
        # simple min-max normalization per-frame
        mn = int(arr.min()) if arr.size else 0
        mx = int(arr.max()) if arr.size else 1
        if mx <= mn:
            mx = mn + 1
        arr8 = ((arr.astype(np.float32) - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
        arr = arr8

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # Expand channels
    if arr.shape[2] == 1:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    return arr


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] == target_h:
        return img
    scale = target_h / float(img.shape[0])
    target_w = max(1, int(round(img.shape[1] * scale)))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def _find_data_root(f: h5py.File) -> h5py.Group:
    h5py = _h5py()
    if "data" in f:
        grp = f["data"]
        if isinstance(grp, h5py.Group):
            return grp
    return f


def _list_demos(data_root: h5py.Group) -> List[str]:
    demos = [k for k in data_root.keys() if str(k).startswith("demo_")]
    return sorted(demos, key=lambda s: (len(s), s))


def _resolve_demo(data_root: h5py.Group, demo: Optional[str], demo_index: Optional[int]) -> str:
    demos = _list_demos(data_root)
    if not demos:
        raise RuntimeError("No demos found (expected keys like demo_0 under /data)")

    if demo is not None:
        if demo in data_root:
            return demo
        raise RuntimeError(f"Demo '{demo}' not found. Available: {demos[:20]}{'...' if len(demos) > 20 else ''}")

    if demo_index is None:
        return demos[0]

    if demo_index < 0 or demo_index >= len(demos):
        raise RuntimeError(f"demo-index out of range: {demo_index}, available 0..{len(demos)-1}")

    return demos[demo_index]


def _resolve_obs_group(demo_group: h5py.Group) -> h5py.Group:
    h5py = _h5py()
    if "obs" in demo_group and isinstance(demo_group["obs"], h5py.Group):
        return demo_group["obs"]
    raise RuntimeError("Demo group has no 'obs' subgroup")


def _resolve_image_keys(obs_group: h5py.Group, keys: Optional[Sequence[str]]) -> List[str]:
    if keys:
        missing = [k for k in keys if k not in obs_group]
        if missing:
            avail = sorted(list(obs_group.keys()))
            raise RuntimeError(
                f"Requested obs keys not found: {missing}. Available obs keys include: {avail[:30]}{'...' if len(avail) > 30 else ''}"
            )
        return list(keys)

    # Heuristic defaults
    candidates = [
        "eye_in_hand_rgb",
        "wrist_rgb",
        "hand_rgb",
        "agentview_rgb",
        "global_rgb",
        "front_rgb",
    ]
    found = [k for k in candidates if k in obs_group]
    if found:
        # prefer 1-2 streams
        return found[:2]

    # fallback: any *_rgb
    rgb_like = [k for k in obs_group.keys() if str(k).endswith("_rgb")]
    if rgb_like:
        return sorted(rgb_like)[:2]

    avail = sorted(list(obs_group.keys()))
    raise RuntimeError(f"No RGB obs keys found. Available obs keys: {avail[:30]}{'...' if len(avail) > 30 else ''}")


def _infer_length(demo_group: h5py.Group, obs_group: h5py.Group, image_keys: Sequence[str]) -> int:
    h5py = _h5py()
    # Prefer actions length if exists
    if "actions" in demo_group and isinstance(demo_group["actions"], h5py.Dataset):
        return int(demo_group["actions"].shape[0])

    # Else use first image dataset length
    ds0 = obs_group[image_keys[0]]
    if not isinstance(ds0, h5py.Dataset) or len(ds0.shape) < 1:
        raise RuntimeError(f"Obs key '{image_keys[0]}' is not a dataset with time dimension")
    return int(ds0.shape[0])


def _safe_get_actions(demo_group: h5py.Group, idx: int) -> Optional[np.ndarray]:
    h5py = _h5py()
    if "actions" not in demo_group:
        return None
    ds = demo_group["actions"]
    if not isinstance(ds, h5py.Dataset) or ds.shape[0] <= idx:
        return None
    try:
        return np.asarray(ds[idx])
    except Exception:
        return None


def _format_actions(actions: Optional[np.ndarray]) -> str:
    if actions is None:
        return ""
    a = np.asarray(actions).reshape(-1)
    if a.size == 0:
        return ""
    # Show up to first 8 dims
    shown = ",".join([f"{float(x):+.3f}" for x in a[:8]])
    suffix = "..." if a.size > 8 else ""
    return f" | a=[{shown}{suffix}]"


def _read_frame(obs_group: h5py.Group, key: str, idx: int) -> np.ndarray:
    h5py = _h5py()
    ds = obs_group[key]
    if not isinstance(ds, h5py.Dataset):
        raise RuntimeError(f"Obs key '{key}' is not a dataset")
    if ds.shape[0] <= idx:
        raise RuntimeError(f"Index {idx} out of range for '{key}', len={ds.shape[0]}")
    return np.asarray(ds[idx])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Visualize HDF5 demos (robomimic/LIBERO-style).")
    ap.add_argument("h5", help="Path to HDF5 file")
    ap.add_argument("--demo", default=None, help="Demo name like demo_0")
    ap.add_argument("--demo-index", type=int, default=None, help="Demo index (default: 0)")
    ap.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Obs image keys to view, e.g. eye_in_hand_rgb agentview_rgb (default: auto)",
    )
    ap.add_argument(
        "--assume",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Stored channel order for *_rgb datasets (default: rgb)",
    )
    ap.add_argument("--start", type=int, default=0, help="Start frame index")
    ap.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
    ap.add_argument("--scale", type=float, default=1.0, help="Resize display scale")
    ap.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    ap.add_argument("--list", action="store_true", help="List demos/obs keys and exit")
    ap.add_argument(
        "--no-file-lock",
        action="store_true",
        help="Disable HDF5 file locking (useful to preview while another process is writing; may show partial data)",
    )

    args = ap.parse_args(argv)

    if args.no_file_lock:
        # Must be set before opening the file. This is helpful on network FS or when another
        # process is holding the lock; you may still see partial/inconsistent data.
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        os.environ.setdefault("HDF5_FILE_LOCKING", "FALSE")

    delay_ms = max(1, int(round(1000.0 / max(1e-3, float(args.fps)))))

    try:
        h5py = _h5py()
        f = h5py.File(args.h5, "r")
    except BlockingIOError as e:
        print(
            "Failed to open HDF5 due to file lock (errno 11).\n"
            "- If a recorder is still running, stop it and try again.\n"
            "- Or re-run with --no-file-lock to bypass locking (may be partial).\n"
            f"File: {args.h5}\nError: {e}",
            file=sys.stderr,
        )
        return 2

    with f:
        data_root = _find_data_root(f)
        demo_name = _resolve_demo(data_root, args.demo, args.demo_index)
        demo_group = data_root[demo_name]
        if not isinstance(demo_group, h5py.Group):
            raise RuntimeError(f"'{demo_name}' is not a group")

        obs_group = _resolve_obs_group(demo_group)

        if args.list:
            demos = _list_demos(data_root)
            obs_keys = sorted(list(obs_group.keys()))
            print(f"H5: {args.h5}")
            print(f"Found demos ({len(demos)}): {demos[:50]}{'...' if len(demos) > 50 else ''}")
            print(f"Demo '{demo_name}' obs keys ({len(obs_keys)}):")
            for k in obs_keys:
                obj = obs_group[k]
                if isinstance(obj, h5py.Dataset):
                    print(f"  - {k}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"  - {k}: <group>")
            if "actions" in demo_group and isinstance(demo_group["actions"], h5py.Dataset):
                ds = demo_group["actions"]
                print(f"actions: shape={ds.shape}, dtype={ds.dtype}")
            return 0

        image_keys = _resolve_image_keys(obs_group, args.keys)
        length = _infer_length(demo_group, obs_group, image_keys)

        start = int(max(0, args.start))
        end = int(args.end) if args.end is not None else length
        end = max(0, min(end, length))
        if start >= end:
            raise RuntimeError(f"Invalid range start={start}, end={end}, length={length}")

        print(f"H5: {args.h5}")
        print(f"Demo: {demo_name} | length={length} | viewing keys={image_keys} | range=[{start},{end})")
        print("Controls: q/ESC quit | space pause | n/p step | [/ ] speed")

        paused = False
        i = start
        while True:
            if i >= end:
                break

            frames: List[np.ndarray] = []
            for k in image_keys:
                raw = _read_frame(obs_group, k, i)
                img = _as_hwc_uint8(raw)
                if args.assume == "rgb":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                label = f"{demo_name}:{k}  {i}/{length-1}"
                frames.append(_put_label(img, label))

            # concat for display
            base_h = min(im.shape[0] for im in frames)
            frames = [_resize_to_height(im, base_h) for im in frames]
            canvas = cv2.hconcat(frames) if len(frames) > 1 else frames[0]

            # optional scale
            if float(args.scale) != 1.0:
                canvas = cv2.resize(
                    canvas,
                    (max(1, int(canvas.shape[1] * float(args.scale))), max(1, int(canvas.shape[0] * float(args.scale)))),
                    interpolation=cv2.INTER_AREA,
                )

            actions = _safe_get_actions(demo_group, i)
            title = f"{demo_name}  t={i}{_format_actions(actions)}"
            cv2.imshow("HDF5 Demo Viewer", canvas)
            cv2.setWindowTitle("HDF5 Demo Viewer", title)

            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key == ord("n"):
                paused = True
                i = min(i + 1, end - 1)
                continue
            if key == ord("p"):
                paused = True
                i = max(i - 1, start)
                continue
            if key == ord("["):
                delay_ms = min(2000, int(delay_ms * 1.25) + 1)
                continue
            if key == ord("]"):
                delay_ms = max(1, int(delay_ms / 1.25))
                continue

            if not paused:
                i += 1

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        raise SystemExit(130)

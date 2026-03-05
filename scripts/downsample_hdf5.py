#!/usr/bin/env python3
"""Downsample LIBERO/robomimic-style HDF5 demos to 10Hz or 20Hz.

示例:
  python scripts/downsample_hdf5.py \
    --input /home/rvl/collect_datasets_ws/data/libero_demos.hdf5 \
    --output /home/rvl/collect_datasets_ws/data/libero_demos_10hz.hdf5 \
    --input-hz 30 --target-hz 10

说明:
- 仅支持目标频率: 10Hz 或 20Hz
- 默认假设原始频率为 30Hz（可通过 --input-hz 修改）
- 保持原有 HDF5 结构，按时间维(第0维)下采样
- 自动更新每个 demo 的 num_samples 属性
"""

from __future__ import annotations

import argparse
import os
from typing import List

import h5py
import numpy as np


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key in src.keys():
        dst[key] = src[key]


def sorted_demo_names(data_group: h5py.Group) -> List[str]:
    names = [k for k in data_group.keys() if str(k).startswith("demo_")]

    def key_fn(name: str):
        try:
            return int(name.split("_")[-1])
        except Exception:
            return name

    return sorted(names, key=key_fn)


def build_indices(num_samples: int, input_hz: float, target_hz: int) -> np.ndarray:
    if num_samples <= 0:
        return np.array([], dtype=np.int64)

    if target_hz >= input_hz:
        return np.arange(num_samples, dtype=np.int64)

    ratio = input_hz / float(target_hz)
    ratio_int = int(round(ratio))

    if abs(ratio - ratio_int) < 1e-6 and ratio_int > 0:
        return np.arange(0, num_samples, ratio_int, dtype=np.int64)

    target_count = max(1, int(round(num_samples * float(target_hz) / float(input_hz))))
    idx = np.linspace(0, num_samples - 1, target_count)
    idx = np.unique(np.round(idx).astype(np.int64))

    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
    return idx


def copy_dataset(
    src_ds: h5py.Dataset,
    dst_parent: h5py.Group,
    name: str,
    indices: np.ndarray,
    original_n: int,
) -> None:
    if src_ds.ndim >= 1 and src_ds.shape[0] == original_n:
        data = src_ds[indices, ...]
    else:
        data = src_ds[()]

    kwargs = {}
    if src_ds.compression is not None:
        kwargs["compression"] = src_ds.compression
    if src_ds.compression_opts is not None:
        kwargs["compression_opts"] = src_ds.compression_opts
    if src_ds.shuffle:
        kwargs["shuffle"] = True
    if src_ds.fletcher32:
        kwargs["fletcher32"] = True

    dst_ds = dst_parent.create_dataset(name, data=data, dtype=src_ds.dtype, **kwargs)
    copy_attrs(src_ds.attrs, dst_ds.attrs)


def copy_group_with_downsample(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    indices: np.ndarray,
    original_n: int,
) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)

    for key in src_group.keys():
        obj = src_group[key]
        if isinstance(obj, h5py.Group):
            child = dst_group.create_group(key)
            copy_group_with_downsample(obj, child, indices, original_n)
        elif isinstance(obj, h5py.Dataset):
            copy_dataset(obj, dst_group, key, indices, original_n)


def infer_original_n(demo_group: h5py.Group) -> int:
    if "actions" in demo_group and isinstance(demo_group["actions"], h5py.Dataset):
        return int(demo_group["actions"].shape[0])

    attr_n = demo_group.attrs.get("num_samples", None)
    if attr_n is not None:
        return int(attr_n)

    for key in demo_group.keys():
        obj = demo_group[key]
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
            return int(obj.shape[0])
        if isinstance(obj, h5py.Group):
            for k2 in obj.keys():
                ds = obj[k2]
                if isinstance(ds, h5py.Dataset) and ds.ndim >= 1:
                    return int(ds.shape[0])

    return 0


def downsample_file(input_path: str, output_path: str, input_hz: float, target_hz: int) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        copy_attrs(src.attrs, dst.attrs)

        if "data" not in src:
            raise RuntimeError("Input HDF5 has no /data group")

        src_data = src["data"]
        if not isinstance(src_data, h5py.Group):
            raise RuntimeError("/data is not a group")

        dst_data = dst.create_group("data")
        copy_attrs(src_data.attrs, dst_data.attrs)

        demo_names = sorted_demo_names(src_data)
        if not demo_names:
            raise RuntimeError("No demo_* groups found under /data")

        print(f"Found demos: {len(demo_names)}")
        print(f"Downsample: {input_hz}Hz -> {target_hz}Hz")

        for name in demo_names:
            src_demo = src_data[name]
            if not isinstance(src_demo, h5py.Group):
                continue

            original_n = infer_original_n(src_demo)
            indices = build_indices(original_n, input_hz, target_hz)
            new_n = int(len(indices))

            dst_demo = dst_data.create_group(name)
            copy_group_with_downsample(src_demo, dst_demo, indices, original_n)
            dst_demo.attrs["num_samples"] = new_n

            print(f"  {name}: {original_n} -> {new_n}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample HDF5 demos to 10Hz or 20Hz.")
    parser.add_argument("--input", required=True, help="Input HDF5 path")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument(
        "--target-hz",
        type=int,
        required=True,
        choices=[10, 20],
        help="Target sampling rate (10 or 20)",
    )
    parser.add_argument(
        "--input-hz",
        type=float,
        default=30.0,
        help="Original sampling rate in Hz (default: 30)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.input_hz <= 0:
        raise ValueError("--input-hz must be > 0")

    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(f"Output exists: {args.output} (use --force to overwrite)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    downsample_file(args.input, args.output, args.input_hz, args.target_hz)

    print(f"Done. Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

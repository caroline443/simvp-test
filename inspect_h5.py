"""
H5 文件结构探查脚本
===================
在 Windows 机器上运行，用于确认 SEVIR h5 文件的内部 key、数据形状和像素值范围。

用法：
    python inspect_h5.py --path "F:/zyx/dataset/sevir_data/SEVIR_VIL_RANDOMEVENTS_2017_0501_0831.h5"
"""

import argparse
import numpy as np
import h5py


def inspect(fpath: str):
    print(f"\n{'='*60}")
    print(f"文件路径: {fpath}")
    print(f"{'='*60}")

    with h5py.File(fpath, "r") as f:
        print(f"\n[1] 顶层 Keys（共 {len(f.keys())} 个）:")
        for k in f.keys():
            ds = f[k]
            print(f"    key='{k}'  shape={ds.shape}  dtype={ds.dtype}")

        print(f"\n[2] 逐 Key 详细信息:")
        for k in f.keys():
            ds = f[k]
            print(f"\n  --- key='{k}' ---")
            print(f"    shape:  {ds.shape}")
            print(f"    dtype:  {ds.dtype}")
            print(f"    ndim:   {ds.ndim}")

            # 读取第一个事件的前几帧做统计
            if ds.ndim == 4:
                # 预期形状: [n_events, n_frames, H, W]
                n_events, n_frames, H, W = ds.shape
                print(f"    解读:   [n_events={n_events}, n_frames={n_frames}, H={H}, W={W}]")
                sample = ds[0, :5].astype(np.float32)  # 第0个事件，前5帧
                print(f"    像素值范围（第0事件前5帧）:")
                print(f"      min={sample.min():.2f}  max={sample.max():.2f}  mean={sample.mean():.2f}")
                print(f"      零值像素占比: {(sample == 0).mean() * 100:.1f}%")

            elif ds.ndim == 3:
                # 可能是 [n_events, H, W] 或 [n_frames, H, W]
                d0, H, W = ds.shape
                print(f"    解读:   [dim0={d0}, H={H}, W={W}]")
                sample = ds[0].astype(np.float32)
                print(f"    像素值范围（第0帧）:")
                print(f"      min={sample.min():.2f}  max={sample.max():.2f}  mean={sample.mean():.2f}")

            elif ds.ndim == 2:
                print(f"    解读:   可能是元数据表格，shape={ds.shape}")
                try:
                    print(f"    前3行: {ds[:3]}")
                except Exception:
                    pass

            elif ds.ndim == 1:
                print(f"    解读:   一维数组，长度={ds.shape[0]}")
                print(f"    前5个值: {ds[:5]}")

        print(f"\n[3] 文件属性（attrs）:")
        if len(f.attrs) == 0:
            print("    （无文件级属性）")
        for attr_k, attr_v in f.attrs.items():
            print(f"    {attr_k}: {attr_v}")

    print(f"\n{'='*60}")
    print("探查完成。请将以上输出发给开发人员。")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=r"F:/zyx/dataset/sevir_data/SEVIR_VIL_RANDOMEVENTS_2017_0501_0831.h5",
        help="h5 文件路径",
    )
    args = parser.parse_args()
    inspect(args.path)

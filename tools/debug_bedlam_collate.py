import argparse
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

from hmr4d.dataset.bedlam.bedlam import BedlamDatasetV2
from hmr4d.utils.pylogger import Log


def _shape_of(v: Any):
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[len={len(v)}]"
    if isinstance(v, Mapping):
        return f"dict(keys={list(v.keys())})"
    return type(v).__name__


def _meta_of(sample: Mapping[str, Any]):
    m = sample.get("meta", {}) if isinstance(sample, Mapping) else {}
    if isinstance(m, Mapping):
        return {
            "idx": m.get("idx", None),
            "vid": m.get("vid", None),
            "start_end": m.get("start_end", None),
            "data_name": m.get("data_name", None),
        }
    return {"meta": m}


def _debug_collate_mixed(values, key_path: str, full_batch):
    if any(v is None for v in values):
        return None

    if all(isinstance(v, Mapping) for v in values):
        out = {}
        keys = set()
        for v in values:
            keys.update(v.keys())
        for k in keys:
            next_path = f"{key_path}.{k}" if key_path else k
            out[k] = _debug_collate_mixed([v.get(k, None) for v in values], next_path, full_batch)
        return out

    if all(isinstance(v, torch.Tensor) for v in values):
        shapes = [tuple(v.shape) for v in values]
        first_shape = shapes[0]
        if any(s != first_shape for s in shapes):
            Log.error(f"[COLLATE-ERROR] Tensor shape mismatch at key='{key_path}'")
            Log.error(f"[COLLATE-ERROR] Batch shapes: {shapes}")
            for i, (v, s) in enumerate(zip(values, shapes)):
                meta = _meta_of(full_batch[i])
                Log.error(
                    f"[COLLATE-ERROR] sample={i} shape={s} "
                    f"idx={meta['idx']} start_end={meta['start_end']} vid={meta['vid']}"
                )
                if v.numel() == 0:
                    Log.error(f"[COLLATE-ERROR] sample={i} has EMPTY tensor at key='{key_path}'")
            raise RuntimeError(f"Collate tensor shape mismatch at key='{key_path}'")
        return torch.stack([v.clone() for v in values], dim=0)

    # Fallback: no special handling needed.
    return values


def debug_collate_fn(batch):
    out = {}
    keys = set()
    for d in batch:
        keys.update(d.keys())
    for k in keys:
        if k.startswith("meta"):
            out[k] = [d.get(k, None) for d in batch]
        else:
            values = [d.get(k, None) for d in batch]
            out[k] = _debug_collate_mixed(values, k, batch)
    out["B"] = len(batch)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--load-image", action="store_true")
    parser.add_argument("--video-root", type=str, default="/data/datasets/bedlam_download")
    parser.add_argument("--saved-frame-root", type=str, default=None)
    parser.add_argument("--saved-frame-ext", type=str, default="jpg")
    parser.add_argument("--image-crop-size", type=int, default=512)
    parser.add_argument("--mid-indices", nargs="+", default=["all60", "maxspan60"])
    args = parser.parse_args()

    ds = BedlamDatasetV2(
        mid_indices=args.mid_indices,
        load_image=args.load_image,
        video_root=args.video_root,
        saved_frame_root=args.saved_frame_root,
        saved_frame_ext=args.saved_frame_ext,
        image_crop_size=args.image_crop_size,
    )

    n = len(ds)
    start = max(0, args.start)
    end = n if args.end < 0 else min(args.end, n)
    if end <= start:
        raise ValueError(f"Invalid range: start={start}, end={end}, len={n}")

    subset = Subset(ds, list(range(start, end)))

    Log.info(
        f"Debugging Bedlam collate: subset=[{start}, {end}) size={len(subset)} "
        f"batch_size={args.batch_size} num_workers={args.num_workers}"
    )

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=debug_collate_fn,
    )

    try:
        for bi, batch in enumerate(loader):
            if bi % 10 == 0:
                Log.info(f"[OK] batch_idx={bi} B={batch['B']}")
            if args.max_batches > 0 and (bi + 1) >= args.max_batches:
                break
    except Exception as e:
        Log.error(f"Stopped at failing batch with error: {type(e).__name__}: {e}")
        raise

    Log.info("Finished without collate errors in checked range.")


if __name__ == "__main__":
    main()

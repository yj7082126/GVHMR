import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from hmr4d.dataset.bedlam.bedlam import BedlamDatasetV2  # 52,788 samples
from hmr4d.utils.preproc import Extractor
from hmr4d.utils.video_io_utils import read_video_np


def build_output_path(base_path: Path, start_ind: int, end_ind: int, file_idx: int) -> Path:
    stem = base_path.stem
    suffix = base_path.suffix if base_path.suffix else ".pt"
    return base_path.with_name(f"{stem}_{start_ind}_{end_ind}_part{file_idx:04d}{suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-ind", type=int, default=1000)
    parser.add_argument("--end-ind", type=int, default=37537)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=1600)
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/datasets/bedlam_dinov3_feats/dinov3_start_feats.pt",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.save_every <= 0:
        raise ValueError("--save-every must be > 0")

    dataset = BedlamDatasetV2()
    video_dir = Path("/data/datasets/bedlam_download/")

    extractor = Extractor(extractor_type="dinov3")
    output_base = Path(args.output_path)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(4)

    total_to_process = max(0, args.end_ind - args.start_ind)
    if total_to_process == 0:
        raise ValueError("Empty index range: --end-ind must be greater than --start-ind")

    chunk_samples = defaultdict(dict)
    file_idx = 0
    chunk_processed = 0

    pbar = tqdm(total=total_to_process, desc="Extracting DINOv3 features")

    for batch_start in range(args.start_ind, args.end_ind, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.end_ind)
        indices = list(range(batch_start, batch_end))

        frames_list = []
        bbx_list = []
        metas = []

        for i in indices:
            sample = dataset[i]
            meta = sample["meta"]
            mid = Path(meta["vid"])
            video_path = video_dir / "/".join(mid.parts[-3:-1]) / f"{mid.stem}.mp4"

            start = int(meta["start_end"][0])
            frame_np = read_video_np(video_path, start_frame=start, end_frame=start + 1)
            if frame_np.shape[0] == 0:
                continue

            frames_list.append(frame_np[0])
            bbx_list.append(sample["bbx_xys"][0].cpu().numpy())
            metas.append((meta, start))

        if len(frames_list) == 0:
            pbar.update(len(indices))
            continue

        frames_np = np.stack(frames_list, axis=0)
        bbx_xys_np = np.stack(bbx_list, axis=0)
        bbx_xys = torch.from_numpy(bbx_xys_np).float()

        features = extractor.extract_video_features(
            frames_np,
            bbx_xys,
            path_type="np",
            img_ds=1.0,
            batch_size=args.batch_size,
        )

        features_np = features.cpu().numpy()

        for row_idx, (meta, start) in enumerate(metas):
            row = {
                "vid": meta["vid"],
                "ind": [start],
                "feat": features_np[row_idx : row_idx + 1],
            }
            chunk_samples[meta["idx"]] = row
            chunk_processed += 1

            if chunk_processed >= args.save_every:
                output_path = build_output_path(output_base, args.start_ind, args.end_ind, file_idx)
                torch.save(dict(chunk_samples), output_path)
                chunk_samples.clear()
                chunk_processed = 0
                file_idx += 1

        pbar.update(len(indices))

    pbar.close()

    if chunk_samples:
        output_path = build_output_path(output_base, args.start_ind, args.end_ind, file_idx)
        torch.save(dict(chunk_samples), output_path)

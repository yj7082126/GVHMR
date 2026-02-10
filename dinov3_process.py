import argparse
from pathlib import Path
from collections import defaultdict
import time
import signal
from contextlib import contextmanager
import logging
from datetime import datetime
import gc
import queue
import torch.multiprocessing as mp

import numpy as np
import torch
from tqdm import tqdm

from hmr4d.dataset.bedlam.bedlam import BedlamDatasetV2  # 52,788 samples
from hmr4d.dataset.bedlam.utils import mid2vname
from hmr4d.utils.pylogger import Log
from hmr4d.utils.preproc import Extractor
from hmr4d.utils.video_io_utils import read_video_np


def build_output_path(base_path: Path, start_ind: int, end_ind: int, file_idx: int) -> Path:
    stem = base_path.stem
    suffix = base_path.suffix if base_path.suffix else ".pt"
    return base_path.with_name(f"{stem}_{start_ind}_{end_ind}_part{file_idx:04d}{suffix}")


def attach_file_logger(log_path: Path):
    log_path = Path(log_path)
    for h in Log.handlers:
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path:
            return
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    Log.addHandler(fh)


def setup_file_logger(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_stamp}.log"
    attach_file_logger(log_path)
    return log_path


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


@contextmanager
def timeout(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    prev_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def worker_loop(task_q, result_q, cfg):
    # Each worker process writes to the same run log file.
    attach_file_logger(Path(cfg["log_path"]))
    np.random.seed(cfg["seed"])

    dataset = BedlamDatasetV2()
    extractor = Extractor(extractor_type="dinov3")
    video_dir = Path(cfg["video_dir"])

    while True:
        task = task_q.get()
        if task is None:
            break

        batch_start = task["batch_start"]
        batch_end = task["batch_end"]
        indices = task["indices"]
        worker_batch_size = task["worker_batch_size"]
        read_timeout_sec = task["read_timeout_sec"]
        slow_threshold_sec = task["slow_threshold_sec"]

        batch_t0 = time.time()
        skipped = 0
        rows = []
        frames_list = []
        bbx_list = []
        metas = []

        try:
            for i in indices:
                data_t0 = time.time()
                sample = dataset[i]
                data_dt = time.time() - data_t0

                meta = sample["meta"]
                video_path = video_dir / mid2vname(meta["vid"])
                start = int(meta["start_end"][0])

                read_t0 = time.time()
                with timeout(read_timeout_sec):
                    frame_np = read_video_np(video_path, start_frame=start, end_frame=start + 1)
                read_dt = time.time() - read_t0

                if frame_np.shape[0] == 0:
                    skipped += 1
                    Log.warning(
                        f"[SKIP] idx={i} mid={meta['vid']} start={start} path={video_path} reason=empty_frame"
                    )
                    continue

                frames_list.append(frame_np[0])
                bbx_list.append(sample["bbx_xys"][0].cpu().numpy())
                metas.append((meta, start))

                if data_dt > slow_threshold_sec or read_dt > slow_threshold_sec:
                    Log.warning(
                        f"[SLOW] idx={i} data={data_dt:.2f}s read={read_dt:.2f}s "
                        f"mid={meta['vid']} start={start}"
                    )

            infer_dt = 0.0
            if len(frames_list) > 0:
                frames_np = np.stack(frames_list, axis=0)
                bbx_xys_np = np.stack(bbx_list, axis=0)
                bbx_xys = torch.from_numpy(bbx_xys_np).float()

                infer_t0 = time.time()
                features = extractor.extract_video_features(
                    frames_np,
                    bbx_xys,
                    path_type="np",
                    img_ds=1.0,
                    batch_size=worker_batch_size,
                )
                infer_dt = time.time() - infer_t0
                features_np = features.cpu().numpy()

                for row_idx, (meta, start) in enumerate(metas):
                    rows.append(
                        {
                            "idx": meta["idx"],
                            "vid": meta["vid"],
                            "ind": [start],
                            "feat": features_np[row_idx : row_idx + 1],
                        }
                    )

            result_q.put(
                {
                    "ok": True,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "rows": rows,
                    "skipped_samples": skipped,
                    "prepared": len(frames_list),
                    "infer_dt": infer_dt,
                    "total_dt": time.time() - batch_t0,
                }
            )

        except Exception as e:
            result_q.put(
                {
                    "ok": False,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "error": f"{type(e).__name__}: {e}",
                    "skipped_samples": skipped,
                    "total_dt": time.time() - batch_t0,
                }
            )


def start_worker(ctx, cfg):
    task_q = ctx.Queue(maxsize=2)
    result_q = ctx.Queue(maxsize=2)
    proc = ctx.Process(target=worker_loop, args=(task_q, result_q, cfg), daemon=True)
    proc.start()
    return proc, task_q, result_q


def stop_worker(proc, task_q):
    if proc is None:
        return
    try:
        if proc.is_alive():
            task_q.put(None)
            proc.join(timeout=5)
    except Exception:
        pass
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-ind", type=int, default=1048)
    parser.add_argument("--end-ind", type=int, default=37537)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=1600)
    parser.add_argument("--read-timeout-sec", type=int, default=30)
    parser.add_argument("--batch-timeout-sec", type=int, default=120)
    parser.add_argument("--slow-threshold-sec", type=float, default=3.0)
    parser.add_argument("--verbose-timing", action="store_true")
    parser.add_argument("--log-dir", type=str, default=None)
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

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    output_base = Path(args.output_path)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir is not None else output_base.parent / "logs"
    log_path = setup_file_logger(log_dir)
    Log.info(f"Execution log file: {log_path}")

    total_to_process = max(0, args.end_ind - args.start_ind)
    if total_to_process == 0:
        raise ValueError("Empty index range: --end-ind must be greater than --start-ind")

    worker_cfg = {
        "video_dir": "/data/datasets/bedlam_download/",
        "seed": 4,
        "log_path": str(log_path),
    }

    ctx = mp.get_context("spawn")
    proc, task_q, result_q = start_worker(ctx, worker_cfg)

    chunk_samples = defaultdict(dict)
    file_idx = 0
    chunk_processed = 0
    skipped_samples = 0
    skipped_batches = 0

    pbar = tqdm(total=total_to_process, desc="Extracting DINOv3 features")

    for batch_start in range(args.start_ind, args.end_ind, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.end_ind)
        indices = list(range(batch_start, batch_end))

        if not proc.is_alive():
            skipped_batches += 1
            Log.warning(f"[BATCH-SKIP] {batch_start}-{batch_end - 1} worker died; restarting worker")
            cleanup_memory()
            proc, task_q, result_q = start_worker(ctx, worker_cfg)
            pbar.update(len(indices))
            continue

        task_q.put(
            {
                "batch_start": batch_start,
                "batch_end": batch_end,
                "indices": indices,
                "worker_batch_size": args.batch_size,
                "read_timeout_sec": args.read_timeout_sec,
                "slow_threshold_sec": args.slow_threshold_sec,
            }
        )

        try:
            result = result_q.get(timeout=args.batch_timeout_sec)
        except queue.Empty:
            skipped_batches += 1
            Log.warning(
                f"[BATCH-SKIP] {batch_start}-{batch_end - 1} exceeded {args.batch_timeout_sec}s; "
                "terminating and restarting worker"
            )
            stop_worker(proc, task_q)
            cleanup_memory()
            proc, task_q, result_q = start_worker(ctx, worker_cfg)
            pbar.update(len(indices))
            continue

        skipped_samples += int(result.get("skipped_samples", 0))
        if not result.get("ok", False):
            skipped_batches += 1
            Log.warning(
                f"[BATCH-SKIP] {batch_start}-{batch_end - 1} worker error after {result.get('total_dt', 0.0):.2f}s "
                f"reason={result.get('error', 'unknown')}"
            )
            cleanup_memory()
            pbar.update(len(indices))
            continue

        rows = result.get("rows", [])
        for row in rows:
            idx = row.pop("idx")
            chunk_samples[idx] = row
            chunk_processed += 1

            if chunk_processed >= args.save_every:
                output_path = build_output_path(output_base, args.start_ind, args.end_ind, file_idx)
                torch.save(dict(chunk_samples), output_path)
                Log.info(f"[SAVE] part={file_idx:04d} samples={len(chunk_samples)} path={output_path}")
                chunk_samples.clear()
                chunk_processed = 0
                file_idx += 1

        pbar.update(len(indices))
        if args.verbose_timing:
            Log.info(
                f"[BATCH] {batch_start}-{batch_end - 1} prepared={result.get('prepared', 0)} "
                f"infer={result.get('infer_dt', 0.0):.2f}s total={result.get('total_dt', 0.0):.2f}s "
                f"skipped_samples={skipped_samples} skipped_batches={skipped_batches}"
            )

    pbar.close()

    if chunk_samples:
        output_path = build_output_path(output_base, args.start_ind, args.end_ind, file_idx)
        torch.save(dict(chunk_samples), output_path)
        Log.info(f"[SAVE] part={file_idx:04d} samples={len(chunk_samples)} path={output_path}")

    stop_worker(proc, task_q)
    cleanup_memory()
    Log.info(f"Done. skipped_samples={skipped_samples} skipped_batches={skipped_batches}")

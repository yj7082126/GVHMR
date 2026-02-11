import argparse
from pathlib import Path
import logging
from datetime import datetime
import gc
import queue

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from hmr4d.dataset.bedlam.bedlam import BedlamDatasetV2
from hmr4d.dataset.bedlam.utils import mid2vname
from hmr4d.utils.pylogger import Log


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


def resolve_output_dir(video_path: Path, save_root: str | None) -> Path:
    if save_root:
        return Path(save_root) / video_path.parent.name / video_path.stem
    return video_path.parent.parent / "vis" / video_path.stem


def worker_loop(task_q, result_q, cfg):
    attach_file_logger(Path(cfg["log_path"]))
    np.random.seed(cfg["seed"])

    dataset_kwargs = {}
    if cfg["try_load_image"]:
        dataset_kwargs["load_image"] = True
    try:
        dataset = BedlamDatasetV2(**dataset_kwargs)
    except TypeError:
        dataset = BedlamDatasetV2()

    video_dir = Path(cfg["video_dir"])

    while True:
        task = task_q.get()
        if task is None:
            break

        batch_start = task["batch_start"]
        batch_end = task["batch_end"]
        indices = task["indices"]
        frames_per_sample = task["frames_per_sample"]
        save_root = task["save_root"]
        start_source = task["start_source"]
        jpeg_quality = task["jpeg_quality"]
        slow_threshold_sec = task["slow_threshold_sec"]

        batch_t0 = datetime.now().timestamp()
        skipped_samples = 0
        saved_frames = 0

        try:
            for i in indices:
                sample_t0 = datetime.now().timestamp()
                sample = dataset[i]
                meta = sample["meta"]

                video_path = video_dir / mid2vname(meta["vid"])

                if start_source == "valid_range":
                    start, _ = dataset.mid_to_valid_range[str(meta["vid"])]
                else:
                    start = int(meta["start_end"][0])

                out_dir = resolve_output_dir(video_path, save_root)
                out_dir.mkdir(parents=True, exist_ok=True)

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    skipped_samples += 1
                    Log.warning(f"[SKIP] idx={i} cannot_open video={video_path}")
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                saved = 0
                while saved < frames_per_sample:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out_path = out_dir / f"{start + saved:05d}.jpg"
                    ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                    if not ok:
                        Log.warning(f"[SKIP] idx={i} failed_write path={out_path}")
                        break
                    saved += 1

                cap.release()
                saved_frames += saved
                if saved == 0:
                    skipped_samples += 1
                    Log.warning(f"[SKIP] idx={i} no_frames video={video_path} start={start}")

                dt = datetime.now().timestamp() - sample_t0
                if dt > slow_threshold_sec:
                    Log.warning(f"[SLOW] idx={i} took={dt:.2f}s video={video_path} start={start} saved={saved}")

            result_q.put(
                {
                    "ok": True,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "saved_frames": saved_frames,
                    "skipped_samples": skipped_samples,
                    "total_dt": datetime.now().timestamp() - batch_t0,
                }
            )
        except Exception as e:
            result_q.put(
                {
                    "ok": False,
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "error": f"{type(e).__name__}: {e}",
                    "saved_frames": saved_frames,
                    "skipped_samples": skipped_samples,
                    "total_dt": datetime.now().timestamp() - batch_t0,
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
    parser.add_argument("--start-ind", type=int, default=0)
    parser.add_argument("--end-ind", type=int, default=37537)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-sample", type=int, default=10)
    parser.add_argument("--batch-timeout-sec", type=int, default=120)
    parser.add_argument("--slow-threshold-sec", type=float, default=2.0)
    parser.add_argument("--start-source", type=str, default="valid_range", choices=["valid_range", "sampled_meta"])
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--verbose-timing", action="store_true")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--video-dir", type=str, default="/data/datasets/bedlam_download/")
    parser.add_argument("--use-load-image", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.frames_per_sample <= 0:
        raise ValueError("--frames-per-sample must be > 0")
    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be in [1, 100]")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    log_base = Path(args.log_dir) if args.log_dir else Path(args.save_root) if args.save_root else Path("./logs")
    log_path = setup_file_logger(log_base)
    Log.info(f"Execution log file: {log_path}")

    total_to_process = max(0, args.end_ind - args.start_ind)
    if total_to_process == 0:
        raise ValueError("Empty index range: --end-ind must be greater than --start-ind")

    worker_cfg = {
        "video_dir": args.video_dir,
        "seed": 4,
        "log_path": str(log_path),
        "try_load_image": args.use_load_image,
    }

    ctx = mp.get_context("spawn")
    proc, task_q, result_q = start_worker(ctx, worker_cfg)

    skipped_samples = 0
    skipped_batches = 0
    total_saved_frames = 0

    pbar = tqdm(total=total_to_process, desc="Saving BEDLAM frames (OpenCV)")

    for batch_start in range(args.start_ind, args.end_ind, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.end_ind)
        indices = list(range(batch_start, batch_end))

        if not proc.is_alive():
            skipped_batches += 1
            Log.warning(f"[BATCH-SKIP] {batch_start}-{batch_end - 1} worker died; restarting")
            cleanup_memory()
            proc, task_q, result_q = start_worker(ctx, worker_cfg)
            pbar.update(len(indices))
            continue

        task_q.put(
            {
                "batch_start": batch_start,
                "batch_end": batch_end,
                "indices": indices,
                "frames_per_sample": args.frames_per_sample,
                "save_root": args.save_root,
                "start_source": args.start_source,
                "jpeg_quality": args.jpeg_quality,
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
        total_saved_frames += int(result.get("saved_frames", 0))

        if not result.get("ok", False):
            skipped_batches += 1
            Log.warning(
                f"[BATCH-SKIP] {batch_start}-{batch_end - 1} worker error after {result.get('total_dt', 0.0):.2f}s "
                f"reason={result.get('error', 'unknown')}"
            )
            cleanup_memory()
            pbar.update(len(indices))
            continue

        pbar.update(len(indices))
        if args.verbose_timing:
            Log.info(
                f"[BATCH] {batch_start}-{batch_end - 1} total={result.get('total_dt', 0.0):.2f}s "
                f"saved_frames_total={total_saved_frames} skipped_samples={skipped_samples} "
                f"skipped_batches={skipped_batches}"
            )

    pbar.close()
    stop_worker(proc, task_q)
    cleanup_memory()
    Log.info(
        f"Done. saved_frames={total_saved_frames} skipped_samples={skipped_samples} skipped_batches={skipped_batches}"
    )

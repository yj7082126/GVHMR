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


def sample_end_frame_indices(range1, range2, frames_per_sample, tail_span, rng: np.random.Generator):
    # valid clip range is [range1, range2), so last valid frame index is range2 - 1
    if range2 <= range1:
        return []
    end_last = range2 - 1
    tail_start = max(range1, range2 - tail_span)
    candidates = np.arange(tail_start, end_last + 1, dtype=np.int64)
    if len(candidates) == 0:
        return []
    if len(candidates) <= frames_per_sample:
        chosen = candidates
    else:
        chosen = np.sort(rng.choice(candidates, size=frames_per_sample, replace=False))
    return chosen.tolist()


def get_saved_frame_indices_in_dir(frame_dir: Path):
    if not frame_dir.exists():
        return []
    out = []
    for fp in frame_dir.glob("*.jpg"):
        try:
            out.append(int(fp.stem))
        except ValueError:
            continue
    return sorted(out)


def ensure_coverage_bins(cap, out_dir: Path, range1: int, range2: int, stride: int, jpeg_quality: int, saved_set: set[int]):
    """
    Ensure each [b, b+stride) bin in [range1, range2) has at least one saved frame.
    If uncovered, save frame at bin start.
    """
    if range2 <= range1:
        return 0
    if stride <= 0:
        return 0

    saved = 0
    for b in range(range1, range2, stride):
        e = min(b + stride, range2)
        covered = any((x >= b and x < e) for x in saved_set)
        if covered:
            continue

        target = b
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ret, frame = cap.read()
        if not ret:
            continue

        out_path = out_dir / f"{int(target):05d}.jpg"
        ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            saved += 1
            saved_set.add(int(target))
    return saved


def worker_loop(task_q, result_q, cfg):
    attach_file_logger(Path(cfg["log_path"]))
    rng = np.random.default_rng(cfg["seed"])

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
        mode = task["mode"]
        tail_span = task["tail_span"]
        coverage_stride = task["coverage_stride"]
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
                    range1, range2 = dataset.mid_to_valid_range[str(meta["vid"])]
                    start = int(range1)
                else:
                    start = int(meta["start_end"][0])
                    range1, range2 = dataset.mid_to_valid_range[str(meta["vid"])]

                out_dir = resolve_output_dir(video_path, save_root)
                out_dir.mkdir(parents=True, exist_ok=True)
                pre_saved = set(get_saved_frame_indices_in_dir(out_dir))

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    skipped_samples += 1
                    Log.warning(f"[SKIP] idx={i} cannot_open video={video_path}")
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                saved = 0

                if mode in ("start", "both"):
                    saved_start = 0
                    while saved_start < frames_per_sample:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out_path = out_dir / f"{start + saved_start:05d}.jpg"
                        ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                        if not ok:
                            Log.warning(f"[SKIP] idx={i} failed_write path={out_path}")
                            break
                        saved_start += 1
                        pre_saved.add(int(start + saved_start - 1))
                    saved += saved_start

                if mode in ("end", "both"):
                    end_indices = sample_end_frame_indices(
                        int(range1), int(range2), frames_per_sample=frames_per_sample, tail_span=tail_span, rng=rng
                    )
                    saved_end = 0
                    for frame_idx in end_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        out_path = out_dir / f"{int(frame_idx):05d}.jpg"
                        ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                        if not ok:
                            Log.warning(f"[SKIP] idx={i} failed_write path={out_path}")
                            continue
                        saved_end += 1
                        pre_saved.add(int(frame_idx))
                    saved += saved_end

                # Always enforce 5-frame coverage (or provided stride) over valid range.
                saved_cov = ensure_coverage_bins(
                    cap=cap,
                    out_dir=out_dir,
                    range1=int(range1),
                    range2=int(range2),
                    stride=int(coverage_stride),
                    jpeg_quality=jpeg_quality,
                    saved_set=pre_saved,
                )
                saved += saved_cov

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
    parser.add_argument("--mode", type=str, default="both", choices=["start", "end", "both"])
    parser.add_argument("--coverage-stride", type=int, default=5, help="Ensure at least one saved frame in each stride bin.")
    parser.add_argument(
        "--tail-span",
        type=int,
        default=None,
        help="When mode includes end, sample from last `tail_span` valid frames. Default=max(2*frames_per_sample, 20).",
    )
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
    if args.coverage_stride <= 0:
        raise ValueError("--coverage-stride must be > 0")
    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be in [1, 100]")
    if args.tail_span is not None and args.tail_span <= 0:
        raise ValueError("--tail-span must be > 0")

    tail_span = args.tail_span if args.tail_span is not None else max(2 * args.frames_per_sample, 20)

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

    pbar = tqdm(total=total_to_process, desc=f"Saving BEDLAM {args.mode}-frames (OpenCV)")

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
                "mode": args.mode,
                "tail_span": tail_span,
                "coverage_stride": args.coverage_stride,
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

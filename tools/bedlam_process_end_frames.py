import argparse
from pathlib import Path
import logging
from datetime import datetime
import gc
import queue

import imageio.v3 as iio
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
    # Keep using the same saved-frame folder so dataset can load both start/end frames by frame index.
    return video_path.parent.parent / "vis" / video_path.stem


def get_tail_frame_indices(range1, range2, end_frame_window):
    # valid clip range is [range1, range2), so last valid frame index is range2 - 1
    if range2 <= range1:
        return []
    tail_start = max(range1, range2 - end_frame_window)
    return list(range(int(tail_start), int(range2)))


def get_required_end_indices(range1, range2, end_frame_window, max_motion_frames):
    """
    Save targets for end-frame coverage:
    1) all tail frames in [range2-end_frame_window, range2)
    2) bridge frame at (start + max_motion_frames), clamped to valid range
    """
    tail = get_tail_frame_indices(range1, range2, end_frame_window=end_frame_window)
    if range2 <= range1:
        return tail
    elif range2 - range1 > max_motion_frames:
        tail = get_tail_frame_indices(range1, range1+max_motion_frames, end_frame_window=end_frame_window // 2)
    return sorted(tail)
    # bridge_idx = int(min(range2 - 1, range1 + int(max_motion_frames)))
    # return sorted(set(tail + [bridge_idx]))


def worker_loop(task_q, result_q, cfg):
    attach_file_logger(Path(cfg["log_path"]))
    dataset_kwargs = {}
    if cfg["try_load_image"]:
        dataset_kwargs["load_image"] = True
    if cfg["end_frame_window"] is not None:
        dataset_kwargs["end_frame_window"] = int(cfg["end_frame_window"])
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
        end_frame_window = task["end_frame_window"]
        save_root = task["save_root"]
        jpeg_quality = task["jpeg_quality"]
        overwrite = task["overwrite"]
        slow_threshold_sec = task["slow_threshold_sec"]

        batch_t0 = datetime.now().timestamp()
        skipped_samples = 0
        saved_frames = 0

        try:
            for i in indices:
                sample_t0 = datetime.now().timestamp()
                mid = dataset.idx2meta[i]
                range1, range2 = dataset.mid_to_valid_range[str(mid)]
                frame_indices = get_required_end_indices(
                    range1,
                    range2,
                    end_frame_window=end_frame_window,
                    max_motion_frames=getattr(dataset, "max_motion_frames", 120),
                )

                video_path = video_dir / mid2vname(mid)
                out_dir = resolve_output_dir(video_path, save_root)
                out_dir.mkdir(parents=True, exist_ok=True)

                if len(frame_indices) == 0:
                    skipped_samples += 1
                    Log.warning(f"[SKIP] idx={i} empty_end_candidates mid={mid} range=({range1},{range2})")
                    continue

                if overwrite:
                    missing_indices = frame_indices
                else:
                    missing_indices = [f for f in frame_indices if not (out_dir / f"{int(f):05d}.jpg").exists()]
                if len(missing_indices) < 3:
                    continue

                saved = 0
                for frame_idx in missing_indices:
                    try:
                        frame = iio.imread(video_path, index=int(frame_idx), plugin="pyav")
                    except Exception as e:
                        Log.warning(f"[SKIP] idx={i} failed_read frame={frame_idx} video={video_path} err={type(e).__name__}: {e}")
                        continue

                    out_path = out_dir / f"{int(frame_idx):05d}.jpg"
                    try:
                        # imageio write is backend-dependent for quality; keep best-effort argument.
                        iio.imwrite(out_path, frame, quality=jpeg_quality)
                        saved += 1
                    except TypeError:
                        try:
                            iio.imwrite(out_path, frame)
                            saved += 1
                        except Exception as e:
                            Log.warning(f"[SKIP] idx={i} failed_write path={out_path} err={type(e).__name__}: {e}")
                    except Exception as e:
                        Log.warning(f"[SKIP] idx={i} failed_write path={out_path} err={type(e).__name__}: {e}")

                saved_frames += saved
                if saved == 0 and len(missing_indices) > 0:
                    skipped_samples += 1
                    Log.warning(
                        f"[SKIP] idx={i} no_end_frames_saved video={video_path} "
                        f"range=({range1},{range2}) missing={len(missing_indices)}"
                    )

                dt = datetime.now().timestamp() - sample_t0
                if dt > slow_threshold_sec:
                    Log.warning(
                        f"[SLOW] idx={i} took={dt:.2f}s video={video_path} "
                        f"tail=[{frame_indices[0]},{frame_indices[-1]}] "
                        f"saved={saved}/{len(missing_indices)}"
                    )

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
    parser.add_argument("--end-frame-window", type=int, default=20, help="Save full tail window [range2-window, range2).")
    parser.add_argument("--batch-timeout-sec", type=int, default=120)
    parser.add_argument("--slow-threshold-sec", type=float, default=2.0)
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose-timing", action="store_true")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--video-dir", type=str, default="/data/datasets/bedlam_download/")
    parser.add_argument("--use-load-image", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.end_frame_window <= 0:
        raise ValueError("--end-frame-window must be > 0")
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

    Log.info(
        f"Saving BEDLAM end-frames with end_frame_window={args.end_frame_window}, "
        f"batch_size={args.batch_size}, overwrite={args.overwrite}"
    )

    worker_cfg = {
        "video_dir": args.video_dir,
        "seed": 4,
        "log_path": str(log_path),
        "try_load_image": args.use_load_image,
        "end_frame_window": args.end_frame_window,
    }

    ctx = mp.get_context("spawn")
    proc, task_q, result_q = start_worker(ctx, worker_cfg)

    skipped_samples = 0
    skipped_batches = 0
    total_saved_frames = 0

    pbar = tqdm(total=total_to_process, desc="Saving BEDLAM end-frames (imageio+pyav)")

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
                "end_frame_window": args.end_frame_window,
                "save_root": args.save_root,
                "jpeg_quality": args.jpeg_quality,
                "overwrite": args.overwrite,
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

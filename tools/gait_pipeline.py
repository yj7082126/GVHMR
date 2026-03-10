#!/usr/bin/env python3
"""
GVHMR end-to-end pipeline: preprocessing → 3D pose estimation → gait analysis.

Outputs per video
-----------------
  <stem>.pt                       – raw GVHMR prediction results
  <stem>_vitpose.csv              – 2D keypoints from ViTPose
  <stem>_coco_camera_joints.csv   – projected 2D SMPLX joints (camera space)
  <stem>_coco_global_joints.csv   – 3D SMPLX joints (global space, metres)
  <stem>_step_lengths.csv         – per-step lengths in cm (walk_out / walk_back)
  <stem>_1_debug.png              – overlay at ~5% of video
  <stem>_2_debug.png              – overlay at video midpoint

Usage
-----
  python gvhmr_pipeline.py video.mp4
  python gvhmr_pipeline.py /path/to/videos/
  python gvhmr_pipeline.py video.mp4 --output /path/to/results/ --gpu 0
  python gvhmr_pipeline.py /path/to/videos/ --skip-existing
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
import logging
import cv2
from datetime import datetime
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# ── GVHMR / HMR4D imports ────────────────────────────────────────────────────
import hydra
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from hmr4d import PROJ_ROOT
from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log
# gvhmr_pl_demo registers itself in Hydra's ConfigStore on import via
# MainStore.store(name="gvhmr_pl_demo", ..., group="model/gvhmr").
# Without this import, demo.yaml's "- model: gvhmr/gvhmr_pl_demo" has nothing
# to resolve and Hydra raises MissingConfigException.
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: F401
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K
from hmr4d.utils.preproc.tracker import pick_main_walking_id
from hmr4d.utils.preproc.vitpose_pytorch.src.vitpose_infer.pose_utils.pose_viz import joints_dict
from hmr4d.utils.geo_transform import compute_cam_angvel, move_to_start_point_face_z
from hmr4d.utils.body_model import BodyModelSMPLX
from hmr4d.utils.body_model.smplx_lite import SmplxLiteV437Coco17
from hmr4d.utils.vis.renderer import Renderer
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_kpts_with_conf_batch

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}
COCO_KP_DICT = joints_dict()["coco"]["keypoints"]

def add_file_logger(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%m/%d %H:%M:%S")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    Log.addHandler(fh)
    
# ─────────────────────────────────────────────────────────────────────────────
# Model loading (done once, reused across all videos)
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo")

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(PROJ_ROOT /cfg.ckpt_path)
    model = model.eval().cuda()

    smplx = BodyModelSMPLX(
        model_path=PROJ_ROOT / "inputs/checkpoints/body_models",
        model_type="smplx",
        gender="neutral",
        num_pca_comps=12,
        flat_hand_mean=False,
    ).cuda()
    smplx_coco = SmplxLiteV437Coco17().cuda()

    tracker          = Tracker()
    vitpose_extractor = VitPoseExtractor()
    extractor        = Extractor()

    Log.info("Models loaded.\n")
    return model, smplx, smplx_coco, tracker, vitpose_extractor, extractor, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Gait analysis: foot-strike detection → step lengths
# ─────────────────────────────────────────────────────────────────────────────

def compute_step_lengths(df_global: pd.DataFrame, fps: float,
                         output_path: Path, stem: str) -> pd.DataFrame:
    """
    Detect foot strikes from global ankle heights and compute step length + width.

    Coordinate system after move_to_start_point_face_z:
      X = lateral (side-to-side), Y = vertical (up), Z = forward (walking direction)

    For displacement dp = (dX, dY, dZ) between consecutive alternating heel strikes:
      step_length = |dZ|  — forward progress (clinical definition)
      step_width  = |dX|  — mediolateral foot separation

    Turn-around frame = frame of maximum hip-centre Z (peak forward travel).
    Steps before turn → 'walk_out'; after → 'walk_back'.
    """
    for col in ("x", "y", "z"):
        df_global[col] = df_global[col].astype(float)

    def get_traj(jidx: int):
        sub = df_global[df_global["joint_idx"] == jidx].sort_values("frame")
        return sub["frame"].values, sub[["x", "y", "z"]].values.astype(float)

    # COCO-17:  11=L-hip  12=R-hip  15=L-ankle  16=R-ankle
    frames_arr, la = get_traj(15)
    _,           ra = get_traj(16)
    _,           lh = get_traj(11)
    _,           rh = get_traj(12)

    hip_center = (lh + rh) / 2

    # Smooth (Gaussian, σ ≈ 2-3 frames at 30 fps)
    sigma   = max(3, int(fps * 0.08))
    la_y_s  = gaussian_filter1d(la[:, 1], sigma)
    ra_y_s  = gaussian_filter1d(ra[:, 1], sigma)
    hip_z_s = gaussian_filter1d(hip_center[:, 2], sigma)

    # Heel-strike = local minimum of ankle height
    min_dist   = max(5, int(fps * 0.25))  # ≥ 0.25 s between strikes
    prominence = 0.015                    # ≥ 1.5 cm oscillation

    l_strikes, _ = signal.find_peaks(-la_y_s, distance=min_dist, prominence=prominence)
    r_strikes, _ = signal.find_peaks(-ra_y_s, distance=min_dist, prominence=prominence)

    turn_frame = frames_arr[int(np.argmax(hip_z_s))]

    all_strikes = (
        [{"frame": int(frames_arr[fi]), "foot": "L", "pos": la[fi]} for fi in l_strikes] +
        [{"frame": int(frames_arr[fi]), "foot": "R", "pos": ra[fi]} for fi in r_strikes]
    )
    all_strikes.sort(key=lambda s: s["frame"])

    rows = []
    for i in range(1, len(all_strikes)):
        a, b = all_strikes[i - 1], all_strikes[i]
        if a["foot"] == b["foot"]:
            continue  # same foot twice → missed or spurious strike; skip
        dp  = b["pos"] - a["pos"]
        mid = (a["frame"] + b["frame"]) / 2
        rows.append({
            "from_frame":     a["frame"],
            "to_frame":       b["frame"],
            "from_foot":      a["foot"],
            "to_foot":        b["foot"],
            "step_length_cm": round(abs(dp[2]) * 100, 1),  # |ΔZ|
            "step_width_cm":  round(abs(dp[0]) * 100, 1),  # |ΔX|
            "phase":          "walk_out" if mid <= turn_frame else "walk_back",
        })

    df_steps = pd.DataFrame(rows)
    df_steps.to_csv(output_path / f"{stem}_step_lengths.csv", index=False)

    # Console summary
    Log.info(f"  Turn-around detected at frame {turn_frame}")
    for phase, label in [("walk_out", "Walk OUT "), ("walk_back", "Walk BACK"), (None, "Overall  ")]:
        sub = df_steps if phase is None else df_steps[df_steps["phase"] == phase]
        if len(sub) == 0:
            continue
        # for col, name in [("step_length_cm", "length"), ("step_width_cm", "width ")]:
        #     v = sub[col]
        #     Log.info(f"  {label}  {name}  n={len(sub)}  "
        #              f"mean={v.mean():.1f}  std={v.std():.1f}  "
        #              f"min={v.min():.1f}  max={v.max():.1f} cm")

    return df_steps


# ─────────────────────────────────────────────────────────────────────────────
# Full single-video pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_video(video_path: Path, output_path: Path,
                  model, smplx, smplx_coco,
                  tracker, vitpose_extractor, extractor,
                  cfg, skip_existing: bool = False) -> None:

    stem    = video_path.stem
    pt_file = output_path / f"{stem}.pt"

    if skip_existing and pt_file.exists():
        Log.info(f"[{stem}] Skipping (--skip-existing).")
        return

    # fps is always read from the video file (not stored in .pt)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if pt_file.exists():
        # ── Load cached prediction, skip all preprocessing + model inference ──
        Log.info(f"[{stem}]  Found {pt_file.name} — loading cached results, skipping inference.")
        final_results = torch.load(pt_file, map_location="cpu")
        length, width, height = final_results["dimensions"]
        Log.info(f"[{stem}]  {width}x{height}  {length} frames  {fps:.2f} fps")
    else:
        # ── Full pipeline: track → extract → VO → model.predict → save .pt ───
        try:
            length, height, width, _c = iio.improps(video_path, plugin="pyav").shape
        except Exception:
            length = 0
        if length == 0:
            cap = cv2.VideoCapture(str(video_path))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            cap.release()

        Log.info(f"[{stem}]  {width}x{height}  {length} frames  {fps:.2f} fps")

        Log.info(f"[{stem}]  Tracking...")
        track_history = tracker.track(video_path)
        vid_length    = len(track_history)
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = tracker.sort_track_length(
            track_history, video_path)

        best_id, _ = pick_main_walking_id(
            id_to_frame_ids=id_to_frame_ids,
            id_to_bbx_xyxys=id_to_bbx_xyxys,
            total_frames=vid_length,
            video_w=width,
            video_h=height,
        )
        Log.info(f"[{stem}]  Track ID selected: {best_id}")

        frame_ids_t = torch.tensor(id_to_frame_ids[best_id])
        bbx_xyxys_t = torch.tensor(id_to_bbx_xyxys[best_id])
        bbx_xyxy    = tracker.interpolate_smooth_bbx(frame_ids_t, bbx_xyxys_t, length=vid_length)
        bbx_xys     = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

        Log.info(f"[{stem}]  ViTPose + ViT features...")
        vitpose      = vitpose_extractor.extract(str(video_path), bbx_xys)         # (L,17,3)
        vit_features = extractor.extract_video_features(str(video_path), bbx_xys)  # (L,1024)

        Log.info(f"[{stem}]  SimpleVO (camera rotation estimation)...")
        vo_results = SimpleVO(video_path, scale=0.5, step=8, method="sift", f_mm=None).compute()
        R_w2c      = torch.from_numpy(vo_results[:, :3, :3])        # (L,3,3)
        K_fullimg  = estimate_K(width, height).repeat(length, 1, 1) # (L,3,3)

        Log.info(f"[{stem}]  GVHMR inference...")
        data = {
            "length":     torch.tensor(length),
            "bbx_xys":    bbx_xys,
            "kp2d":       vitpose,
            "K_fullimg":  K_fullimg,
            "cam_angvel": compute_cam_angvel(R_w2c),
            "f_imgseq":   vit_features,
        }
        pred = model.predict(data, static_cam=False)

        final_results = {
            "input_video":        str(video_path.absolute()),
            "model_dir":          str(cfg.ckpt_path),
            "dimensions":         (length, width, height),
            "bbx_xyxy":           bbx_xyxy,
            "bbx_xys":            bbx_xys,
            "vitpose":            vitpose,
            "vit_features":       vit_features,
            "R_w2c":              R_w2c,
            "K_fullimg":          K_fullimg,
            "smpl_params_global": {k: v.cpu() for k, v in pred["smpl_params_global"].items()},
            "smpl_params_incam":  {k: v.cpu() for k, v in pred["smpl_params_incam"].items()},
        }
        torch.save(final_results, pt_file)
        Log.info(f"[{stem}]  Saved {pt_file.name}")

    # ── From here: use final_results regardless of whether it was loaded or computed ──
    bbx_xyxy = final_results["bbx_xyxy"]
    vitpose  = final_results["vitpose"]

    # Renderer (per-video; depends on width/height)
    renderer = Renderer(width, height, device="cuda",
                        faces=smplx.faces, K=estimate_K(width, height))

    # ── Export: vitpose.csv ───────────────────────────────────────────────────
    Log.info(f"[{stem}]  Exporting CSVs...")
    rows = []
    for frame in range(len(vitpose)):
        for i, row in enumerate(vitpose[frame]):
            rows.append([frame, i, COCO_KP_DICT.get(i, "None"),
                         f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.4f}"])
    pd.DataFrame(rows, columns=["frame", "joint_idx", "joint_name", "x", "y", "confidence"]).to_csv(
        output_path / f"{stem}_vitpose.csv", index=False)

    # ── Export: coco_camera_joints.csv (projected 2D) ────────────────────────
    _, smplx_coco_camera_joints = smplx_coco(**{k: v.cuda() for k, v in final_results["smpl_params_incam"].items()})
    smplx_coco_camera_joints, _ = renderer.project_points_to_full_image(smplx_coco_camera_joints)
    rows = []
    for frame in range(len(smplx_coco_camera_joints)):
        for i, row in enumerate(smplx_coco_camera_joints[frame]):
            rows.append([frame, i, COCO_KP_DICT.get(i, "None"),
                         f"{row[0]:.6f}", f"{row[1]:.6f}"])
    pd.DataFrame(rows, columns=["frame", "joint_idx", "joint_name", "x", "y"]).to_csv(
        output_path / f"{stem}_coco_camera_joints.csv", index=False)

    # ── Export: coco_global_joints.csv (3D, root-normalised) ─────────────────
    smplx_coco_global_verts, smplx_coco_global_joints = smplx_coco(
        **{k: v.cuda() for k, v in final_results["smpl_params_global"].items()}, return_all_verts=True)
    smplx_coco_global_verts, smplx_coco_global_joints = move_to_start_point_face_z(
        smplx_coco_global_verts[..., :132, :],
        smplx_coco.smplx2coco17_interestd.T,
        hip_j=[11, 12],
        shoulder_j=[5, 6],
    )
    rows = []
    for frame in range(len(smplx_coco_global_joints)):
        for i, row in enumerate(smplx_coco_global_joints[frame]):
            rows.append([frame, i, COCO_KP_DICT.get(i, "None"),
                         f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.6f}"])
    df_global = pd.DataFrame(rows, columns=["frame", "joint_idx", "joint_name", "x", "y", "z"])
    df_global.to_csv(output_path / f"{stem}_coco_global_joints.csv", index=False)

    # ── Gait / step analysis ──────────────────────────────────────────────────
    Log.info(f"[{stem}]  Step length + width analysis...")
    compute_step_lengths(df_global, fps, output_path, stem)

    # ── Debug images ──────────────────────────────────────────────────────────
    # "1" = ~5% into video (subject just started walking)
    # "2" = midpoint (subject mid-walk)
    Log.info(f"[{stem}]  Rendering debug images...")
    smplx_out = smplx(**{k: v.cuda() for k, v in final_results["smpl_params_incam"].items()})

    debug_targets = {
        "1": max(0, int(length * 0.05)),  # ~5% mark
        "2": int(length * 0.5),                 # midpoint
    }
    for time_label, frame_idx in debug_targets.items():
        frame_idx = min(frame_idx, length - 10)
        img_raw   = cv2.resize(iio.imread(video_path, index=frame_idx), (width, height))
        img_annot = draw_bbx_xyxy_on_image_batch(
            bbx_xyxy[frame_idx:frame_idx + 1], [img_raw], thickness=8)[0]
        img_annot = draw_kpts_with_conf_batch(
            [img_annot[..., ::-1]],
            vitpose[frame_idx:frame_idx + 1, ..., :2],
            vitpose[frame_idx:frame_idx + 1, ...,  2],
            thickness=8)[0][..., ::-1]
        img_mesh  = renderer.render_mesh(smplx_out.vertices[frame_idx].cuda(), img_raw)
        out_png   = output_path / f"{stem}_{time_label}_debug.png"
        Image.fromarray(np.concatenate([img_annot, img_mesh], axis=1)).save(out_png)
        Log.info(f"[{stem}]  Saved {out_png.name}")

    Log.info(f"[{stem}]  Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix in VIDEO_EXTENSIONS)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GVHMR end-to-end: pose estimation + gait step analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path,
                        help="Video file or directory containing videos")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory  "
                             "(default: <input_dir>/<input_stem>_results/)")
    parser.add_argument("--gpu", default="1",
                        help="CUDA_VISIBLE_DEVICES value  (default: 1)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos whose .pt output already exists")
    args = parser.parse_args()

    # Set GPU before any CUDA initialisation
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Always run from the repo root so that relative paths in configs and
    # checkpoints (e.g. "inputs/checkpoints/...") resolve correctly.
    os.chdir(Path(__file__).resolve().parent)

    input_path = args.input.resolve()
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    if args.output is None:
        args.output = input_path.parent / f"{input_path.stem}_results"
    args.output.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_path)
    if not videos:
        sys.exit(f"No video files found at: {input_path}")

    log_file = args.output / f"gait_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    add_file_logger(log_file)
    Log.info(f"dataset_path={input_path}")
    Log.info(f"output_path={args.output}")
    Log.info(f"log_file={log_file}")
    
    Log.info(f"Found {len(videos)} video(s).")
    Log.info(f"Output → {args.output}\n")

    model, smplx, smplx_coco, tracker, vitpose_extractor, extractor, cfg = load_models()

    failed = []
    for vp in videos:
        try:
            process_video(
                vp, args.output,
                model, smplx, smplx_coco,
                tracker, vitpose_extractor, extractor,
                cfg,
                skip_existing=args.skip_existing,
            )
        except Exception:
            Log.exception(f"[ERROR] Failed: {vp.name}")
            traceback.print_exc()
            failed.append(vp.name)

    if failed:
        Log.error(f"\nFailed ({len(failed)}): {failed}")
    else:
        Log.info("All videos processed successfully.")


if __name__ == "__main__":
    main()

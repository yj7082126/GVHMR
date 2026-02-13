#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
import cv2
import hydra
import imageio.v3 as iio
import lovely_tensors as lt
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_module

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K
from hmr4d.utils.preproc.vitpose_pytorch.src.vitpose_infer.pose_utils.pose_viz import joints_dict
from hmr4d.utils.geo_transform import compute_cam_angvel, move_to_start_point_face_z
from hmr4d.utils.body_model import BodyModelSMPLX
from hmr4d.utils.body_model.smplx_lite import SmplxLiteV437Coco17
from hmr4d.utils.vis.renderer import Renderer
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: F401 imported for hydra instantiate side effects
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_kpts_with_conf_batch

lt.monkey_patch()

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gait inference on all videos in dataset_path")
    parser.add_argument("dataset_path", type=Path, help="Dataset root path containing videos")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output directory (default: <dataset_path>/results)",
    )
    parser.add_argument(
        "--track_ids",
        type=int,
        nargs="+",
        default=None,
        help="Track IDs to use. Default: longest available track per video",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos when <output>/<video_stem>.pt already exists",
    )
    return parser.parse_args()


def add_file_logger(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%m/%d %H:%M:%S")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    Log.addHandler(fh)


def get_video_paths(videos_dir: Path) -> list[Path]:
    video_paths = [
        p
        for p in sorted(videos_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    return video_paths


def resolve_video_shape(video_path: Path) -> tuple[int, int, int]:
    length, height, width, _ = iio.improps(video_path, plugin="pyav").shape
    if length == 0:
        video = cv2.VideoCapture(str(video_path))
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        video.release()
    return int(length), int(height), int(width)


def resolve_track_data(
    tracker: Tracker,
    video_path: Path,
    selected_track_ids: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    track_history = tracker.track(video_path)
    vid_length = len(track_history)
    id_to_frame_ids, id_to_bbx_xyxys, id_sorted = tracker.sort_track_length(track_history, video_path)

    if not id_sorted:
        raise RuntimeError("No track was detected")

    for track_id in id_sorted:
        lst = id_to_frame_ids[track_id]
        ranges = np.split(lst, np.where(np.diff(lst) != 1)[0] + 1)
        ranges = [(x[0], x[-1]) if len(x) > 1 else (x[0], x[0]) for x in ranges]
        Log.info(f"track id {track_id}: {len(lst)} frames, ranges: {ranges}")

    if selected_track_ids:
        valid_track_ids = [t for t in selected_track_ids if t in id_to_frame_ids]
        missing = sorted(set(selected_track_ids) - set(valid_track_ids))
        if missing:
            Log.warning(f"Requested track_ids missing in this video: {missing}")
        if not valid_track_ids:
            raise RuntimeError("None of the requested track_ids are present")
        track_ids = valid_track_ids
    else:
        track_ids = [id_sorted[0]]

    frame_ids, bbx_xyxys = [], []
    for track_id in track_ids:
        frame_ids.append(torch.tensor(id_to_frame_ids[track_id]))
        bbx_xyxys.append(torch.tensor(id_to_bbx_xyxys[track_id]))
    frame_ids = torch.cat(frame_ids)
    bbx_xyxys = torch.cat(bbx_xyxys)

    bbx_xyxy = tracker.interpolate_smooth_bbx(frame_ids, bbx_xyxys, length=vid_length)
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
    return bbx_xyxy, bbx_xys, vid_length


def run_one_video(
    video_path: Path,
    output_path: Path,
    tracker: Tracker,
    vitpose_extractor: VitPoseExtractor,
    extractor: Extractor,
    model,
    cfg,
    smplx,
    smplx_coco,
    selected_track_ids: list[int] | None,
    i = 60
) -> None:
    length, height, width = resolve_video_shape(video_path)
    Log.info(f"Start video: {video_path} ({width}x{height}x{length})")

    bbx_xyxy, bbx_xys, vid_length = resolve_track_data(tracker, video_path, selected_track_ids)
    vitpose = vitpose_extractor.extract(str(video_path), bbx_xys)
    vit_features = extractor.extract_video_features(str(video_path), bbx_xys)

    simple_vo = SimpleVO(video_path, scale=0.5, step=8, method="sift", f_mm=None)
    vo_results = simple_vo.compute()
    R_w2c = torch.from_numpy(vo_results[:, :3, :3])
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys,
        "kp2d": vitpose,
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": vit_features,
    }
    pred = model.predict(data, static_cam=False)
    renderer_c = Renderer(width, height, device="cuda", faces=smplx.faces, K=estimate_K(width, height))

    final_results = {
        "input_video": str(video_path.absolute()),
        "model_dir": str(cfg.ckpt_path),
        "dimensions": (length, width, height),
        "bbx_xyxy": bbx_xyxy,
        "bbx_xys": bbx_xys,
        "vitpose": vitpose,
        "vit_features": vit_features,
        "R_w2c": R_w2c,
        "K_fullimg": K_fullimg,
        "smpl_params_global": {k: v.cpu() for k, v in pred["smpl_params_global"].items()},
        "smpl_params_incam": {k: v.cpu() for k, v in pred["smpl_params_incam"].items()},
    }
    torch.save(final_results, output_path / f"{video_path.stem}.pt")

    df = []
    for frame in range(len(final_results["vitpose"])):
        part = final_results["vitpose"][frame]
        sub_df = [
            [frame, i, joints_dict()["coco"]["keypoints"].get(i, "None"), f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.4f}"]
            for i, row in enumerate(part)
        ]
        df.extend(sub_df)
    df = pd.DataFrame(df, columns=["frame", "joint_idx", "joint_name", "x", "y", "confidence"])
    df.to_csv(output_path / f"{video_path.stem}_vitpose.csv", index=False)

    _, smplx_coco_camera_joints = smplx_coco(**pred["smpl_params_incam"])
    smplx_coco_camera_joints, _ = renderer_c.project_points_to_full_image(  # noqa: F841
        smplx_coco_camera_joints
    )

    smplx_coco_global_verts, smplx_coco_global_joints = smplx_coco(
        **pred["smpl_params_global"], return_all_verts=True
    )
    smplx_coco_global_verts, smplx_coco_global_joints = move_to_start_point_face_z(
        smplx_coco_global_verts[..., :132, :],
        smplx_coco.smplx2coco17_interestd.T,
        hip_j=[11, 12],
        shoulder_j=[5, 6],
    )

    df = []
    for frame in range(len(smplx_coco_camera_joints)):
        part = smplx_coco_camera_joints[frame]
        sub_df = [
            [frame, i, joints_dict()["coco"]["keypoints"].get(i, "None"), f"{row[0]:.6f}", f"{row[1]:.6f}"]
            for i, row in enumerate(part)
        ]
        df.extend(sub_df)
    df = pd.DataFrame(df, columns=["frame", "joint_idx", "joint_name", "x", "y"])
    df.to_csv(output_path / f"{video_path.stem}_coco_camera_joints.csv", index=False)

    df = []
    for frame in range(len(smplx_coco_global_joints)):
        part = smplx_coco_global_joints[frame]
        sub_df = [
            [
                frame,
                i,
                joints_dict()["coco"]["keypoints"].get(i, "None"),
                f"{row[0]:.6f}",
                f"{row[1]:.6f}",
                f"{row[2]:.6f}",
            ]
            for i, row in enumerate(part)
        ]
        df.extend(sub_df)
    df = pd.DataFrame(df, columns=["frame", "joint_idx", "joint_name", "x", "y", "z"])
    df.to_csv(output_path / f"{video_path.stem}_coco_global_joints.csv", index=False)

    smplx_out = smplx(**{k: v.to("cuda") for k, v in final_results["smpl_params_incam"].items()})
    
    img_raw = cv2.resize(iio.imread(video_path, index=i), (width, height))
    img_annot = draw_bbx_xyxy_on_image_batch(final_results['bbx_xyxy'][i:i+1], [img_raw], thickness=8)[0]
    img_annot = draw_kpts_with_conf_batch(
        [img_annot[..., ::-1]], final_results['vitpose'][i:i+1, ..., :2], 
                                final_results['vitpose'][i:i+1, ...,  2], thickness=8)[0][...,::-1]
    img_cam = renderer_c.render_mesh(smplx_out.vertices[i].cuda(), img_raw)
    img_debug = Image.fromarray(np.concatenate([img_annot, img_cam], axis=1))
    img_debug.save(output_path / f"{video_path.stem}_{i}_debug.jpg")
    
    Log.info(f"Finished video: {video_path}")


def main() -> None:
    args = parse_args()
    videos_dir = args.dataset_path.expanduser().resolve()
    output_path = (args.output_path or (videos_dir / "results")).expanduser().resolve()
    output_path.mkdir(exist_ok=True, parents=True)

    log_file = output_path / f"gait_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    add_file_logger(log_file)
    Log.info(f"dataset_path={videos_dir}")
    Log.info(f"output_path={output_path}")
    Log.info(f"log_file={log_file}")

    if not videos_dir.exists() or not videos_dir.is_dir():
        raise FileNotFoundError(f"Expected videos directory at: {videos_dir}")

    video_paths = get_video_paths(videos_dir)
    if not video_paths:
        raise FileNotFoundError(f"No videos found in: {videos_dir}")

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo")

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()

    smplx = BodyModelSMPLX(
        model_path="inputs/checkpoints/body_models",
        model_type="smplx",
        gender="neutral",
        num_pca_comps=12,
        flat_hand_mean=False,
    ).cuda()
    smplx_coco = SmplxLiteV437Coco17().cuda()

    tracker = Tracker()
    vitpose_extractor = VitPoseExtractor()
    extractor = Extractor()

    Log.info(f"Found {len(video_paths)} videos")
    for idx, video_path in enumerate(video_paths, start=1):
        out_pt = output_path / f"{video_path.stem}.pt"
        if args.skip_existing and out_pt.exists():
            Log.info(f"[{idx}/{len(video_paths)}] Skip existing output for {video_path}")
            continue

        Log.info(f"[{idx}/{len(video_paths)}] Processing {video_path}")
        try:
            run_one_video(
                video_path=video_path,
                output_path=output_path,
                tracker=tracker,
                vitpose_extractor=vitpose_extractor,
                extractor=extractor,
                model=model,
                cfg=cfg,
                smplx=smplx,
                smplx_coco=smplx_coco,
                selected_track_ids=args.track_ids,
            )
        except Exception as exc:
            Log.exception(f"Failed on {video_path}: {exc}")

    Log.info("All done")


if __name__ == "__main__":
    main()

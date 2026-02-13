from ultralytics import YOLO
from hmr4d import PROJ_ROOT

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from hmr4d.utils.seq_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
)
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.net_utils import moving_average_smooth


class Tracker:
    def __init__(self) -> None:
        # https://docs.ultralytics.com/modes/predict/
        self.yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")

    def track(self, video_path, conf=0.5, imgsize=None):
        track_history = []
        cfg = {
            "device": "cuda",
            "conf": conf,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        if imgsize is not None:
            cfg["imgsz"] = imgsize  # e.g., 640
        results = self.yolo.track(video_path, **cfg)
        # frame-by-frame tracking
        track_history = []
        for result in tqdm(results, total=get_video_lwh(video_path)[0], desc="YoloV8 Tracking"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history

    @staticmethod
    def sort_track_length(track_history, video_path):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def interpolate_smooth_bbx(self, frame_ids, bbx_xyxys, length=50):
        mask = frame_id_to_mask(frame_ids, length)
        bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
        bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
        assert (bbx_xyxy_one_track.sum(1) != 0).all()

        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        return bbx_xyxy_one_track

    def get_one_track(self, video_path, conf=0.5, imgsize=None, track_id=None, 
                      print_res=True):
        # track
        track_history = self.track(video_path, conf=conf, imgsize=imgsize)
        vid_length = len(track_history)
        # parse track_history & use top1 track
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
        if print_res:
            for k in id_sorted:
                lst = id_to_frame_ids[k]
                ranges = np.split(lst, np.where(np.diff(lst)!=1)[0]+1)
                ranges = [(x[0], x[-1]) if len(x) > 1 else (x[0], x[0]) for x in ranges]
                print(f"track id {k}: {len(lst)} frames, ranges: {ranges}")
                
        track_id = id_sorted[0] if track_id is None else track_id
        frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
        bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)

        # interpolate missing frames
        bbx_xyxy_one_track = self.interpolate_smooth_bbx(frame_ids, bbx_xyxys, length=vid_length)
        return bbx_xyxy_one_track

def _smooth_1d(x, k=9):
    if len(x) < 3:
        return x
    k = min(k, len(x) if len(x) % 2 == 1 else len(x) - 1)
    k = max(k, 3)
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=float) / k
    return np.convolve(xpad, ker, mode="valid")

def pick_main_walking_id(
    id_to_frame_ids,
    id_to_bbx_xyxys,
    total_frames,
    video_w,
    video_h,
    min_coverage=0.99,
):
    """
    Returns:
        best_id (or None), diagnostics dict
    """
    diag_len = np.hypot(video_w, video_h)
    candidates = {}

    for pid, frame_ids in id_to_frame_ids.items():
        n = len(frame_ids)
        coverage = n / max(total_frames, 1)
        if coverage < min_coverage or n < 10:
            continue

        bbx = np.asarray(id_to_bbx_xyxys[pid], dtype=float)  # (N,4): x1,y1,x2,y2
        cx = 0.5 * (bbx[:, 0] + bbx[:, 2])
        cy = 0.5 * (bbx[:, 1] + bbx[:, 3])

        # 1) "roughly centered" score (higher is better)
        dist = np.hypot(cx - video_w / 2.0, cy - video_h / 2.0) / (diag_len + 1e-6)
        center_score = 1.0 - np.clip(np.median(dist), 0.0, 1.0)

        # 2) depth-change proxy from bbox area (towards/backwards camera)
        area = np.clip((bbx[:, 2] - bbx[:, 0]) * (bbx[:, 3] - bbx[:, 1]), 1.0, None)
        area_s = _smooth_1d(area, k=11)

        rel = np.diff(area_s) / (area_s[:-1] + 1e-6)  # relative frame-to-frame change
        rel_s = _smooth_1d(rel, k=9)

        up = rel_s > 0.01     # moving closer (bbox grows)
        down = rel_s < -0.01  # moving farther (bbox shrinks)
        sign_changes = np.sum((rel_s[1:] * rel_s[:-1]) < 0)

        has_both_dirs = up.any() and down.any()
        depth_range = (np.percentile(area_s, 90) - np.percentile(area_s, 10)) / (np.median(area_s) + 1e-6)
        walking_depth_score = float(has_both_dirs) * min(depth_range / 0.25, 1.0) * min(sign_changes / 2.0, 1.0)

        # Final score
        score = 0.55 * coverage + 0.30 * center_score + 0.15 * walking_depth_score
        candidates[pid] = {
            "score": score,
            "coverage": coverage,
            "center_score": center_score,
            "walking_depth_score": walking_depth_score,
            "sign_changes": int(sign_changes),
            "depth_range": float(depth_range),
        }

    if not candidates:
        return None, {}

    best_id = max(candidates, key=lambda k: candidates[k]["score"])
    return best_id, candidates

def find_ids_containing_point(id_to_frame_ids, id_to_bbx_xyxys, point_xy, min_hits=1):
    """
    Return IDs whose bbox contains point_xy at least `min_hits` times.
    """
    x, y = point_xy
    matched = []

    for pid, boxes in id_to_bbx_xyxys.items():
        boxes = np.asarray(boxes)  # (N,4) xyxy
        inside = (
            (boxes[:, 0] <= x) & (x <= boxes[:, 2]) &
            (boxes[:, 1] <= y) & (y <= boxes[:, 3])
        )
        hit_count = int(inside.sum())
        if hit_count >= min_hits:
            matched.append((pid, hit_count))

    # sort by hit count desc
    matched.sort(key=lambda t: t[1], reverse=True)
    return matched  # [(id, hit_count), ...]


def merge_tracks(id_to_frame_ids, id_to_bbx_xyxys, ids_to_merge, point_xy=None, new_id=None):
    """
    Merge multiple track IDs into one.
    If duplicate frame exists, keep the bbox whose center is closest to point_xy (if provided),
    otherwise keep the larger-area bbox.
    """
    if not ids_to_merge:
        return id_to_frame_ids, id_to_bbx_xyxys, None

    if new_id is None:
        new_id = min(ids_to_merge)  # stable simple choice

    x_ref, y_ref = point_xy if point_xy is not None else (None, None)

    # frame -> chosen bbox
    frame_to_box = {}

    for pid in ids_to_merge:
        frames = id_to_frame_ids[pid]
        boxes = np.asarray(id_to_bbx_xyxys[pid])

        for f, b in zip(frames, boxes):
            b = np.asarray(b, dtype=float)
            if f not in frame_to_box:
                frame_to_box[f] = b
                continue

            old = frame_to_box[f]
            if point_xy is not None:
                c_old = ((old[0] + old[2]) * 0.5, (old[1] + old[3]) * 0.5)
                c_new = ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
                d_old = (c_old[0] - x_ref) ** 2 + (c_old[1] - y_ref) ** 2
                d_new = (c_new[0] - x_ref) ** 2 + (c_new[1] - y_ref) ** 2
                if d_new < d_old:
                    frame_to_box[f] = b
            else:
                a_old = (old[2] - old[0]) * (old[3] - old[1])
                a_new = (b[2] - b[0]) * (b[3] - b[1])
                if a_new > a_old:
                    frame_to_box[f] = b

    merged_frames = sorted(frame_to_box.keys())
    merged_boxes = np.array([frame_to_box[f] for f in merged_frames], dtype=float)

    # write merged track
    id_to_frame_ids[new_id] = merged_frames
    id_to_bbx_xyxys[new_id] = merged_boxes

    # remove old tracks except new_id
    for pid in ids_to_merge:
        if pid != new_id:
            id_to_frame_ids.pop(pid, None)
            id_to_bbx_xyxys.pop(pid, None)

    return id_to_frame_ids, id_to_bbx_xyxys, new_id


def find_and_optionally_merge(id_to_frame_ids, id_to_bbx_xyxys, point_xy, min_hits=1):
    matches = find_ids_containing_point(id_to_frame_ids, id_to_bbx_xyxys, point_xy, min_hits=min_hits)
    matched_ids = [pid for pid, _ in matches]

    merged_id = None
    if len(matched_ids) > 2:
        id_to_frame_ids, id_to_bbx_xyxys, merged_id = merge_tracks(
            id_to_frame_ids, id_to_bbx_xyxys, matched_ids, point_xy=point_xy
        )

    return {
        "matches": matches,          # before merge
        "merged": len(matched_ids) > 2,
        "merged_id": merged_id,
        "id_to_frame_ids": id_to_frame_ids,
        "id_to_bbx_xyxys": id_to_bbx_xyxys,
    }

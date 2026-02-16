from pathlib import Path
import numpy as np
import torch
import cv2
from hmr4d.utils.pylogger import Log
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from time import time

from hmr4d.configs import MainStore, builds
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines

import hmr4d.utils.matrix as matrix
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.dataset.bedlam.utils import mid2featname, mid2vname
from hmr4d.utils.geo_transform import compute_cam_angvel, apply_T_on_points
from hmr4d.utils.geo.hmr_global import get_T_w2c_from_wcparams, get_c_rootparam, get_R_c2gv
from hmr4d.network.hmr2.utils.preproc import crop_and_resize


class BedlamDatasetV2(ImgfeatMotionDatasetBase):
    """mid_to_valid_range and features are newly generated."""

    MIDINDEX_TO_LOAD = {
        "all60": ("mid_to_valid_range_all60.pt", "imgfeats/bedlam_all60"),
        "maxspan60": ("mid_to_valid_range_maxspan60.pt", "imgfeats/bedlam_maxspan60"),
    }

    def __init__(
        self,
        mid_indices=["all60", "maxspan60"],
        lazy_load=True,  # Load from disk when needed
        random1024=False,  # Faster loading for debugging
        load_image=False,
        load_indices=[0],
        end_frame_window=20,
    ):
        self.root = Path("inputs/BEDLAM/hmr4d_support")
        self.video_root = Path("/data/datasets/bedlam_download")
        
        self.min_motion_frames = 60
        self.max_motion_frames = 120
        self.lazy_load = lazy_load
        self.dataset_name = "bedlam"
        self.random1024 = random1024
        self.load_image = load_image
        self.load_indices = [int(i) for i in load_indices]
        self.end_frame_window = int(end_frame_window)
        self.image_crop_size = 512
        self.mid_to_saved_frames = {}

        # speficify mid_index to handle
        if not isinstance(mid_indices, list):
            mid_indices = [mid_indices]
        self.mid_indices = mid_indices
        assert all([m in self.MIDINDEX_TO_LOAD for m in mid_indices])

        super().__init__()

    def _load_dataset(self):
        Log.info(f"[BEDLAM] Loading from {self.root}")
        tic = time()
        # Load mid to valid range
        self.mid_to_valid_range = {}
        self.mid_to_imgfeat_dir = {}
        for m in self.mid_indices:
            fn, feat_dir = self.MIDINDEX_TO_LOAD[m]
            mid_to_valid_range_ = torch.load(self.root / fn, weights_only=True)
            self.mid_to_valid_range.update(mid_to_valid_range_)
            self.mid_to_imgfeat_dir.update({mid: self.root / feat_dir for mid in mid_to_valid_range_})

        # Load motionfiles
        Log.info(f"[BEDLAM] Start loading motion files")
        if self.random1024:  # Debug, faster loading
            try:
                Log.info(f"[BEDLAM] Loading 1024 samples for debugging ...")
                self.motion_files = torch.load(self.root / "smplpose_v2_random1024.pth", weights_only=True)
            except:
                Log.info(f"[BEDLAM] Not found, saving 1024 samples to disk ...")
                self.motion_files = torch.load(self.root / "smplpose_v2.pth", weights_only=True)
                keys = list(self.motion_files.keys())
                keys = np.random.choice(keys, 1024, replace=False)
                self.motion_files = {k: self.motion_files[k] for k in keys}
                torch.save(self.motion_files, self.root / "smplpose_v2_random1024.pth")
            self.mid_to_valid_range = {k: v for k, v in self.mid_to_valid_range.items() if k in self.motion_files}
        else:
            self.motion_files = torch.load(self.root / "smplpose_v2.pth", weights_only=True)
        Log.info(f"[BEDLAM] Motion files loaded. Elapsed: {time() - tic:.2f}s")

    def _get_saved_frame_dir(self, mid):
        vname = mid2vname(mid)  # {scene}/{seq}.mp4
        video_path = self.video_root / vname
        return video_path.parent.parent / "vis" / video_path.stem

    def _get_saved_frame_indices(self, mid):
        if mid in self.mid_to_saved_frames:
            return self.mid_to_saved_frames[mid]

        frame_dir = self._get_saved_frame_dir(mid)
        if not frame_dir.exists():
            self.mid_to_saved_frames[mid] = []
            return self.mid_to_saved_frames[mid]

        frame_indices = []
        for fp in frame_dir.glob(f"*.jpg"):
            try:
                frame_indices.append(int(fp.stem))
            except ValueError:
                continue
        frame_indices = sorted(frame_indices)
        self.mid_to_saved_frames[mid] = frame_indices
        return frame_indices

    def _load_aligned_images(self, mid, frame_indices, bbx_xys):
        if not self.load_image:
            return None

        if len(frame_indices) != len(bbx_xys):
            Log.info(f"[BEDLAM] frame/bbx mismatch: {len(frame_indices)} vs {len(bbx_xys)} for {mid}")
            return None

        frame_dir = self._get_saved_frame_dir(mid)
        crops = []
        for frame_idx, bbx in zip(frame_indices, bbx_xys):
            frame_path = frame_dir / f"{int(frame_idx):05d}.jpg"
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                Log.info(f"[BEDLAM] saved frame not found/readable: {frame_path}")
                return None
            frame = frame_bgr[..., ::-1]  # BGR -> RGB
            img_crop, _ = crop_and_resize(
                frame,
                bbx[:2].cpu().numpy(),
                float(bbx[2].item()),
                dst_size=self.image_crop_size,
                enlarge_ratio=1.0,
            )
            crops.append(img_crop)
        return torch.from_numpy(np.stack(crops))

    def _resolve_load_indices(self, length):
        if length <= 0:
            return []
        resolved = []
        for i in self.load_indices:
            j = i if i >= 0 else length + i
            j = max(0, min(length - 1, j))
            resolved.append(int(j))
        return resolved

    def _sample_start_end_from_saved(self, mid, range1, range2):
        """Sample clip [start, end) from saved frame indices.
        Priority:
        1) start/end window candidates + feasible length + load_indices fully saved
        2) all saved frames + feasible length + load_indices fully saved
        3) all saved frames + feasible length (if 1/2 impossible)
        4) fallback to valid-range boundaries.
        """
        saved = self._get_saved_frame_indices(mid)
        saved = sorted({int(f) for f in saved if range1 <= int(f) < range2})
        mlength = range2 - range1

        # For short ranges, always use full valid range.
        if mlength <= self.min_motion_frames:
            return int(range1), int(range2)

        min_len_eff = self.min_motion_frames
        max_len_eff = min(self.max_motion_frames, mlength)

        if len(saved) == 0:
            Log.info(f"[BEDLAM] no saved frames for {mid}, fallback valid-range sample")
            effect_max_motion_len = min(self.max_motion_frames, mlength)
            length = np.random.randint(self.min_motion_frames, effect_max_motion_len + 1)  # [low, high)
            start = np.random.randint(range1, range2 - length + 1)
            return start, start + length

        start_candidates = [f for f in saved if range1 <= f < range1 + self.end_frame_window]
        end_candidates = [f for f in saved if range2 - self.end_frame_window <= f < range2]
        if len(start_candidates) == 0:
            start_candidates = saved
        if len(end_candidates) == 0:
            end_candidates = saved

        saved_set = set(saved)
        def _collect_pairs(starts, ends, require_saved_loaded=True):
            pairs = []
            for s in starts:
                for e in ends:
                    if e < s:
                        continue
                    length = e - s + 1
                    if length < min_len_eff or length > max_len_eff:
                        continue
                    if require_saved_loaded:
                        rel = self._resolve_load_indices(length)
                        abs_indices = [s + r for r in rel]
                        if not all((a in saved_set) for a in abs_indices):
                            continue
                    pairs.append((s, e + 1))  # exclusive end
            return pairs

        pairs = _collect_pairs(start_candidates, end_candidates, require_saved_loaded=True)
        if len(pairs) == 0:
            pairs = _collect_pairs(saved, saved, require_saved_loaded=True)
        # if len(pairs) == 0:
        #     pairs = _collect_pairs(saved, saved, require_saved_loaded=False)

        if len(pairs) > 0:
            s, e = pairs[int(np.random.randint(0, len(pairs)))]
        # Last-resort: single saved frame clip (guarantees image frame exists).
        s = int(start_candidates[0])
        e = int(min(range2, s + max_len_eff))
        if (e - s) > self.max_motion_frames:
            e = s + self.max_motion_frames
            if e > range2:
                e = range2
                s = max(range1, e - self.max_motion_frames)
        return s, e

    def _get_idx2meta(self):
        # sum_frame = sum([e-s for s, e in self.mid_to_valid_range.values()])
        self.idx2meta = sorted(list(self.mid_to_valid_range.keys()))
        Log.info(f"[BEDLAM] {len(self.idx2meta)} sequences. ")

    def _load_data(self, idx):
        mid = self.idx2meta[idx]
        # neutral smplx : "pose": (F, 63), "trans": (F, 3), "beta": (10),
        #           and : "skeleton": (J, 3)
        data = self.motion_files[mid].copy()

        # Random select a subset from saved frame candidates.
        range1, range2 = self.mid_to_valid_range[mid]  # [range1, range2)
        start, end = self._sample_start_end_from_saved(mid, range1, range2)
        # Hard cap clip length to max_motion_frames.

        length = end - start
        data["start_end"] = (start, end)
        data["length"] = length
        data["meta"] = {"data_name": self.dataset_name, "idx": idx, "vid": mid, "start_end": (start, end)}
        
        # Update data to a subset
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and len(v.shape) > 1 and k != "skeleton":
                data[k] = v[start:end]

        # Load img(as feature) : {mid -> 'features', 'bbx_xys', 'img_wh', 'start_end'}
        imgfeat_dir = self.mid_to_imgfeat_dir[mid]
        f_img_dict = torch.load(imgfeat_dir / mid2featname(mid), weights_only=True)

        # remap (start, end)
        start_mapped = start - f_img_dict["start_end"][0]
        end_mapped = end - f_img_dict["start_end"][0]

        data["f_imgseq"] = f_img_dict["features"][start_mapped:end_mapped].float()  # (L, 1024)
        data["bbx_xys"] = f_img_dict["bbx_xys"][start_mapped:end_mapped].float()  # (L, 4)
        data["img_wh"] = f_img_dict["img_wh"]  # (2)
        data["kp2d"] = torch.zeros((end - start), 17, 3)  # (L, 17, 3)  # do not provide kp2d
        rel_indices = self._resolve_load_indices(end - start)
        abs_indices = [start + i for i in rel_indices]
        bbx_sel = data["bbx_xys"][torch.as_tensor(rel_indices, dtype=torch.long)] if len(rel_indices) > 0 else data["bbx_xys"][0:0]
        data["image"] = self._load_aligned_images(mid, abs_indices, bbx_sel)
        if data["image"] is None:
            # Fallback: replace missing requested frames with nearest saved frame inside [start, end),
            # and keep image tensor size consistent with len(load_indices).
            saved_clip = sorted(
                {int(f) for f in self._get_saved_frame_indices(mid) if start <= int(f) < end}
            )
            if len(saved_clip) > 0:
                abs_fallback = [min(saved_clip, key=lambda x: abs(x - a)) for a in abs_indices]
                rel_fallback = [max(0, min((end - start) - 1, af - start)) for af in abs_fallback]
                bbx_sel_fb = (
                    data["bbx_xys"][torch.as_tensor(rel_fallback, dtype=torch.long)]
                    if len(rel_fallback) > 0
                    else data["bbx_xys"][0:0]
                )
                data["image"] = self._load_aligned_images(mid, abs_fallback, bbx_sel_fb)
                data["image_frame_indices"] = torch.as_tensor(rel_fallback, dtype=torch.long)
            else:
                Log.info(f"[BEDLAM] image loading failed for {mid}, start_end=({start},{end}), abs_indices={abs_indices}")
                n_img = max(1, len(rel_indices))
                data["image"] = torch.zeros((n_img, self.image_crop_size, self.image_crop_size, 3), dtype=torch.uint8)
                data["image_frame_indices"] = torch.as_tensor(rel_indices if len(rel_indices) > 0 else [0], dtype=torch.long)
        else:
            data["image_frame_indices"] = torch.as_tensor(rel_indices, dtype=torch.long)
        data["f_dinov3_imgseq"], data["f_dinov3_frame"] = None, None

        return data

    def _process_data(self, data, idx):
        length = data["length"]

        # SMPL params in cam
        body_pose = data["pose"][:, 3:]  # (F, 63)
        betas = data["beta"].repeat(length, 1)  # (F, 10)
        global_orient = data["global_orient_incam"]  # (F, 3)
        transl = data["trans_incam"] + data["cam_ext"][:, :3, 3]  # (F, 3), bedlam convention
        smpl_params_c = {"body_pose": body_pose, "betas": betas, "transl": transl, "global_orient": global_orient}

        # SMPL params in world
        global_orient_w = data["pose"][:, :3]  # (F, 3)
        transl_w = data["trans"]  # (F, 3)
        smpl_params_w = {"body_pose": body_pose, "betas": betas, "transl": transl_w, "global_orient": global_orient_w}

        gravity_vec = torch.tensor([0, -1, 0], dtype=torch.float32)  # (3), BEDLAM is ay
        T_w2c = get_T_w2c_from_wcparams(
            global_orient_w=global_orient_w,
            transl_w=transl_w,
            global_orient_c=global_orient,
            transl_c=transl,
            offset=data["skeleton"][0],
        )  # (F, 4, 4)
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], gravity_vec)  # (F, 3, 3)

        # cam_angvel (slightly different from WHAM)
        cam_angvel = compute_cam_angvel(T_w2c[:, :3, :3])  # (F, 6)

        # Returns: do not forget to make it batchable! (last lines)
        max_len = self.max_motion_frames
        return_data = {
            "meta": data["meta"],
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,  # (F, 3, 3)
            "gravity_vec": gravity_vec,  # (3)
            "bbx_xys": data["bbx_xys"],  # (F, 3)
            "K_fullimg": data["cam_int"],  # (F, 3, 3)
            "f_imgseq": data["f_imgseq"],  # (F, D)
            "image": data["image"],  # (F, H, W, 3) or None
            "image_frame_indices": data["image_frame_indices"],  # (N,)
            "f_dinov3_imgseq": data["f_dinov3_imgseq"],  # (F, 1280, 32, 32) or None
            "f_dinov3_frame": data["f_dinov3_frame"],  # (F,) or None
            "kp2d": data["kp2d"],  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "mask": {
                "valid": get_valid_mask(max_len, length),
                "vitpose": False,
                "bbx_xys": True,
                "f_imgseq": True,
                "image": data["image"] is not None,
                "f_dinov3_imgseq": data["f_dinov3_imgseq"] is not None,
                "spv_incam_only": False,
            },
        }
        
        # Batchable
        return_data["smpl_params_c"] = repeat_to_max_len_dict(return_data["smpl_params_c"], max_len)
        return_data["smpl_params_w"] = repeat_to_max_len_dict(return_data["smpl_params_w"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
        return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
        return return_data


group_name = "train_datasets/imgfeat_bedlam"
MainStore.store(name="v2", node=builds(BedlamDatasetV2), group=group_name)
MainStore.store(name="v2_random1024", node=builds(BedlamDatasetV2, random1024=True), group=group_name)
MainStore.store(name="v2_with_image", node=builds(BedlamDatasetV2, load_image=True), group=group_name)

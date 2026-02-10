from pathlib import Path
import numpy as np
import torch
from hmr4d.utils.pylogger import Log
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from time import time

from hmr4d.configs import MainStore, builds
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.vis.renderer_utils import simple_render_mesh_background
from hmr4d.utils.video_io_utils import read_video_np, save_video

import hmr4d.utils.matrix as matrix
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.dataset.bedlam.utils import mid2featname, mid2vname
from hmr4d.utils.geo_transform import compute_cam_angvel, apply_T_on_points
from hmr4d.utils.geo.hmr_global import get_T_w2c_from_wcparams, get_c_rootparam, get_R_c2gv


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
        no_dinov3=True,
        dinov3_dict_path=None,
    ):
        self.root = Path("inputs/BEDLAM/hmr4d_support")
        self.min_motion_frames = 60
        self.max_motion_frames = 120
        self.lazy_load = lazy_load
        self.dataset_name = "bedlam"
        self.random1024 = random1024
        self.no_dinov3 = no_dinov3
        self.dinov3_dict_path = dinov3_dict_path

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
        self._load_dinov3_dict()

    def _load_dinov3_dict(self):
        self.dinov3_feat_dict = None
        if self.no_dinov3 or self.dinov3_dict_path is None:
            return
        dinov3_path = Path(self.dinov3_dict_path)
        if not dinov3_path.exists():
            Log.info(f"[BEDLAM] DinoV3 dict not found: {dinov3_path}")
            return
        raw_dict = torch.load(dinov3_path, weights_only=True)
        self.dinov3_feat_dict = {}
        for vname, items in raw_dict.items():
            frame2feat = {}
            for item in items:
                frame2feat[int(item["index"])] = item["feat"].float()
            self.dinov3_feat_dict[vname] = frame2feat
        Log.info(f"[BEDLAM] Loaded DinoV3 dict: {dinov3_path} ({len(self.dinov3_feat_dict)} videos)")

    def _get_first_dinov3(self, key_candidates, start, end):
        if self.dinov3_feat_dict is None:
            return None, None
        frame2feat = None
        for key in key_candidates:
            if key in self.dinov3_feat_dict:
                frame2feat = self.dinov3_feat_dict[key]
                break
        if frame2feat is None:
            return None, None

        selected_idx = start if start in frame2feat else None
        if selected_idx is None:
            valid_indices = [i for i in frame2feat.keys() if start <= i < end]
            if len(valid_indices) == 0:
                return None, None
            selected_idx = min(valid_indices)

        feat = frame2feat[selected_idx]
        return feat[None], torch.tensor([selected_idx], dtype=torch.long)

    def _get_idx2meta(self):
        # sum_frame = sum([e-s for s, e in self.mid_to_valid_range.values()])
        self.idx2meta = sorted(list(self.mid_to_valid_range.keys()))
        Log.info(f"[BEDLAM] {len(self.idx2meta)} sequences. ")

    def _load_data(self, idx):
        mid = self.idx2meta[idx]
        # neutral smplx : "pose": (F, 63), "trans": (F, 3), "beta": (10),
        #           and : "skeleton": (J, 3)
        data = self.motion_files[mid].copy()

        # Random select a subset
        range1, range2 = self.mid_to_valid_range[mid]  # [range1, range2)
        mlength = range2 - range1
        min_motion_len = self.min_motion_frames
        max_motion_len = self.max_motion_frames

        if mlength < min_motion_len:  # the minimal mlength is 30 when generating data
            start = range1
            length = mlength
        else:
            effect_max_motion_len = min(max_motion_len, mlength)
            length = np.random.randint(min_motion_len, effect_max_motion_len + 1)  # [low, high)
            start = np.random.randint(range1, range2 - length + 1)
        end = start + length
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
        vname = mid2vname(mid)
        featname = mid2featname(mid)
        key_candidates = [mid, vname, Path(vname).stem, featname, Path(featname).stem]
        data["f_dinov3_imgseq"], data["f_dinov3_frame"] = self._get_first_dinov3(
            key_candidates, start, end
        )

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
            "f_dinov3_imgseq": data["f_dinov3_imgseq"],  # (F, 1280, 32, 32) or None
            "f_dinov3_frame": data["f_dinov3_frame"],  # (F,) or None
            "kp2d": data["kp2d"],  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "mask": {
                "valid": get_valid_mask(max_len, length),
                "vitpose": False,
                "bbx_xys": True,
                "f_imgseq": True,
                "f_dinov3_imgseq": data["f_dinov3_imgseq"] is not None,
                "spv_incam_only": False,
            },
        }

        if False:  # check transformation, wis3d: sampled motion (global, incam)
            wis3d = make_wis3d(name="debug-data-bedlam")
            smplx = make_smplx("supermotion")

            # global
            smplx_out = smplx(**smpl_params_w)
            w_gt_joints = smplx_out.joints
            add_motion_as_lines(w_gt_joints, wis3d, name="w-gt_joints")

            # incam
            smplx_out = smplx(**smpl_params_c)
            c_gt_joints = smplx_out.joints
            add_motion_as_lines(c_gt_joints, wis3d, name="c-gt_joints")

            # Check transformation works correctly
            print("T_w2c", (apply_T_on_points(w_gt_joints, T_w2c) - c_gt_joints).abs().max())
            R_c, t_c = get_c_rootparam(
                smpl_params_w["global_orient"], smpl_params_w["transl"], T_w2c, data["skeleton"][0]
            )
            print("transl_c", (t_c - smpl_params_c["transl"]).abs().max())
            R_diff = matrix_to_axis_angle(
                (axis_angle_to_matrix(R_c) @ axis_angle_to_matrix(smpl_params_c["global_orient"]).transpose(-1, -2))
            ).norm(dim=-1)
            print("global_orient_c", R_diff.abs().max())  # < 1e-6

            skeleton_beta = smplx.get_skeleton(smpl_params_c["betas"])
            print("Skeleton", (skeleton_beta[0] - data["skeleton"]).abs().max())  # (1.2e-7)

        if False:  # cam-overlay
            smplx = make_smplx("supermotion")

            # *. original bedlam param
            # mid = self.idx2meta[idx]
            # video_path = "-".join(mid.replace("bedlam_data/", "inputs/bedlam/").split("-")[:-1])
            # npz_file = "inputs/bedlam/processed_labels/20221024_3-10_100_batch01handhair_static_highSchoolGym.npz"
            # params = np.load(npz_file, allow_pickle=True)
            # mid2index = {}
            # for j in tqdm(range(len(params["video_name"]))):
            #     k = params["video_name"][j] + "-" + params["sub"][j]
            #     mid2index[k] = j
            # betas = params['shape'][mid2index[mid]][:length]
            # global_orient_incam = torch.from_numpy(params['pose_cam'][121][:, :3])
            # body_pose = torch.from_numpy(params['pose_cam'][121][:, 3:66])
            # transl_incam = torch.from_numpy(params["trans_cam"][121])
            smplx_out = smplx(**smpl_params_c)

            # ----- Render Overlay ----- #
            mid = self.idx2meta[idx]
            images = read_video_np(self.root / "videos" / mid2vname(mid), data["start_end"][0], data["start_end"][1])
            render_dict = {
                "K": data["cam_int"][:1],  # only support batch-size 1
                "faces": smplx.faces,
                "verts": smplx_out.vertices,
                "background": images,
            }
            img_overlay = simple_render_mesh_background(render_dict)
            save_video(img_overlay, "tmp.mp4", crf=23)

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

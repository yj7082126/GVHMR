from pathlib import Path
import numpy as np
import torch
import cv2
from hmr4d.utils.pylogger import Log

from hmr4d.configs import MainStore, builds
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.utils.net_utils import repeat_to_max_len, repeat_to_max_len_dict
from hmr4d.network.hmr2.utils.preproc import crop_and_resize

class Uni3CAlignedDatasetV1(ImgfeatMotionDatasetBase):
    def __init__(
        self, 
        random = True,
        n_els = -1,
        load_image=True,
        image_crop_size=512,
        load_indices=[0],
    ):
        self.root = Path("inputs/uni3c_aligned")
        self.max_motion_frames = 120
        self.random = random
        self.n_els = n_els
        self.dataset_name = "UNI3C_SynthGenerated"
        self.load_image = load_image
        self.load_indices = [int(i) for i in load_indices]
        self.image_crop_size = image_crop_size
        
        super().__init__()
        
    def _load_dataset(self):
        Log.info(f"[{self.dataset_name}] Loading from {str(self.root)} ...")
        
        self.indices = sorted([str(x.relative_to(self.root)) for x in self.root.glob("*/*")])
        if self.random:
            np.random.shuffle(self.indices)
        Log.info(f"[{self.dataset_name}] Loaded from {len(self.indices)} synthetic samples")

    def _load_aligned_images(self, mid, frame_indices, bbx_xys):
        if not self.load_image:
            return None
        if len(frame_indices) != len(bbx_xys):
            Log.info(f"[{self.dataset_name}] frame/bbx mismatch: {len(frame_indices)} vs {len(bbx_xys)} for {mid}")
            return None

        crops = []
        for frame_idx, bbx in zip(frame_indices, bbx_xys):
            input_path = self.root / f"{mid}/frames/{int(frame_idx):04d}.png"
            frame_bgr = cv2.imread(str(input_path))
            if frame_bgr is None:
                Log.info(f"[{self.dataset_name}] image not found/readable: {input_path}")
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

    def _get_idx2meta(self):
        self.n_els = len(self.indices) if self.n_els == -1 else self.n_els
        self.idx2meta = {k: v for k, v in enumerate(self.indices[:self.n_els])}
        Log.info(f"[{self.dataset_name}] Using {len(self.idx2meta)} synthetic samples")

    def _resolve_load_indices(self, length):
        if length <= 0:
            return []
        resolved = []
        for i in self.load_indices:
            j = i if i >= 0 else length + i
            j = max(0, min(length - 1, j))
            resolved.append(int(j))
        return resolved

    def _load_data(self, idx):
        mid = self.idx2meta[idx]
        batch = torch.load(self.root / f"{mid}/batch_meta.pt")
        del batch['index']
        batch["_dinov3_mid"] = mid
        return batch
    
    def _process_data(self, data, idx):
        max_len = self.max_motion_frames
        mid = data.get("_dinov3_mid", "")
        clip_len = int(data.get("length", data["bbx_xys"].shape[0]))
        rel_indices = self._resolve_load_indices(clip_len)
        bbx_sel = (
            data["bbx_xys"][torch.as_tensor(rel_indices, dtype=torch.long)]
            if len(rel_indices) > 0
            else data["bbx_xys"][0:0]
        )
        image = self._load_aligned_images(mid, rel_indices, bbx_sel)
        data["image"] = image
        data["f_dinov3_imgseq"], data["f_dinov3_frame"] = None, None
        if "mask" not in data:
            data["mask"] = {}
        data["mask"]["image"] = image is not None
        data["mask"]["f_dinov3_imgseq"] = False
        
        data["smpl_params_c"] = repeat_to_max_len_dict(data["smpl_params_c"], max_len)
        data["smpl_params_w"] = repeat_to_max_len_dict(data["smpl_params_w"], max_len)
        data["R_c2gv"] = repeat_to_max_len(data["R_c2gv"], max_len)
        data["bbx_xys"] = repeat_to_max_len(data["bbx_xys"], max_len)
        data["K_fullimg"] = repeat_to_max_len(data["K_fullimg"], max_len)
        data["f_imgseq"] = repeat_to_max_len(data["f_imgseq"], max_len)
        data["kp2d"] = repeat_to_max_len(data["kp2d"], max_len)
        data["cam_angvel"] = repeat_to_max_len(data["cam_angvel"], max_len)
        if "_dinov3_mid" in data:
            del data["_dinov3_mid"]
        return data

group_name = "train_datasets/synth_uni3c_aligned"
MainStore.store(name="v1", node=builds(Uni3CAlignedDatasetV1), group=group_name)

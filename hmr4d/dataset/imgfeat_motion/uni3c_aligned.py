from pathlib import Path
import numpy as np
import torch
from hmr4d.utils.pylogger import Log

from hmr4d.configs import MainStore, builds
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.utils.net_utils import repeat_to_max_len, repeat_to_max_len_dict

class Uni3CAlignedDatasetV1(ImgfeatMotionDatasetBase):
    def __init__(
        self, 
        random = True,
        n_els = -1
    ):
        self.root = Path("inputs/uni3c_aligned")
        self.max_motion_frames = 120
        self.random = random
        self.n_els = n_els
        self.dataset_name = "UNI3C_SynthGenerated"
        
        super().__init__()
        
    def _load_dataset(self):
        Log.info(f"[{self.dataset_name}] Loading from {str(self.root)} ...")
        
        self.indices = sorted([str(x.relative_to(self.root)) for x in self.root.glob("*/*")])
        if self.random:
            np.random.shuffle(self.indices)
        Log.info(f"[{self.dataset_name}] Loaded from {len(self.indices)} synthetic samples")

    def _get_idx2meta(self):
        self.n_els = len(self.indices) if self.n_els == -1 else self.n_els
        self.idx2meta = {k: v for k, v in enumerate(self.indices[:self.n_els])}
        Log.info(f"[{self.dataset_name}] Using {len(self.idx2meta)} synthetic samples")

    def _load_data(self, idx):
        mid = self.idx2meta[idx]
        batch = torch.load(self.root / f"{mid}/batch_meta.pt")
        del batch['index']
        return batch
    
    def _process_data(self, data, idx):
        max_len = self.max_motion_frames
        
        if False:
            from hmr4d.utils.smplx_utils import make_smplx
            from hmr4d.utils.vis.renderer import Renderer
            from hmr4d.utils.video_io_utils import read_video_np, get_writer
            from hmr4d.utils.vis.cv2_utils import draw_kpts_with_conf_batch, draw_kpts_batch

            smplx = make_smplx("supermotion")
            renderer_c = Renderer(batch['meta']['width'], batch['meta']['height'], device="cuda", 
                      faces=smplx.faces, K=batch['K_fullimg'][0])

            verts = smplx(**{k:v.to(device) for k,v in batch['smpl_params_c'].items()}).vertices
            images = read_video_np(root / f"{mid}/final.mp4", start_frame=1)  # (T, H, W, 3), np.uint8

            writer1 = get_writer('tmp.mp4', fps=30, crf=23)
            for j in tqdm(range(120), desc=f"Rendering Global"):
                # black_backg = np.zeros((batch['meta']['height'], batch['meta']['width'], 3)).astype(np.uint8)
                backg = cv2.resize(images[j], (batch['meta']['width'], batch['meta']['height']))

                smpl_rgbs, smpl_depths = renderer_c.render_mesh(
                    verts[j], background=backg, return_depth=True
                )
                
                writer1.write_frame(smpl_rgbs)
            writer1.close()
        
                # Batchable
        
        data["smpl_params_c"] = repeat_to_max_len_dict(data["smpl_params_c"], max_len)
        data["smpl_params_w"] = repeat_to_max_len_dict(data["smpl_params_w"], max_len)
        data["R_c2gv"] = repeat_to_max_len(data["R_c2gv"], max_len)
        data["bbx_xys"] = repeat_to_max_len(data["bbx_xys"], max_len)
        data["K_fullimg"] = repeat_to_max_len(data["K_fullimg"], max_len)
        data["f_imgseq"] = repeat_to_max_len(data["f_imgseq"], max_len)
        data["kp2d"] = repeat_to_max_len(data["kp2d"], max_len)
        data["cam_angvel"] = repeat_to_max_len(data["cam_angvel"], max_len)
        return data

group_name = "train_datasets/synth_uni3c_aligned"
MainStore.store(name="v1", node=builds(Uni3CAlignedDatasetV1), group=group_name)

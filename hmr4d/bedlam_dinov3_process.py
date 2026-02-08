import os, sys
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import torch

from hmr4d.utils.preproc import Extractor
from hmr4d.utils.video_io_utils import read_video_np

device = 'cuda:0'
#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-ind', type=int, default=42)
    parser.add_argument('--end-ind', type=int, default=55)
    args = parser.parse_args()
    
    extractor = Extractor(extractor_type='dinov3')

    #%%
    root = Path("inputs/BEDLAM/hmr4d_support")
    # output_root = (root / "imgfeats_dinov3")
    video_dir = Path("/data/datasets/bedlam_download/")
    processed_dict = defaultdict(list)
    output_root = Path("/data/datasets/bedlam_dinov3_feats/")
    output_root.mkdir(parents=True, exist_ok=True)

    mid_to_valid_range_ = torch.load(root / "mid_to_valid_range_all60.pt")
    for mid in mid_to_valid_range_:
        mid = Path(mid)
        parent = f"bedlam_all60/{mid.parts[-3]}"
        imgfeat_dir = root / f"imgfeats/bedlam_all60/{mid.parts[-3]}/{mid.parts[-1]}.pt"
        videopath_dir = video_dir / "/".join(mid.parts[-3:-1]) / f"{mid.stem}.mp4"
        processed_dict[parent].append( (videopath_dir, imgfeat_dir) )
    mid_to_valid_range_ = torch.load(root / "mid_to_valid_range_maxspan60.pt")
    for mid in mid_to_valid_range_:
        mid = Path(mid)
        parent = f"bedlam_maxspan60/{mid.parts[-3]}"
        imgfeat_dir = root / f"imgfeats/bedlam_maxspan60/{mid.parts[-3]}/{mid.parts[-1]}.pt"
        videopath_dir = video_dir / "/".join(mid.parts[-3:-1]) / f"{mid.stem}.mp4"
        processed_dict[parent].append( (videopath_dir, imgfeat_dir) )
        
    #%%
    for key in list(processed_dict.keys())[args.start_ind:args.end_ind]:
        videofeats = {}
        for (videopath_dir, imgfeat_dir) in tqdm(processed_dict[key]):
            f_img_dict = torch.load(imgfeat_dir)
            start, end = f_img_dict["start_end"]
            frames = read_video_np(videopath_dir, start_frame=start, end_frame=end)
            
            (output_root / Path(key).parent).mkdir(parents=True, exist_ok=True)
            try:
                test = extractor.extract_video_features(frames, f_img_dict["bbx_xys"][start:end], 
                                                    path_type='np', img_ds=1.0, batch_size=16)
                videofeats[imgfeat_dir.stem] = test
                torch.save(videofeats, output_root / f"{key}.pt")  
            except Exception as e:
                print(f"Error saving {key}: {e}")   
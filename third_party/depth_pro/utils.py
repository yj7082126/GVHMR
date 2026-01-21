# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import pillow_heif
from PIL import ExifTags, Image, TiffTags
from pillow_heif import register_heif_opener
import kornia
import torch

register_heif_opener()
LOGGER = logging.getLogger(__name__)


def extract_exif(img_pil: Image) -> Dict[str, Any]:
    """Return exif information as a dictionary.

    Args:
    ----
        img_pil: A Pillow image.

    Returns:
    -------
        A dictionary with extracted EXIF information.

    """
    # Get full exif description from get_ifd(0x8769):
    # cf https://pillow.readthedocs.io/en/stable/releasenotes/8.2.0.html#image-getexif-exif-and-gps-ifd
    img_exif = img_pil.getexif().get_ifd(0x8769)
    exif_dict = {ExifTags.TAGS[k]: v for k, v in img_exif.items() if k in ExifTags.TAGS}

    tiff_tags = img_pil.getexif()
    tiff_dict = {
        TiffTags.TAGS_V2[k].name: v
        for k, v in tiff_tags.items()
        if k in TiffTags.TAGS_V2
    }
    return {**exif_dict, **tiff_dict}


def fpx_from_f35(width: float, height: float, f_mm: float = 50) -> float:
    """Convert a focal length given in mm (35mm film equivalent) to pixels."""
    return f_mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2)


def load_rgb(
    path: Union[Path, str], auto_rotate: bool = True, remove_alpha: bool = True
) -> Tuple[np.ndarray, List[bytes], float]:
    """Load an RGB image.

    Args:
    ----
        path: The url to the image to load.
        auto_rotate: Rotate the image based on the EXIF data, default is True.
        remove_alpha: Remove the alpha channel, default is True.

    Returns:
    -------
        img: The image loaded as a numpy array.
        icc_profile: The color profile of the image.
        f_px: The optional focal length in pixels, extracting from the exif data.

    """
    LOGGER.debug(f"Loading image {path} ...")

    path = Path(path)
    if path.suffix.lower() in [".heic"]:
        heif_file = pillow_heif.open_heif(path, convert_hdr_to_8bit=True)
        img_pil = heif_file.to_pillow()
    else:
        img_pil = Image.open(path)

    img_exif = extract_exif(img_pil)
    icc_profile = img_pil.info.get("icc_profile", None)

    # Rotate the image.
    if auto_rotate:
        exif_orientation = img_exif.get("Orientation", 1)
        if exif_orientation == 3:
            img_pil = img_pil.transpose(Image.ROTATE_180)
        elif exif_orientation == 6:
            img_pil = img_pil.transpose(Image.ROTATE_270)
        elif exif_orientation == 8:
            img_pil = img_pil.transpose(Image.ROTATE_90)
        elif exif_orientation != 1:
            LOGGER.warning(f"Ignoring image orientation {exif_orientation}.")

    img = np.array(img_pil)
    # Convert to RGB if single channel.
    if img.ndim < 3 or img.shape[2] == 1:
        img = np.dstack((img, img, img))

    if remove_alpha:
        img = img[:, :, :3]

    LOGGER.debug(f"\tHxW: {img.shape[0]}x{img.shape[1]}")

    # Extract the focal length from exif data.
    f_35mm = img_exif.get(
        "FocalLengthIn35mmFilm",
        img_exif.get(
            "FocalLenIn35mmFilm", img_exif.get("FocalLengthIn35mmFormat", None)
        ),
    )
    if f_35mm is not None and f_35mm > 0:
        LOGGER.debug(f"\tfocal length @ 35mm film: {f_35mm}mm")
        f_px = fpx_from_f35(img.shape[1], img.shape[0], f_35mm)
    else:
        f_px = None

    return img, icc_profile, f_px

def get_boundaries_mask(disparity, sobel_threshold=0.3):
    def sobel_filter(disp, mode="sobel", beta=10.0):
        sobel_grad = kornia.filters.spatial_gradient(disp, mode=mode, normalized=False)
        sobel_mag = torch.sqrt(sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2)
        alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

        return alpha

    sobel_beta = 10.0
    normalized_disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
    return sobel_filter(normalized_disparity, "sobel", beta=sobel_beta) < sobel_threshold

def get_depth_image(depth, min_depth=0.1, max_depth=250.0):
    inverse_depth = 1 / depth
    max_inv_depth = min(inverse_depth.max(), 1 / min_depth)
    min_inv_depth = max(1 / max_depth, inverse_depth.min())
    inverse_depth_norm = (inverse_depth - min_inv_depth) / (max_inv_depth - min_inv_depth)
    
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_norm)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(color_depth)

def get_points_3d(depth, K):
    height, width = depth.shape
    points2d = torch.stack(torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32), 
    indexing="xy"), -1)  # [h,w,2]
    
    points3d = torch.cat([points2d, torch.ones_like(points2d)[..., 0:1]], dim=-1).reshape(height * width, 3)  # [hw,3]
    points3d = (K.inverse() @ points3d.T * depth.reshape(1, height * width).cpu()).T
    points3d = points3d.cpu().numpy()
    points3d = np.concatenate([points3d, np.ones_like(points3d)[..., 0:1]], axis=-1)
    
    return points3d
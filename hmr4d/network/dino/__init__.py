
from hmr4d import PROJ_ROOT
from .dinov3 import DINOv3Backbone

DINOV3_REPO = PROJ_ROOT / f"inputs/checkpoints/dinov3"  # this is HMR2.0a, follow WHAM

def load_dinov3(model_type="dinov3_vith16plus"):
    assert model_type in [
        "dinov3_vit7b",
        "dinov3_vith16plus",
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
    ]
    model = DINOv3Backbone(dino_dir=str(DINOV3_REPO), name=model_type)
    return model
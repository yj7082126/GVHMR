from PIL import Image
import torch
from diffusers import (
    AutoencoderKL, 
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    FluxControlImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor
from nunchaku import NunchakuFluxTransformer2dModel

class ImageGenerator_Nunchaku:
    
    def __init__(self, 
        sd15_base = "SG161222/Realistic_Vision_V5.1_noVAE",
        sd15_vae = "stabilityai/sd-vae-ft-mse",
        sd15_openpose_cn = "lllyasviel/control_v11p_sd15_openpose",
        flux_nunchaku_model = "nunchaku-tech/nunchaku-flux.1-depth-dev/svdq-int4_r32-flux.1-depth-dev.safetensors",
        depth_model = "LiheYoung/depth-anything-large-hf",
        dtype = torch.float16,
        device = 'cuda'
    ):
        self.device = device
        vae = AutoencoderKL.from_pretrained(sd15_vae, torch_dtype=dtype,)
        controlnet_15 = ControlNetModel.from_pretrained(sd15_openpose_cn, torch_dtype=dtype)
        
        self.pipe1 = StableDiffusionControlNetPipeline.from_pretrained(
            sd15_base, vae=vae, controlnet=controlnet_15, torch_dtype=dtype,
        )
        self.pipe1.scheduler = DPMSolverMultistepScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True,
        )
        self.pipe1.enable_xformers_memory_efficient_attention()
        self.pipe1.enable_model_cpu_offload()
        
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(flux_nunchaku_model)
        self.pipe2 = FluxControlImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        
        self.midas = DepthPreprocessor.from_pretrained(depth_model)
        
    def step1(self, prompt, image, 
        width=768,
        height=480,
        batch_size=1,
        num_inference_steps=50,
        guidance_scale=6.0,
        controlnet_conditioning_scale=1.1,
        control_guidance_end=0.85,
        negative_prompt="low quality, worst quality, blurry, deformed, bad anatomy",
        generator=None,
    ):
        result = self.pipe1(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=width,
            height=height,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_end=control_guidance_end,
            generator=generator,
        )
        return result.images, {
            'prompt': prompt, 'negative_prompt': negative_prompt,
            'width': width, 'height': height, 'batch_size': batch_size,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'controlnet_conditioning_scale': controlnet_conditioning_scale,
            'control_guidance_end': control_guidance_end,
            'seed': generator.initial_seed() if generator is not None else None,
        }
    
    def step2(self, prompt, image, 
        width=1536,
        height=960,
        strength=0.75,
        num_inference_steps=50,
        guidance_scale=10.0,
        generator=None,
    ):
        depth_image = self.midas(image)[0].convert("RGB")
        image = image.resize((width, height), Image.BICUBIC)
        control_image = depth_image.resize((width, height), Image.BICUBIC)

        result = self.pipe2(
            prompt=prompt,
            image=image,
            control_image=control_image,
            width=width,
            height=height,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images, {
            'prompt': prompt, 
            'width': width, 'height': height, 'strength': strength,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'seed': generator.initial_seed() if generator is not None else None,
        }
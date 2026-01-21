from PIL import Image
import torch
from diffusers import (
    AutoencoderKL, 
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler
)
from controlnet_aux import MidasDetector

class ImageGenerator:
    
    def __init__(self, 
        sd15_base = "SG161222/Realistic_Vision_V5.1_noVAE",
        sd15_vae = "stabilityai/sd-vae-ft-mse",
        sd15_openpose_cn = "lllyasviel/control_v11p_sd15_openpose",
        sdxl_refiner_model = "RunDiffusion/Juggernaut-XL-v9",
        sdxl_depth_cn = "diffusers/controlnet-depth-sdxl-1.0",
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
        
        controlnet_xl = ControlNetModel.from_pretrained(sdxl_depth_cn, torch_dtype=dtype)
        self.pipe2 = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            sdxl_refiner_model,
            controlnet=controlnet_xl,
            torch_dtype=dtype, variant='fp16'
        )
        self.pipe2.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe2.scheduler.config)
        self.pipe2.enable_xformers_memory_efficient_attention()
        self.pipe2.enable_model_cpu_offload()
        
        self.midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
        
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
        strength=0.5,
        batch_size=1,
        num_inference_steps=50,
        guidance_scale=3.0,
        controlnet_conditioning_scale=0.75,
        control_guidance_end=0.6,
        negative_prompt="low quality, worst quality, blurry, deformed, bad anatomy",
        generator=None,
    ):
        depth_image = self.midas(image)
        image = image.resize((width, height), Image.BICUBIC)
        control_image = depth_image.resize((width, height), Image.BICUBIC)

        result = self.pipe2(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            control_image=control_image,
            width=width,
            height=height,
            strength=strength,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_end=control_guidance_end,
            generator=generator,
        )
        return result.images, {
            'prompt': prompt, 'negative_prompt': negative_prompt,
            'width': width, 'height': height, 'strength': strength,
            'batch_size': batch_size,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'controlnet_conditioning_scale': controlnet_conditioning_scale,
            'control_guidance_end': control_guidance_end,
            'seed': generator.initial_seed() if generator is not None else None,
        }
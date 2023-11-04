import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler
)

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.controlanimation_pipeline import ControlAnimationPipeline
from animatediff.utils.util import save_videos_grid
# from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
import shutil

from animatediff.utils.util import load_weights

from modules.controlresiduals_pipeline import MultiControlNetResidualsPipeline

from compel import Compel

from modules.utils import get_frames_pil_images

from diffusers import StableDiffusionImg2ImgPipeline

class ControlAnimatePipeline():
    def __init__(self, config):
        # self.args = args
        # self.inference_config = OmegaConf.load(inference_config_path)
        self.inference_config = OmegaConf.load(config.inference_config_path)
        # self.generator = generator
        # self.config  = config #OmegaConf.load(config_path)
        sample_idx = 0
        model_config = config
            
        motion_module = config.motion_module
    
        ### >>> create validation pipeline >>> ###
        tokenizer    = CLIPTokenizer.from_pretrained(model_config.pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_config.pretrained_model_path, subfolder="text_encoder")

        if model_config.vae_path == "":
            vae          = AutoencoderKL.from_pretrained(model_config.pretrained_model_path, subfolder="vae")            
        else:
            vae          = AutoencoderKL.from_single_file(model_config.vae_path)     

        unet         = UNet3DConditionModel.from_pretrained_2d(model_config.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs))

        
        self.multicontrolnetresiduals_pipeline = MultiControlNetResidualsPipeline(list(model_config.controlnets), list(config.cond_scale)) if model_config.controlnets is not None else None
        # self.multicontrolnetresiduals_overlap_pipeline = MultiControlNetResidualsPipeline(list(model_config.overlap_controlnets), list(config.overlap_cond_scale)) if model_config.overlap_controlnets is not None else None


        # if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
        # else: assert False
        schedulers = {
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "DDIMScheduler": DDIMScheduler,
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "LMSDiscreteScheduler": LMSDiscreteScheduler,
            "PNDMScheduler": PNDMScheduler,
        }

        pipeline = ControlAnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=schedulers[config.scheduler](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs)), # **OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs)
        ).to("cuda")

        self.pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = motion_module,
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to("cuda")

        # self.prompt = config.prompt
        # self.n_prompt = config.n_prompt

        self.pipeline.load_textual_inversion("models/TI/easynegative.safetensors", token="easynegative")

        self.prompt = self.pipeline.maybe_convert_prompt(config.prompt, self.pipeline.tokenizer)
        self.n_prompt = self.pipeline.maybe_convert_prompt(config.n_prompt, self.pipeline.tokenizer)

        # print(">>>>>>>>>>>>>.PROMPT AFTER MAYBE? ", self.prompt)

        # self.seed = 

        # self.prompts      = model_config.prompt
        # self.n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
        # init_image   = model_config.init_image if hasattr(model_config, 'init_image') else None

        # random_seeds = model_config.get("seed", [-1])
        # random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        # self.random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

        # torch.manual_seed(config.seed)
        # else: torch.seed()

        # device = "cuda"
        # self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        #     "Lykon/dreamshaper-8", torch_dtype=torch.float16, use_safetensors=True, cache_dir = 'cache',
        #     safety_checker = None,
        # ).to(device)

    def animate(self, input_frames, last_output_frame, config):

        torch.manual_seed(config.seed)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        width , height = input_frames[0].size

        compel_proc = Compel(tokenizer=self.pipeline.tokenizer, text_encoder=self.pipeline.text_encoder)
        prompt_embeds = compel_proc(self.prompt).to('cuda')
        negative_prompt_embeds = compel_proc(self.n_prompt).to('cuda')

        # print(">>>>>>>>>>>>>>>  EMBEDS SIZE ", prompt_embeds.shape) #8697306282698618417

        print(f"# current seed: {torch.initial_seed()}")
        # print(f"# sampling {self.prompt} ...")

        sample = self.pipeline(
            # prompt                  = self.prompt,
            # negative_prompt         = self.n_prompt,
            prompt_embeds           = prompt_embeds,
            negative_prompt_embeds  = negative_prompt_embeds,
            input_frames            = input_frames,
            num_inference_steps     = config.steps,
            strength                = config.strength,
            guidance_scale          = config.guidance_scale,
            width                   = width,
            height                  = height,
            video_length            = config.frame_count,
            generator               = self.generator,
            overlaps                = config.overlaps,
            multicontrolnetresiduals_pipeline = self.multicontrolnetresiduals_pipeline,
            epoch = config.epoch,
            output_dir = config.output_video_dir,
            save_outputs = bool(config.save_frames),
            last_output_frame = last_output_frame

        ).videos

        frames = get_frames_pil_images(sample)

        torch.cuda.empty_cache()

        # img2img = self.img2img_pipe(
        #             prompt_embeds           = prompt_embeds.repeat(len(frames),1,1),
        #             negative_prompt_embeds  = negative_prompt_embeds.repeat(len(frames),1,1),
        #             image                   = frames,
        #             num_inference_steps     = 30,
        #             strength                = 0.25,
        #             guidance_scale          = config.guidance_scale,
        # )

        # frames = img2img.images

        return frames
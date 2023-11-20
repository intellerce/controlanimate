# Adapted from AnimateDiff: https://github.com/guoyww/AnimateDiff

import torch
from compel import Compel
from omegaconf import OmegaConf
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    LCMScheduler
)
from diffusers import AutoencoderKL
from animatediff.utils.util import load_weights
from modules.utils import get_frames_pil_images
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.models.unet import UNet3DConditionModel
from modules.controlresiduals_pipeline import MultiControlNetResidualsPipeline
from animatediff.pipelines.controlanimation_pipeline import ControlAnimationPipeline #, LCMScheduler
from modules.ip_adapter import IPAdapter, IPAdapterPlus, IPAdapterFull


class ControlAnimatePipeline():
    def __init__(self, config):
        self.inference_config = OmegaConf.load(config.inference_config_path)
        model_config = config
        motion_module = config.motion_module

        self.use_lcm = bool(config.use_lcm)
    
        ### >>> create validation pipeline >>> ###
        tokenizer    = CLIPTokenizer.from_pretrained(model_config.pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(model_config.pretrained_model_path, subfolder="text_encoder")

        if model_config.vae_path == "":
            vae          = AutoencoderKL.from_pretrained(model_config.pretrained_model_path, subfolder="vae")           
        else:
            vae          = AutoencoderKL.from_single_file(model_config.vae_path)    

        if not self.use_lcm:
            unet         = UNet3DConditionModel.from_pretrained_2d(model_config.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs))
        else:
            unet         = UNet3DConditionModel.from_pretrained_2d(model_config.pretrained_lcm_model_path, subfolder="unet", use_safetensors=True, unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs))

        
        self.multicontrolnetresiduals_pipeline = MultiControlNetResidualsPipeline(list(model_config.controlnets), list(config.cond_scale), use_lcm = self.use_lcm) if model_config.controlnets is not None else None
        # self.multicontrolnetresiduals_overlap_pipeline = MultiControlNetResidualsPipeline(list(model_config.overlap_controlnets), list(config.overlap_cond_scale)) if model_config.overlap_controlnets is not None else None


        schedulers = {
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "DDIMScheduler": DDIMScheduler,
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "LMSDiscreteScheduler": LMSDiscreteScheduler,
            "PNDMScheduler": PNDMScheduler,
            "LCMScheduler": LCMScheduler
        }


        if not self.use_lcm:
            pipeline = ControlAnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=schedulers[config.scheduler](**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs)), # **OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs)
            ).to("cuda")
        else:
            pipeline = ControlAnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=None,
            ).to("cuda")


        # IP Adapter Addition
        self.use_ipadapter = bool(config.use_ipadapter)
        if self.use_ipadapter:
            image_encoder_path = "models/IP-Adapter/models/image_encoder/"
            ip_ckpt =  "models/IP-Adapter/models/ip-adapter_sd15.bin" # "models/IP-Adapter/models/ip-adapter-plus_sd15.bin" # config.ipadapter_ckpt # "models/IP-Adapter/models/ip-adapter_sd15.bin"
            main_ip_pipe = IPAdapter(pipeline, image_encoder_path, ip_ckpt, 'cuda', num_tokens=4) # IPAdapterPlus(pipeline, image_encoder_path, ip_ckpt, 'cuda', num_tokens=16)
            pipeline.ip_adapter = main_ip_pipe

            if self.multicontrolnetresiduals_pipeline is not None:
                main_ip_pipe.set_ip_adapter_4controlanimate(self.multicontrolnetresiduals_pipeline)


        if not self.use_lcm:
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
        else:
            self.pipeline = load_weights(
                pipeline,
                # motion module
                motion_module_path         = motion_module,
                motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            ).to("cuda")


        if not self.use_lcm: 
            self.pipeline.unet.half()
            self.pipeline.vae.half()
        # IP Adapter Attn Processors should not be replaced with xformer ones... # TODO
        if not self.use_ipadapter: self.pipeline.enable_xformers_memory_efficient_attention()

        if self.multicontrolnetresiduals_pipeline is not None:
            if not self.use_lcm: self.multicontrolnetresiduals_pipeline.controlnet.half()
            if not self.use_ipadapter: self.multicontrolnetresiduals_pipeline.controlnet.enable_xformers_memory_efficient_attention()

        self.pipeline.load_textual_inversion("models/TI/easynegative.safetensors", token="easynegative")

        self.prompt = self.pipeline.maybe_convert_prompt(config.prompt, self.pipeline.tokenizer)
        self.n_prompt = self.pipeline.maybe_convert_prompt(config.n_prompt, self.pipeline.tokenizer)


    def animate(self, input_frames, last_output_frames, config,
                image_prompt_embeds= None,
                uncond_image_prompt_embeds= None,
                ):

        torch.manual_seed(config.seed)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        width , height = config.width, config.height #input_frames[0].size

        compel_proc = Compel(tokenizer=self.pipeline.tokenizer, text_encoder=self.pipeline.text_encoder)
        prompt_embeds = compel_proc(self.prompt).to('cuda')
        negative_prompt_embeds = compel_proc(self.n_prompt).to('cuda')

        
        # print(f"# current seed: {torch.initial_seed()}")
        print("CONFIG: ", config)

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
            last_output_frames = last_output_frames,
            use_lcm = self.use_lcm,
            guess_mode = bool(config.guess_mode),
            ipa_scale = config.ipa_scale,
        ).videos

        frames = get_frames_pil_images(sample)

        torch.cuda.empty_cache()

        return frames
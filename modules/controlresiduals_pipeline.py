##############################################
# INTELLERCE LLC - Oct. - Nov. 2023 
# This codebase is designed and written for research, test and demo purposes only
# and is not recommended for production purposes.

# Created by: Hamed Omidvar
##############################################

import os
import cv2
import torch
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from transformers import pipeline
from diffusers import  ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from controlnet_aux import LineartDetector, LineartAnimeDetector, PidiNetDetector
from controlnet_aux import OpenposeDetector, MLSDdetector, NormalBaeDetector, HEDdetector


class MultiControlNetResidualsPipeline:
    def __init__(self, hf_controlnet_names, cond_scale, use_lcm):
        cache_dir = 'cache'

        self.controlnet_names = hf_controlnet_names

        self.controlnets = []

        for controlnet_name in hf_controlnet_names:
            self.controlnets.append(ControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.float16))

            # ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16),
            # ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),

        self.controlnet = MultiControlNetModel(self.controlnets).to('cuda')

        self.cond_scale = cond_scale

        self.use_lcm = use_lcm
        
        self.ip_adapter = None

        # self.multicontrolnet.to('cpu')

        def canny_processor(image):
            o_image = np.array(image)
            o_image = cv2.Canny(o_image, 100, 200)
            o_image = o_image[:, :, None]
            o_image = np.concatenate([o_image, o_image, o_image], axis=2)
            o_image = Image.fromarray(o_image)
            return o_image
        self.canny_processor = canny_processor
        self.mlsd_processor = MLSDdetector.from_pretrained('lllyasviel/Annotators', cache_dir = cache_dir,)
        self.openpose_processor = OpenposeDetector.from_pretrained('lllyasviel/Annotators', cache_dir = cache_dir,)
        self.hed_processor = HEDdetector.from_pretrained('lllyasviel/Annotators', cache_dir = cache_dir,)
        self.lineart_anime_processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators", cache_dir = cache_dir)
        self.lineart_processor = LineartDetector.from_pretrained("lllyasviel/Annotators", cache_dir = cache_dir)
        self.normalbae_processor =  NormalBaeDetector.from_pretrained("lllyasviel/Annotators", cache_dir = cache_dir)
        self.pidi_processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators', cache_dir = cache_dir)
        self.depth_estimate_processor = pipeline('depth-estimation', cache_dir = cache_dir)

        date_time = datetime.datetime.now()
        self.date_time = date_time.strftime("%Y%m%d_%H%M%S_%f")


    def move_to_device(self,controlnet_model, device):

        if 'mlsd' in controlnet_model:                
            self.mlsd_processor.to(device)

        elif  'openpose' in controlnet_model:
            self.openpose_processor.to(device)
            # o_image.show()

        elif 'hed' in controlnet_model:                
            self.hed_processor.to(device)

        elif 'lineart_anime' in controlnet_model:
            self.lineart_anime_processor.to(device)

        elif 'lineart' in controlnet_model:
            self.lineart_processor.to(device)

        elif 'normalbae' in controlnet_model:                
            self.normalbae_processor.to(device)
        
        elif 'softedge' in controlnet_model:

            self.pidi_processor.to(device)
        elif 'depth' in controlnet_model:
            self.depth_estimator.to(device)


    def prepare_controlnet_input_image(self, controlnet_model, image):
        if 'canny' in controlnet_model:
            o_image = self.canny_processor(image)

        elif 'mlsd' in controlnet_model: 
            w, h = image.size
            detect_resolution=min(h,w)
            image_resolution=min(h,w)                 
            o_image = self.mlsd_processor(image)

        elif  'openpose' in controlnet_model:
            # h, w = image.size
            # detect_resolution=min(h,w)
            # image_resolution=min(h,w)  
            # o_image = self.openpose_processor(image,detect_resolution= detect_resolution, image_resolution=image_resolution, hand_and_face=True)
            
            o_image = self.openpose_processor(image, hand_and_face=True)
            # o_image.show()

        elif 'hed' in controlnet_model:                
            o_image = self.hed_processor(image)

        elif 'lineart_anime' in controlnet_model:
            w, h = image.size
            detect_resolution=min(h,w)
            image_resolution=min(h,w)                     
            o_image = self.lineart_anime_processor(image, detect_resolution= detect_resolution, image_resolution=image_resolution)

        elif 'lineart' in controlnet_model:
            w, h = image.size
            detect_resolution =  min(h,w)
            image_resolution =  min(h,w)             
            o_image = self.lineart_processor(image, detect_resolution= detect_resolution, image_resolution=image_resolution)

        elif 'normalbae' in controlnet_model:                
            o_image = self.normalbae_processor(image)
        
        elif 'softedge' in controlnet_model:
            w, h = image.size
            detect_resolution= min(h,w)
            image_resolution= min(h,w)
            o_image = self.pidi_processor(image, detect_resolution= detect_resolution, image_resolution=image_resolution)

        elif 'depth' in controlnet_model:
            o_image = self.depth_estimator(image)['depth']
            o_image = np.array(o_image)
            o_image = o_image[:, :, None]
            o_image = np.concatenate([image, image, image], axis=2)
            o_image = Image.fromarray(o_image)

        else:
            raise Exception(f"ControlNet model {controlnet_model} is not supported at this time.")

        return o_image



    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        # if do_classifier_free_guidance and not guess_mode:
        #     image = torch.cat([image] * 2)

        return image


    def prepare_images(self, 
                       image,
                       width, 
                       height,
                       batch_size,
                       num_images_per_prompt,
                       device,
                       controlnet
                       ):
            images = []

            for image_ in image:
                # height, width = image_.size
                batch_size = 1
                num_images_per_prompt = 1
                device = 'cuda'
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=True,
                    guess_mode=False,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
            return image


 


    def prep_control_images(self, images,
                            control_image_processor,
                            epoch = 0,
                            output_dir = 'tmp/output',
                            save_outputs = True,
                            do_classifier_free_guidance = True,
                            guess_mode = False,
                            ):

        # print(date_time)

        # self.input_images = images

        output_dir = os.path.join(output_dir, f'controlnet_outputs_{self.date_time}')
        
        self.control_image_processor = control_image_processor
        
        

        self.controlnet.to('cuda')

        prep_images = []
        for ctrl_name in self.controlnet_names:
            out_dir = os.path.join(output_dir, ctrl_name)
            if not os.path.exists(out_dir) and save_outputs:
                os.makedirs(out_dir)

            self.move_to_device(ctrl_name, 'cuda')
            ctrl_images = None
            for i, image in tqdm(enumerate(images)):
                width, height = image.size
                prep_image = self.prepare_controlnet_input_image(ctrl_name, image)
                if save_outputs:
                    prep_image.save(os.path.join(out_dir,"{}_{:04d}.png".format(epoch,i)))
                prep_image = self.prepare_images([prep_image], width, height ,1,1,'cuda',self.controlnet)
                if ctrl_images is not None:
                    ctrl_images = torch.cat([ctrl_images, prep_image[0]])
                else:
                    ctrl_images = prep_image[0]
            
            
             
            if do_classifier_free_guidance and not guess_mode and not self.use_lcm:
                ctrl_images = torch.cat([ctrl_images] * 2)

            prep_images.append(ctrl_images)
        
        self.prep_images = prep_images

        


    def __call__(self, 
                 control_model_input,
                 t,
                 controlnet_prompt_embeds, 
                 frame_count,
                 image_embeds = None,
                 do_classifier_free_guidance = True,
                 guess_mode = True):
        
        control_model_input = rearrange(control_model_input, 'b c f h w -> (b f) c h w' )

        # IP Adapter
        # if self.ip_adapter is not None:

        controlnet_prompt_embeds = torch.cat([controlnet_prompt_embeds] * frame_count)

        down_block_res_samples_multi, mid_block_res_sample_multi = self.controlnet(
            control_model_input.half(), #[:,:,i,:,:].half(),
            t,
            encoder_hidden_states=controlnet_prompt_embeds.half(),
            controlnet_cond=self.prep_images,
            conditioning_scale=self.cond_scale,
            guess_mode=guess_mode,
            return_dict=False,
            )

        # tuples are un-assinagnable so we first convert to lists and then convert back
        down_block_additional_residuals = list(down_block_res_samples_multi) 
        
        # Re-arranging the outputs of controlnet(s) to account for the frames
        for i, tensor in enumerate(down_block_additional_residuals):
            down_block_additional_residuals[i] = rearrange(tensor, '(b f) c h w -> b c f h w', f = frame_count)

    
        mid_block_additional_residual = rearrange(mid_block_res_sample_multi, '(b f) c h w -> b c f h w', f = frame_count)

        down_block_additional_residuals = tuple(down_block_additional_residuals)

        return down_block_additional_residuals, mid_block_additional_residual
                
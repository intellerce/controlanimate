##############################################
# INTELLERCE LLC - Oct. 2023
# The FFMPEG stream ecoding/decoding is based on: https://github.com/Filarius/stable-diffusion-webui/blob/master/scripts/vid2vid.py
##############################################


import json
import os
import PIL
import cv2
import sys
import time
import torch
import zipfile
import asyncio
import argparse
import datetime
import numpy as np
from PIL import Image
from pathlib import Path
# from logger import Logger
# from libs.database import Database
from subprocess import Popen, PIPE
# from libs.utils import ImageUtils
# from libs.banned import Banned
# from libs.stablediffusion import StableDiffusion
from typing import Any, Callable, Dict, List, Optional, Union
from omegaconf import OmegaConf

# Video utils
from modules.utils import video_to_high_fps, get_fps_frame_count_width_height

# Downloader
# from basicsr.utils.download_util import load_file_from_url

from modules.controlanimate_pipeline import ControlAnimatePipeline 

# from modules.utils import get_frames_pil_images

# RAFT library is used for the Warp Fusion part of the vid2vid
# from libs.RAFT.core.raft import RAFT
# from libs.RAFT.core.utils.utils import InputPadder

# from libs.igfpgan import iGFPGAN
## 
# MODNet Requirements:

from PIL import ImageOps        

from modules.upscaler import Upscaler




####################################################################
# The following is the main function of this program

def vid2vid(
        config_path
        ):
    # 
    # user_id:str, model_name:str, input_video_path:str, output_video_dir:str, 
    # prompt: Union[str, List[str]], negative_prompt:Union[str, List[str]]=None, 
    # warp_scale:Optional[float]=0.5, fps:Optional[float]=12, fps_ffmpeg:Optional[float]=30, crf:Optional[float]=17,
    # start_time:Optional[str]="00:00:00", end_time:Optional[str]="00:01:00",
    # num_inference_steps:Optional[int]=20, guidance_scale:Optional[float]=9.5, strength:Optional[float]=0.3, 
    # num_images_per_prompt:Optional[int]=1, scale:Optional[int]=1, seed:Optional[int]=None,
    # state_queue: Optional[asyncio.Queue]=None, videojob: Optional[VideoJob]=None
    """
    (The Vid2Vid_Params object is defined in videoutils.py)
    This is the main function that performs the SD-based vid2vid conversion.
    It uses the libs.stablediffusion library for its SD based img2img conversions.
    As such it shares many of its parameters with the img2img function in this library.
    Default values, however, are updated based on the few experiments.
    The additional parameters are as follows:
    input_video_path: str : This is the path to the video file that is going to be processes.
    output_video_dir: str: This is the dire where the final processed video file will be saved.
                            Two video files will be generated: 
                            1. 'v_' + input video name -> No audio
                            2. 'av_' + + input video name -> With audio
    warp_scale: Optional[float]=0.5: This is the amount of blending of each frame with the
                                    warped version of the previous frame.
    fps:Optional[float]=12: The frame rate at which the input video will be read for diffusion.
                            One common trick is to choose a lower value for FPS and then interpolate
                            the frames to increase the FPS.
    fps_ffmpeg:Optional[float]=30: Frame rate of the output video. If larger than FPS then FFMPEG
                                    interpolation will be used to increase the frame rate.
    crf: float: Between 0 and 51. CRF indicates the level of video quality. 17 is a good value.
    start_time: Optional[str]="00:00:00": Starts the video conversion from the given time.
    end_time: Optional[str]="00:01:00": Ends the video conversion from the given time.

    Remark: In this implementation the code uses the same video size as the input video which might
            need larger VRAM if the video frames are large. It is recommended that the video files are
            preprocessed to a smalled video - perhaps in the front end / user's device.
    """

    date_time = datetime.datetime.now()
    date_time = date_time.strftime("%Y%m%d_%H%M%S_%f")
    print(date_time)

    # use_img2img = 'control' not in model_name
    
    config  = OmegaConf.load(config_path)

    save_frames = bool(config.save_frames)

    upscaler = None
    upscale = 4
    use_face_enhancer = bool(config.use_face_enhancer)

    print("USE FACE ENHANCER:", use_face_enhancer)


    ##################################################
    # Calculating the number of frames to be processed
    start_time = config.start_time.strip()
    end_time = config.end_time.strip()

    x = time.strptime(start_time,'%H:%M:%S')
    x_seconds = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
    y = time.strptime(end_time,'%H:%M:%S')
    y_seconds = datetime.timedelta(hours=y.tm_hour,minutes=y.tm_min,seconds=y.tm_sec).total_seconds()

    input_fps, input_frame_count, width, height = get_fps_frame_count_width_height(config.input_video_path)

    if config.width != 0: width = config.width
    if config.height != 0: height = config.height

    width_64 = width - width%64
    height_64 = height - height%64

    config.W = width_64
    config.H = height_64

    print("VID2VID HEIGHT, WIDTH:", config.H, config.W)


    
    input_duration = input_frame_count/input_fps

    output_duration = min(input_duration, y_seconds - x_seconds)
    intermediate_frame_count = config.fps * output_duration

    print("Frames to be processed:", intermediate_frame_count)
    ###################################################


    if start_time == "":
        start_time = "00:00:00"
    if end_time == "00:00:00":
        end_time = ""

    time_interval = (
        f"-ss {start_time}" + f" -to {end_time}" if len(end_time) else ""
    )

    # width_corrected, height_corrected = img_util.resize_width_height_2(width, height, max_length)
    # print("WIDTH HEIGH:", width, height, width_corrected, height_corrected)

    # print("CORRECTED LENGTH:", width_corrected, height_corrected)



    input_file_path = os.path.normpath(config.input_video_path.strip())
    decoder = ffmpeg(
        " ".join(
            [
                "ffmpeg" +  " -y -loglevel panic",
                f'{time_interval} -i "{input_file_path}"',
                "-vf eq=brightness=0.06:saturation=4",
                f"-s:v {width_64}x{height_64} -r {config.fps}",
                "-f image2pipe -pix_fmt rgb24",
                "-vcodec rawvideo -",
            ]
        ),
        use_stdout=True,
    )
    decoder.start()

    output_file_name = f"v_{os.path.basename(config.input_video_path).split('.')[0]}_{date_time}.mp4"

    # width_corrected2, height_corrected2 = width_corrected*scale, height_corrected*scale


    # width_scaled = min(width_64*scale,2048)
    # height_scaled = int(height_64/width_64*width_scaled)
    if not os.path.exists(config.output_video_dir):
        os.makedirs(config.output_video_dir)

    if upscale > 1 and upscaler is None:
            width_64_out = upscale * width_64
            height_64_out = upscale * height_64

    encoder = ffmpeg(
        " ".join(
            [
                "ffmpeg" +  " -y -loglevel debug",
                "-f rawvideo -pix_fmt rgb24",
                "-vcodec rawvideo",
                f"-s:v {width_64_out}x{height_64_out}",
                f"-r {config.fps}",
                # f"-filter 'minterpolate=\'fps={config.fps}\''",
                "-i - -c:v libx264 -preset fast",
                f'-crf {config.crf} "{config.output_video_dir}/{output_file_name}"',
            ]
        ),
        use_stdin=True,
    )
    encoder.start()

    
    # if len(end_time) > 0:
    #     seconds = ffmpeg.seconds(end_time) - ffmpeg.seconds(start_time)
    #     loops = seconds * int(config.fps)
    #     # state.job_count = loops
    # else:
    #     loops = None

    pull_count = width_64 * height_64 * 3

    frame_count = 1
    in_frame_count = 1
    raw_image = decoder.readout(pull_count)

    if config.seed == -1:
        config.seed = np.random.randint(1,2**16)
        print(">>>> SEED:", config.seed)

    



    # generator = torch.Generator(device="cpu").manual_seed(config.seed)
    animate_pipeline = ControlAnimatePipeline(config)
    
    overlap_frame = None

    

    config.overlap = False
    config.overlaps = 0

    overlap_length = config.overlap_length

    overlap_frames = []
    original_frame_count = config.frame_count

    secondary_inference_steps = config.steps #  int((1-1/5)*config.steps)

    epoch = 0


    ### MAIN LOOP: 
    while len(raw_image):
        pil_images_batch = []
        add_frames_count = original_frame_count
        if len(overlap_frames) > 0:
            pil_images_batch += overlap_frames
            add_frames_count -= len(overlap_frames)
            config.overlap = True
            config.overlaps = len(overlap_frames)

        # print("HELLOW!!!!!!!!!!!!!!!!!!!!!!!!!")


        for i in range(add_frames_count):
            if len(raw_image):
                pil_image = Image.fromarray(
                    np.uint8(raw_image).reshape((height_64, width_64, 3)), mode="RGB"
                    )
                pil_images_batch.append(pil_image)

            raw_image = decoder.readout(pull_count)

        print("ADDING FRAMES:", len(pil_images_batch) , len(overlap_frames))

        config.L = len(pil_images_batch)
        config.frame_count = len(pil_images_batch)

        # if len(overlap_frames) > 0: config.steps = secondary_inference_steps

        if len(overlap_frames) > 0: config.strength = config.overlap_strength

        config.epoch = epoch
        epoch+=1

        frames = animate_pipeline.animate(pil_images_batch, config)

        

        # img2img = img2img_pipe()


        # if overlap_frame:        
        #     frames[0] = Image.blend(frames[0], overlap_frame, 0.5)
        #     frames[1] = Image.blend(frames[1], frames[0], 0.5)

        frames_out_upsacled = []

        if upscale > 1 and upscaler is None:
                upscaler = Upscaler(upscale, use_face_enhancer=use_face_enhancer)

        for frame in frames[len(overlap_frames):]:
            frame_out = frame
            if upscaler is not None:
                frame_out = upscaler(frame_out)

            frames_out_upsacled.append(frame_out)


        if save_frames:
            dir = os.path.join(config.output_video_dir, f'vid2vid_frames_{date_time}')
            dir_in_frames = os.path.join(config.output_video_dir, f'vid2vid_input_frames_{date_time}')
            if not os.path.exists(dir_in_frames):
                os.makedirs(dir_in_frames)
            if not os.path.exists(dir):
                os.makedirs(dir)
                with open(os.path.join(dir, 'info.json'), 'w') as f:
                    params = OmegaConf.to_container(config, resolve=True)
                    # params = {key: value for (key, value) in config.__dict__.items() if key != 'sd' and key != 'videojob' and key != '_metadata'}
                    print("PARAMS:", params)
                    json.dump(params, f, indent=2)
            for frame in pil_images_batch[len(overlap_frames):]:
                frame.save(os.path.join(dir_in_frames,"{:04d}.png".format(in_frame_count)))
                in_frame_count+=1
            for frame in frames_out_upsacled:
                frame.save(os.path.join(dir,"{:04d}.png".format(frame_count)))
                frame_count+=1

    
        for frame in frames_out_upsacled:
            encoder.write(np.asarray(frame.convert('RGB').resize((width_64_out,height_64_out))))


        # if overlap_frames is not None:
        #     for i in range(len(overlap_frames)):
        #         frames[i] = Image.blend(frames[i], overlap_frames[i], 0.9)

        # blended_bg = Image.blend(output, init_img_w_bg, bg_blend_scale)

        if overlap_length > 0:
            overlap_frames = frames[-overlap_length:]
        # overlap_frame = frames[-1]


    encoder.write_eof()
    

    # prev_frame_raw = None
    # prev_frame = None
    
    # frame = 1
    # while raw_image is not None and len(raw_image) > 0:
    #     torch.cuda.empty_cache() 
    #     image_PIL = Image.fromarray(
    #         np.uint8(raw_image).reshape((height_corrected, width_corrected, 3)), mode="RGB"
    #     )
    #     image_PIL = img_util.resize_max_length(image_PIL, max_length).convert('RGB')

    #     print ("iamge_PIL size", image_PIL.size)

    #     state = frame/intermediate_frame_count # TODO Store me in a JOB object

    #     image_PIL_w_bg = image_PIL.copy()
    #     # init_image = image_PIL.copy()

    #     # Separating the person from the bg
    #     # _, mask_orig = infer2(modnet_network, image_PIL_w_bg)

    #     # Stable Diffusion img2img - Or any img2img conversion


    #     if save_frames:
    #         dir = f'./tmp/vid2vid_frames_{date_time}'
    #         if not os.path.exists(dir):
    #             os.makedirs(dir)
    #             with open(os.path.join(dir, 'info.json'), 'w') as f:
    #                 params = {key: value for (key, value) in vid2vid_params.__dict__.items() if key != 'sd' and key != 'videojob'}
    #                 print("PARAMS:", vid2vid_params.__dict__.items())
    #                 json.dump(params, f, indent=2)
    #         output.save(os.path.join(dir,"{:04d}.png".format(frame)))


    #     # Writing the frame in the video file
    #     encoder.write(np.asarray(output_warped.convert('RGB').resize((width_corrected2,height_corrected2))))
    #     frame += 1

    #     # Reading next frame
    #     raw_image = decoder.readout(pull_count)
    # # Making sure to close the stream
    # encoder.write_eof()

    # Waiting one second to make sure the process is complete
    # await asyncio.sleep(2)
     
    # Waiting for io processes ...
    time.sleep(10)

    # Adding audio to the final video
    output_w_audio_file_name = 'a' + output_file_name
    final_process = video_to_high_fps(output_w_audio_file_name, # Name of the final output
                    os.path.join(config.output_video_dir,output_file_name), # Video to add audio
                    input_file_path, # Input video to use its audio
                    config.output_video_dir, # Save location
                    time_interval,
                    config.fps_ffmpeg, 
                    config.crf
                    )

    return final_process

############################################################################

# The following class takes care of FFMPEG processes for decoding and encoding video files
class ffmpeg:
    def __init__(
        self,
        cmdln,
        use_stdin=False,
        use_stdout=False,
        use_stderr=False,
        print_to_console=True,
    ):
        self._process = None
        self._cmdln = cmdln
        self._stdin = None

        if use_stdin:
            self._stdin = PIPE

        self._stdout = None
        self._stderr = None

        if print_to_console:
            self._stderr = sys.stdout
            self._stdout = sys.stdout

        if use_stdout:
            self._stdout = PIPE

        if use_stderr:
            self._stderr = PIPE

        self._process = None

    def start(self):
        self._process = Popen(
            self._cmdln, stdin=self._stdin, stdout=self._stdout, stderr=self._stderr, shell=True
        )

    def readout(self, cnt=None):
        if cnt is None:
            buf = self._process.stdout.read()
        else:
            buf = self._process.stdout.read(cnt)
        arr = np.frombuffer(buf, dtype=np.uint8)

        return arr

    def readerr(self, cnt):
        buf = self._process.stderr.read(cnt)
        return np.frombuffer(buf, dtype=np.uint8)

    def write(self, arr):
        bytes = arr.tobytes()
        self._process.stdin.write(bytes)

    def write_eof(self):
        if self._stdin != None:
            self._process.stdin.close()

    def is_running(self):
        return self._process.poll() is None

    @staticmethod
    def seconds(input="00:00:00"):
        [hours, minutes, seconds] = [int(pair) for pair in input.split(":")]
        return hours * 3600 + minutes * 60 + seconds

############################################################################

if __name__ == '__main__':
    # Running some basic tests...
    vid2vid(
         config_path='configs/prompts/vid/4-RealisticVision.yaml')
    

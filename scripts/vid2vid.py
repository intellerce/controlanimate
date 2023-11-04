##############################################
# INTELLERCE LLC - Oct. - Nov. 2023 
# This codebase is designed and written for research, test and demo purposes only
# and is not recommended for production purposes.

# The FFMPEG stream ecoding/decoding was ispired from: 
# https://github.com/Filarius/video2video

# This code will work only when the repo's root is added to the PYTHONPATH.
# export PYTHONPATH=$PYTHONPATH:"./"
##############################################



import os
import json
import time
import datetime
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from modules.upscaler import Upscaler
#  from typing import Any, Callable, Dict, List, Optional, Union # TODO
from modules.controlanimate_pipeline import ControlAnimatePipeline 
from modules.utils import video_to_high_fps, get_fps_frame_count_width_height, FFMPEGProcessor


####################################################################
# The following is the main function of this program

def vid2vid(
        config_path
        ):
    """
    This function converts an input video into an output video based on the
    parameters provided in the config file.
    PARAMS:
    config_path: str -> Path to the config file.
    """

    date_time = datetime.datetime.now()
    date_time = date_time.strftime("%Y%m%d_%H%M%S_%f")
    print(date_time)

    
    config  = OmegaConf.load(config_path)

    save_frames = bool(config.save_frames)

    upscaler = None
    upscale = float(config.upscale)
    use_face_enhancer = bool(config.use_face_enhancer)


    ##################################################
    # Figuring out the number of frames to be processed
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
    
    input_duration = input_frame_count/input_fps

    output_duration = min(input_duration, y_seconds - x_seconds)
    intermediate_frame_count = config.fps * output_duration

    print("Frames to be processed:", intermediate_frame_count)
    ###################################################


    if start_time == "": start_time = "00:00:00"
    if end_time == "00:00:00": end_time = ""

    cmd_time_string = (f"-ss {start_time}" + f" -to {end_time}" if len(end_time) else "")

    input_file_path = os.path.normpath(config.input_video_path.strip())
    ffmpeg_decoder = FFMPEGProcessor(
                " ".join(
                    [
                        "ffmpeg" +  " -y -loglevel quiet",
                        f'{cmd_time_string} -i "{input_file_path}"',
                        "-vf eq=brightness=0.06:saturation=4",
                        f"-s:v {width_64}x{height_64} -r {config.fps}",
                        "-f image2pipe -pix_fmt rgb24",
                        "-vcodec rawvideo -",
                    ]
                ),
                std_out=True,
            )

    output_file_name = f"Video_{os.path.basename(config.input_video_path).split('.')[0]}_{date_time}.mp4"

    if not os.path.exists(config.output_video_dir):
        os.makedirs(config.output_video_dir)

    if upscale > 1 and upscaler is None:
            width_64_out = int(upscale * width_64)
            height_64_out = int(upscale * height_64)

    ffmpeg_encoder = FFMPEGProcessor(
                " ".join(
                    [
                        "ffmpeg" +  " -y -loglevel quiet",
                        "-f rawvideo -pix_fmt rgb24",
                        "-vcodec rawvideo",
                        f"-s:v {width_64_out}x{height_64_out}",
                        f"-r {config.fps}",
                        "-i - -c:v libx264 -preset fast",
                        f'-crf {config.crf} "{config.output_video_dir}/{output_file_name}"',
                    ]
                ),
                std_in=True,
            )

    read_byte_count = width_64 * height_64 * 3

    frame_count = 1
    in_frame_count = 1
    raw_image = ffmpeg_decoder.read(read_byte_count)

    if config.seed == -1:
        config.seed = np.random.randint(1,2**16)
        print(">>>> SEED:", config.seed)

    
    animate_pipeline = ControlAnimatePipeline(config)
    config.overlap = False
    config.overlaps = 0
    overlap_length = config.overlap_length
    overlap_frames = []
    original_frame_count = config.frame_count
    overlap_input_frames = []
    # secondary_inference_steps = config.steps #  int((1-1/5)*config.steps)
    epoch = 0
    last_output_frame = None # This frame is used to cause similarity between epochs
    ### MAIN LOOP: 
    while len(raw_image):
        pil_images_batch = []
        add_frames_count = original_frame_count
        if len(overlap_frames) > 0:
            pil_images_batch += overlap_input_frames
            add_frames_count -= len(overlap_frames)
            config.overlap = True
            config.overlaps = len(overlap_frames)

        for i in range(add_frames_count):
            if len(raw_image):
                pil_image = Image.fromarray(
                    np.uint8(raw_image).reshape((height_64, width_64, 3)), mode="RGB"
                    )
                pil_images_batch.append(pil_image)

            raw_image = ffmpeg_decoder.read(read_byte_count)

        # print("ADDING FRAMES:", len(pil_images_batch) , len(overlap_frames))

        config.L = len(pil_images_batch)
        config.frame_count = len(pil_images_batch)

        if len(overlap_frames) > 0: config.strength = config.overlap_strength

        config.epoch = epoch
        epoch+=1

        frames = animate_pipeline.animate(pil_images_batch, last_output_frame, config)

        last_output_frame = frames[-1]


        for i, frame in enumerate(overlap_frames):
            frames[i] = Image.blend(frames[i], frame, (len(overlap_frames)-i-0.5)/len(overlap_frames))

        if overlap_length > 0:
            overlap_frames = frames[-overlap_length:]
            overlap_input_frames = pil_images_batch[-overlap_length:]


        frames_out_upsacled = []
        if upscale > 1 and upscaler is None:
            upscaler = Upscaler(upscale, use_face_enhancer=use_face_enhancer, upscale_first=bool(config.upscale_first))
        for frame in frames[:(len(pil_images_batch)-len(overlap_frames))]:
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
                    json.dump(params, f, indent=2)
            for frame in pil_images_batch[:(len(pil_images_batch)-len(overlap_frames))]:
                frame.save(os.path.join(dir_in_frames,"{:04d}.png".format(in_frame_count)))
                in_frame_count+=1
            for frame in frames_out_upsacled:
                frame.save(os.path.join(dir,"{:04d}.png".format(frame_count)))
                frame_count+=1

    
        for frame in frames_out_upsacled:
            ffmpeg_encoder.write(np.asarray(frame.convert('RGB').resize((width_64_out,height_64_out))))

    ffmpeg_encoder.close()
    
    # Waiting for io processes ...
    time.sleep(5)

    # Adding audio to the final video
    output_w_audio_file_name = 'Audio' + output_file_name
    final_process = video_to_high_fps(output_w_audio_file_name, # Name of the final output
                    os.path.join(config.output_video_dir,output_file_name), # Video to add audio
                    input_file_path, # Input video to use its audio
                    config.output_video_dir, # Save location
                    cmd_time_string,
                    config.fps_ffmpeg, 
                    config.crf
                    )

    return final_process

if __name__ == '__main__':
    # Running some basic tests...
    vid2vid(
         config_path='configs/prompts/SampleConfig.yaml')
    

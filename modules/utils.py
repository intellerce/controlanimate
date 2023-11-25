##############################################
# INTELLERCE LLC - Oct. - Nov. 2023 
# This codebase is designed and written for research, test and demo purposes only
# and is not recommended for production purposes.

# Created by: Hamed Omidvar
##############################################

import os
import cv2
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from subprocess import Popen, PIPE
import torch.nn.functional as F

from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


def video_to_high_fps(output_name, video_file_path, audio_path, processed_file_save_dir, time_interval, fps_ffmpeg, crf = 17, ffmpeg_path = '/usr/bin/ffmpeg'):
    """
    The purpose of this function is increase the framerate of the input video (e.g., generated from output frames) by interpolation.
    It also adds audio to the input video.
    params:
    output_name: Name (not path) of the output file. E.g., av_s3g3hg7vc0kls.mp4
    video_file_path: Path to the video file that we want to add audio to.
    audio_path: Path to the video/audio file (e.g., input video file) whose audio should be added to the input
    processed_file_save_dir: Path to the directory where the output will be saved.
    time_interval: The interval of the audio: E.g., "-ss 00:00:00 -to 00:00:01"
    fps_ffmpeg: The framerate of the output file. If larger than input file then interpolations will be used.
    crf:  Quality index.
    """

    assert os.path.exists(video_file_path), 'Invalid video file path.'
    assert os.path.exists(audio_path), 'Invalid audio file path.'
    
    cmd = [ffmpeg_path, '-i', video_file_path,
                '-vn ' + time_interval  +  ' -i', audio_path,
                '-q:v 0',
                f'-crf {crf}',
                f'-vf "minterpolate=fps={fps_ffmpeg}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" ',
                '-shortest',
                '-c:v libx264 -preset fast',
                '-strict -2',
                '-fflags shortest',
                '-loglevel debug -y',
                '"' + os.path.join(processed_file_save_dir, output_name) + '"'
                ]
    cmd = ' '.join(cmd)
    os.system(cmd)

    return True


def get_fps_frame_count_width_height(video_file_path):
    """
    The purpose of this function is to detect the FPS, frame count and size of the frames.
    """
    assert os.path.exists(video_file_path), 'Invalid video file path.'
    video = cv2.VideoCapture(video_file_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return fps, frame_count, width, height

def get_frames_pil_images(videos: torch.Tensor, rescale=False, n_rows=6, fps=8):
    frames = rearrange(videos, "b c t h w -> (t b) c h w")
    outputs = []
    for x in frames:
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)
        outputs.append(x)
    
    return outputs

# The following class takes care of FFMPEG processes for decoding and encoding video files
class FFMPEGProcessor:
    def __init__(
            self,
            cmd,
            std_in = False,
            std_out = False
            ):
        
        self.process = Popen(
            cmd, stdin= PIPE if std_in else None,
            stdout = PIPE if std_out else None,
            shell=True
        )

    def read(self, count):
        buffer = self.process.stdout.read(count)
        out_array = np.frombuffer(buffer, dtype=np.uint8)
        return out_array
    
    def write(self, in_array):
        bytes = in_array.tobytes()
        self.process.stdin.write(bytes)

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()


def match_colors(input_frames, ref_frame):
    """
    This function matches the color of the output frames with the color of the last frame of the 
    previous batch of frames.
    """
    cm = ColorMatcher()
    img_ref = Normalizer(np.asarray(ref_frame)).type_norm()
    outputs = []
    for frame in input_frames:
        frame = Normalizer(np.asarray(frame)).type_norm()
        img_res = cm.transfer(src=frame, ref=img_ref, method='hm-mkl-hm') # 'default', 'hm', 'reinhard', 'mvgd', 'mkl', 'hm-mvgd-hm', 'hm-mkl-hm'
        img_res = Normalizer(img_res).uint8_norm()
        outputs.append(Image.fromarray(img_res))

    return outputs


if __name__ == '__main__':

    start_time = "00:00:00"
    end_time = "00:00:02"

    time_interval = (
        f"-ss {start_time}" + f" -to {end_time}" if len(end_time) else ""
    )

    final_process = video_to_high_fps('test.mp4', # Name of the final output
                'tmp/output/v_walk.mp4', # Video to add audio
                'tmp/walk.mp4', # Input video to use its audio
                'tmp/output', # Save location
                time_interval,
                60, 
                17
                )
    





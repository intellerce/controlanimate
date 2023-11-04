##############################################
# INTELLERCE LLC - Oct. - Nov. 2023 
# This codebase is designed and written for research, test and demo purposes only
# and is not recommended for production purposes.

# Created by: Hamed Omidvar
##############################################


import os
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

class Upscaler():
    def __init__(self, scale, use_face_enhancer = True, upscale_first = False):
        model_name = 'RealESRGAN_x4plus_anime_6B'  #'RealESRGAN_x4plus_anime_6B'RealESRNet_x4plus
        self.scale = scale
        self.use_face_enhancer = use_face_enhancer

        self.upscale_first = False

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']

        # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        # netscale = 4
        # file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']


        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=None)
        
        if self.use_face_enhancer:
            from gfpgan import GFPGANer
            self.face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler= None if self.upscale_first else self.upsampler) # 
                # bg_upsampler=None) # self.upsampler

    def __call__(self, pil_image):
        assert  self.scale > 1 and self.scale < 8, 'Error: Invalid scale value.'
        if self.use_face_enhancer:
            if self.scale > 1 and self.upscale_first:
                output, _ = self.upsampler.enhance(np.asarray(pil_image), outscale=self.scale)
            else:
                output = np.asarray(pil_image)
            _, _, output = self.face_enhancer.enhance(output, has_aligned=False, only_center_face=False, paste_back=True)
        elif self.scale > 1:
            output, _ = self.upsampler.enhance(np.asarray(pil_image), outscale=self.scale)

        return Image.fromarray(output)


if __name__ == '__main__':
    img = Image.open('tmp/0001.png')
    upscaler = Upscaler(4)
    upscaler(img).show()
import os
from huggingface_hub import snapshot_download, hf_hub_download



sd_path = os.path.join("models" ,"LCM_Dreamshaper_v7")
os.makedirs(sd_path, exist_ok=True)
hf_hub_download(repo_id="SimianLuo/LCM_Dreamshaper_v7", subfolder="unet", filename="diffusion_pytorch_model.safetensors", local_dir=sd_path)
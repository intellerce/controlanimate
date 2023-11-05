import os
from huggingface_hub import snapshot_download


sd_path = "models/StableDiffusion/stable-diffusion-v1-5"
os.makedirs(sd_path, exist_ok=True)

snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", 
                revision="fp16",
                local_dir=sd_path
                )
import os
from huggingface_hub import snapshot_download, hf_hub_download



sd_path = os.path.join("models" ,"DreamBooth_LoRA")
os.makedirs(sd_path, exist_ok=True)
hf_hub_download(repo_id="latent-consistency/lcm-lora-sdv1-5", filename="pytorch_lora_weights.safetensors", local_dir=sd_path)

os.rename(os.path.join(sd_path, "pytorch_lora_weights.safetensors"),os.path.join(sd_path,'lcm_lora.safetensors'))
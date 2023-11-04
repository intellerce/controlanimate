from huggingface_hub import snapshot_download

snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", 
                revision="fp16",
                local_dir="models/StableDiffusion/stable-diffusion-v1-5"
                )
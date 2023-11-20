import os
from huggingface_hub import snapshot_download


ipa_path = "models/IP-Adapter"
os.makedirs(ipa_path, exist_ok=True)

snapshot_download(repo_id="h94/IP-Adapter", 
                local_dir=ipa_path
                )
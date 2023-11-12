#!/bin/bash


# Downloading the required models
bash download_bashscripts/0-MotionModule.sh
# bash download_bashscripts/0-StableDiffusion1-5.sh
python download_bashscripts/DownloadSD.py
python download_bashscripts/DownloadLCM.py
python download_bashscripts/DownloadLCMLoRA.py
bash download_bashscripts/DreamShaper8.sh
bash download_bashscripts/VAE.sh





#!/bin/bash


# Downloading the required models
bash download_bashscripts/0-MotionModule.sh
bash download_bashscripts/0-StableDiffusion1-5.sh
bash download_bashscripts/DreamShaper8.sh
bash download_bashscripts/VAE.sh



conda env create -f env.yml


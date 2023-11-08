# ControlAnimate

- Combining AnimateDiff with Multi-ControlNet and Img2Img for Vid2Vid applications.
This small library is especially focused on Vid2Vid applications by using ControlNet (or Multi-ControlNet) to guide the video generation and AnimateDiff for concistency.
- In addition it uses Img2Img for creating more consistent videos (after the first epoch).
Similar to AnimateDiff it allows the use of DreamBooth/LoRA models in addition to the Stable Diffusion 1.5 base model. 
- This is an initial release so please expect potential issues and bugs. Feedback, suggestions, and feature requests are welcome.

![ControlAnimate](./readme/fig1_wlogo.png?raw=true "ControlAnimate")


## News

- :fire: Nov. 7, 2023 - Now supporting Latent Consistency Model (LCM) - Achieving 10X performance gain!


## Compatibility and Requirements
- This codebase was tested on Linux (Ubuntu 22.04) only.
It was tested on an Intel machine with NVIDIA Gefore RTX 3090 (24 GB VRAM) and requires at least 16 GB of RAM.

## Installation
- Make sure you have Anaconda installed (https://www.anaconda.com/download).
- Also make sure that FFMPEG is properly installed and set up (you can follow these guides for the installation: 
["Guide 1"](https://ubuntuhandbook.org/index.php/2023/03/ffmpeg-6-0-released-how-to-install-in-ubuntu-22-04-20-04/safest-way-to-install-latest-stable-ffmpeg-4-3-on-ubuntu-20-04-ppa-not-wor) and if there are still issues this: 
["Guide 2"](https://community.wolfram.com/groups/-/m/t/2188963) - You can set the FFMPEG path in the configs/prompts yaml files)

```
git clone git@github.com:intellerce/controlanimate.git
cd ControlAnimate

bash download.sh

conda env create -f env.yml
```

### Vid2Vid
- After setting the config file 'configs/prompts/SampleConfig.yaml', simply run the following (don't forget to point to a valid input video file):
```
conda activate controlanimate
bash start.sh
```
Tested on a machine with a single RTX 3090.

## Prompt Weighting
- Prompt weighting is based on Compel. You can use + or (...)+ for importance or add weights like this: (cat)1.2
Similarly you can use the negative sign (-) to reduce the weight or use weights below 1.
Please refer to https://github.com/damian0815/compel/blob/main/Reference.md for more info.

## Results
- Four ControlNets and Latent Overlapping
[![ControlAnimate](./readme/result1.jpg?raw=true)](https://youtu.be/i2YFW2JSGQU "ControlAnimate")
- LCM (No ControlNet)
[![ControlAnimate](./readme/result_lcm.jpg?raw=true)](https://youtu.be/4xAlnOzsj3o "ControlAnimate")

## Known Issues
This is an initial release so please expect some potential bugs and issues.
Currently, memory optimizations using xformers does not work properly (for the Motion Module only) and leads to some unclear errors. So it has been partially disabled in this release (for the Motion Modules).
The code was tested on Linux and will not work on Windows currently (at least the FFMPEGProcessor needs to be modified).

## Todo
- [x] GitHub Release
- [ ] Bug Fixes and Improvements
- [ ] Fixing xformers Issues and GPU Memory Optimization
- [ ] Windows Support
- [ ] Interface


## Contact Us
**Hamed Omidvar, Ph.D.**: [hamed.omidvar@intellerce.com](mailto:hamed.omidvar@intellerce.com)  
**Vahideh Akhlaghi, Ph.D.**: [vahideh.akhlaghi@intellerce.com](mailto:vahideh.akhlaghi@intellerce.com)  


## License
This codebase is released under the Apache v2.0 license. For the licenses of the codebases that this repository is based on please refer to their corresponding Github/Website pages.

## Acknowledgements
This codebase was built upon and/or inspired by the following repositories:
[AnimateDiff](https://github.com/guoyww/AnimateDiff)
[Diffusers](https://github.com/huggingface/diffusers)
[Video2Video](https://github.com/Filarius/video2video)

The authors would like to thank Kalin Ovtcharov (Extropolis Corp.) for invaluable feedback and suggestions.

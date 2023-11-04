# ControlAnimate

Combining AnimateDiff with Multi-ControlNet and Img2Img for Vid2Vid applications.
This small library is especially focused on Vid2Vid applications and allows one to use ControlNet (or Multi-ControlNet) to guide the video generation. 
In addition it uses Img2Img for creating more consistent videos (after the first epoch).
Similar to AnimateDiff it allows the use of DreamBooth/LoRA models in addition to the Stable Diffusion 1.5 base model. 
- This is an initial release so please expect potential issues and bugs. Feedback, suggestions, and feature request are welcome.

![ControlAnimate](./readme/fig1_wlogo.png?raw=true "ControlAnimate")


## Compatibility and Requirements
This codebase was tested on Linux (Ubuntu 22.04) only.
It was tested on an Intel machine with NVIDIA Gefore RTX 3090 (24 GB VRAM) and requires at least 16 GB of RAM.

## Installation
Make sure you have Anaconda installed (https://www.anaconda.com/download).

```
git clone https://github.com/intellerce/ControlAnimate.git
cd ControlAnimate

bash install.sh
```

Tested on a single RTX 3090.

### Vid2Vid
After setting the config file 'configs/prompts/SampleConfig.yaml', simply run the following:
```
bash start.sh
```


## Prompt Weighting
Prompt weighting is based on Compel. You can use + or (...)+ for importance or add weights like this: (cat)1.2
Similarly you can use the negative sign (-) to reduce the weight or use weights below 1.
Please refer to ??? for more info.

## A Few Results
...


## Known Issues
This is an initial release so please expect some potential bugs and issues.
Currently memory optimization using xformers does not work properly and leads to some unclear errors so it is disabled in this release.
The code was tested on Linux and will not work on Windows currently (at least the FFMPEGProcessor needs to be updated).

## Todo
- [x] GitHub Release
- [ ] Windows Support
- [ ] Fixing xformers Issues and GPU Memory Optimization
- [ ] Interface


## Contact Us
**Hamed Omidvar, Ph.D.**: [hamed.omidvar@intellerce.com](mailto:hamed.omidvar@intellerce.com)  


## License
This codebase is released under the Apache v2.0 license. For the licenses of the codebases that this repository is based on please refer to their corresponding Github/Website pages.

## Acknowledgements
This codebase was built upon the following repositories:
[AnimateDiff](https://github.com/guoyww/AnimateDiff)
[Diffusers](https://github.com/huggingface/diffusers)
[Video2Video](https://github.com/Filarius/video2video)

The author would like to thank Kalin Ovtcharov (Extropolis Corp.) for suggestions and invaluable feedback.

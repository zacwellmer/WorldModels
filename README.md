# World Models
This repo reproduces the [original implementation](https://github.com/hardmaru/WorldModelsExperiments) of [World Models](https://arxiv.org/abs/1803.10122). This implementation uses TensorFlow 2.2.

## Docker
The easiest way to handle dependencies is with [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker). Follow the instructions below to generate and attach to the container.
```
docker image build -t wm:1.0 -f docker/Dockerfile.wm .
docker container run -p 8888:8888 --gpus '"device=0"' --detach -it --name wm wm:1.0
docker attach wm
```

## Visualizations
To visualize the environment from the agents perspective or generate synthetic observations use the [visualizations jupyter notebook](WorldModels/visualizations.ipynb). It can be launched from your container with the following:
```
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 --allow-root
```

Real Frame Sample             |  Reconstructed Real Frame  |  Imagined Frame
:-------------------------:|:-------------------------:|:-------------------------:|
![alt-text-1](imgs/true_frame.png "Real Frame")| ![alt-text-2](imgs/reconstructed_frame.png "Reconstructed Frame") | ![alt-text-3](imgs/imagined.png "Imagined Frame")

Ground Truth (CarRacing)             |  Reconstructed
:-------------------------:|:-------------------------:
<img src="imgs/true_traj.gif" alt="drawing" width="500"/> | <img src="imgs/reconstruct_traj.gif" alt="drawing" width="500"/>

Ground Truth Environment (DoomTakeCover)   |  Dream Environment
:-------------------------:|:-------------------------:
<img src="imgs/doom_real_traj.gif" alt="drawing" width="500"/> | <img src="imgs/doom_dream_traj.gif" alt="drawing" width="500"/>

## Reproducing Results From Scratch
These instructions assume a machine with a 64 core cpu and a gpu. If running in the cloud it will likely financially make more sense to run the extraction and controller processes on a cpu machine and the VAE, preprocessing, and RNN tasks on a GPU machine.

### DoomTakeCover-v0
**CAUTION** The doom environment leaves some processes hanging around. In addition to running the doom experiments, the script kills processes including 'vizdoom' in the name (be careful with this if you are not running in a container).
To reproduce results for DoomTakeCover-v0 run the following bash script.
```
bash launch_scripts/wm_doom.bash
```

### CarRacing-v0
To reproduce results for CarRacing-v0 run the following bash script
```
bash launch_scripts/carracing.bash
```

## Disclaimer
I have not run this for long enough(~45 days wall clock time) to verify that we produce the same results on CarRacing-v0 as the original implementation.

Average return curves comparing the original implementation and ours. The shaded area represents a standard deviation above and below the mean. 

![alt text](imgs/og_carracing_comparison.png "CarRacing-v0 comparison")

For simplicity, the Doom experiment implementation is slightly different than the original
* We do not use weighted cross entropy loss for done predictions 
* We train the RNN with sequences that always begin at the start of an episode (as opposed to random subsequences)
* We sample whether the agent dies (as opposed to a deterministic cut-off)

|  |\tau | Returns Dream Environment  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Returns Actual Environment  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
|------|------|------|------|
|   D. Ha Original  | 1.0 | 1145 +/- 690 | 868 +/- 511 |
|   Eager  |  1.0 | 1465 +/- 633 | 849 +/- 499 |

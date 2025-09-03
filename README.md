# End-to-End Pipelines for Scalable 3D Motion Mining in Dance Archives from Monocular Footage
[![Website](https://www.couleur.org/PREMIERE/JMTA/website.svg)](https://www.couleur.org/PREMIERE/JMTA/)

## 📖 Overview

## 🌍 Installation
------------

```py
conda create -n premiere python=3.11
conda activate premiere
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::ffmpeg
pip install -r requirement.txt
```

Tested on Linux Ubuntu 22.04, 24.04 and Windows 11.

Works with a NVIDIA GPU (ampere architecture or higher) with at least 16GB of VRAM.

## Model Checkpoints
------------

```
wget https://www.couleur.org/premiere-files/models.zip
unzip to MODELS_DIR
```
then add the following 
```
export MODELS_PATH="MODELS_DIR"
```

## Meta-scripts
------------

See [premiereFullVideoProcessing.md](premiereFullVideoProcessing.md) for the full [PREMIERE-Pipeline.oy](premiereFullVideoProcessing.py) script documentation.

See [multiPersonProcessing.md](multiPersonProcessing.md) for the [MultiPerson-Pipeline.py](multiPersonProcessing.py) script documentation.


## Simple Usage
------------


```py
python premiereFullVideoProcessing.py --video ../videos/D0-21.mp4 --directory ../results/D0-21
```

```py
python multiPersonProcessing.py --video ../videos/D0-21.mp4 --directory ../results/D0-21
```


## Funding
------------

```
This work was supported by the HORIZON-CL2-2021-HERITAGE-000201-04 under Grant number 101061303 - PREMIERE
```

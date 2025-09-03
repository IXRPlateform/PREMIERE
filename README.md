# PREMIERE Processing Graph
======

Installation
------------

```py
conda create -n premiere python=3.11
conda activate premiere
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::ffmpeg
pip install -r requirement.txt
```

Models
------------

```
wget https://www.couleur.org/premiere-files/models.zip
unzip to MODELS_DIR
then
export MODELS_PATH="MODELS_DIR"
```


Usage
------------

Simple processing

```py
python premiereFullVideoProcessing.py --video ../videos/D0-21.mp4 --directory ../results/D0-21
```


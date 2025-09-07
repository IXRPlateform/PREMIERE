# End-to-End Pipelines for Scalable 3D Motion Mining in Dance Archives from Monocular Footage
[![Website](https://www.couleur.org/PREMIERE/JMTA/website.svg)](https://www.couleur.org/PREMIERE/JMTA/)


## 📖 Overview

Digital dance and theater archives are difficult to search and study because few tools can accurately extract 3D motion from the monocular, often degraded footage that dominates heritage collections. We present two cloud-ready pipelines dedicated to the [PREMIERE Horizon project](https://premiere-project.eu/) that transform these videos into temporally dense SMPL-X reconstructions, per-frame segmentation masks, depth maps, and estimated camera trajectories.


<div style="display: flex; gap: 16px; align-items: flex-start;">
	<img src="MultiPerson-Pipeline.svg" alt="MultiPerson Pipeline Overview" style="height:220px;">
	<img src="PREMIERE-Pipeline.svg" alt="PREMIERE-Pipeline Overview" style="height:220px;">
</div>
<div align="center" style="margin-top: 8px; font-size: 15px; color: #555;">
<em>Left – MultiPerson Pipeline. Right – PREMIERE Pipeline.</em>
</div>


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
```
unzip to MODELS_DIR
then add the following environement variable:
Linux:
```
export MODELS_PATH="MODELS_DIR"
```
Windows:
```
set MODELS_PATH="MODELS_DIR"
```

## Meta-scripts
------------

See [premiereFullVideoProcessing.md](premiereFullVideoProcessing.md) for the full [PREMIERE-Pipeline](premiereFullVideoProcessing.py) script documentation.

See [multiPersonProcessing.md](multiPersonProcessing.md) for the [MultiPerson-Pipeline](multiPersonProcessing.py) script documentation.


## Simple Usage
------------

```py
python premiereFullVideoProcessing.py --video inputVideoFile.mp4 --directory outputDirectory
```
to process a full video with the PREMIERE-Pipeline.

or

```py
python multiPersonProcessing.py --video inputVideoFile.mp4 --directory outputDirectory
```
to process a video sequence with the MultiPerson-Pipeline. 

## Visualization
------------

You can visualize the final outputs of theses two pipelines with our visualization script [appVisualization.py](appVisualization.py) based on Flask and [Three.js](https://threejs.org).

The raw cleaned data (`nlf-final.pkl`):
```py
python appVisualization.py outputDirectory/nlf-final.pkl
```
The final filtered data (`nlf-final-filtered.pkl`):
```py
python appVisualization.py outputDirectory/nlf-final-filtered.pkl
```


## Funding
------------

This work was supported by the HORIZON-CL2-2021-HERITAGE-000201-04 under Grant number 101061303 - PREMIERE


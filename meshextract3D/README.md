# PREMIERE 3D mesh extraction
======

## Installation
------------
The environment and required packages can be installed **with or without CUDA dependency**.  
It is **recommended** to install without CUDA dependency first, and ensure everything is installed and all the scripts runs smoothly. The installation assumes you have already conda (environment package manager) installed in your system. 

Without cuda dependency (Recommended)
```py
conda create -n extract3D python=3.11
conda activate extract3D
conda install pytorch==2.3.1 
pip install -r requirement.txt
```

With cuda dependency (Optional, Intented for using GPU for fast processing)
```py
conda create -n extract3D python=3.11
conda activate extract3D
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirement.txt
```


------------

SMPLX Model Download
------------
Download the  zipped model folder for smplx named `Model_smplx.zip` using the following URL in any browser: https://www.couleur.org/premiere-data/

After downloading, unzip this file. Then export the model directory path in the terminal:

In linux:
```bash
export MODELS_PATH="path/to/Model_smplx"
```

In windows:
```bash
$env:MODELS_PATH="path\to\Model_smplx"
```
## Usage

### Extract 3D Files  (**GLB**/**OBJ**) for specific Scenes  

This script extracts 3D mesh files in **GLB**/**OBJ** format from specific scenes (single or multiple). The saved meshes will be inside the  folder named `body`.

 
To extract the 3D mesh for specific scenes, provide the scene numbers as arguments. Ensure there is a space between each scene number for multiple scene 3D mesh extraction.  Additionally, you can provide an argument to extract glb or obj which is described in example below.

For example, the following command extracts the  3D mesh in **GLB** format for **scene 5 and scene 10**:
```py
python extract3Dmesh.py --input_pkl nlf-final-filtered-floorc.pkl --input_dir D0/annotationdata/scenepkl/ --output_3dformat glb --scene_number 5 10
```

For example, the following command extracts the  3D mesh in **OBJ** format for **scene 5 and scene 10**:
```py
python extract3Dmesh.py --input_pkl nlf-final-filtered-floorc.pkl --input_dir D0/annotationdata/scenepkl/ --output_3dformat obj --resolution_thresh 0.5 --scene_number 5 10
```

#### Obj Resolution
`--resolution_thresh`: This argument is used to control the amount of mesh decimation for the .obj files. For example a value of 0.5 indicates only 50% of triangle faces are retained (50% are removed), a value of 0.1 means 10% of faces are retained (90% are removed). The size of the obj files also decreases in the proportional way when this resolution is decreased.  By default (when the argument is not provided) the this value would be 1.0 and no decimation would occur.
### Extracting 3D Files for all  Scenes 
To extract 3D mesh for all the scene, do not provide the argument `--scene_number`. This will extract 3D meshes for all the scenes automatically. For example:

```py
python extract3Dmesh.py --input_pkl nlf-final-filtered-floorc.pkl --input_dir D0/annotationdata/scenepkl/ --output_3dformat obj --resolution_thresh 0.5
```

Arguments description:

`--input_pkl`:The name of the PKL file used to extract the 3D mesh. You can check the PKL file name inside a scene folder within the directory: For example: <video_name>/annotationdata/scenepkl/scene_<scene_number>/


`--input_dir`: It is the directory path where the pkl files of the scenes are present 



`--output_3dformat`: It is the argument to choose the type of mesh file. For extracting in glb use  `glb` as the keyword in this argument or for obj use  `obj` as the keyword in this argument as shown in the example above. If this `--output_3dformat` argument is not mentioned then it extracts in glb by default
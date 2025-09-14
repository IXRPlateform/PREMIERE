# Model Setup Instructions

Create a `models` folder in your project directory.  
Download each model weight as described below, rename the files if necessary, and place them into the specified structure.

---

## List of Models to Download

### 1. VGGT
- **Model Link:** [VGGT-1B](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt)  
- **Save as:**  
  ```
  models/vggt/model.pt
  ```


---

### 2. NLF
- **Model Link:** [nlf_l](https://bit.ly/nlf_l_pt)  
- **Save as:**  
  ```
  models/nlf/nlf_l.pt
  ```


---

### 3. Moge
- **Model Link:** [moge-vitl](https://huggingface.co/Ruicheng/moge-vitl/tree/main)  
- **Save the model name as:**  
  ```
  models/moge/moge-large.pt
  ```


---

### 4. SMPL-X
- **Register here:** [SMPL-X Registration](https://smpl-x.is.tue.mpg.de/register.php)  
- **Download page:** [SMPL-X Downloads](https://smpl-x.is.tue.mpg.de/download.php)  
- **Download:** *SMPL-X model V1.1 (NPZ + PKL)*  
- **Instructions:**  
- Unzip `models_smplx_v1_1.zip`.  
- Copy the following files into the target folder:  
  ```
  models/smplx/SMPLX_FEMALE.npz
  models/smplx/SMPLX_MALE.npz
  models/smplx/SMPLX_NEUTRAL.npz
  ```

---

### 5. SMPL
- **Download page:** [SMPL Downloads](https://smpl.is.tue.mpg.de/download.php)  
- **Download:** *Version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)*  
- **Instructions:**  
- The file will be downloaded as `SMPL_python_v.1.1.0.zip`.  
- Unzip and copy the following file as:  
  ```
  models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
  ```

---

### 6. SAM2
- **Repo page:** [SAM2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#download-checkpoints)  
- **Download:** `sam2.1_hiera_large.pt`  
- **Save as:**  
  ```
  models/sam2/sam2.1_hiera_large.pt
  ```

---

### 7. WiLoR
- **Weights:** [wilor_final.ckpt](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt)  
- **Config file:** [model_config.yaml](https://github.com/rolpotamias/WiLoR/blob/main/pretrained_models/model_config.yaml)  
- **Save as:**  
 ```
  models/wilor/wilor_final.ckpt
  models/wilor/model_config.yaml
  ```

---



### 8. MANO
- **Register here:** [MANO Registration](https://mano.is.tue.mpg.de/)  
- **Download page:** [MANO Downloads](https://mano.is.tue.mpg.de/download.php)  
- **Download:** `Models and Code` (format: `mano_v*_*.zip`)  
- **Instructions:**  
- Unzip the package.  
- Copy the following files as:  
  ```
  models/mano/MANO_LEFT.pkl
  models/mano/MANO_RIGHT.pkl
  ```

---

## Final Folder Structure

```plaintext
models/
├── vggt/
│   └── model.pt
├── moge/
│   └── moge-large.pt
├── smplx/
│   ├── SMPLX_FEMALE.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_NEUTRAL.npz
├── smpl/
│   └── basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
├── sam2/
│   └── sam2.1_hiera_large.pt
├── wilor/
│   ├── wilor_final.ckpt
│   └── model_config.yaml
├── mano/
│   ├── MANO_LEFT.pkl
│   └── MANO_RIGHT.pkl
└── nlf/   (*ToDo*)

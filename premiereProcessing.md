# Processing Pipeline Documentation

![PREMIERE-Pipeline Overview](PREMIERE-Pipeline.svg)

## 1. Overview

This script orchestrates a multi-step pipeline for analyzing videos. It includes VGGT and MoGe analysis, 3D pose estimation, segmentation, optional shadow removal, tracking, outlier filtering, and camera compensation. Each step is controlled via command-line arguments, allowing for flexible execution.

---

## 2. Command-Line Arguments

| Argument                  | Type    | Default         | Description                                                                                                                  |
|---------------------------|---------|-----------------|------------------------------------------------------------------------------------------------------------------------------|
| `--directory`             | `str`   | `None`          | Directory for storing output files. **Required**.                                                                            |
| `--video`                 | `str`   | `None`          | Path to the video file to process. **Required**.                                                                             |
| `--fov`                   | `float` | `0`             | Field of view override for 3D pose estimation. Defaults to auto-estimation if `0`.                                           |
| `--rbfkernel`             | `str`   | `"linear"`      | RBF kernel for final filtering. Choices: `["linear", "multiquadric"]`.                                                       |
| `--rbfsmooth`             | `float` | `-1`            | Smoothness parameter for RBF filtering. Defaults computed if `< 0`.                                                          |
| `--rbfepsilon`            | `float` | `-1`            | Epsilon parameter for RBF filtering. Defaults computed if `< 0`.                                                             |
| `--step`                  | `int`   | `0`             | Start at a specific step (0 = all steps).                                                                                    |
| `--batchsize`             | `int`   | `25`            | Batch size for 3D pose estimation.                                                                                           |
| `--displaymode`           | flag    | `False`         | Enables display mode during processing.                                                                                      |
| `--handestimation`        | flag    | `False`         | Enables hand pose estimation based on Wilor if this flag is set.                                                             |
| `--removeshadows`         | flag    | `False`         | Enables shadow removal steps.                                                                                                |
| `--removeshadowsthreshold`| `float` | `0.00015`       | Threshold for shadow removal.                                                                                               |
| `--detectionthreshold`    | `float` | `0.3`           | Human detection threshold.                                                                                                  |
| `--computedepth`          | flag    | `False`         | Compute the depth data if this flag is set.                                                                                 |
| `--framesampling`         | `float` | `0.5`           | Frame sampling in seconds for the MoGe and MAST3r analysis.                                                                 |
| `--scalefactor`           | `float` | `0`             | Scale factor for camera compensation.                                                                                        |
| `--camtoltranslation`     | `float` | `0.2`           | Camera translation tolerance (in meters) for detecting if the camera is moving.                                              |
| `--camtolrotation`        | `float` | `2`             | Camera rotation tolerance (in degrees) for detecting if the camera is rotating.                                              |
| `--camtolfov`             | `float` | `3`             | Camera FOV tolerance (in degrees) for detecting if the camera FOV is changing.                                               |

---

## 3. Pipeline Steps

### Step 0: VGGT Analysis
- **Condition**: Executes if `--step <= 0`.
- **Scripts**: 
  - `videoCameraPosesAnalysisVGGT.py`
  - `interpolateCameraPosesVGGT.py`
- **Outputs**:
  - `vggt.pkl`, `vggt.glb`, `vggt-interpolated.pkl`
- **Purpose**: Analyzes camera poses and interpolates them.

### Extract Data from VGGT Analysis
- **Always executed after Step 0 (if not skipped)**.
- **Purpose**: Extracts camera information (FOV, fixed/dynamic status).

### Step 1: MoGe Analysis
- **Condition**: Executes if `--step <= 1`.
- **Script**: `floorAnalysisMoge.py`
- **Outputs**:
  - `moge.pkl`
- **Purpose**: Analyzes floor and depth information.

### Step 2: 3D Pose Estimation (NLF)
- **Condition**: Executes if `--step <= 2`.
- **Script**: Custom command based on dynamic/static FOV
- **Outputs**:
  - `nlf.pkl`: Raw 3D pose estimates.
- **Purpose**: Detects humans and estimates 3D poses.

### Extract Human Statistics
- **Always executed after Step 2 (if not skipped)**.
- **Purpose**: Calculates min, max, and majority human counts.

### Step 3: Cleaning Poses & Injecting Camera Properties
- **Condition**: Executes if `--step <= 3`.
- **Scripts**:
  - `cleanFramesPkl.py`
  - `computeCameraProperties.py`
  - `injectCameraPropertiesPkl.py`
- **Outputs**:
  - `nlf-clean.pkl`: Cleaned poses.
  - `video.json`: Camera properties.

### Step 4: SAM2.1 Humans Segmentation
- **Condition**: Executes if `--step <= 4`.
- **Script**: `sam21NLF.py`
- **Outputs**:
  - `nlf-videoSegmentation.mp4`: Human segmentation video.

### Additional Steps for Shadow Processing (Optional)
- **Condition**: Executes if `--step` is appropriate AND `--removeshadows = True`.
- **Scripts**:
  - Steps for adding homogeneity and removing shadows
- **Outputs**:
  - Shadow-processed PKL files

### Step 8: Byte Track Tracking
- **Condition**: Executes if `--step <= 8`.
- **Script**: `trackingBTPkl.py`
- **Outputs**:
  - Tracking data integrated into PKL files

### Hand Estimation (Optional)
- **Condition**: Executes if `--handestimation = True`.
- **Script**: Wilor-based hand estimation
- **Outputs**:
  - Hand pose data added to PKL files

### Step 10: Remove Outlier in PKL
- **Condition**: Executes if `--step <= 10`.
- **Script**: Outlier removal algorithms
- **Purpose**: Cleans the pose data by removing anomalies

### Step 11: Radial Basis Function Interpolation and Filtering
- **Condition**: Executes if `--step <= 11`.
- **Script**: RBF filtering implementation
- **Purpose**: Smooths and interpolates pose data

### Step 12: Copy Final PKL Files
- **Condition**: Executes if `--step <= 12`.
- **Purpose**: Creates final output files
- **Outputs**:
  - `nlf-final.pkl`
  - `nlf-final-filtered.pkl`

### Step 13: Camera Compensation
- **Condition**: Executes if `--step <= 13`.
- **Purpose**: Adjusts poses based on camera movement
- **Outputs**:
  - `nlf-final-camerac.pkl`
  - `nlf-final-filtered-camerac.pkl`
  - `nlf-final-floorc.pkl`
  - `nlf-final-filtered-floorc.pkl`

---

## 4. Execution Flow

1. Start with VGGT analysis (`--step=0`) to determine camera properties.
2. Continue with MoGe analysis (`--step=1`) for floor and depth information.
3. Perform NLF 3D pose estimation (`--step=2`).
4. Clean and process the data through various stages:
   - Shadow removal (if `--removeshadows` is enabled)
   - Tracking and depth processing
   - Hand estimation (if `--handestimation` is enabled)
5. Final steps include outlier removal, RBF filtering, and camera compensation.

---
## 5. Summary

1. **Argument Parsing**: The script configures the pipeline through comprehensive command-line options, allowing control over FOV detection, camera tolerances, shadow removal, and more.
   
2. **Initial Camera and Scene Analysis**:
   - **VGGT Analysis**: Determines camera poses and movement patterns.
   - **MoGe Analysis**: Extracts floor information and depth data.
   - **3D Pose Estimation**: Uses NLF model for human pose detection.

3. **Data Enhancement and Cleaning**:
   - **Segmentation**: Identifies humans using SAM2.1.
   - **Shadow Processing**: Optional removal of shadow-based false detections.
   - **Tracking**: Maintains consistent human IDs across frames.

4. **Advanced Features**:
   - **Hand Estimation**: Optional enhancement for detailed hand poses.
   - **Outlier Removal**: Filters anomalous pose data.
   - **RBF Filtering**: Smooths and interpolates for more natural motion.
   - **Camera Compensation**: Adjusts poses based on camera movement.

5. **Final Outputs**:
   - Multiple PKL files representing different processing stages and compensation levels.
   - Both raw cleaned data (`nlf-final.pkl`) and filtered versions (`nlf-final-filtered.pkl`).
   - Camera-compensated versions for advanced applications.

The pipeline represents a sophisticated approach to video processing for motion analysis, with options for various levels of detail and quality in the final outputs. Its modular design allows for both complete processing and targeted analysis of specific aspects of human motion.

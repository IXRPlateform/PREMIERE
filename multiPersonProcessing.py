"""
Multi-person 3D pose estimation and tracking pipeline.

This script processes a video to extract 3D human poses, track people over time,
and refine the results. It performs:
- Camera parameter estimation (using VGGT)
- Floor plane detection (using MoGe)
- 3D pose estimation (using NLF)
- Tracking and segmentation
- Temporal filtering and outlier removal
- Optional hand pose estimation
- Camera motion compensation
"""

# Standard library imports
import os
import sys
import shutil
import pickle
import json
import time
import subprocess
from argparse import ArgumentParser

# Third-party imports
import cv2
import numpy as np
import colorama

# Local imports
from premiere.functionsCommon import loadPkl2
from premiere.functionsVGGT import estimanteVGGTScaleFactor

from colorama import init, Fore, Style
init(autoreset=True)

# Record the start time
start_time = time.time()

def getMajorityValue(humansPerFrames):
    """
    Determine the most common value in an array.
    
    Args:
        humansPerFrames: NumPy array containing values to analyze
        
    Returns:
        The most frequent value in the array, or None if array is empty
    """
    if humansPerFrames.size == 0:
        return None
    values, counts = np.unique(humansPerFrames.flatten(), return_counts=True)
    majority_value = values[np.argmax(counts)]
    return majority_value


def getHumanNumber(dataPKL):
    """
    Analyze the number of humans detected across video frames.
    
    Args:
        dataPKL: Dictionary containing detection data with 'allFrameHumans' key
        
    Returns:
        tuple: (most_common_number, maximum_number, minimum_number)
    """
    humansPerFrames = np.empty([len(dataPKL['allFrameHumans']), 1], dtype=int)
    for i in range(len(dataPKL['allFrameHumans'])):
        humansPerFrames[i] = len(dataPKL['allFrameHumans'][i])
    
    humanNumber = getMajorityValue(humansPerFrames)
    maxNbHumans = max(humansPerFrames)
    minNbHumans = min(humansPerFrames)
    
    return humanNumber, maxNbHumans, minNbHumans


if __name__ == "__main__":
    # ================================================================
    # Parse command line arguments
    # ================================================================
    parser = ArgumentParser()

    parser.add_argument("--directory", type=str, default=None, 
                        help="Directory to store the processed files")
    parser.add_argument("--video", type=str, default=None, 
                        help="Video file to process")
    parser.add_argument("--fov", type=float, default=0, 
                        help="Field of view for the 3D pose estimation")
    parser.add_argument("--camtoltranslation", type=float, default=.2, 
                        help="Camera translation tolerance (in meter) for detecting camera movement")
    parser.add_argument("--camtolrotation", type=float, default=2, 
                        help="Camera rotation tolerance (in degree) for detecting camera rotation")
    parser.add_argument("--camtolfov", type=float, default=3, 
                        help="Camera fov tolerance (in degree) for detecting fov changes")
    parser.add_argument("--framesampling", type=float, default=.5, 
                        help="Frame sampling in second for the MoGe and MAST3r analysis")
    parser.add_argument("--rbfkernel", type=str, default="linear", 
                        choices=["linear", "multiquadric", "univariatespline"], 
                        help="RBF kernel to use for the 3D pose estimation filtering")
    parser.add_argument("--rbfsmooth", type=float, default=-1, 
                        help="Smoothness for the RBF kernel")
    parser.add_argument("--rbfepsilon", type=float, default=-1, 
                        help="Epsilon for the RBF kernel")
    parser.add_argument("--step", type=int, default=0, 
                        help="Step to process (default: 0 for all steps)")
    parser.add_argument("--batchsize", type=int, default=25, 
                        help="Batch size for the nlf 3D pose estimation")
    parser.add_argument("--displaymode", action="store_true", 
                        help="Display mode activated if this flag is set")
    parser.add_argument("--handestimation", action="store_true", 
                        help="Inject hand estimation based on Wilor if this flag is set")
    parser.add_argument("--detectionthreshold", type=float, default=0.3,
                        help="Threshold for detecting humans")
    parser.add_argument("--dispersionthreshold", type=float, default=.1, 
                        help="Threshold for human dispersion in segmentation/tracking")
    parser.add_argument("--copyvideo", action="store_true", 
                        help="Copy the video to output directory if set")
    parser.add_argument("--scalefactor", type=float, default=0, 
                        help="Scale factor for the camera compensation (0 = auto)")
    parser.add_argument("--webdirectory", type=str, default=None, 
                        help="Directory to store the processed files for static website visualization")
    parser.add_argument("--webtarget", type=str, default="nlf-final-filtered-camerac", 
                        choices=["nlf", "nlf-clean", "nlf-final", "nlf-final-filtered", "nlf-final-floorc", "nlf-final-filtered-floorc", "nlf-final-camerac", "nlf-final-filtered-camerac"], 
                        help="Target for the web directory")

    print("\n############################################################")
    print("# Arguments")
    print("############################################################")
    args = parser.parse_args()
    
    # Print all arguments for logging purposes
    print("Directory:", args.directory)
    print("Video:", args.video)
    print("Fov:", args.fov)
    print("Framesampling:", args.framesampling)
    print("Rbfsmooth:", args.rbfsmooth)
    print("Rbfepsilon:", args.rbfepsilon)
    print("Rbfkernel:", args.rbfkernel)
    print("Displaymode:", args.displaymode)
    print("Step:", args.step)
    print("Dispersionthreshold:", args.dispersionthreshold)
    print("Detectionthreshold:", args.detectionthreshold)
    print("Handestimation:", args.handestimation)
    print("Batchsize:", args.batchsize)
    print("Copyvideo:", args.copyvideo)
    print("Camtoltranslation:", args.camtoltranslation)
    print("Camtolrotation:", args.camtolrotation)
    print("Camtolfov:", args.camtolfov)
    print("Scalefactor:", args.scalefactor)
    if (args.webdirectory is not None):
        print("Web directory:", args.webdirectory)
        print("Web target:", args.webtarget)

    # ================================================================
    # Validate and prepare input parameters
    # ================================================================
    videoFileName = "\"" + args.video + "\""  # Add quotes for shell commands
    
    # Convert boolean flags to integers for downstream scripts
    displayMode = 1 if args.displaymode else 0
    
    # Extract parameters for readability
    handEstimation = args.handestimation 
    rbfkernel = args.rbfkernel
    rbfsmooth = args.rbfsmooth
    rbfepsilon = args.rbfepsilon
    fov = args.fov
    dispersionthreshold = args.dispersionthreshold
    detectionThreshold = args.detectionthreshold
    type = "nlf"  # Method for 3D pose estimation
    copyVideo = args.copyvideo
    frameSampling = args.framesampling
    camtoltranslation = args.camtoltranslation
    camtolrotation = args.camtolrotation
    camtolfov = args.camtolfov
    scaleFactor = args.scalefactor

    # Check required arguments
    if args.directory is None:
        print("Please provide a directory")
        sys.exit(1)
    if args.video is None:
        print("Please provide a video")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        print(f"Created directory: {args.directory}")
    
    # ================================================================
    # Analyze video properties
    # ================================================================
    print("\n############################################################")
    print("# Video information")
    print("############################################################")
    video = cv2.VideoCapture(args.video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video width:", width)
    print("Video height:", height)
    print("Video fps:", fps)
    print("Video frames count:", frames_count)
    video.release()
    
    # Copy the video file if requested
    if copyVideo:
        shutil.copyfile(args.video, os.path.join(args.directory, "video.mp4"))

    # ================================================================
    # Set automatic parameters based on video properties
    # ================================================================
    # Set RBF smoothing parameters based on video frame rate
    if rbfsmooth < 0:
        if rbfkernel == "linear":
            if fps > 100:
                rbfsmooth = 0.02
            elif fps > 60:
                rbfsmooth = 0.01
            else:
                rbfsmooth = 0.005
        elif rbfkernel == "univariatespline":
            if fps > 60:
                rbfsmooth = 0.5
            else:
                rbfsmooth = 0.25
        elif rbfkernel == "multiquadric":
            if fps > 60:
                rbfsmooth = 0.000025
            else:
                rbfsmooth = 0.00001
                
    # Set RBF epsilon parameter based on video frame rate
    if rbfepsilon < 0:
        if rbfkernel == "multiquadric":
            if fps > 60:
                rbfepsilon = 20
            else:
                rbfepsilon = 25
      
    # Calculate frame sampling based on desired time interval
    videoSampling = int(fps * frameSampling)
    print("Video sampling:", videoSampling)

    # ================================================================
    # Step 0: VGGT camera analysis
    # ================================================================
    # Define output file paths
    output_vggt_pkl = os.path.join(args.directory, "vggt.pkl")
    output_vggt_glb = os.path.join(args.directory, "vggt.glb")
    output_vggt_interpolated_pkl = os.path.join(args.directory, "vggt-interpolated.pkl")
    
    print("\n############################################################")
    print("# Step 0: VGGT analysis")
    print("############################################################")
    if args.step <= 0:
        print()
        # Run camera pose analysis
        command_videoCameraPosesAnalysis = f"python videoCameraPosesAnalysisVGGT.py {videoFileName} {videoSampling} {output_vggt_pkl} {output_vggt_glb} {fov} {camtoltranslation} {camtolrotation} {camtolfov}"
        print("Processing VGGT analysis...")
        print(command_videoCameraPosesAnalysis)
        result = subprocess.run(command_videoCameraPosesAnalysis, shell=True)
        if result.returncode != 0:
            print("\nError in VGGT analysis")
            sys.exit(1)
            
        print()
        # Interpolate camera poses for smooth transitions
        command_interpolateCameraPoses = f"python interpolateCameraPosesVGGT.py {output_vggt_pkl} {output_vggt_interpolated_pkl} {rbfkernel} {rbfsmooth} {rbfepsilon}"
        print("Processing camera poses interpolation...")
        print(command_interpolateCameraPoses)
        result = subprocess.run(command_interpolateCameraPoses, shell=True)
        if result.returncode != 0:
            print("\nError in camera poses interpolation")
            sys.exit(1)

    # ================================================================
    # Extract camera parameters from VGGT analysis
    # ================================================================
    print("\n############################################################")
    print("# Extract data from VGGT analysis")
    print("############################################################")
    print()
    print("Extracting data from VGGT analysis...")
    print("Reading VGGT Pkl file:", output_vggt_pkl)
    
    with open(output_vggt_pkl, 'rb') as file:
        vggtPKL = pickle.load(file)
    
    # Determine if camera FOV is dynamic or fixed
    fov_x_degrees = 0
    dynamicFov = not vggtPKL["fov_fixed"]
    if dynamicFov:
        print("Dynamic fov")
        fov_x_degrees = vggtPKL['fov_x_degrees'][0] 
    else:
        fov_x_degrees = vggtPKL['average_fov_x_degrees']
        print("Fov fixed:", fov_x_degrees)

    # Determine if camera position is fixed or moving
    cameraFixed = vggtPKL["camera_fixed"]
    if cameraFixed:
        print("Camera fixed")
    else:
        print("Camera moving")
      
    # ================================================================
    # Step 1: MoGe floor analysis
    # ================================================================
    output_moge_pkl = os.path.join(args.directory, "moge.pkl")
    print("\n############################################################")
    print("# Step 1: MoGe floor analysis")
    print("############################################################")
    if args.step <= 1:
        print()
        command_floorAnalysisMoge = f"python floorAnalysisMoge.py {videoFileName} {output_moge_pkl} {fov_x_degrees}"
        print("Processing MoGe analysis...")
        print(command_floorAnalysisMoge)
        result = subprocess.run(command_floorAnalysisMoge, shell=True)
        if result.returncode != 0:
            print("\nError in MoGe floor analysis")
            sys.exit(1)

    # ================================================================
    # Step 2: Extract 3D poses with NLF
    # ================================================================
    output_type_pkl = os.path.join(args.directory, f"{type}.pkl")
    print("\n############################################################")
    print("# Step 2: Extract 3D poses with NLF")
    print("############################################################")
    if args.step <= 2:
        print()
        # Use different command depending on if camera has dynamic or fixed FOV
        if dynamicFov:
            command_videoNLF = f"python videoNLF.py --video {videoFileName} --out_pkl {output_type_pkl} --camerafile {output_vggt_interpolated_pkl} --det_thresh {detectionThreshold} --batchsize {args.batchsize}"
        else:
            command_videoNLF = f"python videoNLF.py --video {videoFileName} --out_pkl {output_type_pkl} --fov {fov_x_degrees} --det_thresh {detectionThreshold} --batchsize {args.batchsize}"
        
        print("Processing NLF poses estimation...") 
        print(command_videoNLF)
        result = subprocess.run(command_videoNLF, shell=True)
        if result.returncode != 0:
            print("\nError in NLF pose estimation")
            sys.exit(1)    
        
    # ================================================================
    # Determine number of humans in the video
    # ================================================================
    print("\n############################################################")
    print("# Extract total human number")
    print("############################################################")
    print()
    print("Read pkl:", output_type_pkl)
    
    with open(output_type_pkl, 'rb') as file:
        dataPKL = pickle.load(file) 
   
    print("Frames:", len(dataPKL['allFrameHumans'])) 
    humanNumber, maxNbHumans, minNbHumans = getHumanNumber(dataPKL)
    print('humanNumber:', humanNumber)
    print('maxNbHumans:', maxNbHumans)
    print('minNbHumans:', minNbHumans)    

    output_cleaned_pkl = os.path.join(args.directory, f"{type}-clean.pkl")
    threshold = 0.5
        
    # ================================================================
    # Step 3: Clean poses and inject camera properties
    # ================================================================
    output_video_json = os.path.join(args.directory, "video.json")
    print("\n############################################################")
    print("# Step 3: Clean poses")
    print("############################################################")
    if args.step <= 3:
        print()
        # Clean frames to retain consistent number of humans
        command_cleanFramesPkl = f"python cleanFramesPkl.py {output_type_pkl} {output_cleaned_pkl} {humanNumber} {threshold}"
        print("Processing cleaned pkl...") 
        print(command_cleanFramesPkl)
        result = subprocess.run(command_cleanFramesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in cleaning poses")
            sys.exit(1)
            
        print()
        # Compute camera properties from multiple analyses
        output_video_json = os.path.join(args.directory, "video.json")
        command_computeCameraProperties = f"python computeCameraProperties.py {output_vggt_pkl} {output_moge_pkl} {output_cleaned_pkl} {fov_x_degrees} {output_video_json}"
        print("Processing camera properties...")
        print(command_computeCameraProperties)
        result = subprocess.run(command_computeCameraProperties, shell=True)
        if result.returncode != 0:
            print("\nError in camera properties")
            sys.exit(1)
            
        print()
        # Inject camera properties into both PKL files
        command_injectCameraPropertiesPkl = f"python injectCameraPropertiesPkl.py {output_video_json} {output_type_pkl} {output_type_pkl}"
        print(f"Inject camera properties in {output_type_pkl}")
        print(command_injectCameraPropertiesPkl)
        result = subprocess.run(command_injectCameraPropertiesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in camera properties")
            sys.exit(1)
            
        command_injectCameraPropertiesPkl = f"python injectCameraPropertiesPkl.py {output_video_json} {output_cleaned_pkl} {output_cleaned_pkl}"
        print(f"Inject camera properties in {output_cleaned_pkl}")
        print(command_injectCameraPropertiesPkl)
        result = subprocess.run(command_injectCameraPropertiesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in inject camera properties")
            sys.exit(1)

    # ================================================================
    # Step 4: 3D Tracking
    # ================================================================
    # Adjust threshold based on frame rate
    threshold = 0.5
    if fps > 60:
        threshold = 0.4
        
    output_tracking_pkl = os.path.join(args.directory, f"{type}-clean-track.pkl")
    print("\n############################################################")
    print("# Step 4: 3D Tracking")
    print("############################################################")
    if args.step <= 4:
        print()
        command_tracking3DPkl = f"python tracking3DPkl.py {output_cleaned_pkl} {output_tracking_pkl} {threshold}"
        print("Processing tracking pkl...") 
        print(command_tracking3DPkl)
        result = subprocess.run(command_tracking3DPkl, shell=True)
        if result.returncode != 0:
            print("\nError in 3D tracking")
            sys.exit(1)
        
    # ================================================================
    # Step 5: Add SAM2.1 tracking for improved segmentation
    # ================================================================
    # Adjust track minimum size based on frame rate
    trackMinSize = 30
    if fps < 50:
        trackMinSize = 5
        
    output_seg_pkl = os.path.join(args.directory, f"{type}-clean-track-seg.pkl")
    output_video_segmentation = os.path.join(args.directory, f"{type}-videoSegmentation.mp4")
    print("\n############################################################")
    print("# Step 5: Add SAM2.1 tracking")
    print("############################################################")
    if args.step <= 5:
        print()
        command_fusionMultiPersonTracking = f"python sam21MultiPerson.py {output_tracking_pkl} {videoFileName} {humanNumber} {trackMinSize} {output_seg_pkl} {output_video_segmentation} {dispersionthreshold} {displayMode}"
        print("Processing fusion...") 
        print(command_fusionMultiPersonTracking)
        result = subprocess.run(command_fusionMultiPersonTracking, shell=True)
        if result.returncode != 0:
            print("\nError in SAM2.1 tracking")
            sys.exit(1)
      
    # ================================================================
    # Step 6: Tracks fusion to consolidate multiple tracks
    # ================================================================
    output_fusion_pkl = os.path.join(args.directory, f"{type}-clean-track-seg-fusion.pkl")
    output_final_pkl = output_fusion_pkl
    
    # Adjust track minimum size based on frame rate
    trackMinSize = 30
    if fps < 50:
        trackMinSize = 10
        
    print("\n############################################################")
    print("# Step 6: Tracks fusion")
    print("############################################################")
    if args.step <= 6:
        print()
        command_tracksFusion = f"python tracksFusion.py {output_seg_pkl} {output_fusion_pkl} 10"
        print("Processing track fusion...") 
        print(command_tracksFusion)
        result = subprocess.run(command_tracksFusion, shell=True)
        if result.returncode != 0:
            print("\nError in track fusion")
            sys.exit(1)

    # ================================================================
    # Step 7: Remove outliers in pose data
    # ================================================================
    output_final_outlier_pkl = os.path.join(args.directory, f"{type}-clean-track-seg-fusion-outlier.pkl")
    print("\n############################################################")
    print("# Step 7: Remove outlier in pkl")
    print("############################################################")
    if args.step <= 7:
        print()
        command_removeoutlier = f"python removeOutlier.py {output_fusion_pkl} {output_final_outlier_pkl} 0"
        print("Processing Outlier removal...") 
        print(command_removeoutlier)
        result = subprocess.run(command_removeoutlier, shell=True)
        if result.returncode != 0:
            print("\nError in outlier removal")
            sys.exit(1)

    # ================================================================
    # Step 8: Optional hand pose estimation
    # ================================================================
    if handEstimation:
        print("\n############################################################")
        print("# Step 8: Inject hand estimation based on Wilor in pkl")
        print("############################################################")
        previous_output_final_outlier_pkl = output_final_outlier_pkl
        output_final_outlier_pkl = os.path.join(args.directory, f"{type}-clean-track-seg-fusion-outlier-handestimation.pkl")
        if args.step <= 8:
            print()
            command_injectHands = f"python injectHandsPkl.py {previous_output_final_outlier_pkl} {videoFileName} {output_final_outlier_pkl} {displayMode}"
            print("Processing hand estimation...")
            print(command_injectHands)
            result = subprocess.run(command_injectHands, shell=True)
            if result.returncode != 0:
                print("\nError in hand estimation")
                sys.exit(1)

    # ================================================================
    # Step 9: Apply RBF filtering for temporal smoothness
    # ================================================================
    # Set output file path based on whether hand estimation was done
    if handEstimation:
        output_final_outlier_filtered_pkl = os.path.join(args.directory, f"{type}-clean-track-seg-fusion-outlier-handestimation-filtered.pkl")
    else:
        output_final_outlier_filtered_pkl = os.path.join(args.directory, f"{type}-clean-track-seg-fusion-outlier-filtered.pkl")
        
    print("\n############################################################")
    print("# Step 9: RBF and Filtering")
    print("############################################################")
    if args.step <= 9:
        print()
        command_rbfFiltering = f"python RBFFilterSMPLX.py {output_final_outlier_pkl} {output_final_outlier_filtered_pkl} {rbfkernel} {rbfsmooth} {rbfepsilon}"
        print("Processing RBF and filtering...") 
        print(command_rbfFiltering)
        result = subprocess.run(command_rbfFiltering, shell=True)
        if result.returncode != 0:
            print("\nError in RBF and filtering")
            sys.exit(1)

    # ================================================================
    # Copy final PKL files to standard names
    # ================================================================
    print("\n############################################################")
    print("# Copy final pkl files")
    print("############################################################")
    print()
    output_destination_pkl = os.path.join(args.directory, f"{type}-final.pkl")
    output_destination_filtered_pkl = os.path.join(args.directory, f"{type}-final-filtered.pkl")

    print("Copying final pkl files...")
    print("Final pkl")
    print("From:", output_final_pkl)
    print("To:", output_destination_pkl)
    shutil.copyfile(output_final_pkl, output_destination_pkl)
    
    print("Final filtered pkl")
    print("From:", output_final_outlier_filtered_pkl)
    print("To:", output_destination_filtered_pkl)
    shutil.copyfile(output_final_outlier_filtered_pkl, output_destination_filtered_pkl)

    # ================================================================
    # Step 10: Camera compensation for world-space coordinates
    # ================================================================
    output_destination_camerac_pkl = os.path.join(args.directory, f"{type}-final-camerac.pkl")
    output_destination_camerac_filtered_pkl = os.path.join(args.directory, f"{type}-final-filtered-camerac.pkl")
    output_destination_floorc_pkl = os.path.join(args.directory, f"{type}-final-floorc.pkl")
    output_destination_floorc_filtered_pkl = os.path.join(args.directory, f"{type}-final-filtered-floorc.pkl")
    print("\n############################################################")
    print("# Step 10: Camera compensation")
    print("############################################################")
    if args.step <= 10:
        # Auto-estimate scale factor if not provided
        if scaleFactor == 0:
            print(output_destination_filtered_pkl, output_vggt_interpolated_pkl)
            posePKL, vggtPKL = loadPkl2(output_destination_filtered_pkl, output_vggt_interpolated_pkl)
            print("Scale factor estimation...")
            scaleFactor = estimanteVGGTScaleFactor(posePKL, vggtPKL)
            print("Scale factor:", scaleFactor)
            
        print()
        # Apply camera compensation to unfiltered poses
        command_cameraCompensation = f"python cameraCompensation.py {output_destination_pkl} {output_destination_camerac_pkl} {output_vggt_interpolated_pkl} {output_moge_pkl} {scaleFactor} 1 0.1"
        print("Processing camera compensation...")
        print(command_cameraCompensation)
        result = subprocess.run(command_cameraCompensation, shell=True)
        if result.returncode != 0:
            print("\nError in camera compensation on poses")
            sys.exit(1)
            
        # Apply camera compensation to filtered poses
        command_cameraCompensation = f"python cameraCompensation.py {output_destination_filtered_pkl} {output_destination_camerac_filtered_pkl} {output_vggt_interpolated_pkl} {output_moge_pkl} {scaleFactor} 1 0.1"
        print("Processing camera compensation on filter poses...")
        print(command_cameraCompensation)
        result = subprocess.run(command_cameraCompensation, shell=True)
        if result.returncode != 0:
            print("\nError in camera compensation")
            sys.exit(1)

        print()
        # Apply camera floor compensation to unfiltered poses
        command_cameraFloorCompensation = f"python cameraFloorCompensation.py {output_destination_pkl} {output_destination_floorc_pkl} {output_moge_pkl}"
        print("Processing camera floor compensation...")
        print(command_cameraFloorCompensation)
        result = subprocess.run(command_cameraFloorCompensation, shell=True)
        if result.returncode != 0:
            print("\nError in camera compensation on poses")
            sys.exit(1)
            
        # Apply camera floor compensation to filtered poses
        command_cameraFloorCompensation = f"python cameraFloorCompensation.py {output_destination_filtered_pkl} {output_destination_floorc_filtered_pkl} {output_moge_pkl}"
        print("Processing camera floor compensation on filter poses...")
        print(command_cameraFloorCompensation)
        result = subprocess.run(command_cameraFloorCompensation, shell=True)
        if result.returncode != 0:
            print("\nError in camera compensation")
            sys.exit(1)

    # ================================================================
    # Step 11: Convert PKL for static web visualization
    # ================================================================
    print("\n############################################################")
    print("# Step 11: Convert PKL for static web visualization")
    print("############################################################")
    if args.step <= 11:
        # Check required arguments
        if args.webdirectory is not None:
            # Create output directory if it doesn't exist
            if not os.path.exists(args.webdirectory):
                os.makedirs(args.webdirectory)
                print(f"Created directory: {args.webdirectory}")
            shutil.copyfile(args.video, os.path.join(args.webdirectory, "video.mp4"))
            output_webdestination = os.path.join(args.webdirectory, "video")
            source_pkl = os.path.join(args.directory, args.webtarget+".pkl")
            command_convertCapture3D = f"python convertCapture3D.py {source_pkl} {output_webdestination} 0 0 1 1 1 0 neutral"
            print("Processing PKL to web...")
            print(command_convertCapture3D)
            result = subprocess.run(command_convertCapture3D, shell=True)
            if result.returncode != 0:
                print("\nError in PKL to web conversion")
                sys.exit(1)

    # Record the end time
    end_time = time.time()

    print()

    print("\n############################################################")
    print("# Results")
    print("############################################################")
    # Calculate and print the total execution time
    execution_time = end_time - start_time
    print(f"Total execution time of multiPersonProcessing: {execution_time:.2f} seconds")

    # Convert the execution time to hours, minutes, and seconds
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print the total execution time in hours, minutes, and seconds
    print(f"Total execution time of multiPersonProcessing: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
    try:
        with open(output_video_json, 'r') as json_file:
            video_data = json.load(json_file)
            # Process and print each item directly instead of using json.dumps
            items = list(video_data.items())
            for i, (key, value) in enumerate(items):
                # Apply color formatting for specific fields
                if key == 'floor_detected':
                    colored_value = f"{Fore.GREEN if value else Fore.RED}{value}{Style.RESET_ALL}"
                elif key == 'camera_motion_coherent':
                    colored_value = f"{Fore.GREEN if value else Fore.RED}{value}{Style.RESET_ALL}"
                elif key == 'camera_fixed':
                    colored_value = f"{Fore.YELLOW if value else Fore.BLUE}{value}{Style.RESET_ALL}"
                elif key == 'dynamic_fov':
                    colored_value = f"{Fore.YELLOW if value else Fore.BLUE}{value}{Style.RESET_ALL}"
                else:
                    # Keep other fields as is
                    colored_value = value
                
                # Add comma if not the last item
                end_char = "" if i == len(items) - 1 else ","
                print(f"    {key}: {colored_value}{end_char}")
            
    except FileNotFoundError:
        print(f"Error: The file {output_video_json} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {output_video_json} contains invalid JSON.")
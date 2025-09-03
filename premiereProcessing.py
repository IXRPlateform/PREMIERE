"""
premiereProcessing.py 3D pose estimation and tracking pipeline.

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
import json
import cv2
import pickle
import subprocess
import numpy as np
import time
import shutil

from argparse import ArgumentParser

from premiere.functionsCommon import loadPkl2
from premiere.functionsVGGT import estimanteVGGTScaleFactor

from colorama import init, Fore, Style
init(autoreset=True)

# Record the start time
start_time = time.time()

def getMajorityValue(humansPerFrames):
    if humansPerFrames.size == 0:
        return None
    values, counts = np.unique(humansPerFrames.flatten(), return_counts=True)
    majority_value = values[np.argmax(counts)]
    return majority_value

def getHumanNumber(dataPKL):
    humansPerFrames = np.empty([len(dataPKL['allFrameHumans']), 1],dtype=int)
    for i in range(len(dataPKL['allFrameHumans'])):
        humansPerFrames[i] = len(dataPKL['allFrameHumans'][i])
    humanNumber = getMajorityValue(humansPerFrames)
    maxNbHumans = max(humansPerFrames)
    minNbHumans = min(humansPerFrames)
    return humanNumber, maxNbHumans, minNbHumans

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--directory", type=str, default=None, help="Directory to store the processed files")
    parser.add_argument("--video", type=str, default=None, help="Video file to process")
    parser.add_argument("--fov", type=float, default=0, help="Field of view for the 3D pose estimation")
    parser.add_argument("--rbfkernel", type=str, default="linear", choices=["linear", "multiquadric"], help="RBF kernel to use for the 3D pose estimation filtering")
    parser.add_argument("--rbfsmooth", type=float, default=-1, help="Smoothness for the RBF kernel")
    parser.add_argument("--rbfepsilon", type=float, default=-1, help="Epsilon for the RBF kernel")
    parser.add_argument("--step", type=int, default=0, help="Step to process (default: 0 for all steps)")
    parser.add_argument("--batchsize", type=int, default=25, help="Batch size for the nlf 3D pose estimation")
    parser.add_argument("--displaymode", action="store_true", help="Display mode activated if this flag is set")
    parser.add_argument("--handestimation", action="store_true", help="Inject hand estimation based on Wilor if this flag is set")
    parser.add_argument("--removeshadows", action="store_true", help="Remove the human shadows if this flag is set")
    parser.add_argument("--removeshadowsthreshold", type=float, default=.00015,help="Threshold for removing the human shadows")
    parser.add_argument("--detectionthreshold", type=float, default=0.3,help="Threshold for detecting the human")
    parser.add_argument("--computedepth", action="store_true", help="Compute the depth data if this flag is set")
    parser.add_argument("--framesampling", type=float, default=.5 , help="Frame sampling in second for the MoGe and MAST3r analysis")
    parser.add_argument("--scalefactor", type=float, default=0, help="Scale factor for the camera compensation")
    parser.add_argument("--camtoltranslation", type=float, default=.2 , help="Camera translation tolerance (in meter) for detecting if the camera it moving in the video")
    parser.add_argument("--camtolrotation", type=float, default=2 , help="Camera rotation tolerance (in degree) for detecting if the camera it rotating in the video")
    parser.add_argument("--camtolfov", type=float, default=3 , help="Camera fov tolerance (in degree) for detecting if the camera is changing its fov in the video")
    parser.add_argument("--webdirectory", type=str, default=None, help="Directory to store the processed files for static website visualization")
    parser.add_argument("--webtarget", type=str, default="nlf-final-filtered-camerac", choices=["nlf", "nlf-clean", "nlf-final", "nlf-final-filtered", "nlf-final-floorc", "nlf-final-filtered-floorc", "nlf-final-camerac", "nlf-final-filtered-camerac"], help="Target for the web directory")
        
    print ("\n############################################################")
    print ("# Arguments")
    print ("############################################################")
    args = parser.parse_args()
    print ("Type: NLF")
    print ("Directory: ", args.directory)
    print ("Video: ", args.video)
    print ("Fov: ", args.fov)
    print ("Framesampling: ", args.framesampling)
    print ("Rbfsmooth: ", args.rbfsmooth)
    print ("Rbfepsilon: ", args.rbfepsilon)
    print ("Rbfkernel: ", args.rbfkernel)
    print ("Displaymode: ", args.displaymode)
    print ("Handestimation: ", args.handestimation)
    print ("Batchsize: ", args.batchsize)
    print ("Removeshadows: ", args.removeshadows)
    print ("Step: ", args.step)
    print ("Detectionthreshold: ", args.detectionthreshold)
    print ("Scalefactor: ", args.scalefactor)
    if (args.removeshadows):
        print ("Removeshadowsthreshold: ", args.removeshadowsthreshold)
    if (args.webdirectory is not None):
        print("Web directory:", args.webdirectory)
        print("Web target:", args.webtarget)
    
    videoFileName = "\""+args.video+"\""

    if args.displaymode:
        displayMode = 1
    else:
        displayMode = 0
    handEstimation = args.handestimation
    removeshadows = args.removeshadows
    rbfkernel = args.rbfkernel
    rbfsmooth = args.rbfsmooth
    rbfepsilon = args.rbfepsilon
    fov = args.fov
    type = "nlf"
    removeShadowsThreshold = args.removeshadowsthreshold
    detectionThreshold = args.detectionthreshold
    frameSampling = args.framesampling
    camtoltranslation = args.camtoltranslation
    camtolrotation = args.camtolrotation
    camtolfov = args.camtolfov
    scaleFactor = args.scalefactor

    computeDepth = removeshadows or args.computedepth
    print("computeDepth: ", computeDepth)

    if args.directory is None:
        print("Please provide a directory")
        sys.exit(1)
    if args.video is None:
        print("Please provide a video")
        sys.exit(1)
    
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        print(f"Created directory: {args.directory}")
    
    print ("\n############################################################")
    print ("# Video information")
    print ("############################################################")
    video = cv2.VideoCapture(args.video)
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video width: ", width)
    print("Video height: ", height)
    print("Video fps: ", fps)
    print("Video frames count: ", frames_count)
    video.release()
  
    if rbfsmooth < 0:
        if (rbfkernel == "linear"):
            if fps > 100:
                rbfsmooth = 0.02
            elif fps > 60:
                rbfsmooth = 0.01
            else:
                rbfsmooth = 0.005
        elif (rbfkernel == "univariatespline"):
            if fps > 60:
                rbfsmooth = 0.5
            else:
                rbfsmooth = 0.25
        elif (rbfkernel == "multiquadric"):
            if fps > 60:
                rbfsmooth = 0.000025
            else:
                rbfsmooth = 0.00001
                
    if rbfepsilon < 0:
        if (rbfkernel == "multiquadric"):
            if fps > 60:
                rbfepsilon = 20
            else:
                rbfepsilon = 25
      
    videoSampling = int(fps * frameSampling)
    print("Video sampling: ", videoSampling)

    output_vggt_pkl = os.path.join(args.directory, "vggt.pkl")
    output_vggt_glb = os.path.join(args.directory, "vggt.glb")
    output_vggt_interpolated_pkl = os.path.join(args.directory, "vggt-interpolated.pkl")
    print ("\n############################################################")
    print ("# Step 0: VGGT analysis")
    print ("############################################################")
    if args.step <= 0:
        print()
        command_videoCameraPosesAnalysis = f"python videoCameraPosesAnalysisVGGT.py {videoFileName} {videoSampling} {output_vggt_pkl} {output_vggt_glb} {fov} {camtoltranslation} {camtolrotation} {camtolfov}"
        print("Processing VGGT analysis...")
        print(command_videoCameraPosesAnalysis)
        result = subprocess.run(command_videoCameraPosesAnalysis, shell=True)
        if result.returncode != 0:
            print("\nError in VGGT analysis")
            sys.exit(1)
        print()
        command_interpolateCameraPoses = f"python interpolateCameraPosesVGGT.py {output_vggt_pkl} {output_vggt_interpolated_pkl} {rbfkernel} {rbfsmooth} {rbfepsilon}"
        print("Processing camera poses interpolation...")
        print(command_interpolateCameraPoses)
        result = subprocess.run(command_interpolateCameraPoses, shell=True)
        if result.returncode != 0:
            print("\nError in camera poses interpolation")
            sys.exit(1)

    print ("\n############################################################")
    print ("# Extract data from VGGT analysis")
    print ("############################################################")
    print()
    print ("Extracting data from VGGT analysis...")
    print ("Reading VGGT Pkl file: ", output_vggt_pkl)
    with open(output_vggt_pkl, 'rb') as file:
        vggtPKL = pickle.load(file)
    
    fov_x_degrees = 0
    dynamicFov = not vggtPKL["fov_fixed"]
    if dynamicFov:
        print("Dynamic fov")
        fov_x_degrees = vggtPKL['fov_x_degrees'][0]
        print("Fov dynamic[0]: ", fov_x_degrees)
    else:
        fov_x_degrees = vggtPKL['average_fov_x_degrees']
        print("Fov fixed: ", fov_x_degrees)

    cameraFixed = vggtPKL["camera_fixed"]
    if cameraFixed:
        print("Camera fixed")
    else:
        print("Camera moving")

    print ("\n############################################################")
    print ("# Step 1: MoGe analysis")
    print ("############################################################")
    print (" Methods: MoGe model for 3D/depth estimation + custom algorithms")
    print ("############################################################")
    output_moge_pkl = os.path.join(args.directory, "moge.pkl")
    if args.step <= 1:
        print()
        command_floorAnalysisMoge = f"python floorAnalysisMoge.py {videoFileName} {output_moge_pkl} {fov_x_degrees}"
        print("Processing MoGe analysis...")
        print(command_floorAnalysisMoge)
        result = subprocess.run(command_floorAnalysisMoge, shell=True)
        if result.returncode != 0:
            print("\nError in MoGe floor analysis")
            sys.exit(1)

    print ("\n############################################################")
    print ("# Step 2: 3D poses estimation with NLF")
    print ("############################################################")
    print (" Method: NLF model for 3D pose estimation")
    print ("############################################################")
    output_type_pkl = os.path.join(args.directory, type+".pkl")
    if args.step <= 2:
        print()
        print()
        if dynamicFov:
            command_videoNLF = f"python videoNLF.py --video {videoFileName} --out_pkl {output_type_pkl} --camerafile {output_vggt_interpolated_pkl} --det_thresh {detectionThreshold} --batchsize {args.batchsize}"
        else:
            command_videoNLF = f"python videoNLF.py --video {videoFileName} --out_pkl {output_type_pkl} --fov {fov_x_degrees} --det_thresh {detectionThreshold} --batchsize {args.batchsize}"
        print("Processing NLF...") 
        print(command_videoNLF)
        result = subprocess.run(command_videoNLF, shell=True)
        if result.returncode != 0:
            print("\nError in NLF pose estimation")
            sys.exit(1)       
        
    print ("\n############################################################")
    print ("# Extract total human statisitics")
    print ("############################################################")
    print (" Method: Simple data extraction form Step 1")
    print ("############################################################")
    print()
    # Open the pkl file
    print ("Read pkl: ",output_type_pkl)
    file = open(output_type_pkl, 'rb')
    dataPKL = pickle.load(file) 
    file.close()
   
    print("Frames: ", len(dataPKL['allFrameHumans'])) 
    humanNumber, maxNbHumans, minNbHumans = getHumanNumber(dataPKL)
    print('maxNbHumans: ', maxNbHumans)
    print('minNbHumans: ', minNbHumans)    

    output_cleaned_pkl = os.path.join(args.directory, type+"-clean.pkl")
    threshold = 0.4
        
    print ("\n############################################################")
    print ("# Step 3: Cleaning poses and compute/inject camera properties")
    print ("############################################################")
    print (" Method: Custom algorithms for cleaning poses + camera properties injection")
    print ("############################################################")
    output_video_json = os.path.join(args.directory, "video.json")
    if args.step <= 3:
        print()
        command_cleanFramesPkl = f"python cleanFramesPkl.py {output_type_pkl} {output_cleaned_pkl} {humanNumber} {threshold}"
        print("Processing cleaned pkl...") 
        print(command_cleanFramesPkl)
        result = subprocess.run(command_cleanFramesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in cleaning poses")
            sys.exit(1)
        print()
        command_computeCameraProperties = f"python computeCameraProperties.py {output_vggt_pkl} {output_moge_pkl} {output_cleaned_pkl} {fov_x_degrees} {output_video_json}"
        print("Processing camera properties...")
        print(command_computeCameraProperties)
        result = subprocess.run(command_computeCameraProperties, shell=True)
        if result.returncode != 0:
            print("\nError in camera properties")
            sys.exit(1)
        print()
        command_injectCameraPropertiesPkl = f"python injectCameraPropertiesPkl.py {output_video_json} {output_type_pkl} {output_type_pkl}"
        print("Inject camera properties in", output_type_pkl)
        print(command_injectCameraPropertiesPkl)
        result = subprocess.run(command_injectCameraPropertiesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in camera properties")
            sys.exit(1)
        command_injectCameraPropertiesPkl = f"python injectCameraPropertiesPkl.py {output_video_json} {output_cleaned_pkl} {output_cleaned_pkl}"
        print("Inject camera properties in", output_cleaned_pkl)
        print(command_injectCameraPropertiesPkl)
        result = subprocess.run(command_injectCameraPropertiesPkl, shell=True)
        if result.returncode != 0:
            print("\nError in inject camera properties")
            sys.exit(1)
        

    print ("\n############################################################")
    print ("# Step 4: SAM2.1 humans segmentation")
    print ("############################################################")
    print (" Method: SAM2.1 model for humans segmentation")
    print ("############################################################")
    output_video_segmentation = os.path.join(args.directory, type+"-videoSegmentation.mp4")
    if args.step <= 4:
        print()
        command_segmentation = f"python sam21NLF.py {videoFileName} {output_cleaned_pkl} {output_video_segmentation} {displayMode}"
        print("Processing fusion...") 
        print(command_segmentation)
        result = subprocess.run(command_segmentation, shell=True)
        if result.returncode != 0:
            print("\nError in segmentation")
            sys.exit(1)

    if computeDepth:
        print ("\n############################################################")
        print ("# Step 5: Extract depth data")
        print ("############################################################")
        print (" Method: Custom algorithms for depth extraction")
        print ("############################################################")
        output_depth_pkl = os.path.join(args.directory, "videoDepth.pkl")
        output_depth_video = os.path.join(args.directory, "videoDepth.mp4")
        if args.step <= 5:
            print()
            command_videoProcessMoge = f"python videoProcessMoge.py {videoFileName} {output_moge_pkl} {output_depth_video} {output_depth_pkl} {fov_x_degrees}"
            print("Processing depth data...") 
            print(command_videoProcessMoge)
            result = subprocess.run(command_videoProcessMoge, shell=True)
            if result.returncode != 0:
                print("\nError in depth extraction")
                sys.exit(1)

    removedShadowsStr = ""
    if removeshadows:
        removedShadowsStr = "-removedshadows"

    output_shadows_pkl = output_cleaned_pkl
    if removeshadows:
        print ("\n############################################################")
        print ("# Step 6: Add humans homogeneity")
        print ("############################################################")
        print (" Method: Custom algorithms for adding homogeneity for each human segmented in Step 3 with the depth data from Step 4 and the original video")
        print ("############################################################")
        if args.step <= 6:
            print()
            output_shadows_pkl = os.path.join(args.directory, type+"-clean"+removedShadowsStr+".pkl")
            command_addHumanHomogeneityPkl = f"python addHumanHomogeneityPkl.py {output_cleaned_pkl} {output_depth_video} {output_video_segmentation} {output_depth_pkl} {videoFileName} {output_shadows_pkl} {displayMode}"
            print("Processing add human homogeneity...") 
            print(command_addHumanHomogeneityPkl)
            result = subprocess.run(command_addHumanHomogeneityPkl, shell=True)
            if result.returncode != 0:
                print("\nError in add human homogeneity")
                sys.exit(1)
        print ("\n############################################################")
        print ("# Step 7: Remove Shadows")
        print ("############################################################")
        print (" Method: Custom algorithms for removing the cleaned pkl the poses corresponding of shadows")
        print ("############################################################")
        if args.step <= 7:
            output_shadows_pkl = os.path.join(args.directory, type+"-clean"+removedShadowsStr+".pkl")
            command_removeShadowsPkl = f"python removeShadowsPkl.py {output_shadows_pkl} {output_shadows_pkl} {removeShadowsThreshold}"
            print("Processing shadow detection...") 
            print(command_removeShadowsPkl)
            result = subprocess.run(command_removeShadowsPkl, shell=True)
            if result.returncode != 0:
                print("\nError in shadow detection")
                sys.exit(1)


    print ("\n############################################################")
    print ("# Step 8: Byte Track Tracking")
    print ("############################################################")
    print (" Method: Byte Track model for tracking humans")
    print ("############################################################")
    
    output_tracking_pkl = os.path.join(args.directory, type+"-clean"+removedShadowsStr+"-track.pkl")
    output_final_pkl = output_tracking_pkl
    if args.step <= 8:
        print()
        command_trackingBTPkl = f"python trackingBTPkl.py {output_shadows_pkl} {videoFileName} {output_tracking_pkl} {displayMode}"
        print("Processing tracking pkl...") 
        print(command_trackingBTPkl)
        result = subprocess.run(command_trackingBTPkl, shell=True)
        if result.returncode != 0:
            print("\nError in BT Tracking")
            sys.exit(1)

    if handEstimation:
        print ("\n############################################################")
        print ("# Step 9: Inject hand detection based on Wilor in pkl")
        print ("############################################################")
        print (" Method: Custom algorithms for injecting hand detection based on Wilor in a tracking pose pkl")
        print ("############################################################")
        previous_output_final_pkl = output_final_pkl
        output_final_pkl = os.path.join(args.directory, type+"-clean"+removedShadowsStr+"-track-handestimation.pkl")
        if args.step <= 9:
            print()
            command_injectHands = f"python injectHandsPkl.py {previous_output_final_pkl} {videoFileName} {output_final_pkl} {displayMode}"
            print("Processing hand detection...")
            print(command_injectHands)
            result = subprocess.run(command_injectHands, shell=True)
            if result.returncode != 0:
                print("\nError in hand detection")
                sys.exit(1)

    print ("\n############################################################")
    print ("# Step 10: Remove outlier in pkl")
    print ("############################################################")
    print (" Method: Custom algorithms for removing outliers in a pose pkl")
    print ("############################################################")
    if handEstimation:
        output_final_outlier_pkl = os.path.join(args.directory, type+"-clean"+removedShadowsStr+"-track-handestimation-outlier.pkl")
    else:
        output_final_outlier_pkl = os.path.join(args.directory, type+"-clean"+removedShadowsStr+"-track-outlier.pkl")
    if args.step <= 10:
        print()
        command_removeoutlier = f"python removeOutlier.py {output_final_pkl} {output_final_outlier_pkl} 0"
        print("Processing Outlier removal...") 
        print(command_removeoutlier)
        result = subprocess.run(command_removeoutlier, shell=True)
        if result.returncode != 0:
            print("\nError in removing outlier")
            sys.exit(1)


    print ("\n############################################################")
    print ("# Step 11: Radial Basis Function Interpolation and Filtering")
    print ("############################################################")
    print (" Method: Custom algorithms for RBF interpolation and filtering in a pose pkl")
    print ("############################################################")
    if handEstimation:
        output_final_outlier_filtered_pkl = os.path.join(args.directory, type+"-clean"+removedShadowsStr+"-track-handestimation-outlier-filtered.pkl")
    else:
        output_final_outlier_filtered_pkl = os.path.join(args.directory, type+"-clean"+removedShadowsStr+"-track-outlier-filtered.pkl")
    if args.step <= 11:
        print()
        command_rbfFiltering = f"python RBFFilterSMPLX.py {output_final_outlier_pkl} {output_final_outlier_filtered_pkl} {rbfkernel} {rbfsmooth} {rbfepsilon}"
        print("Processing RBF and filtering...") 
        print(command_rbfFiltering)
        result = subprocess.run(command_rbfFiltering, shell=True)
        if result.returncode != 0:
            print("\nError in RBF and filtering")
            sys.exit(1)

    print ("\n############################################################")
    print ("# Step 12: Copy final pkl files")
    print ("############################################################")
    print (" Method: Copy the final pkl files in the output directory")
    print ("############################################################")
    print()
    output_destination_pkl = os.path.join(args.directory, type+"-final.pkl")
    output_destination_filtered_pkl = os.path.join(args.directory, type+"-final-filtered.pkl")

    print("Copying final pkl files...")
    print ("Final pkl")
    print("From: ",output_final_pkl)
    print("To: ",output_destination_pkl)
    shutil.copyfile(output_final_pkl, output_destination_pkl)
    print ("Final filtered pkl")
    print("From: ",output_final_outlier_filtered_pkl)
    print("To: ",output_destination_filtered_pkl)
    shutil.copyfile(output_final_outlier_filtered_pkl, output_destination_filtered_pkl)

    print ("\n############################################################")
    print ("# Step 13: Camera compensation")
    print ("############################################################")
    output_destination_camerac_pkl = os.path.join(args.directory, type+"-final-camerac.pkl")
    output_destination_camerac_filtered_pkl = os.path.join(args.directory, type+"-final-filtered-camerac.pkl")
    output_destination_floorc_pkl = os.path.join(args.directory, type+"-final-floorc.pkl")
    output_destination_floorc_filtered_pkl = os.path.join(args.directory, type+"-final-filtered-floorc.pkl")
    if args.step <= 13:
        if scaleFactor == 0:
            print (output_destination_filtered_pkl, output_vggt_interpolated_pkl)
            posePKL, vggtPKL = loadPkl2(output_destination_filtered_pkl, output_vggt_interpolated_pkl)
            print ("Scale factor estimation...")
            scaleFactor = estimanteVGGTScaleFactor(posePKL, vggtPKL)
            print("Scale factor: ", scaleFactor)
        print()
        command_cameraCompensation = f"python cameraCompensation.py {output_destination_pkl} {output_destination_camerac_pkl} {output_vggt_interpolated_pkl} {output_moge_pkl} {scaleFactor} 1 0.1"
        print("Processing camera compensation...")
        print(command_cameraCompensation)
        result = subprocess.run(command_cameraCompensation, shell=True)
        if result.returncode != 0:
            print("\nError in camera compensation on poses")
            sys.exit(1)
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

    print("\n############################################################")
    print("# Step 14: Convert PKL for static web visualization")
    print("############################################################")
    if args.step <= 14:
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
    print(f"Total execution time of premiereProcessing: {execution_time:.2f} seconds")

    # Convert the execution time to hours, minutes, and seconds
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print the total execution time in hours, minutes, and seconds
    print(f"Total execution time of premiereProcessing: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
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
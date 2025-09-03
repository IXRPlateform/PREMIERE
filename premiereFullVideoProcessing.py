#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
premiereFullVideoProcessing.py
------------------------------
A script that:
1. Detects scenes in a video using PySceneDetect (ContentDetector by default).
2. Splits the video into scene-based sub-videos.
3. (Optionally) processes each scene by calling an external script (premiereProcessing.py).
4. (Optionally) fuses the resulting .pkl files across scenes into a single set of files.

Usage:
------
python premiereFullVideoProcessing.py \
    --directory /path/to/output \
    --video /path/to/video.mp4 \
    [other arguments]

Requirements:
-------------
- Python 3.x
- scenedetect
- opencv-python (cv2)
- pickle
- (Optional) premiereProcessing.py script in the same environment/path
"""

import os
import sys
import subprocess
import shutil
import cv2
import pickle
import time
from argparse import ArgumentParser

# PySceneDetect imports
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg


def readPklFile(fileName):
    """
    Safely read a .pkl file and return its content.
    """
    if not os.path.exists(fileName):
        raise FileNotFoundError(f"Could not find PKL file: {fileName}")
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def savePklFile(fileName, allDataPKL):
    """
    Save data to a .pkl file safely.
    """
    with open(fileName, 'wb') as f:
        pickle.dump(allDataPKL, f)


def saveFusionFile(directory, scene_dirs, fusionFileName, fileName):
    """
    Creates a fused PKL file by concatenating 'allFrameHumans' data from multiple
    scenes, while also tracking per-scene floor angles, Z offsets, and 
    start/end frame indices in arrays: scene_start, scene_end, scene_floor_angle_deg,
    scene_floor_Zoffset.
    """
    # Read the first scene's PKL file as the "base".
    basePklFile = os.path.join(directory, scene_dirs[0], fileName)
    basePkl = readPklFile(basePklFile)

    # Initialize the 'scene_*' lists and store the original base scene info.
    basePkl['scene_start'] = [0]
    basePkl['scene_end'] = [len(basePkl['allFrameHumans']) - 1]
    basePkl['scene_floor_angle_deg'] = [basePkl['floor_angle_deg']]
    basePkl['scene_floor_Zoffset'] = [basePkl['floor_Zoffset']]

    # Set overall floor data to 0.0 in the "fusion" context,
    # so the per-scene lists handle actual floors.
    basePkl['floor_angle_deg'] = 0.0
    basePkl['floor_Zoffset'] = 0.0

    # We track the last index to correctly offset the start/end for subsequent scenes.
    count = len(basePkl['allFrameHumans'])

    # Iterate over all other scenes and extend the base data.
    for i in range(1, len(scene_dirs)):
        scene_path = os.path.join(directory, scene_dirs[i], fileName)
        print("Fusion file reading:", scene_path)
        intpulPkl = readPklFile(scene_path)

        # Extend and update the "scene_*" lists.
        basePkl['allFrameHumans'].extend(intpulPkl['allFrameHumans'])
        basePkl['scene_floor_angle_deg'].append(intpulPkl['floor_angle_deg'])
        basePkl['scene_floor_Zoffset'].append(intpulPkl['floor_Zoffset'])
        basePkl['scene_start'].append(count)
        basePkl['scene_end'].append(count + len(intpulPkl['allFrameHumans']) - 1)

        count += len(intpulPkl['allFrameHumans'])

    # Finally, save the fused PKL file.
    savePklFile(os.path.join(directory, fusionFileName), basePkl)


def saveFusion(directory, scene_dirs):
    """
    Fuses two specific sets of PKL files across scenes: 'nlf-final.pkl' and
    'nlf-final-filtered.pkl'. Results are saved as:
    - fusion-nlf-final.pkl
    - fusion-nlf-final-filtered.pkl
    """
    print("Pkl files fusion in progress...")
    saveFusionFile(directory, scene_dirs, "fusion-nlf-final.pkl", "nlf-final.pkl")
    saveFusionFile(directory, scene_dirs, "fusion-nlf-final-filtered.pkl", "nlf-final-filtered.pkl")


def getVideoFrameNumber(videoFileName):
    """
    Returns the total number of frames in a video using OpenCV.
    """
    if not os.path.exists(videoFileName):
        raise FileNotFoundError(f"Video file not found: {videoFileName}")
    video = cv2.VideoCapture(videoFileName)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {videoFileName}")
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return count


def main():
    """
    Main execution function:
    1. Parses arguments.
    2. Decides whether to detect and split scenes, process them, or fuse PKL files.
    3. Iterates over scenes or a specific scene, calling `premiereProcessing.py`.
    4. (Optional) Summarizes frames across all scenes and compares to original video frame count.
    5. (Optional) Fuses PKL files.
    """
    start_time = time.time()  # Start measuring time

    # 1. Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--directory", type=str, default=None,
                        help="Directory to store the processed files")
    parser.add_argument("--video", type=str, default=None,
                        help="Video file to process")

    # Additional arguments for controlling 3D pose estimation / RBF usage
    parser.add_argument("--fov", type=float, default=0,
                        help="Field of view for the 3D pose estimation")
    parser.add_argument("--rbfkernel", type=str, default="linear",
                        choices=["linear", "multiquadric"],
                        help="RBF kernel to use for the 3D pose estimation filtering")
    parser.add_argument("--rbfsmooth", type=float, default=-1,
                        help="Smoothness for the RBF kernel")
    parser.add_argument("--rbfepsilon", type=float, default=-1,
                        help="Epsilon for the RBF kernel")

    # Scene/step management
    parser.add_argument("--step", type=int, default=-1,
                        help="Step to process (default: -1 for all steps)")
    parser.add_argument("--scene", type=int, default=-1,
                        help="Scene to process (default: -1 for all scenes)")
    parser.add_argument("--batchsize", type=int, default=25,
                        help="Batch size for the nlf 3D pose estimation")

    # Flag arguments
    parser.add_argument("--displaymode", action="store_true",
                        help="Display mode activated if this flag is set")
    parser.add_argument("--handestimation", action="store_true",
                        help="Inject hand estimation if this flag is set")
    parser.add_argument("--removeshadows", action="store_true",
                        help="Remove the human shadows if this flag is set")
    parser.add_argument("--removeshadowsthreshold", type=float, default=0.00015,
                        help="Threshold for removing the human shadows")
    parser.add_argument("--detectionthreshold", type=float, default=0.3,
                        help="Threshold for detecting the human")
    parser.add_argument("--onlygeneratesscenes", action="store_true",
                        help="Generate only the scenes if this flag is set")
    parser.add_argument("--donotgeneratesscenes", action="store_true",
                        help="Will not generate the scenes if this flag is set")
    parser.add_argument("--fusionpklfiles", action="store_true",
                        help="Generate the fusion of the PKL files if this flag is set")


    parser.add_argument("--framesampling", type=float, default=.5 , 
                        help="Frame sampling in second for the MoGe and MAST3r analysis")
    parser.add_argument("--scalefactor", type=float, default=0, 
                        help="Scale factor for the camera compensation")
    parser.add_argument("--camtoltranslation", type=float, default=.2 , 
                        help="Camera translation tolerance (in meter) for detecting if the camera it moving in the video")
    parser.add_argument("--camtolrotation", type=float, default=2 , 
                        help="Camera rotation tolerance (in degree) for detecting if the camera it rotating in the video")
    parser.add_argument("--camtolfov", type=float, default=3 , 
                        help="Camera fov tolerance (in degree) for detecting if the camera is changing its fov in the video")

    args = parser.parse_args()

    # 2. Validate important arguments
    if args.directory is None:
        print("Error: Please provide a directory (--directory).")
        sys.exit(1)
    if args.video is None:
        print("Error: Please provide a video file path (--video).")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"Error: Video file does not exist at: {args.video}")
        sys.exit(1)

    directory = args.directory
    videoFileName = args.video
    scene = args.scene
    step = args.step

    # Decide if we should generate scenes
    onlyGeneratesScenes = args.onlygeneratesscenes
    donotgeneratesscenes = args.donotgeneratesscenes
    fusionpklfiles = args.fusionpklfiles

    # By default, if step == -1 (meaning "all steps"), we set step = 0 in practice 
    # and also decide to generate scenes, unless the user explicitly said "do not generate scenes."
    generatesScenes = False
    if step == -1:
        step = 0
        generatesScenes = True
    if donotgeneratesscenes or fusionpklfiles:
        generatesScenes = False
    if scene != -1:
        # If a scene is specified, we assume we do NOT generate all scenes.
        generatesScenes = False

    # If user said "onlyGeneratesScenes", override the logic to generate scenes, 
    # unless we are fusing or skipping as well.
    if onlyGeneratesScenes:
        generatesScenes = True

    # 3. Scene generation block
    if generatesScenes:
        # Create main directory if missing
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        # Detect scenes in the main video
        video_obj = open_video(videoFileName, backend='pyav')
        scene_manager = SceneManager()
        # You can switch out to AdaptiveDetector() if needed
        # scene_manager.add_detector(ContentDetector())
        scene_manager.add_detector(AdaptiveDetector(min_scene_len=10, window_width=5))

        print("Detecting scenes...")
        scene_manager.detect_scenes(video_obj, show_progress=True)

        scene_list = scene_manager.get_scene_list()
        if len(scene_list) == 0:
            # If no scenes detected, store entire video in a single "scene_000".
            print("No scenes detected, storing entire video as scene_000.")
            sceneDir = os.path.join(directory, "scene_000")
            if not os.path.exists(sceneDir):
                os.makedirs(sceneDir)
                print(f"Created directory: {sceneDir}")
            destinationPath = os.path.join(sceneDir, "video.mp4")
            shutil.copy(videoFileName, destinationPath)
        else:
            # If scenes are detected, split them using ffmpeg
            print(f"{len(scene_list)} scenes detected.")
            template = os.path.join(directory, "scene_$SCENE_NUMBER.mp4")
            print("Splitting video into scenes using ffmpeg...")

            # Splitting the video into scenes with no audio track (using -an)
            split_video_ffmpeg(
                input_video_path=videoFileName,
                scene_list=scene_list,
                output_file_template=template,
                show_progress=True,
                arg_override='-map 0:v -c:v libx264 -preset fast -crf 18 -an'
            )

            # After splitting, reorganize files into directories scene_XXX/video.mp4
            for i in range(len(scene_list)):
                sceneFileName = template.replace("$SCENE_NUMBER", f'{i+1:03d}')
                sceneDir = os.path.splitext(sceneFileName)[0]
                destinationPath = os.path.join(sceneDir, "video.mp4")

                if not os.path.exists(sceneDir):
                    os.makedirs(sceneDir)
                    print(f"Created directory: {sceneDir}")
                # Move the .mp4 to the final location
                shutil.move(sceneFileName, destinationPath)

    # 4. Processing block (calls external 'premiereProcessing.py') if not only generating scenes
    if not onlyGeneratesScenes and not fusionpklfiles:
        # If user did not specify a particular scene, we process all
        if scene == -1:
            scene_dirs = [
                d for d in os.listdir(directory)
                if os.path.isdir(os.path.join(directory, d)) and d.startswith("scene_")
            ]
            scene_dirs.sort()

            print("All scene directories:", scene_dirs)
            for sdir in scene_dirs:
                sceneDir = os.path.join(directory, sdir)
                scnVideoFileName = os.path.join(sceneDir, "video.mp4")
                if not os.path.exists(scnVideoFileName):
                    print(f"Warning: video.mp4 not found in {sceneDir}, skipping.")
                    continue

                command = [
                    "python", "premiereProcessing.py",
                    "--directory", sceneDir,
                    "--video", scnVideoFileName,
                    "--fov", str(args.fov),
                    "--rbfkernel", args.rbfkernel,
                    "--rbfsmooth", str(args.rbfsmooth),
                    "--rbfepsilon", str(args.rbfepsilon),
                    "--step", str(args.step),
                    "--batchsize", str(args.batchsize),
                    "--removeshadowsthreshold", str(args.removeshadowsthreshold),
                    "--detectionthreshold", str(args.detectionthreshold),
                    "--framesampling", str(args.framesampling),
                    "--scalefactor", str(args.scalefactor),
                    "--camtoltranslation", str(args.camtoltranslation),
                    "--camtolrotation", str(args.camtolrotation),
                    "--camtolfov", str(args.camtolfov)
                ]

                if args.displaymode:
                    command.append("--displaymode")
                if args.handestimation:
                    command.append("--handestimation")
                if args.removeshadows:
                    command.append("--removeshadows")

                print("Running:", " ".join(command))
                result = subprocess.run(command)
                if result.returncode != 0:
                    print(f"Error running premiereProcessing for {sceneDir}. Return code: {result.returncode}")
                    sys.exit(1)

        else:
            # Process a single user-specified scene
            sceneDirName = f"scene_{scene:03d}"
            sceneDir = os.path.join(directory, sceneDirName)
            scnVideoFileName = os.path.join(sceneDir, "video.mp4")

            if not os.path.exists(sceneDir):
                print(f"Error: Scene directory {sceneDir} does not exist.")
                sys.exit(1)
            if not os.path.exists(scnVideoFileName):
                print(f"Error: video.mp4 not found in {sceneDir}.")
                sys.exit(1)

            command = [
                "python", "premiereProcessing.py",
                "--directory", sceneDir,
                "--video", scnVideoFileName,
                "--fov", str(args.fov),
                "--rbfkernel", args.rbfkernel,
                "--rbfsmooth", str(args.rbfsmooth),
                "--rbfepsilon", str(args.rbfepsilon),
                "--step", str(args.step),
                "--batchsize", str(args.batchsize),
                "--removeshadowsthreshold", str(args.removeshadowsthreshold),
                "--detectionthreshold", str(args.detectionthreshold),
                "--framesampling", str(args.framesampling),
                "--scalefactor", str(args.scalefactor),
                "--camtoltranslation", str(args.camtoltranslation),
                "--camtolrotation", str(args.camtolrotation),
                "--camtolfov", str(args.camtolfov)
            ]

            if args.displaymode:
                command.append("--displaymode")
            if args.handestimation:
                command.append("--handestimation")
            if args.removeshadows:
                command.append("--removeshadows")

            print("Running:", " ".join(command))
            result = subprocess.run(command)
            if result.returncode != 0:
                print(f"Error running premiereProcessing for {sceneDir}. Return code: {result.returncode}")
                sys.exit(1)

    # 5. If not processing but summarizing frames or fusing PKLs
    else:
        scene_dirs = [
            d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d)) and d.startswith("scene_")
        ]
        scene_dirs.sort()

        totalFrameNumber = 0
        for sdir in scene_dirs:
            sceneDir = os.path.join(directory, sdir)
            scnVideoFileName = os.path.join(sceneDir, "video.mp4")
            if not os.path.exists(scnVideoFileName):
                print(f"Warning: video.mp4 not found in {sceneDir}, skipping frames count.")
                continue
            frameNumber = getVideoFrameNumber(scnVideoFileName)
            totalFrameNumber += frameNumber
            print(f"Scene video file name: {scnVideoFileName}, Frames: {frameNumber}")

        referenceFrameNumber = getVideoFrameNumber(args.video)
        print(f"Reference video frame number: {referenceFrameNumber}")
        print(f"Total frames across all scenes: {totalFrameNumber}")

        # Fusion step if requested
        if fusionpklfiles:
            saveFusion(directory, scene_dirs)

    # 6. Print total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time of premiereFullVideoProcessing: {execution_time:.2f} seconds")

    # 7. Breakdown of execution time in hours/minutes/seconds
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")


if __name__ == "__main__":
    main()

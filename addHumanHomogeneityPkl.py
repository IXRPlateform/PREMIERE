"""
Add homogeneity scores to detected humans in a pickle file based on depth and color variance.

This script processes:
1. A pickle file containing human detections per frame.
2. A depth video and its associated pickle file with min/max depth values per frame.
3. A segmentation video indicating object regions in each frame.
4. An input video for color data.

For each human in the segmentation video, the script calculates a "homogeneity score" based on:
- Variance of depth values within the region.
- Variance of color values within the region.

The results are saved back into the input pickle file under the 'homogeneity' field for each human.

**Usage**:
    python addHumanHomogeneityPkl.py <input_pkl> <input_depth_video> <input_seg_video> <input_depth_pkl> <input_video> <output_pkl> <display: 0 No, 1 Yes>

:param input_pkl: Path to the input pickle file containing detected humans.
:param input_depth_video: Path to the depth video file.
:param input_seg_video: Path to the segmentation video file.
:param input_depth_pkl: Path to the depth metadata pickle file (min/max depth values per frame).
:param input_video: Path to the RGB video file.
:param output_pkl: Path to the output pickle file to save updated human data.
:param display: 0 or 1 (whether to visualize frames during processing).
"""

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.stats import median_abs_deviation
from numpy import linalg
from tqdm import tqdm

# Import common functions
from premiere.functionsCommon import loadPkl, savePkl
from premiere.functionsMoge import colorizeDepthImage
from premiere.functionsDepth import unpackDepthImage

def main():
    """Main function to add homogeneity scores to detected humans."""
    # Validate command-line arguments
    if len(sys.argv) != 8:
        print("Usage: python addHumanHomogeneityPkl.py <input_pkl> <input_depth_video> <input_seg_video> <input_depth_pkl> <input_video> <output_pkl> <display: 0 No, 1 Yes>")
        sys.exit(1)

    # Parse arguments
    inputPklName = sys.argv[1]
    videoInDepthName = sys.argv[2]
    videoInMaskName = sys.argv[3]
    videoInDepthPklName = sys.argv[4]
    videoInName = sys.argv[5]
    outputPklName = sys.argv[6]
    display = int(sys.argv[7]) == 1

    # Open input videos
    videoDepth = cv2.VideoCapture(videoInDepthName)
    videoMask = cv2.VideoCapture(videoInMaskName)
    video = cv2.VideoCapture(videoInName)

    # Load input pickle file containing human data using functionsCommon.loadPkl
    print("Reading input pickle:", inputPklName)
    dataPKL = loadPkl(inputPklName)
    allFrameHumans = dataPKL['allFrameHumans']

    # Load depth metadata pickle (min/max depth per frame) using functionsCommon.loadPkl
    print("Reading depth metadata pickle:", videoInDepthPklName)
    videoDepthPKL = loadPkl(videoInDepthPklName)

    # Retrieve video properties
    width = int(videoDepth.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoDepth.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(videoDepth.get(cv2.CAP_PROP_FPS))
    frames_count = int(videoDepth.get(cv2.CAP_PROP_FRAME_COUNT))

    # Validate video streams
    if not videoDepth.isOpened():
        print("[!] Error opening the depth video.")
        sys.exit(1)

    if not videoMask.isOpened():
        print("[!] Error opening the mask video.")
        sys.exit(1)

    if not video.isOpened():
        print("[!] Error opening the RGB video.")
        sys.exit(1)

    # Initialize progress bar
    pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

    try:
        count = 0
        while videoDepth.isOpened():
            retDepth, frameDepth = videoDepth.read()
            retMask, frameMask = videoMask.read()
            ret, frame = video.read()

            if not retDepth or not retMask or not ret:
                break

            max_value_in_frameMask = np.max(frameMask)

            # Get min/max depth for the current frame from the depth metadata pickle
            min_depth = videoDepthPKL[count][0]
            max_depth = videoDepthPKL[count][1]

            # Unpack depth and mask from the depth frame
            depth, mask = unpackDepthImage(frameDepth, min_depth, max_depth, brg2rgb=True)
            
            # Prepare visualization if display is enabled
            if display:
                visuDepth = colorizeDepthImage(depth)

            # Apply erosion to the segmentation mask
            kernel = np.ones((1, 1), np.uint8)
            erodedMask = cv2.erode(frameMask, kernel, iterations=1)
            maskSeg = erodedMask == 0

            # Update visualization if enabled
            if display:
                visuDepth[maskSeg] = 0

            # Process each segmented region in the current frame
            for i in range(1, max_value_in_frameMask + 1):
                # Check if the human index exists in current frame
                if i > len(allFrameHumans[count]):
                    print(f"Warning: Human index {i} exceeds number of humans in frame {count}")
                    continue
                    
                maskSegId = erodedMask[:, :, 0] == i  # Ensure the mask is 2D
                depth_values_in_maskSeg = depth[maskSegId]
                colors_in_maskSeg = frame[maskSegId]

                # Compute homogeneity score based on depth and color variance
                score = 0
                if depth_values_in_maskSeg.size > 0:
                    variance = np.var(depth_values_in_maskSeg)
                    varianceColor = np.var(colors_in_maskSeg, axis=0)
                    varianceColorNorm = linalg.norm(varianceColor) / 1000.0
                    score = variance * varianceColorNorm

                # Update the homogeneity score in the human detection data
                allFrameHumans[count][i - 1]['homogeneity'] = score

            # Display visualization if enabled
            if display:
                cv2.imshow('Depth Frame', visuDepth)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            count += 1
            pbar.update(1)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nError during processing: {e}")
    finally:
        pbar.close()

        # Save updated pickle with added homogeneity scores using functionsCommon.savePkl
        print(f"Saving updated pickle to {outputPklName}")
        savePkl(outputPklName, dataPKL)

        # Release video resources
        videoDepth.release()
        videoMask.release()
        video.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessed {count} frames. Added homogeneity scores to human detections.")

if __name__ == "__main__":
    main()

"""
Process a video using the MoGe model to extract depth information.

This script:
1. Takes a video file as input.
2. Processes each frame with the MoGe depth estimation model.
3. Generates a new video with packed depth frames.
4. Saves depth metadata to a pickle file.

Usage:
  python videoProcessMoge.py <input_video> <moge_pkl> <output_video> <output_pkl> <fov>
  or
  python videoProcessMoge.py <input_video> <moge_pkl> <output_video> <output_pkl> <fov> <fovfile>

:param input_video: Path to the input video file.
:param moge_pkl: Path to the MoGe pickle file with parameters.
:param output_video: Path for the output video with depth visualization.
:param output_pkl: Path for the output pickle with depth metadata.
:param fov: Field of view in degrees (0 to use average FOV from MoGe PKL).
:param fovfile: (Optional) Path to a pickle file with per-frame FOV values.
"""

import os
import sys
import cv2
import av
import numpy as np
import torch
import time
from tqdm import tqdm

# Import common functions
from premiere.functionsCommon import loadPkl, savePkl
from premiere.functionsMoge import initMoGeModel
from premiere.functionsDepth import packDepthImage, computeMinMaxDepth

models_path = os.environ["MODELS_PATH"]

def main():
    """Main function to process a video using MoGe model."""
    if len(sys.argv) != 6 and len(sys.argv) != 7:
        print("Usage: python videoProcessMoge.py <input_video> <moge_pkl> <output_video> <output_pkl> <fov>")
        print ("or")
        print("Usage: python videoProcessMoge.py <input_video> <moge_pkl> <output_video> <output_pkl> <fov> <fovfile>")
        sys.exit(1)

    # Parse command line arguments
    videoInName = sys.argv[1]
    mogePklName = sys.argv[2]
    videoOutName = sys.argv[3]
    outputPklName = sys.argv[4]
    fov_x_degrees = float(sys.argv[5])
    useDynamicFOV = False
    fovPKL = None
    
    # Load optional per-frame FOV data
    if len(sys.argv) == 7:
        fovfile = sys.argv[6]
        fovPKL = loadPkl(fovfile)
        useDynamicFOV = True

    # Load MoGe parameters using functionsCommon.loadPkl
    print("Reading PKL:", mogePklName)
    mogePKL = loadPkl(mogePklName)
          
    # Calculate average FOV if not specified
    if fov_x_degrees == 0:
        fov_x_degrees = 0
        for i in range(len(mogePKL)):
            fov_x_degrees += mogePKL[i]['fov_x_degrees']
        fov_x_degrees /= len(mogePKL)

    print("Fov_x:", fov_x_degrees)

    # Initialize MoGe model
    device_name = 'cuda'
    device, model = initMoGeModel(device_name)

    # Open input video
    video = cv2.VideoCapture(videoInName)
    if not video.isOpened():
        print('[!] Error opening the video')
        sys.exit(1)
        
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up output video container with PyAV
    outputContainer = av.open(videoOutName, mode='w')
    
    # Configure codec options for high-quality lossless encoding
    codec_options = {
        'lossless': '1',
        'preset': 'veryslow',
        'crf': '0',
        'threads': '12',
    }
    
    # Add output video stream with H.264 codec
    outputStream = outputContainer.add_stream('libx264rgb', options=codec_options)
    outputStream.width = width
    outputStream.height = height
    outputStream.pix_fmt = 'rgb24'  # Standard pixel format for libx264
    outputStream.thread_type = 'AUTO'
    
    # Verify stream is properly configured
    if not outputStream.codec_context.is_open:
        outputStream.codec_context.open()

    # Initialize data storage for depth metadata
    allFrameData = []
    count = 0

    # Progress bar for visualization
    pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)
    
    try:
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                # Convert frame to tensor
                image_tensor = torch.tensor(frame / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
                
                # Use dynamic FOV if available
                if useDynamicFOV:
                    fov_x_degrees = fovPKL[count]
                
                # Run MoGe inference
                output = model.infer(image_tensor, fov_x=fov_x_degrees)

                # Extract depth and mask
                depth = output['depth'].cpu().numpy()
                mask = output['mask'].cpu().numpy()

                # Compute min/max depth for normalization
                min_depth, max_depth = computeMinMaxDepth(depth, mask)
                
                # Store depth metadata
                data = [min_depth, max_depth]
                allFrameData.append(data)

                # Create colored depth visualization
                colorImage = packDepthImage(depth, mask, min_depth, max_depth)

                # Convert OpenCV image to AV frame
                outframe = av.VideoFrame.from_ndarray(colorImage, format='rgb24')
                outframe = outframe.reformat(format='rgb24')
                
                # Encode frame and write to output
                packets = outputStream.encode(outframe)
                for packet in packets:
                    outputContainer.mux(packet)
                
                count += 1
                pbar.update(1)
            else:
                break
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nError during video processing: {e}")
    finally:
        pbar.close()
        
        # Save depth metadata to pickle file using functionsCommon.savePkl
        savePkl(outputPklName, allFrameData)
        
        # Encode remaining frames
        packets = outputStream.encode(None)
        for packet in packets:
            outputContainer.mux(packet)
        
        # Clean up
        outputStream.close()
        outputContainer.close()
        video.release()
        
        print(f"\nProcessed {count} frames. Output saved to {videoOutName} and {outputPklName}.")

if __name__ == "__main__":
    main()

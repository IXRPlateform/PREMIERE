"""
Segment humans from a video using the SAM 2.1 model and annotations from a pickle file.

This script takes:
1. An input video file.
2. A pickle file (typically containing per-frame detections/annotations).
3. An output video mask filename (MP4 or similar).
4. An integer flag (0 or 1) indicating whether to display frames interactively.

**Usage**: 
    python sam21NLF.py <input_video> <input_pkl> <output_videomask> <display: 0 No, 1 Yes>

Example:
    python sam21NLF.py ../videos/example.mp4 ./detections.pkl ./masksOutput.mp4 1

:param input_video: Path to the input video file.
:param input_pkl: Path to the input pickle file containing 'allFrameHumans' data.
:param output_videomask: Path to the output masked video file.
:param display: 0 or 1; if 1, frames with masks overlaid will be displayed during processing.

The script:
1. Loads a SAM 2.1 model from state_dict.
2. Reads a video frame by frame.
3. For each frame, retrieves human detections from the pickle file.
4. Attempts to generate a segmentation mask for each detection using the SAM model.
5. Renders a combined mask where each detected human is assigned a unique (random) color.
6. Optionally displays the processed frame with alpha-blended masks.
7. Writes the mask frames as a video using PyAV (libx264).
"""

import sys
import cv2
import av
import os
import torch
import numpy as np
import random

from tqdm import tqdm
from sam2.make_sam import make_sam_from_state_dict
from sam2.demo_helpers.shared_ui_layout import make_hires_mask_uint8
from sam2.demo_helpers.video_data_storage import SAM2VideoObjectResults
from collections import defaultdict

# Import common functions
from premiere.functionsCommon import loadPkl, savePkl

models_path = os.environ["MODELS_PATH"]

def generate_random_colors(n):
    """
    Generate a dictionary of random RGB colors.

    :param n: Number of distinct colors to generate.
    :type n: int
    :return: A dictionary where keys are stringified indices, and values are (R, G, B) tuples.
    :rtype: dict
    """
    colors = {}
    for i in range(n):
        colors[str(i)] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return colors

def main():
    """
    Main function to segment humans from a video based on a pickle of per-frame detections.

    This function:
      1. Parses command line arguments (input_video, input_pkl, output_videomask, display).
      2. Loads the input video with OpenCV.
      3. Loads the per-frame detection data from the input pickle (expects 'allFrameHumans').
      4. Creates an output video container using PyAV with H.264 encoding.
      5. Loads and prepares the SAM 2.1 model on CPU or GPU if available.
      6. Iterates through video frames, for each frame:
         a) Reads detection info.
         b) Initializes or updates segmentation masks in the SAM model.
         c) Applies each mask with a distinct color to the frame.
         d) Optionally displays the masked frame.
         e) Writes the mask to the output video.
      7. Cleans up and closes the output streams when done or on exception.

    Command line usage:
        python sam21NLF.py <input_video> <input_pkl> <output_videomask> <display: 0 or 1>
    """
    if len(sys.argv) != 5:
        print("Usage: python sam21NLF.py <input_video> <input_pkl> <output_videomask> <display: 0 No, 1 Yes>")
        sys.exit(1)

    # Parse command line arguments
    videoInName = sys.argv[1]
    input_pkl = sys.argv[2]
    videoMaskName = sys.argv[3]
    display = (int(sys.argv[4]) == 1)

    # Load the pickle file using functionsCommon.loadPkl
    print(f"Reading pickle file: {input_pkl}")
    dataPKL = loadPkl(input_pkl)
    allFrameHumans = dataPKL['allFrameHumans']

    # Open the video for reading
    video = cv2.VideoCapture(videoInName)
    if not video.isOpened():
        print('[!] Error opening video.')
        sys.exit(1)

    # Retrieve video properties
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Prepare an output container with PyAV
    outputVideoContainer = av.open(videoMaskName, mode='w')
    # Codec options for high-quality (or lossless) encoding using libx264
    codec_options = {
        'lossless': '1',
        'preset': 'veryslow',
        'crf': '0',
        'threads': 'auto',
    }
    outputStream = outputVideoContainer.add_stream('libx264rgb', options=codec_options)
    outputStream.width = width
    outputStream.height = height
    outputStream.pix_fmt = 'rgb24'  
    outputStream.thread_type = 'AUTO'
    if not outputStream.codec_context.is_open:
        outputStream.codec_context.open()

    print(f"Output configuration: {outputStream.codec_context.options}")
    print(f"Output format: {outputVideoContainer.format.name}")
    print('[+] Processing video...\n')

    # Progress bar to visualize progress
    pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

    # Determine device (GPU if available, else CPU) and dtype
    device, dtype = ("cpu", torch.float32)
    if torch.cuda.is_available():
        device, dtype = ("cuda", torch.bfloat16)

    # Load the SAM 2.1 model from the provided state_dict
    model_name = os.path.join(models_path, 'sam2', 'sam2.1_hiera_large.pt')
    print("Loading model...")
    model_config_dict, sammodel = make_sam_from_state_dict(model_name)

    print("Setting up SAM model...")
    sammodel.to(device=device, dtype=dtype)

    # Prepare color palette for segmentation
    white = (255, 255, 255)
    random.seed(42)
    colors = generate_random_colors(256)

    # Dict to store per-object memory for video segmentation 
    # (i.e., to help track objects across frames)
    memory_per_obj_dict = defaultdict(SAM2VideoObjectResults.create)

    # Configuration for image encoding (resizing, etc.) in SAM
    imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}

    try:
        count = 0
        # Read frames until video is exhausted
        while video.isOpened():
            ret, frame = video.read()
            if not ret or count >= len(allFrameHumans):
                # End of video or no more annotations
                break

            humans = allFrameHumans[count]
            points = []
            colorsStr = []

            # For each human, pick a single point that lies within the video frame 
            # (e.g., the 'loc' or a joint) to use as an initial segmentation prompt.
            for i in range(len(humans)):
                human = humans[i]
                point = [
                    human['loc'][0]/width,
                    human['loc'][1]/height
                ]
                # Check if point lies in normalized [0,1] range
                if 0 < point[0] < 1 and 0 < point[1] < 1:
                    points.append(point)
                    colorsStr.append(str(i))
                else:
                    # If main 'loc' was out of frame, try searching among keypoints
                    for j in range(len(human['j2d_smplx'])):
                        point = [
                            human['j2d_smplx'][j][0]/width,
                            human['j2d_smplx'][j][1]/height
                        ]
                        if 0 < point[0] < 1 and 0 < point[1] < 1:
                            points.append(point)
                            colorsStr.append(str(i))
                            break

            # Encode the frame for use in the SAM pipeline
            encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)

            # Initialize the combined mask images for each object
            combined_mask_result = np.zeros(frame.shape, dtype=np.uint8)
            combined_mask_id_result = np.zeros(frame.shape, dtype=np.uint8)

            if points:
                # Re-initialize memory if needed for each frame
                memory_per_obj_dict = defaultdict(SAM2VideoObjectResults.create)

                for i, point in enumerate(points):
                    obj_prompts = {
                        'box_tlbr_norm_list': [],
                        'fg_xy_norm_list': [point],
                        'bg_xy_norm_list': []
                    }
                    obj_key_name = colorsStr[i]
                    init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(
                        encoded_imgs_list, 
                        **obj_prompts,
                        mask_index_select=2
                    )
                    memory_per_obj_dict[obj_key_name].store_prompt_result(count, init_mem, init_ptr)

                # For each object memory entry, compute or update masks
                for obj_key_name, obj_memory in memory_per_obj_dict.items():
                    obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
                        encoded_imgs_list, 
                        **obj_memory.to_dict()
                    )

                    # obj_score < 0 often indicates a bad or missing detection
                    obj_score = obj_score.item()
                    if obj_score < 0:
                        print(f"Bad object score for {obj_key_name}! Skipping memory storage...")
                        continue

                    # Store memory to help with tracking as the video progresses
                    obj_memory.store_result(count, mem_enc, obj_ptr)

                    # Interpolate the mask predictions to the full frame resolution
                    obj_mask = torch.nn.functional.interpolate(
                        mask_preds[:, best_mask_idx, :, :],
                        size=combined_mask_result.shape[:2],
                        mode="bilinear"
                    )
                    obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()

                    color = colors[obj_key_name]
                    obj_id = int(obj_key_name)

                    combined_mask_result[obj_mask_binary] = color
                    combined_mask_id_result[obj_mask_binary] = obj_id + 1  # zero is background

            # Optional display
            if display:
                # For visualization, overlay the mask on the original frame
                mask = np.any(combined_mask_result != [0, 0, 0], axis=-1).astype(np.uint8)
                segmentedFrame = frame.copy()
                alpha = 0.5
                # Blend colors into the original frame
                segmentedFrame[mask == 1] = cv2.addWeighted(
                    frame[mask == 1], 
                    1 - alpha, 
                    combined_mask_result[mask == 1], 
                    alpha,
                    0
                )

                # Annotate each human with an index
                for i in range(len(humans)):
                    human = humans[i]
                    cv2.putText(
                        segmentedFrame,
                        str(i),
                        (int(human['loc'][0]), int(human['loc'][1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        white,
                        2,
                        cv2.LINE_AA
                    )

                cv2.imshow('mask', segmentedFrame)
                cv2.waitKey(5)

            # Convert the mask to a PyAV VideoFrame and write
            outframe = av.VideoFrame.from_ndarray(combined_mask_id_result, format='rgb24')
            outframe = outframe.reformat(format='rgb24')
            packets = outputStream.encode(outframe)
            for packet in packets:
                outputVideoContainer.mux(packet)

            count += 1
            pbar.update(1)

    except KeyboardInterrupt:
        # If interrupted, exit gracefully
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup
        pbar.close()
        video.release()
        
        # Encode any remaining frames
        packets = outputStream.encode(None)
        for packet in packets:
            outputVideoContainer.mux(packet)
        outputStream.close()
        outputVideoContainer.close()

if __name__ == "__main__":
    main()

# Example commands:
# python sam21NLF.py ../videos/D0-talawa_technique_intro-Scene-021.mp4 ./pkl/D0-21.pkl ./result.mp4 1
# python sam21NLF.py F:/MyDrive/Tmp/Tracking/T3-2-1024.MP4 ./pkl/yolo-T3-2-1024.pkl ./resultSeg.mp4 1
# python sam21NLF.py ../videos/D6-5min-1-Scene-008.mp4 ./pkl/yolo-D6-008.pkl ./resultSeg-D6-008.mp4 0

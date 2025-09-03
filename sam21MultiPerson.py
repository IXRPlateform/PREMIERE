#!/usr/bin/env python
"""
This script processes a video by performing video segmentation on detected human poses.
It loads pose data from a pickle file, filters object tracks based on a minimum length and dispersion,
sets up a segmentation model (SAM 2.1), processes video frames (forward & backward in time),
and finally writes the output segmentation results to a video file while also saving the updated pose data.
"""

import pickle
import sys
import cv2
import os
import av
import numpy as np
import torch
from sam2.make_sam import make_sam_from_state_dict
from sam2.demo_helpers.video_data_storage import SAM2VideoObjectResults
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.distance import pdist

# Read the models path from environment variables
models_path = os.environ["MODELS_PATH"]


def initializeVideoMasking(allFrameHumans, index, frame_idx, memory_per_obj_dict, sammodel, reverse=False):
    """
    Initializes video masking for objects present in a given frame.
    
    For each human detected in the specified frame, this function:
      - Skips those with id -1 if 'reverse' mode is enabled.
      - Constructs an object prompt with normalized bounding-box coordinates.
      - Calls the SAM model's initialize method to create initial mask and memory encoding.
      - Stores the resulting memory data (for future tracking).
    
    Args:
        allFrameHumans (list): List containing detection information for each frame.
        index (int): Frame index used for human selection.
        frame_idx (int): The current frame index being processed.
        memory_per_obj_dict (defaultdict): Dictionary mapping object keys to their stored memory and prompts.
        sammodel: The SAM model instance used for segmentation.
        reverse (bool): When True, filter out objects with id -1.
    """
    # Loop over each detected human in the specified frame index
    for human in range(len(allFrameHumans[index])):
        # In reverse mode, skip objects with id -1 (invalid objects)
        if reverse and allFrameHumans[index][human]['id'] == -1:
            continue

        # Retrieve normalized location for prompt
        locNorm = allFrameHumans[index][human]['locNorm']
        # Prepare the prompt dictionary for the SAM model; here only the top-left coordinate is used
        obj_prompts = {
            'box_tlbr_norm_list': [],
            'fg_xy_norm_list': [(locNorm[0], locNorm[1])],
            'bg_xy_norm_list': []
        }
        
        # Generate a key name for the object based on its index in the list
        obj_key_name = str(human)
        # Call the SAM model's initialization method for video masking.
        # This returns the initial mask prediction, memory encoding, and pointer.
        init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(encoded_imgs_list, **obj_prompts, mask_index_select=2)
        # Store the prompt results (memory and pointer) for the object for future steps.
        memory_per_obj_dict[obj_key_name].store_prompt_result(frame_idx, init_mem, init_ptr)


def processFrame(frame, frame_idx, encoded_imgs_list, memory_per_obj_dict, sammodel, colors):
    """
    Processes a single video frame to update tracking and segmentation.
    
    For every tracked object, the function:
      - Calls the SAM model step for video masking using stored memory.
      - Checks if the segmentation score is acceptable (ignores low-score results).
      - Updates the object's memory with the current frame's result.
      - Resizes and binarizes the predicted mask.
      - Compares the mask with human detections based on coordinates and assigns an object ID.
      - Composites a color-coded segmentation mask and a mask with object IDs.
    
    Args:
        frame (ndarray): The current video frame.
        frame_idx (int): The index number of the current frame.
        encoded_imgs_list: Preprocessed/encoded image representation for the SAM model.
        memory_per_obj_dict (defaultdict): The dictionary holding memory information for each tracked object.
        sammodel: The SAM model instance.
        colors (dict): A dictionary mapping object keys to BGR color tuples.
    
    Returns:
        tuple: A tuple containing:
            - combined_mask_id_result (ndarray): Segmentation mask with object IDs.
            - combined_mask_result (ndarray): Color-coded segmentation mask.
    """
    # Create empty arrays for the final combined mask and object ID mask
    combined_mask_result = np.zeros(frame.shape, dtype=np.uint8)
    combined_mask_id_result = np.zeros(frame.shape, dtype=np.uint8)
    
    # Process each object's stored memory (tracked object)
    for obj_key_name, obj_memory in memory_per_obj_dict.items():
        # Call the SAM model to perform one step of video masking for the object
        obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(encoded_imgs_list, **obj_memory.to_dict())
        
        # Skip processing if the object score is below threshold (likely due to occlusion or poor prediction)
        if obj_score.item() < 0:
            continue
        
        # Update the stored memory with current frame's results
        obj_memory.store_result(frame_idx, mem_enc, obj_ptr)
        
        # Resize the predicted mask to match the frame dimensions using bilinear interpolation
        obj_mask = torch.nn.functional.interpolate(
            mask_preds[:, best_mask_idx, :, :],
            size=combined_mask_result.shape[:2],
            mode="bilinear"
        )
        
        # Threshold the mask predictions to create a binary mask
        obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()
        
        # Convert the object key into an integer ID (used for matching with detections)
        idsam = int(obj_key_name)
        
        # Loop through all human detections in the current frame and assign the object id
        for j in range(len(allFrameHumans[frame_idx])):
            loc = allFrameHumans[frame_idx][j]['loc']
            x1, y1 = loc[0], loc[1]
            # Check if the detection coordinate lies within the mask area
            if 0 <= int(y1) < obj_mask_binary.shape[0] and 0 <= int(x1) < obj_mask_binary.shape[1]:
                if obj_mask_binary[int(y1), int(x1)]:
                    allFrameHumans[frame_idx][j]['idsam'] = idsam
                    break
        
        # Retrieve the color for the current object and update the combined mask results
        color = colors[obj_key_name]
        combined_mask_result[obj_mask_binary] = color
        combined_mask_id_result[obj_mask_binary] = idsam + 1   # Store object id (offset by 1)
    
    return combined_mask_id_result, combined_mask_result


# Ensure correct number of command-line arguments are provided
if len(sys.argv) < 9:
    print("Usage: python sam21MultiHMR.py <pkl_path> <video_path> <human> <track_size_min> <output_pkl_path> <output_video_path> <dispersion threshold> <display: 0 No, 1 Yes>")
    sys.exit(1)

# Parse command-line arguments
input_pkl_path = sys.argv[1]
video_path = sys.argv[2]
human = int(sys.argv[3])
trackSizeMin = int(sys.argv[4])
output_pkl_path = sys.argv[5]
output_video_path = sys.argv[6]
dispersionthreshold = float(sys.argv[7])
display = int(sys.argv[8]) == 1

# Load human pose data from the pickle file
print("Reading pose pickle:", input_pkl_path)
with open(input_pkl_path, 'rb') as file:
    dataPKL = pickle.load(file)

# Retrieve per-frame human detections
allFrameHumans = dataPKL['allFrameHumans']

# Determine the maximum number of humans detected and the maximum track ID across all frames
maxHumans = 0
maxId = 0
maxIndex = -1
for i in range(len(allFrameHumans)):
    currentHumans = len(allFrameHumans[i])
    if currentHumans > maxHumans:
        maxHumans = currentHumans
        maxIndex = i
    for human_info in allFrameHumans[i]:
        maxId = max(maxId, human_info['id'])

print('maxHumans:', maxHumans)
print('maxId:', maxId)
print('maxIndex:', maxIndex)

# Compute the size (length) of each track (i.e., number of frames where each ID appears)
tracksSize = np.zeros(maxId + 1, dtype=int)
for frame in allFrameHumans:
    for human_info in frame:
        tracksSize[human_info['id']] += 1
print('Tracks Size:', tracksSize)

# Create a list to hold the tracks; each track is a NumPy array containing [frame_index, human_index] pairs
tracks = []
for i in range(maxId + 1):
    tracks.append(np.zeros((tracksSize[i], 2), dtype=int))

# Create an array to track the current position (or count) for each track
tracksCurrentPosition = np.zeros(maxId + 1, dtype=int)
for i in range(len(allFrameHumans)):
    for j in range(len(allFrameHumans[i])):
        idToProcess = allFrameHumans[i][j]['id']
        tracks[idToProcess][tracksCurrentPosition[idToProcess]] = [i, j]
        tracksCurrentPosition[idToProcess] += 1

# Remove tracks that have fewer frames than the minimum threshold by marking them as invalid (-1)
for t in range(len(tracks)):
    if tracksSize[t] < trackSizeMin:
        for i in range(tracksSize[t]):
            frame_i, human_index = tracks[t][i]
            allFrameHumans[frame_i][human_index]['id'] = -1

# Normalize human locations based on video dimensions and remove detections outside the valid range
for i in range(len(allFrameHumans)):
    for human_info in allFrameHumans[i]:
        x, y = human_info['loc'][0], human_info['loc'][1]
        # Normalize by video dimensions
        x_norm = x / dataPKL['video_width']
        y_norm = y / dataPKL['video_height']
        human_info['locNorm'] = [x_norm, y_norm]
        # Invalidate detections that fall outside the normalized [0, 1] range
        if x_norm < 0 or y_norm < 0 or x_norm > 1 or y_norm > 1:
            human_info['id'] = -1

# Build a list of human locations (normalized) for each frame; ignore invalid detections (-1)
allHumansLoc = []
for frame in allFrameHumans:
    frame_loc = []
    for human_info in frame:
        if human_info['id'] == -1:
            continue
        x, y = human_info['loc'][0], human_info['loc'][1]
        x_norm = x / dataPKL['video_width']
        y_norm = y / dataPKL['video_height']
        frame_loc.append([x_norm, y_norm])
        human_info['locNorm'] = [x_norm, y_norm]
    allHumansLoc.append(frame_loc)

# Compute lengths (number of detections) per frame for later reference
lengths = [len(frame) for frame in allFrameHumans]
maxLengths = max(lengths)
print('maxLengths:', maxLengths)
print('maxLengthsIndex:', lengths.index(maxLengths))

# Determine the frame with the maximum valid humans that meets a dispersion condition.
# The dispersion condition checks whether the minimum distance between any two human detections
# in the frame is larger than a given threshold.
for i in range(maxIndex, len(allHumansLoc)):
    if len(allHumansLoc[i]) == maxHumans and maxHumans > 0:
        coords = np.array(allHumansLoc[i])  # Shape: (N, 2)
        if len(coords) > 1:
            distances = pdist(coords)
            min_dist = np.min(distances)
        else:
            min_dist = float('inf')  # Single detection: no dispersion issue
        if min_dist > dispersionthreshold:
            maxIndex = i
            break

if maxIndex == -1:
    print("No frame satisfies the conditions (maxHumans present and heads sufficiently separated).")
maxIndex += 1  # Include the frame found
print("Optimal maxIndex found:", maxIndex)

# Reset SAM detection IDs to -1 for all human detections in every frame
for frame in allFrameHumans:
    for human_info in frame:
        human_info['idsam'] = -1

# Configuration for image encoding (shared across frames)
imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}

# Set up a dictionary to store prompt and recent memory data for each tracked object.
# Memory data is stored using a defaultdict with SAM2VideoObjectResults as the default factory.
memory_per_obj_dict = defaultdict(SAM2VideoObjectResults.create)

# Define a color mapping for each object id (represented as strings)
colors = {
    "0": (255, 0, 255),
    "1": (0, 255, 0),
    "2": (0, 0, 255),
    "3": (255, 0, 0),
    "4": (255, 255, 0),
    "5": (0, 255, 255),
    "6": (255, 0, 255),
    "7": (0, 128, 0),
    "8": (0, 0, 128),
    "9": (128, 0, 0),
    "10": (255, 255, 255),
    "11": (0, 255, 0),
    "12": (0, 0, 255),
    "13": (255, 0, 0),
    "14": (255, 255, 0),
    "15": (0, 255, 255),
    "16": (255, 0, 255),
    "17": (0, 128, 0),
    "18": (0, 0, 128),
    "19": (128, 0, 0),
}

# Open the video to verify availability of frames and get video properties.
vcap = cv2.VideoCapture(video_path)
ok_frame, first_frame = vcap.read()
if not ok_frame:
    raise IOError(f"Unable to read video frames: {video_path}")

vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fps = vcap.get(cv2.CAP_PROP_FPS)
width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_size = (first_frame.shape[1], first_frame.shape[0])
print("total_frames:", total_frames)
print("frame_size:", frame_size)

# Load the SAM 2.1 model and move it to the appropriate device
print("Loading model...")
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

model_name = os.path.join(models_path, 'sam2', 'sam2.1_hiera_large.pt')
model_config_dict, sammodel = make_sam_from_state_dict(model_name)
sammodel.to(device=device, dtype=dtype)

# Set up the output video container using av and configure the codec parameters.
outputVideoContainer = av.open(output_video_path, mode='w')
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

# Lists to store previous frames and combined mask results for output video.
previousFrames = []
combined_mask_results = []
combined_id_mask_results = []

# Initialize progress bar for frame processing.
pbar = tqdm(total=total_frames, unit=' frames', dynamic_ncols=True, position=0, leave=True)

# ---------------- Process initial frames ---------------- #
# Read frames up to maxIndex+1 and store them (to later process in reverse order)
for frame_idx in range(maxIndex + 1):
    ok_frame, frame = vcap.read()
    previousFrames.append(frame)

# Process frames in reverse order (from maxIndex back to frame 0)
for frame_idx in reversed(range(maxIndex + 1)):
    frame = previousFrames[frame_idx]
    # Encode the frame for the SAM model using shared image encoding configuration
    encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)
    
    # On a particular frame condition, initialize video masking in reverse mode
    if (maxIndex - 1) == frame_idx:
        initializeVideoMasking(allFrameHumans, maxIndex, frame_idx, memory_per_obj_dict, sammodel, reverse=True)
    
    # Process the current frame to obtain segmentation results
    combined_mask_id_result, combined_mask_result = processFrame(frame, frame_idx, encoded_imgs_list, memory_per_obj_dict, sammodel, colors)
    combined_mask_results.append(combined_mask_result)
    combined_id_mask_results.append(combined_mask_id_result)
    
    # If display is enabled, show the original frame and the mask side by side
    if display:
        sidebyside_frame = np.hstack((frame, combined_mask_result))
        sidebyside_frame = cv2.resize(sidebyside_frame, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow("Video Segmentation Result - q to quit", sidebyside_frame)
        cv2.waitKey(1)
    
    pbar.update(1)

# Write the reversed frames with segmentation results to the output video
for frame_idx in reversed(range(len(combined_mask_results))):
    outframe = av.VideoFrame.from_ndarray(combined_id_mask_results[frame_idx], format='rgb24')
    outframe = outframe.reformat(format='rgb24')
    packets = outputStream.encode(outframe)
    for packet in packets:
        outputVideoContainer.mux(packet)

# ---------------- Process remaining frames ---------------- #
for frame_idx in range(maxIndex + 1, total_frames):
    ok_frame, frame = vcap.read()
    if not ok_frame:
        break
    
    # Encode the frame for segmentation
    encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)
    
    # Initialize video masking prompts at specific frames
    if ((maxIndex - 1) == frame_idx) or (frame_idx == 1):
        initializeVideoMasking(allFrameHumans, maxIndex, frame_idx, memory_per_obj_dict, sammodel)
    
    # Process the current frame to update segmentation and tracking
    combined_mask_id_result, combined_mask_result = processFrame(frame, frame_idx, encoded_imgs_list, memory_per_obj_dict, sammodel, colors)
    
    # Write the segmentation result frame to the output video
    outframe = av.VideoFrame.from_ndarray(combined_mask_id_result, format='rgb24')
    outframe = outframe.reformat(format='rgb24')
    packets = outputStream.encode(outframe)
    for packet in packets:
        outputVideoContainer.mux(packet)
    
    # If display is enabled, show a side-by-side view of the original frame and segmentation
    if display:
        sidebyside_frame = np.hstack((frame, combined_mask_result))
        sidebyside_frame = cv2.resize(sidebyside_frame, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow("Video Segmentation Result - q to quit", sidebyside_frame)
        cv2.waitKey(1)
    
    pbar.update(1)
pbar.close()

# Save the updated pose data (including segmentation IDs) into an output pickle file
with open(output_pkl_path, 'wb') as handle:
    pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Cleanup: flush final packets, release resources, and destroy display windows.
packets = outputStream.encode(None)
for packet in packets:
    outputVideoContainer.mux(packet)
vcap.release()
cv2.destroyAllWindows()

#!/usr/bin/env python
"""
This script processes an input video to estimate camera poses using a VGGT model.
It performs the following steps:
  1. Loads and preprocesses video frames.
  2. Loads a pretrained VGGT model.
  3. Processes frames in sliding windows to estimate extrinsic (pose) and intrinsic matrices.
  4. Aligns the local poses into a global camera trajectory.
  5. Computes fields of view (FOV) for each frame.
  6. Computes relative camera positions/rotations.
  7. Exports a GLB file visualizing the camera trajectory.
  8. Saves the camera pose data into a pickle file.
  
Usage:
    python video_camera_poses.py <input_video> <frame_sampling> <output_vggt_pkl> <fov>
"""

import os
import sys
import cv2
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import trimesh
from tqdm import tqdm

from PIL import Image
from torchvision import transforms as TF
from scipy.spatial.transform import Rotation as R

# Import the VGGT model and related utilities
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from colorama import init, Fore, Style
init(autoreset=True)

# Get the models path from environment variable
models_path = os.environ["MODELS_PATH"]


def get_available_cuda_memory_gb():
    """
    Returns the available (free) memory on the current CUDA device in gigabytes (GB).
    Uses torch.cuda.memory.get_memory_info if available, otherwise computes based on reserved memory.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.cuda.current_device()
    try:
        mem_info = torch.cuda.memory.get_memory_info(device)
        free_memory = mem_info.free / (1024 ** 3)  # Convert bytes to GB
    except AttributeError:
        device_props = torch.cuda.get_device_properties(device)
        total_memory = device_props.total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        free_memory = total_memory - reserved

    return free_memory

def compute_fov_degree(intrinsic_matrix, width, height):
    """
    Computes the horizontal, vertical, and diagonal fields of view (in degrees)
    given a camera intrinsic matrix and image resolution.

    Parameters:
      intrinsic_matrix (np.array): 3x3 camera intrinsic matrix.
      width (int): Image width in pixels.
      height (int): Image height in pixels.

    Returns:
      Tuple[float, float, float]: (horizontal FOV, vertical FOV, diagonal FOV) in degrees.
    """
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    fov_horizontal = 2 * np.arctan((width / 2) / fx) * (180 / np.pi)
    fov_vertical = 2 * np.arctan((height / 2) / fy) * (180 / np.pi)
    diagonal = np.sqrt(width**2 + height**2)
    focal_avg = (fx + fy) / 2
    fov_diagonal = 2 * np.arctan((diagonal / 2) / focal_avg) * (180 / np.pi)

    return fov_horizontal, fov_vertical, fov_diagonal


def compute_fov(intrinsic_matrix, width, height):
    """
    Computes the horizontal, vertical, and diagonal fields of view (in radians)
    given a camera intrinsic matrix and image resolution.

    Parameters:
      intrinsic_matrix (np.array): 3x3 camera intrinsic matrix.
      width (int): Image width in pixels.
      height (int): Image height in pixels.

    Returns:
      Tuple[float, float, float]: (horizontal FOV, vertical FOV, diagonal FOV) in radians.
    """
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    fov_horizontal = 2 * np.arctan((width / 2) / fx)
    fov_vertical = 2 * np.arctan((height / 2) / fy)
    diagonal = np.sqrt(width**2 + height**2)
    focal_avg = (fx + fy) / 2
    fov_diagonal = 2 * np.arctan((diagonal / 2) / focal_avg)

    return fov_horizontal, fov_vertical, fov_diagonal



# def add_frame(imgs, frame):
#     """
#     Preprocess a video frame:
#       - Convert BGR (OpenCV) to RGB.
#       - Convert to a PIL image.
#       - Resize the image to have a fixed width of 518 while keeping the aspect ratio.
#       - Adjust the height to be divisible by 14 and crop vertically if needed.
#     The processed image is appended to the list 'imgs'.
#     """
#     # Convert BGR to RGB and then to a PIL Image
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(frame_rgb)

#     width, height = img.size
#     new_width = 518
#     # Maintain aspect ratio; ensure new height is divisible by 14
#     new_height = round(height * (new_width / width) / 14) * 14
#     img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

#     # Crop vertically if the new height is greater than 518
#     if new_height > 518:
#         start_y = (new_height - 518) // 2
#         img = img.crop((0, start_y, new_width, start_y + 518))

#     imgs.append(img)


def add_frame(imgs, frame):
    """
    Preprocess a video frame:
      - Convert BGR (OpenCV) to RGB.
      - Convert to a PIL image.
      - Resize the image to have a fixed width of 518 while keeping the aspect ratio.
      - Adjust the height to be divisible by 14 and crop vertically if needed.
    The processed image is appended to the list 'imgs'.
    """
    # Convert BGR to RGB and then to a PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    width, height = img.size
    target_size = 518

    if width >= height:  # Landscape or square orientation
        new_width = target_size
        # Maintain aspect ratio; ensure new height is divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        # Crop vertically if the new height is greater than target_size
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img.crop((0, start_y, new_width, start_y + target_size))
    else:  # Portrait orientation
        new_height = target_size
        # Maintain aspect ratio; ensure new width is divisible by 14
        new_width = round(width * (new_height / height) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        
        # Crop horizontally if the new width is greater than target_size
        if new_width > target_size:
            start_x = (new_width - target_size) // 2
            img = img.crop((start_x, 0, start_x + target_size, new_height))

    imgs.append(img)


def align_camera_poses(all_extrinsics, extrinsics_references, window_size, window_overlap):
    """
    Align camera poses estimated locally within sliding windows to obtain a global camera trajectory.
    
    Parameters:
      all_extrinsics (np.array): Array of shape (N, 3, 4) with local extrinsic matrices.
      extrinsics_references (np.array): Array of shape (num_windows, 3, 4) with reference extrinsics.
      window_size (int): Number of frames per window.
      window_overlap (int): Number of overlapping frames between consecutive windows.
    
    Returns:
      Tuple[np.array, np.array]: (rotations_global, positions_global) where rotations are 3x3 matrices
                                 and positions are 3D vectors.
    """
    rotations_global = []
    positions_global = []
    global_poses = []  # Will hold 4x4 homogeneous transformation matrices

    # def to_homogeneous(ext):
    #     """Convert a 3x4 extrinsic matrix to a 4x4 homogeneous matrix."""
    #     return np.vstack([ext, np.array([0, 0, 0, 1], dtype=ext.dtype)])

    num_windows = extrinsics_references.shape[0]
    if num_windows < 1:
        return np.array(rotations_global), np.array(positions_global)

    total_frames = window_size + (num_windows - 1) * (window_size - window_overlap)
    idx = 0  # Index in all_extrinsics

    # Process first window (no alignment needed)
    count_window0 = window_size if len(all_extrinsics) >= window_size else len(all_extrinsics)
    for i in range(count_window0):
        # T_local = to_homogeneous(all_extrinsics[idx])
        T_local = all_extrinsics[idx]
        global_poses.append(T_local)
        idx += 1

    # Process subsequent windows with alignment using overlapping frames
    for w in range(1, num_windows):
        count_window = window_size - window_overlap
        if idx + count_window > len(all_extrinsics):
            count_window = len(all_extrinsics) - idx

        global_start = window_size + (w - 1) * (window_size - window_overlap)
        ref_global_idx = global_start + (window_overlap - 1)
        # ref_global_idx = w * (window_size - window_overlap) + window_overlap - 1
        if ref_global_idx >= len(global_poses):
            ref_global_idx = len(global_poses) - 1

        T_global_ref = global_poses[ref_global_idx]
        T_local_ref = extrinsics_references[w]
        # T_local_ref = to_homogeneous(extrinsics_references[w])
        T_align = T_global_ref @ np.linalg.inv(T_local_ref)

        for i in range(count_window):
            T_local = all_extrinsics[idx]
            # T_local = to_homogeneous(all_extrinsics[idx])
            T_global = T_align @ T_local
            global_poses.append(T_global)
            idx += 1

    # Trim global poses to match expected total frames
    global_poses = global_poses[:total_frames]
    for T in global_poses:
        # print (T)
        rotations_global.append(T[:3, :3])
        positions_global.append(T[:3, 3])

    return np.array(rotations_global), np.array(positions_global)


def extract_intrinsics(all_intrinsics, window_size, window_overlap, total_frames):
    """
    Extract intrinsic matrices from per-window intrinsics, handling the overlapping structure.

    Parameters:
      all_intrinsics (np.array): Array of shape (N, 3, 3) with intrinsic matrices.
      window_size (int): Number of frames per window.
      window_overlap (int): Number of overlapping frames between windows.
      total_frames (int): Total number of frames processed.
    
    Returns:
      np.array: Array of intrinsic matrices for each processed frame.
    """
    intrinsics_global = []
    idx = 0

    # Process first window directly
    count_window0 = window_size if len(all_intrinsics) >= window_size else len(all_intrinsics)
    for i in range(count_window0):
        if idx < len(all_intrinsics):
            intrinsics_global.append(all_intrinsics[idx].copy())
            idx += 1

    # Process subsequent windows while skipping redundant (overlapping) frames
    while idx < len(all_intrinsics):
        count_window = window_size - window_overlap
        if idx + count_window + window_overlap > len(all_intrinsics):
            count_window = max(0, len(all_intrinsics) - idx - window_overlap)

        # Skip overlapping frames
        idx += min(window_overlap, len(all_intrinsics) - idx)
        for j in range(count_window):
            if idx < len(all_intrinsics):
                intrinsics_global.append(all_intrinsics[idx].copy())
                idx += 1

    if len(intrinsics_global) > total_frames:
        intrinsics_global = intrinsics_global[:total_frames]

    return np.array(intrinsics_global)


def export_camera_trajectory_glb(rotations, positions, output_filename, pyramid_scale=1.0):
    """
    Exports a GLB file visualizing the camera trajectory.
    Each camera is represented as a wireframe pyramid with a rainbow gradient color from first to last frame.
    
    Parameters:
      rotations (np.array): List of 3x3 rotation matrices.
      positions (np.array): List of 3D position vectors.
      output_filename (str): File path for the output GLB.
      pyramid_scale (float): Scale factor for the pyramid size.
    """
    # Correction des axes pour la visualisation
    def fix_coordinate_system(rotations, positions):
        """
        Corrige l'orientation des caméras pour la visualisation 3D.
        Inverse les axes X et Y et ajuste les rotations en conséquence.
        """
        corrected_positions = positions.copy()
        corrected_rotations = rotations.copy()
        
        # Matrice de transformation pour inverser les axes X et Y
        transform = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        
        # Appliquer la transformation aux positions
        corrected_positions = corrected_positions @ transform
        
        # Ajuster les matrices de rotation
        for i in range(len(corrected_rotations)):
            corrected_rotations[i] = transform @ corrected_rotations[i]
        
        return corrected_rotations, corrected_positions
    
    # Appliquer la correction des axes
    # corrected_rotations, corrected_positions = fix_coordinate_system(rotations, positions)
    
    corrected_rotations = rotations
    corrected_positions = positions


    # Function to generate rainbow colors based on parameter t (0 to 1)
    def rainbow_color(t):
        """
        Returns RGB color from rainbow spectrum based on t (0 to 1).
        """
        # Simple HSV to RGB conversion for rainbow effect
        h = t  # hue (0 to 1)
        s = 1.0  # saturation
        v = 1.0  # value
        
        # HSV to RGB conversion
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        return [r + m, g + m, b + m, 1.0]  # RGBA format

    # Define a local wireframe pyramid (camera frustum) with 5 vertices.
    width = 0.1 * pyramid_scale   # half-width of the base
    height = 0.075 * pyramid_scale  # half-height of the base
    length = 0.2 * pyramid_scale   # distance from apex to base

    local_vertices = np.array([
        [0.0,  0.0,   0.0],           # apex (camera center)
        [-width, -height, -length],   # bottom-left
        [ width, -height, -length],   # bottom-right
        [ width,  height, -length],   # top-right
        [-width,  height, -length]     # top-left
    ])

    # Define edges connecting the vertices of the pyramid
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # from apex to base
        (1, 2), (2, 3), (3, 4), (4, 1)    # base perimeter
    ]

    # Create a scene to hold all geometries
    scene = trimesh.Scene()
    
    # Create individual pyramid for each camera position with appropriate color
    num_cameras = len(corrected_rotations)
    
    # Create trajectory line first (with gradient colors)
    positions_np = np.array(corrected_positions)
    if positions_np.shape[0] > 1:
        # Create path segments with individual colors
        for i in range(len(positions_np) - 1):
            t_val = i / max(1, num_cameras - 1)
            color = rainbow_color(t_val)
            
            # Create a segment
            segment = np.array([positions_np[i], positions_np[i+1]])
            path = trimesh.load_path([segment])
            
            # Set entity color
            path.entities[0].color = color
            
            # Add to scene
            scene.add_geometry(path, node_name=f"trajectory_segment_{i}")
    
    # Now add the camera pyramids
    for i, (R_mat, t) in enumerate(zip(corrected_rotations, corrected_positions)):
        # Calculate relative position in sequence (0 to 1)
        t_val = i / max(1, num_cameras - 1)
        # Get rainbow color for this camera
        color = rainbow_color(t_val)
        
        # Transform local vertices to global coordinates
        global_vertices = (R_mat @ local_vertices.T).T + t
        
        # Create individual line segments for this pyramid
        for start_idx, end_idx in edges:
            segment = np.array([global_vertices[start_idx], global_vertices[end_idx]])
            
            # Create a path for this segment
            line_path = trimesh.load_path([segment])
            
            # Set the color for this segment
            line_path.entities[0].color = color
            
            # Add to the scene
            scene.add_geometry(line_path, node_name=f"camera_{i}_edge_{start_idx}_{end_idx}")

    # Export the scene
    glb_data = trimesh.exchange.gltf.export_glb(scene)
    with open(output_filename, "wb") as f:
        f.write(glb_data)
    
    print(f"GLB file exported to '{output_filename}' with rainbow-colored cameras")


def load_video_frames(video_path, frame_sampling):
    """
    Loads and preprocesses video frames based on the sampling rate.

    Parameters:
      video_path (str): Path to the input video.
      frame_sampling (int): Interval between frames to sample. If -1, uses 1 frame per second.

    Returns:
      tuple: (list of preprocessed PIL images, list of [frame_index, absolute_frame_number], metadata)
             where metadata is a tuple (width, height, fps, total_frame_count).
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("[!] Error opening the video.")
        sys.exit(1)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_sampling == -1:
        frame_sampling = fps  # Use 1 frame per second

    print(f"Total frames in video: {frames_count}")
    print(f"Frame sampling rate: {frame_sampling}")

    # Estimate the total number of frames to process
    total_frames = (frames_count - 1) // frame_sampling + 2
    if frames_count <= frame_sampling:
        total_frames = 2

    print(f"Number of frames to process: {total_frames}")

    imgs = []
    frame_numbers = []
    pbar = tqdm(total=total_frames, unit=' frames', dynamic_ncols=True, desc="Loading frames")
    for i in range(total_frames):
        # Use the last frame for the final sample
        frame_to_process = frames_count - 1 if i == total_frames - 1 else i * frame_sampling
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_to_process)
        ret, frame = video.read()
        if ret:
            add_frame(imgs, frame)
            frame_numbers.append(frame_to_process)
            pbar.update(1)
        else:
            print(f"Error reading frame {frame_to_process}")
            sys.exit(1)
    pbar.close()

    return imgs, frame_numbers, (width, height, fps, frames_count)


def load_model(model_path, device, dtype):
    """
    Loads the VGGT model from a given checkpoint path.

    Parameters:
      model_path (str): Path to the model checkpoint.
      device (str): Device to load the model onto.
      dtype: Torch data type (e.g., torch.float16).

    Returns:
      VGGT model in evaluation mode.
    """
    model = VGGT()
    state_dict = torch.load(model_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def process_batches(imgs, window_size, window_overlap, model, device, dtype, to_tensor):
    """
    Processes preprocessed images in sliding windows to predict extrinsic and intrinsic matrices.

    Parameters:
      imgs (list): List of preprocessed PIL images.
      window_size (int): Number of frames per window.
      window_overlap (int): Overlap frames between consecutive windows.
      model: VGGT model.
      device (str): Device string (e.g., "cuda").
      dtype: Torch data type.
      to_tensor: Torchvision transform to convert PIL images to tensors.

    Returns:
      tuple: (all_extrinsics, extrinsics_references, all_intrinsics)
    """
    nb_windows = len(imgs) // (window_size - window_overlap)
    if len(imgs) % (window_size - window_overlap) != 0:
        nb_windows += 1

    all_extrinsics = []
    extrinsics_references = []
    all_intrinsics = []
    all_depth_maps = []

    windowsToProcess = 0
    for i in range(nb_windows):
        startpos = i * (window_size - window_overlap)
        endpos = min(startpos + window_size, len(imgs))
        if startpos != endpos-1:
            windowsToProcess += 1

    pbar = tqdm(total=len(imgs), unit=' frames', dynamic_ncols=True, desc="Processing batches")
    for i in range(windowsToProcess):
        if i!=windowsToProcess-1:
            startpos = i * (window_size - window_overlap)
            endpos = min(startpos + window_size, len(imgs))
        else:
            startpos = i * (window_size - window_overlap)
            endpos = len(imgs)

        batch_images = []
        for j in range(startpos, endpos):
            img_tensor = to_tensor(imgs[j]).unsqueeze(0)  # Add batch dimension
            batch_images.append(img_tensor)

        if batch_images:
            count = 0
            tensors_batch = torch.cat(batch_images, dim=0).to(device).to(dtype)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # Add extra dimension to simulate batch of batches
                    tensors_batch = tensors_batch[None]
                    aggregated_tokens_list, ps_idx = model.aggregator(tensors_batch)
                # Predict camera parameters (pose encoding)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, tensors_batch.shape[-2:])

                extrinsic = extrinsic.cpu().numpy().squeeze()
                intrinsic = intrinsic.cpu().numpy().squeeze()

                # Predict Depth Maps
                depth_map, _ = model.depth_head(aggregated_tokens_list, tensors_batch, ps_idx)
                depth_map = depth_map.cpu().numpy().squeeze()

                batch_size = extrinsic.shape[0]

                extrinsics_matrices = np.zeros((batch_size, 4, 4))
                extrinsics_matrices[:, :3, :4] = extrinsic
                extrinsics_matrices[:, 3, 3] = 1
                for c in range(batch_size):
                    world_to_camera = extrinsics_matrices[c]
                    camera_to_world = np.linalg.inv(world_to_camera)
                    extrinsics_matrices[c] = camera_to_world

                # extrinsic = extrinsics_matrices[:, :3 :,]
                # print (f"Extrinsic: {extrinsic}")
                if i == 0:
                    # For the first window, store the first extrinsic as reference and all frames
                    extrinsics_references.append(extrinsics_matrices[0].copy())
                    for j in range(batch_size):
                        all_extrinsics.append(extrinsics_matrices[j].copy())
                        all_intrinsics.append(intrinsic[j].copy())
                        all_depth_maps.append(depth_map[j].copy())
                        count = count + 1
                else:
                    # For subsequent windows, store a reference from the overlapping frames
                    ref_idx = min(window_overlap - 1, batch_size - 1)
                    extrinsics_references.append(extrinsics_matrices[ref_idx].copy())
                    # Append only the non-overlapping frames
                    for j in range(window_overlap, batch_size):
                        all_extrinsics.append(extrinsics_matrices[j].copy())
                        all_intrinsics.append(intrinsic[j].copy())
                        all_depth_maps.append(depth_map[j].copy())
                        count = count + 1
            pbar.update(count)
    pbar.close()

    return np.array(all_extrinsics), np.array(extrinsics_references), np.array(all_intrinsics), np.array(all_depth_maps)

def process_batches_intrinsics(imgs, model, device, dtype, to_tensor):
    
    all_intrinsics = []
    print (f"Processing {len(imgs)} frames")
    # This list will hold the metadata for each processed frame

    pbar = tqdm(total=len(imgs), unit=' frames', dynamic_ncols=True, position=0, leave=True)
    for i in range(len(imgs)):
        img = imgs[i]
        img_tensor = to_tensor(img).unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor[None]
        batch_images = [ img_tensor ]
        tensorsBatch = torch.cat(batch_images, dim=0).to(device).to(dtype)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # img_tensor = img_tensor[None]
                aggregated_tokens_list, ps_idx = model.aggregator(tensorsBatch)
            
            # Predict Cameras
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            _, intrinsic = pose_encoding_to_extri_intri(pose_enc, tensorsBatch.shape[-2:])

            intrinsic = intrinsic.cpu().numpy().squeeze()
            # print (f"Frame {i}: Intrinsic: {intrinsic}")
            all_intrinsics.append(intrinsic.copy())
    
            pbar.update(1)
    pbar.close()
    return np.array(all_intrinsics)    


def is_camera_motion_coherent(positions_global, rotations_global, plot=False, max_allowed_outliers=2):
    """
    Checks if the camera motion is globally coherent by detecting abrupt translation or rotation jumps.

    Parameters:
        positions_global (np.ndarray): (N, 3) camera positions over time.
        rotations_global (np.ndarray): (N, 3, 3) rotation matrices over time.
        plot (bool): If True, plots translation and rotation deltas.
        max_allowed_outliers (int): Maximum number of allowed abrupt frames.

    Returns:
        bool: True if motion is coherent, False otherwise.
    """
    # --- Translation jumps ---
    disp = np.diff(positions_global, axis=0)  # frame-to-frame vector difference
    disp_norm = np.linalg.norm(disp, axis=1)
    disp_mean = np.mean(disp_norm)
    disp_std = np.std(disp_norm)
    disp_thresh = disp_mean + 4 * disp_std
    trans_outliers = np.where(disp_norm > disp_thresh)[0]

    # --- Rotation jumps ---
    rots = R.from_matrix(rotations_global)
    rel_rots = rots[:-1].inv() * rots[1:]
    rot_angles = np.degrees(rel_rots.magnitude())  # in degrees
    rot_mean = np.mean(rot_angles)
    rot_std = np.std(rot_angles)
    rot_thresh = rot_mean + 4 * rot_std
    rot_outliers = np.where(rot_angles > rot_thresh)[0]

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(disp_norm, label="Translation Δ", color="blue")
        ax[0].axhline(disp_thresh, color="red", linestyle="--", label="3σ Threshold")
        ax[0].set_ylabel("Translation Δ")
        ax[0].legend()
        ax[0].set_title("Frame-to-frame Translation Change")

        ax[1].plot(rot_angles, label="Rotation Δ (deg)", color="green")
        ax[1].axhline(rot_thresh, color="red", linestyle="--", label="3σ Threshold")
        ax[1].set_ylabel("Rotation Δ (deg)")
        ax[1].set_xlabel("Frame index")
        ax[1].legend()
        ax[1].set_title("Frame-to-frame Rotation Change")

        plt.tight_layout()
        plt.show()

    total_outliers = len(trans_outliers) + len(rot_outliers)
    return total_outliers <= max_allowed_outliers

def main():
    # Validate command-line arguments
    if len(sys.argv) != 9:
        print("Usage: python video_camera_poses.py <input_video> <frame_sampling> <output_vggt_pkl> <output_vggt_glb> <fov> <tol_translation> <tol_rotation> <tol_fov>")
        sys.exit(1)

    video_in_name = sys.argv[1]
    frame_sampling = int(sys.argv[2])
    output_vggt_pkl_name = sys.argv[3]
    output_vggt_glb_name = sys.argv[4]
    global_fov =  np.radians(float(sys.argv[5]))
    tol_translation = float(sys.argv[6])
    tol_rotation = float(sys.argv[7])
    tol_rotation = np.radians(tol_rotation)
    tol_fov = float(sys.argv[8])
    tol_fov = np.radians(tol_fov)

    # Load and preprocess video frames
    imgs, frame_numbers, metadata = load_video_frames(video_in_name, frame_sampling)
    width, height, fps, frames_count = metadata

    # Load the VGGT model from checkpoint
    model_path = os.path.join(models_path, "vggt", "model.pt")
    print(f"Loading model from {model_path}")
    device = "cuda"
    # Choose the data type based on device capability (here we force float16)
    dtype = torch.float16
    model = load_model(model_path, device, dtype)
    model_width, model_height = 518, 294

    available_memory_gb = get_available_cuda_memory_gb()
    print(f"Available CUDA memory: {available_memory_gb:.3f} GB")

    # Set window size and overlap based on available memory
    if available_memory_gb > 16:
        window_size = 60
        window_overlap = 30
    elif available_memory_gb > 8:
        window_size = 40
        window_overlap = 20
    else:
        window_size = 10
        window_overlap = 5

    # Process frames in sliding windows to get extrinsics and intrinsics
    to_tensor = TF.ToTensor()
    all_extrinsics, extrinsics_references, all_intrinsics, all_depth_maps = process_batches(
        imgs, window_size, window_overlap, model, device, dtype, to_tensor
    )

    print (f"all_extrinsics: {all_extrinsics.shape}")
    print (f"extrinsics_references: {extrinsics_references.shape}")
    print (f"all_intrinsics: {all_intrinsics.shape}")
    
    # new_intrinsics = process_batches_intrinsics(imgs, model, device, dtype, to_tensor)
    
    # print (f"new_intrinsics: {new_intrinsics.shape}")

    # Verify and fix shapes for extrinsics and intrinsics if necessary
    fixed_extrinsics = []
    for ext in all_extrinsics:
        if isinstance(ext, np.ndarray) and ext.shape == (4, 4):
            fixed_extrinsics.append(ext)
        else:
            print(f"Ignoring extrinsic with shape: {ext.shape if hasattr(ext, 'shape') else 'non-array'}")

    fixed_extrinsics_references = []
    for ref in extrinsics_references:
        if isinstance(ref, np.ndarray) and ref.shape == (4, 4):
            fixed_extrinsics_references.append(ref)
        elif isinstance(ref, np.ndarray) and len(ref.shape) == 4 and ref.shape[1:] == (4, 4):
            fixed_extrinsics_references.append(ref[0].copy())
            print(f"Converted reference shape from {ref.shape} to (4, 4)")
        else:
            print(f"Ignoring extrinsics reference with shape: {ref.shape if hasattr(ref, 'shape') else 'non-array'}")

    fixed_intrinsics = []
    for intr in all_intrinsics:
        if isinstance(intr, np.ndarray) and intr.shape == (3, 3):
            fixed_intrinsics.append(intr)
        else:
            print(f"Ignoring intrinsic with shape: {intr.shape if hasattr(intr, 'shape') else 'non-array'}")

    all_extrinsics = np.array(fixed_extrinsics)
    extrinsics_references = np.array(fixed_extrinsics_references)
    all_intrinsics = np.array(fixed_intrinsics)

    # print (f"all_extrinsics final: {all_extrinsics.shape}")

    # print(f"Final shape of extrinsics: {all_extrinsics.shape}")
    # print(f"Final shape of extrinsics references: {extrinsics_references.shape}")
    # print(f"Final shape of intrinsics: {all_intrinsics.shape}")

    # Align camera poses to build the global trajectory
    rotations_global, positions_global = align_camera_poses(
        all_extrinsics, extrinsics_references, window_size, window_overlap
    )
    # intrinsic_global = extract_intrinsics(all_intrinsics, window_size, window_overlap, len(imgs))
    # print("intrinsic_global")
    # print(intrinsic_global)

    # positions_global[:, 1] = -positions_global[:, 1]
    # positions_global[:, 2] = -positions_global[:, 2]

    np.set_printoptions(formatter={'float': lambda x: f"{x:.5f}"})
    # positions_global = positions_global[:, [0, 1, 2]]  # Permute X Z Y → Z Y X
    positions_global = positions_global



    if len(rotations_global) != len(imgs):
        print(f"WARNING: Number of poses ({len(rotations_global)}) does not match number of images ({len(imgs)})")

    # Convert rotation matrices to Euler angles
    rotations_global_euler = np.array([R.from_matrix(rot).as_euler('xyz') for rot in rotations_global])

    # Compute FOVs for each intrinsic matrix
    fovh_list, fovv_list, fovd_list = [], [], []
    for i in range (len(all_intrinsics)):
        intr = all_intrinsics[i]
        # new_intr = new_intrinsics[i]
        fov_h, fov_v, fov_d = compute_fov(intr, model_width, model_height)
        # new_fovh, new_fovv, new_fovd = compute_fov(new_intr, model_width, model_height)
        # print (f"Frame {i}: Old FOV: {np.degrees(fov_h)} New FOV: {np.degrees(new_fovh)}")
        fovh_list.append(fov_h)
        fovv_list.append(fov_v)
        fovd_list.append(fov_d)
    fovh = np.array(fovh_list)
    fovv = np.array(fovv_list)
    fovd = np.array(fovd_list)
    fovh_degree = np.degrees(fovh)
    fovv_degree = np.degrees(fovv)
    fovd_degree = np.degrees(fovd)

    # Override horizontal FOV if a non-zero global FOV is provided
    if global_fov != 0:
        fovh = np.full(len(fovh), global_fov)
        fovh_degree = np.full(len(fovh_degree), np.degrees(global_fov))

    # Compute relative positions (with respect to the first frame)
    reference_position = positions_global[0]
    relative_positions = positions_global - reference_position

    # Compute relative rotations (using composition with the inverse of the reference rotation)
    reference_rotation = R.from_euler('xyz', rotations_global_euler[0])
    reference_rotation_inv = reference_rotation.inv()
    relative_rotations = []
    for angles in rotations_global_euler:
        current_rotation = R.from_euler('xyz', angles)
        rel_rot = reference_rotation_inv * current_rotation
        relative_rotations.append(rel_rot.as_euler('xyz'))
    relative_rotations = np.array(relative_rotations)

    # Convert relative rotations to quaternions
    relative_quaternions = np.array([R.from_euler('xyz', euler).as_quat() for euler in relative_rotations])

    translation_fixed = np.allclose(positions_global, positions_global[0], atol=tol_translation)
    rotation_fixed = np.allclose(rotations_global_euler, rotations_global_euler[0], atol=tol_rotation)
    print(f"Translation fixed: {translation_fixed}, Rotation fixed: {rotation_fixed}")
    # print(rotations_global_euler)
    # print(rotations_global_euler[0])
    
    fovh_fixed = np.allclose(fovh, np.mean(fovh), atol=tol_fov)
    print(f"FOV fixed: {fovh_fixed}")
    
    camera_fixed = translation_fixed and rotation_fixed
    print(f"Camera fixed: {camera_fixed}")  
    
    average_fovh = np.mean(fovh)
    
    coherent = is_camera_motion_coherent(positions_global, rotations_global, plot=False)
    
    if coherent:
        print("Camera motion is coherent")
    else:
        print(Fore.RED + "Camera motion is NOT coherent"+ Style.RESET_ALL)
    
    # Export the camera trajectory as a GLB file
    export_camera_trajectory_glb(rotations_global, positions_global, output_vggt_glb_name)

    # Prepare data for saving
    camera_poses_data = {
        "frames": frame_numbers,
        "video": video_in_name,
        "video_width": width,
        "video_height": height,
        "video_fps": fps,
        "frame_sampling": frame_sampling,
        "total_frames": frames_count,
        "frames_processed": len(imgs),
        'depth_maps': all_depth_maps,
        "rotationMatrices": rotations_global,
        "positions": positions_global,
        "rotations": rotations_global_euler,
        "relative_positions": relative_positions,
        "relative_rotations": relative_rotations,
        "relative_quaternions": relative_quaternions,
        "fov_x": fovh,
        "fov_y": fovv,
        "fov_d": fovd,
        "average_fov_x": average_fovh,
        "average_fov_x_degrees": np.degrees(average_fovh),
        "fov_x_degrees": fovh_degree,
        "fov_y_degrees": fovv_degree,
        "fov_d_degrees": fovd_degree,
        "global_fov_x": global_fov,
        "camera_fixed": camera_fixed,
        "fov_fixed": fovh_fixed,
        "camera_translation_fixed": translation_fixed,
        "camera_rotation_fixed": rotation_fixed,
        "tol_translation": tol_translation,
        "tol_rotation": tol_rotation,
        "tol_fov": tol_fov,
        "camera_motion_coherent": coherent,
    }

    # print (relative_positions)
    # print (relative_rotations)

    with open(output_vggt_pkl_name, "wb") as f:
        pickle.dump(camera_poses_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Camera poses saved in {output_vggt_pkl_name}")

    # Release video resource
    cv2.VideoCapture(video_in_name).release()


if __name__ == "__main__":
    main()

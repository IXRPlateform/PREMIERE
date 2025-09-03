import sys
import os
import torch
import time
import numpy as np
import PIL
import cv2
import pickle
import smplx
import roma
import math

import torchvision  # Must import this for the model to load without error
import torchvision.transforms.functional as transform

from argparse import ArgumentParser
from tqdm import tqdm
from premiere.functionsCommon import projectPoints3dTo2d, keypointsBBox3D

def readBatch(video, batch_size):
    """
    Read a batch of frames from a video capture object.

    This function reads up to `batch_size` frames from the provided `video`
    object. If frames are available, each frame is converted into an RGB
    float tensor on GPU and returned in a list. If the end of the video is
    reached or `video.read()` fails, the function returns `ret = False`.

    :param video: An OpenCV VideoCapture object.
    :type video: cv2.VideoCapture
    :param batch_size: Number of frames to read for the batch.
    :type batch_size: int
    :return: A tuple where the first value indicates whether frames were read
             successfully (`True` or `False`), and the second value is a list
             of frames as tensors.
    :rtype: Tuple[bool, List[torch.Tensor]]
    """
    batch = []
    ret = True

    # Loop to read `batch_size` frames from the video
    for i in range(batch_size):
        ret, frame = video.read()
        if ret:
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert RGB numpy array to float tensor and permute dimensions from (H, W, C) to (C, H, W)
            tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1)
            # Move the tensor to GPU
            tensor_image = tensor.cuda()
            batch.append(tensor_image)
        else:
            # If we can't read the frame, break and mark ret as False
            ret = False
            break

    # If no frames were read, also return ret = False
    if len(batch) == 0:
        ret = False

    return ret, batch


def updateHumanFromSMPLX(human, modelSMPLX, scaleX=1.0, scaleY=1.0, scaleZ=1.0, offsetZ=0.0):
    """
    Recomputes a human body model with SMPLX and applies transforms (scale + offset).

    This function takes a dictionary describing a human's shape parameters,
    SMPLX model pose parameters, and 3D translation. It then re-computes
    the 3D keypoints using the SMPLX model, applies rotation, and adjusts
    translations and scale. The resulting 3D joint positions are stored back
    into the `human` dictionary.

    :param human: Dictionary containing human parameters, including:
                  - 'shape' (np.ndarray): The betas for body shape (dim = 10).
                  - 'rotvec' (np.ndarray): Rotational vectors for body and hands (dim = 55 x 3).
                  - 'transl' (np.ndarray): The root translation (pelvis).
    :type human: dict
    :param modelSMPLX: SMPLX model loaded via smplx.create.
    :type modelSMPLX: smplx.body_models.SMPLX
    :param scaleX: Scale factor in X dimension.
    :type scaleX: float, optional
    :param scaleY: Scale factor in Y dimension.
    :type scaleY: float, optional
    :param scaleZ: Scale factor in Z dimension.
    :type scaleZ: float, optional
    :param offsetZ: Additional offset applied in the Z dimension after scaling.
    :type offsetZ: float, optional
    :return: Updated `human` dictionary with new 3D SMPLX keypoints and other updated parameters.
    :rtype: dict
    """
    # Shape parameters (betas)
    betas = torch.from_numpy(np.expand_dims(human['shape'], axis=0)).cuda()
    
    # Expression is set to zeros by default here
    expression = torch.zeros(1, 10).cuda()
    
    # Pose is the full set of rotational vectors (rotvec)
    pose = torch.from_numpy(np.expand_dims(human['rotvec'], axis=0)).cuda()
    bs = pose.shape[0]

    # Prepare kwargs for SMPLX forward pass
    kwargs_pose = {
        'betas': betas,
        'return_verts': False,
        'pose2rot': True
    }
    
    # Global orientation set to zero
    t = torch.zeros(1, 3).cuda()
    kwargs_pose['global_orient'] = t.repeat(bs, 1)
    
    # Body pose (body joints)
    kwargs_pose['body_pose'] = pose[:, 1:22].flatten(1)

    # Face and hand pose    
    kwargs_pose['jaw_pose'] = pose[:,22:23].flatten(1)
    kwargs_pose['leye_pose'] = pose[:,23:24].flatten(1)
    kwargs_pose['reye_pose'] = pose[:,24:25].flatten(1)    
    
    # Left and right hand pose
    kwargs_pose['left_hand_pose'] = pose[:, 25:40].flatten(1)
    kwargs_pose['right_hand_pose'] = pose[:, 40:55].flatten(1)
    
    # Run the SMPLX model to get joint positions
    output = modelSMPLX(**kwargs_pose)
    
    # Extract 3D joints 
    j3d = output.joints
    # Convert the root orientation from rotvec to rotation matrix
    R = roma.rotvec_to_rotmat(pose[:, 0])
    
    # Translate joints so that the pelvis is at the origin, then rotate
    pelvis = j3d[:, [0]]
    j3d = (R.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)
    
    # Person center is the joint index for the pelvis or relevant center (here index 15)
    person_center = j3d[:, [15]]
    # person_center = pelvis
    
    # Copy transl from `human`, then adjust scale and offset
    transCPU = human['transl'].copy()
    transCPU[0] = transCPU[0] * scaleX
    transCPU[1] = transCPU[1] * scaleY
    transCPU[2] = (transCPU[2] * scaleZ) + offsetZ
    
    # Convert transl to torch and subtract person_center
    trans = torch.from_numpy(transCPU).cuda()
    trans = trans - person_center
    
    # Update joint positions
    j3d = j3d + trans
    
    # Store updated keypoints in the human dictionary
    keypoints = j3d.detach().cpu().numpy().squeeze()
    human['j3d_smplx'] = keypoints[0:127]  # some internal indexing
    human['transl_pelvis'] = keypoints[0]
    human['expression'] = np.zeros(10, dtype=np.float32)  # expression placeholder
    return human

def process_human_detection(pred, frame_idx, human_idx, current_fov, modelSMPLX, video_width, video_height):
    """Process a single human detection and return a populated human dictionary."""
    human = {}
    # Shape (betas)
    human['shape'] = pred['betas'][frame_idx][human_idx].detach().cpu().numpy().squeeze()
    # Pose, reshaped to 55 x 3
    human['rotvec'] = pred['pose'][frame_idx][human_idx].detach().cpu().numpy().squeeze().reshape(55, 3)
    # Arbitrary ID
    human['id'] = -1
    # Joints 3D
    human['j3d_smplx'] = pred['joints3d'][frame_idx][human_idx].detach().cpu().numpy().squeeze().reshape(55, 3)
    # Convert from mm to m (arbitrary scaling)
    human['j3d_smplx'] /= 1000.0
    # The head is at index 15
    human['transl'] = human['j3d_smplx'][15]
    human['trans'] = pred['trans'][frame_idx][human_idx].detach().cpu().numpy().squeeze()
    human['fov'] = current_fov
    human['occluded'] = False
    
    # Refine the SMPLX with updated transformations
    human = updateHumanFromSMPLX(human, modelSMPLX)
    
    # Project the 3D points to 2D
    proj_2d = projectPoints3dTo2d(
        human['j3d_smplx'], 
        fov=current_fov,  # Use the same FOV for projection
        width=video_width, 
        height=video_height
    )
    human['j2d_smplx'] = proj_2d
    
    # Bounding box from the detection
    bbox = pred['boxes'][frame_idx][human_idx].detach().cpu().numpy().squeeze()
    human['bbox'] = bbox[0:4]
    human['score'] = bbox[4]
    
    # Compute the 3D bounding box from keypoints
    min_coords, max_coords = keypointsBBox3D(human['j3d_smplx'])
    human['bbox3d'] = [min_coords, max_coords]
    
    # Store the pelvis transl and the 2D location of joint 15
    human['transl_pelvis'] = human['j3d_smplx'][0].reshape(1,3)
    human['loc'] = human['j2d_smplx'][15]
    
    # Uncertainties from detection
    human['joint_uncertainties'] = pred['joint_uncertainties'][frame_idx][human_idx].detach().cpu().numpy().squeeze()
    
    return human


def main():
    # Argument parser for command line usage
    parser = ArgumentParser()
    parser.add_argument("--video", type=str, default='')
    parser.add_argument("--out_pkl", type=str, default='allFrame.pkl')
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--fov", type=float, default=70)
    parser.add_argument("--batchsize", type=int, default=20)
    parser.add_argument("--camerafile", type=str, default='')

    # Parse arguments from command line
    args = parser.parse_args()
       
    # Ensure we have a CUDA-enabled machine
    assert torch.cuda.is_available()

    # Load the path to the NLF model
    models_path = os.environ["MODELS_PATH"]
    model_name = os.path.join(models_path, 'nlf', 'nlf_l.pt')

    det_thresh = args.det_thresh
    fov = args.fov
    camerafile = args.camerafile
    fovPKL = None
    useDynamicFOV = False
    if camerafile != '':
        with open(camerafile, 'rb') as file:
            fovPKL = pickle.load(file)
            useDynamicFOV = True

    print("Loading NLF model...")
    print(model_name)
    # Load the TorchScript model for NLF
    model = torch.jit.load(model_name).cuda().eval()
    print("NLF model loaded")

    # Creating a VideoCapture object to read the video
    video = cv2.VideoCapture(args.video)

    # Gather video properties
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = video.get(cv2.CAP_PROP_FPS)

    print('[INFO] Loading SMPLX model')
    gender = 'neutral'
    modelSMPLX = smplx.create(
        models_path, 'smplx',
        gender=gender,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        ext='npz'
    ).cuda()
    print('[INFO] SMPLX model loaded')

    keypointsNumber = 55
    batch_size = args.batchsize
    allFrameHumans = []

    cameraFovs = []
    frame_count = 0

    pbar = tqdm(total=total_frames, unit=' frames', dynamic_ncols=True, position=0, leave=True, file=sys.stdout)
    while video.isOpened():
        ret, batch = readBatch(video, batch_size)
        
        if len(batch) == 0:
            # No more frames to read
            break
            
        if useDynamicFOV:
            # Process each frame individually with its own FOV value
            for i, frame in enumerate(batch):
                # Get FOV for current frame
                current_fov = float(fovPKL['interpolated_fov_x'][frame_count]) * 180.0 / math.pi
                    
                # Add batch dimension to single frame
                frame_tensor = frame.unsqueeze(0)
                
                with torch.inference_mode():
                    # Process a single frame with its specific FOV
                    pred = model.detect_smpl_batched(
                        frame_tensor,
                        model_name='smplx',
                        default_fov_degrees=current_fov,  # Pass as float
                        detector_threshold=args.det_thresh
                    )
                
                # Handle the single frame's predictions
                allHumans = []
                if len(pred['trans'][0]) > 0:  # Check if any humans were detected
                    for j in range(len(pred['trans'][0])):
                        human = process_human_detection(
                            pred, 0, j, current_fov, modelSMPLX, 
                            video_width, video_height
                        )
                        allHumans.append(human)
                
                # Add this frame's humans to the overall list
                allFrameHumans.append(allHumans)
                cameraFovs.append(current_fov)
                
                # Increment frame counter after processing each frame
                frame_count += 1
                
                # Update progress bar for each frame
                pbar.update(1)
                
            sys.stdout.flush()
        else:
            # Stack the batch into a 4D tensor (B, C, H, W)
            tensorBatch = torch.stack(batch, dim=0)

            with torch.inference_mode():
                # Run model detection
                pred = model.detect_smpl_batched(
                    tensorBatch, 
                    model_name='smplx', 
                    default_fov_degrees=fov, 
                    detector_threshold=args.det_thresh
                )

            # For each frame in the batch
            for i in range(len(pred['trans'])):
                allHumans = []
                # For each detected human
                for j in range(len(pred['trans'][i])):
                    human = process_human_detection(
                        pred, i, j, fov, modelSMPLX, 
                        video_width, video_height
                    )
                    allHumans.append(human)
                
                allFrameHumans.append(allHumans)
                cameraFovs.append(fov)

            # Update the progress bar
            pbar.update(len(batch))
            frame_count += len(batch)
            sys.stdout.flush()
    
    pbar.close()
    
    # Gather all data into a dictionary
    allData = {
        'allFrameHumans': allFrameHumans,
        "camera_fovs": np.array(cameraFovs),
        'model_name': model_name,
        'model_type': 'nlf',
        'video': args.video,
        'fov_x_deg': args.fov,
        'video_width': video_width,
        'video_height': video_height,
        'video_fps': video_fps,
        'keypoints_number': keypointsNumber
    }

    # Save the data to a pickle file
    with open(args.out_pkl, 'wb') as handle:
        pickle.dump(allData, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done')

if __name__ == "__main__":
    main()

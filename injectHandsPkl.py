import os
import sys
import cv2
import torch
import roma
import smplx
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Import PREMIERE utilities
from premiere.functionsSMPLX import updateHumanFromSMPLX
from premiere.functionsCommon import projectPoints3dTo2d, loadPkl, savePkl

# -------------- WiLoR-specific imports --------------
from wilor.models import WiLoR, load_wilor
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils import recursive_to

# ----------------------------------------------------------------------
# 1) Camera projection utilities
# ----------------------------------------------------------------------

def project_full_img(points, cam_trans, focal_length, img_res):
    """
    Projects 3D points (camera coords) into the 2D image coordinate system.
    
    Args:
        points: 3D points in camera coordinate system
        cam_trans: Camera translation vector
        focal_length: Camera focal length
        img_res: Image resolution [width, height]
        
    Returns:
        2D projected points in image coordinates
    """
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3)
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]

    points = points + cam_trans
    points = points / points[..., -1:]  # perspective divide
    V_2d = (K @ points.T).T
    return V_2d[..., :-1]

def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    """
    Convert bounding-box-based camera parameters to full-image parameters.
    
    Args:
        cam_bbox: Bounding box camera parameters
        box_center: Center of the bounding box
        box_size: Size of the bounding box
        img_size: Image dimensions
        focal_length: Camera focal length
        
    Returns:
        Full image camera parameters
    """
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.

    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

# ----------------------------------------------------------------------
# 2) Bounding box utilities
# ----------------------------------------------------------------------

def compute_bounding_box(points, frame_h, frame_w, increase_size=0.2):
    """
    Compute an expanded bounding box around 2D points, clamped to image boundaries.
    
    Args:
        points: 2D points to build a bounding box around
        frame_h: Frame height
        frame_w: Frame width
        increase_size: How much to expand the bounding box (ratio)
        
    Returns:
        Bounding box as [min_x, min_y, max_x, max_y]
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Expand bounding box by 'increase_size' (e.g. 20%)
    w = max_x - min_x
    h = max_y - min_y
    min_x -= increase_size * w
    max_x += increase_size * w
    min_y -= increase_size * h
    max_y += increase_size * h

    # Clamp to image boundaries
    min_x = max(0, min_x)
    max_x = min(frame_w, max_x)
    min_y = max(0, min_y)
    max_y = min(frame_h, max_y)

    return [min_x, min_y, max_x, max_y]

def compute_hand_bboxes(human, frame_shape, increase_size=0.2):
    """
    Compute bounding boxes for both hands of a human.
    
    Args:
        human: Human dictionary containing keypoints
        frame_shape: Tuple of (height, width) of the frame
        increase_size: How much to expand the bounding boxes
        
    Returns:
        List of bounding boxes and whether each box is for the right hand
    """
    points2D = human['j2d_smplx']
    bboxes = []
    is_right = []

    # Compute bounding box for left hand
    selected_points_left = np.vstack([points2D[25:40], points2D[20]])  # left hand + wrist
    bbox_left = compute_bounding_box(selected_points_left, frame_shape[0], frame_shape[1], increase_size) 
    bboxes.append(bbox_left)
    is_right.append(False)

    # Compute bounding box for right hand
    selected_points_right = np.vstack([points2D[40:55], points2D[21]])  # right hand + wrist
    bbox_right = compute_bounding_box(selected_points_right, frame_shape[0], frame_shape[1], increase_size) 
    bboxes.append(bbox_right)
    is_right.append(True)
    
    return bboxes, is_right

# ----------------------------------------------------------------------
# 3) Rotation utilities
# ----------------------------------------------------------------------

def rotmat_to_axis_angle(rotmats):
    """
    Convert rotation matrices (B, 3, 3) to axis-angle (B, 3).
    
    Args:
        rotmats: Rotation matrices of shape (B, 3, 3)
        
    Returns:
        Axis-angle representation of shape (B, 3)
    """
    return roma.rotmat_to_rotvec(rotmats)

def extract_local_finger_pose(global_hand_rotmat, finger_rotmats):
    """
    Convert hand rotations to local finger axis-angles for SMPL-X.
    
    Args:
        global_hand_rotmat: Global hand rotation matrix
        finger_rotmats: Finger rotation matrices (15, 3, 3)
        
    Returns:
        Local finger axis-angles as (1, 45) tensor
    """
    # Convert to axis-angle per finger
    finger_axisangles = rotmat_to_axis_angle(finger_rotmats)  # (15, 3)
    # Flatten to (1, 45)
    finger_axisangles = finger_axisangles.view(1, -1)
    return finger_axisangles

# ----------------------------------------------------------------------
# 4) Model loading
# ----------------------------------------------------------------------

def load_wilor_model(models_path):
    """
    Load the WiLoR hand pose estimation model.
    
    Args:
        models_path: Path to the models directory
        
    Returns:
        Loaded WiLoR model and its configuration
    """
    model_name = os.path.join(models_path, 'wilor', 'wilor_final.ckpt')
    model_config = os.path.join(models_path, 'wilor', 'model_config.yaml')

    print('[INFO] Loading WiLoR model...')
    model, model_cfg = load_wilor(checkpoint_path=model_name, cfg_path=model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print('[INFO] WiLoR model loaded')
    
    return model, model_cfg

def load_smplx_model(models_path, gender='neutral'):
    """
    Load the SMPL-X body model.
    
    Args:
        models_path: Path to the models directory
        gender: Gender for the SMPL-X model ('neutral', 'male', or 'female')
        
    Returns:
        Loaded SMPL-X model
    """
    print('[INFO] Loading SMPL-X model...')
    model = smplx.create(
        models_path, 'smplx',
        gender=gender,
        use_pca=False, flat_hand_mean=True,
        num_betas=10,
        ext='npz').cuda()
    print('[INFO] SMPL-X model loaded')
    
    return model

# ----------------------------------------------------------------------
# 5) Hand processing
# ----------------------------------------------------------------------

def process_hands_with_wilor(frame, bboxes, is_right, model, model_cfg, device, display=False):
    """
    Process hand images with WiLoR to estimate hand poses.
    
    Args:
        frame: Current video frame
        bboxes: Bounding boxes for hands
        is_right: Boolean array indicating if each box is a right hand
        model: The WiLoR model
        model_cfg: WiLoR model configuration
        device: Computing device (CPU/GPU)
        display: Whether to visualize the detected hand keypoints
        
    Returns:
        List of local finger axis-angles for each hand
    """
    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=1.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_hands_pose = []  # will store the local finger axis-angles for each hand

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        # Adjust sign for right-hand bounding boxes
        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam']
        pred_cam[:,1] = multiplier * pred_cam[:,1]

        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE 
            * img_size.max()
        )
        pred_cam_t_full = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size, scaled_focal_length
        ).detach().cpu().numpy()

        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            # Global-orient of the hand (3x3)
            hand_global_rotmat = out['pred_mano_params']['global_orient'][n]
            # Finger rotations (15 x 3 x 3)
            finger_rotmats = out['pred_mano_params']['hand_pose'][n]

            # Convert to local finger axis-angles
            local_finger_aa = extract_local_finger_pose(hand_global_rotmat, finger_rotmats)
            all_hands_pose.append(local_finger_aa.detach().cpu().numpy())

            # If "display" is on, project & draw
            if display:
                joints3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                is_right_hand = batch['right'][n].cpu().numpy()
                # Flip x if right hand
                joints3d[:,0] = (2*is_right_hand - 1)*joints3d[:,0]
                cam_t = pred_cam_t_full[n]
                joints2d = project_full_img(
                    torch.tensor(joints3d), 
                    torch.tensor(cam_t), 
                    scaled_focal_length, 
                    img_size[n]
                )
                joints2d = joints2d.detach().cpu().numpy()

                color = (0,255,0) if is_right_hand else (255,0,0)
                for p in joints2d:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 1, color, -1)

    return all_hands_pose

def update_human_hand_poses(human, left_hand_pose, right_hand_pose, use_expression):
    """
    Update the human pose with new hand poses.
    
    Args:
        human: Human dictionary containing pose data
        left_hand_pose: Left hand pose parameters (45,)
        right_hand_pose: Right hand pose parameters (45,)
        use_expression: Whether facial expressions are used
        
    Returns:
        Updated human dictionary
    """
    pose = human['rotvec'].copy()  # Create a copy to avoid modifying the original

    # Determine start indices based on whether expressions are used
    if use_expression:
        left_start, right_start = 23, 37
    else:
        left_start, right_start = 25, 40
    
    # Update left hand joints (15 joints, each with 3 axis-angle components)
    for i in range(15):
        pose[left_start + i] = -left_hand_pose[i*3:(i+1)*3]  # Negative for left hand

    # Update right hand joints
    for i in range(15):
        pose[right_start + i] = right_hand_pose[i*3:(i+1)*3]
        
    # Update the human dictionary with the new pose
    human['rotvec'] = pose
    return human

def update_smplx_keypoints(humans, model_smplx, use_expression, video_width, video_height, fov):
    """
    Update SMPLX models for all humans in all frames.
    
    Args:
        humans: List of lists of human dictionaries
        model_smplx: SMPL-X model
        use_expression: Whether facial expressions are used
        video_width: Video width in pixels
        video_height: Video height in pixels
        fov: Field of view in degrees
        
    Returns:
        Updated humans list
    """
    print("Updating SMPLX models...")
    pbar = tqdm(total=len(humans), unit=' frames', dynamic_ncols=True)
    
    for i in range(len(humans)):
        for j in range(len(humans[i])):
            # Update human from SMPLX
            human = humans[i][j]
            humans[i][j] = updateHumanFromSMPLX(human, model_smplx, use_expression)
            
            # Project 3D joints to 2D
            proj_2d = projectPoints3dTo2d(
                humans[i][j]['j3d_smplx'], 
                fov=fov, 
                width=video_width, 
                height=video_height
            )
            humans[i][j]['j2d_smplx'] = proj_2d
            
        pbar.update(1)
    pbar.close()
    
    return humans

# ----------------------------------------------------------------------
# 6) Main script
# ----------------------------------------------------------------------

def main():
    """Main function to process hand poses and update the PKL file."""
    
    # Parse command line arguments
    if len(sys.argv) != 5:
        print("Usage: python injectHandsPkl.py <input_pkl> <video> <output_pkl> <display: 0 or 1>")
        sys.exit(1)

    input_pkl_path = sys.argv[1]
    video_path = sys.argv[2]
    output_pkl_path = sys.argv[3]
    display = (int(sys.argv[4]) == 1)

    # Load PKL data
    try:
        dataPKL = loadPkl(input_pkl_path)
        allFrameHumans = dataPKL['allFrameHumans']
    except Exception as e:
        print(f"[ERROR] Failed to load PKL file: {e}")
        sys.exit(1)

    # Open video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print('[ERROR] Could not open video file')
        sys.exit(1)

    # Get video properties
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Store video dimensions in dataPKL if not already present
    dataPKL['video_width'] = video_width
    dataPKL['video_height'] = video_height

    # Load models
    try:
        models_path = os.environ["MODELS_PATH"]
        model_wilor, model_cfg = load_wilor_model(models_path)
        model_smplx = load_smplx_model(models_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        sys.exit(1)

    # Determine if we use expressions based on model type
    use_expression = not (dataPKL['model_type'] in ["hmr2", "nlf"])

    # Process video frames
    print('[+] Processing video frames for hand poses...')
    pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True)

    count = 0
    increase_size = 0.2  # Percentage to increase hand bounding boxes

    while video.isOpened():
        ret, frame = video.read()
        if not ret or count >= len(allFrameHumans):
            break

        humans = allFrameHumans[count]
        
        for human in humans:
            # Compute bounding boxes for both hands
            bboxes, is_right = compute_hand_bboxes(human, frame.shape, increase_size)
            
            # Process hands with WiLoR
            hands_poses = process_hands_with_wilor(
                frame, bboxes, is_right, model_wilor, model_cfg, device, display
            )

            # Update human with new hand poses if both hands were detected
            if len(hands_poses) == 2:
                left_hand_pose = hands_poses[0][0]  # shape (45,)
                right_hand_pose = hands_poses[1][0]  # shape (45,)
                human = update_human_hand_poses(human, left_hand_pose, right_hand_pose, use_expression)

        # Display frame if requested
        if display:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        pbar.update(1)
        count += 1

    pbar.close()
    video.release()
    if display:
        cv2.destroyAllWindows()

    # Update SMPLX models and project to 2D
    allFrameHumans = update_smplx_keypoints(
        allFrameHumans, model_smplx, use_expression, 
        video_width, video_height, dataPKL['fov_x_deg']
    )

    # Update PKL data
    dataPKL['allFrameHumans'] = allFrameHumans

    # Save the updated PKL file
    try:
        savePkl(output_pkl_path, dataPKL)
        print(f'[INFO] Hand poses injected and saved to {output_pkl_path}')
    except Exception as e:
        print(f"[ERROR] Failed to save PKL file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

import sys
import pickle
import numpy as np
import cv2
import math

from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution
from scipy.optimize import minimize_scalar

from premiere.functionsMoge import computeInitialRotation
from premiere.functionsCommon import loadPkl3, keypointsBBox3D

from colorama import init, Fore, Style
init(autoreset=True)

def stabilize_floor_contact(allFrameHumans, contact_threshold=0.005, max_offset=0.02, alpha=0.2, Kp=0.8, Kd=0.4):
    """
    Stabilizes floor contact by correcting only upwards using an adaptive PD controller.
    
    For each frame, the procedure is as follows:
      1. Detect the floor level from foot joints (typical indices for ankle and toe).
      2. Smooth the floor trajectory with an exponential filter.
      3. Compute a robust global level (5th percentile over all frames).
      4. For each frame, set the desired level as the maximum between the smoothed level and the global level.
      5. For the frame, for each human, compute error = (desired level - lowest foot position).
         Then, aggregate these errors (e.g., by taking the maximum value) to obtain a global error.
      6. Compute the offset via a PD law: offset = Kp * error + Kd * (error - previous_error), then clip to max_offset.
      7. Apply this offset to all humans whose individual error exceeds the contact_threshold.
    
    Args:
        allFrameHumans: List of frames, each frame being a list of dictionaries representing humans.
        contact_threshold: Minimum threshold to trigger a correction for a human.
        max_offset: Maximum allowed offset per frame.
        alpha: Exponential smoothing coefficient for the floor trajectory.
        Kp: Proportional gain of the PD controller.
        Kd: Derivative gain of the PD controller.
        
    Returns:
        allFrameHumans with adjusted vertical (Y) positions.
    """
    # Typical indices for ankle and toe in SMPLX
    foot_joints = [7, 8, 10, 11]

    # 1. Detect the floor level per frame from feet
    floor_levels = []
    for humans in allFrameHumans:
        if len(humans) == 0:
            floor_levels.append(None)
            continue
        lowest_points = []
        for human in humans:
            foot_y = human['j3d_smplx'][foot_joints, 1]
            lowest_points.append(np.min(foot_y))
        floor_levels.append(np.min(lowest_points) if lowest_points else None)
    
    # 2. Smooth the floor trajectory with exponential filter
    smoothed_floor = []
    prev = None
    for level in floor_levels:
        if level is None:
            smoothed_floor.append(prev)
        elif prev is None:
            prev = level
            smoothed_floor.append(level)
        else:
            prev = alpha * level + (1 - alpha) * prev
            smoothed_floor.append(prev)
    
    # 3. Compute the global floor level (5th percentile over all feet)
    all_foot_y = []
    for humans in allFrameHumans:
        for human in humans:
            foot_y = human['j3d_smplx'][foot_joints, 1]
            all_foot_y.extend(foot_y)
    global_floor = np.percentile(all_foot_y, 5) if all_foot_y else 0.0

    # 4. Initialize previous global error for PD control
    previous_global_error = 0.0

    # 5. Iterate over frames (assuming the sequence is chronological)
    for i, humans in enumerate(allFrameHumans):
        if len(humans) == 0 or smoothed_floor[i] is None:
            continue
        # Desired level is the maximum between smoothed and global level
        desired_floor = max(smoothed_floor[i], global_floor)
        
        # For each human in the frame, compute individual error
        errors = []
        for human in humans:
            foot_y = human['j3d_smplx'][foot_joints, 1]
            lowest_foot = np.min(foot_y)
            errors.append(desired_floor - lowest_foot)
        
        # Aggregate frame error (e.g., take the maximum value)
        frame_error = max(errors) if errors else 0.0
        
        # Compute error derivative with respect to previous frame
        error_derivative = frame_error - previous_global_error
        
        # Compute offset via PD controller
        offset = Kp * frame_error + Kd * error_derivative
        # Only correct upwards (positive offset) and limit amplitude
        offset = np.clip(offset, 0, max_offset)
        
        # Apply correction for each human if their individual error exceeds the threshold
        for human in humans:
            foot_y = human['j3d_smplx'][foot_joints, 1]
            lowest_foot = np.min(foot_y)
            individual_error = desired_floor - lowest_foot
            if individual_error > contact_threshold:
                human['j3d_smplx'][:, 1] += offset
                # Update translation (here, joint 15 is used as reference)
                human['transl'] = human['j3d_smplx'][15]
        
        previous_global_error = frame_error

    return allFrameHumans

# Description: This script is used to convert the poses from the camera frame to the floor frame.
def main():
    if len(sys.argv) < 6 or len(sys.argv) > 8:
        print("Usage: python cameraCompensation.py <input_pkl> <output_pkl> <vggt_interpolated_pkl> <moge_pkl> <factor> [floor_stabilize=0] [contact_threshold=0.05]")
        sys.exit(1)

    input_pkl = sys.argv[1]
    outputpklName = sys.argv[2]
    vggt_pkl = sys.argv[3]
    moge_pkl = sys.argv[4]
    factor = -float(sys.argv[5])  # Now mandatory

    # Load data
    dataPKL, vggtPKL, mogePKL = loadPkl3(input_pkl, vggt_pkl, moge_pkl)
    allFrameHumans = dataPKL['allFrameHumans']
    
    # Optional parameters with default values
    floor_stabilize = 0
    contact_threshold = 0.05
    
    if len(sys.argv) > 6:
        floor_stabilize = int(sys.argv[6])
    if len(sys.argv) > 7:
        contact_threshold = float(sys.argv[7])

    cameraMotionCoherent = vggtPKL["camera_motion_coherent"]

    if cameraMotionCoherent:  
        print(f"Using camera scale factor: {factor}")
    else:
        print(Fore.RED +"Camera motion is not coherent. No scale factor applied."+ Style.RESET_ALL)
        print(Fore.RED +"Camera position and rotation will be set to 0.0"+ Style.RESET_ALL)

    pivot = np.array([0, 0, 0])
    rotationCameraDegrees = mogePKL['rotation_camera_degrees']
    R_np = computeInitialRotation(rotationCameraDegrees)

    video_width = dataPKL.get('video_width', 640)
    video_height = dataPKL.get('video_height', 480)

    # Then apply camera transformations for each frame
    for i in range(len(allFrameHumans)):
        # Get camera pose for this frame
        cameraPositionSrc = vggtPKL['interpolated_relative_positions'][i]
        cameraPosition = np.zeros(3, dtype=np.float32)
        cameraPosition[0] = -cameraPositionSrc[0]*factor
        cameraPosition[1] = cameraPositionSrc[1]*factor
        cameraPosition[2] = -cameraPositionSrc[2]*factor
        cameraRotation = vggtPKL['interpolated_relative_rotations'][i]
        
        # Convert Euler angles to rotation matrix
        camera_rot_matrix, _ = cv2.Rodrigues(cameraRotation)
        
        for j in range(len(allFrameHumans[i])):
            human = allFrameHumans[i][j]
            human['j3d_smplx'] = (R_np @ (human['j3d_smplx'] - pivot).T).T + pivot
            global_rot_vec_np = human['rotvec'][0]
            global_rot_mat, _ = cv2.Rodrigues(global_rot_vec_np)
            corrected_global_rot_mat = R_np @ global_rot_mat
            corrected_global_rot_vec, _ = cv2.Rodrigues(corrected_global_rot_mat)
            human['rotvec'][0] = corrected_global_rot_vec.reshape(1, 3)

            # Apply rotation to all joints
            human['j3d_smplx'] = (camera_rot_matrix @ (human['j3d_smplx'] - pivot).T).T + pivot
            
            # Apply translation to all joints

            human['j3d_smplx'] = human['j3d_smplx'] + cameraPosition
            
            # Update the global orientation
            global_rot_vec_np = human['rotvec'][0]
            global_rot_mat, _ = cv2.Rodrigues(global_rot_vec_np)
            corrected_global_rot_mat = camera_rot_matrix @ global_rot_mat
            corrected_global_rot_vec, _ = cv2.Rodrigues(corrected_global_rot_mat)
            human['rotvec'][0] = corrected_global_rot_vec.reshape(1, 3)
            
            # Update translation to match head position after all transforms
            human['transl'] = human['j3d_smplx'][15]


    # Apply optional floor stabilization
    if floor_stabilize > 0:
        allFrameHumans = stabilize_floor_contact(allFrameHumans, contact_threshold)
        
    # Calculate the floor offset (minimum Y coordinate of all joints)
    all3DJoints = []
    for i in range(len(allFrameHumans)):
        humans = allFrameHumans[i]
        # if len(humans) == 0:
        #     continue
        for h in humans:
            all3DJoints.append(h['j3d_smplx'])

    min_y = 0
    all3DJoints = np.array(all3DJoints)
    # y_coords = -all3DJoints[:, :, 1].flatten()
    # min_y = np.min(y_coords)
    # print ("min_y: ", min_y)
      
    # Define foot joint indices in SMPLX model
    foot_joints = [7, 8, 10, 11]  # Left ankle, right ankle, left toe, right toe

    # Extract only foot joints for more accurate floor detection
    foot_y_coords = []
    for i in range(len(allFrameHumans)):
        humans = allFrameHumans[i]
        for h in humans:
            foot_y = -h['j3d_smplx'][foot_joints, 1]
            foot_y_coords.extend(foot_y)

    foot_y_coords = np.array(foot_y_coords)

    # Use 5th percentile instead of minimum to be robust against outliers
    min_y = np.percentile(foot_y_coords, 5) if len(foot_y_coords) > 0 else 0
    print("Robust min_y (5th percentile of foot joints):", min_y)
    # Apply floor offset to all skeletons so the floor is at Y=0
    print("Applying floor offset to align floor at Y=0...")
    for i in range(len(allFrameHumans)):
        humans = allFrameHumans[i]
        for human in humans:
            # Apply the Y offset to all joints (note: the negative sign because we used -Y earlier)
            human['j3d_smplx'][:, 1] += min_y  # Adding because min_y accounts for the negative
            # Update translation to reflect the new position
            human['transl'] = human['j3d_smplx'][15]  # Assuming 15 is the head/reference joint
            # Update the 3D bounding box based on the adjusted joints
            min_coords, max_coords = keypointsBBox3D(human['j3d_smplx'])
            human['bbox3d'] = [min_coords, max_coords]

    globalCameraRotations = []
    globalCameraTranslations = []

    # Store camera transforms for each frame
    print("Storing camera transforms for visualization...")
    for i in range(len(allFrameHumans)):
        # Get the original camera pose (without negation for visualization purposes)
        # cameraPosition = vggtPKL['interpolated_relative_positions'][i]*factor
        cameraPositionSrc = vggtPKL['interpolated_relative_positions'][i]
        cameraPosition = np.zeros(3, dtype=np.float32)
        cameraPosition[0] = -cameraPositionSrc[0]*factor
        cameraPosition[1] = cameraPositionSrc[1]*factor-min_y
        cameraPosition[2] = cameraPositionSrc[2]*factor
        cameraRotation = vggtPKL['interpolated_relative_rotations'][i]

        # Convert Euler angles to rotation matrix
        camera_rot_matrix, _ = cv2.Rodrigues(cameraRotation)
        corrected_global_rot_mat = R_np @ camera_rot_matrix
        corrected_global_rot_vec, _ = cv2.Rodrigues(corrected_global_rot_mat)

        globalCameraRotations.append(corrected_global_rot_vec)
        globalCameraTranslations.append(cameraPosition)
    
    # Add to the output data
    dataPKL['camera_rotations'] = np.array(globalCameraRotations)
    dataPKL['camera_translations'] = np.array(globalCameraTranslations)
    
    # Set floor offset to 0 since we've applied it to the data
    dataPKL['camera_rotation_deg'] = [0, 0, 0]
    dataPKL['floor_angle_deg'] = 0
    dataPKL['floor_Zoffset'] = 0 # Set to 0 since we've applied the offset
    
    # Save the pkl file
    print ("write pkl file: ", outputpklName)
    with open(sys.argv[2], 'wb') as handle:
        pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL) 

if __name__ == '__main__':
    main()

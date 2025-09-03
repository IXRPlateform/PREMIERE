import sys
import pickle
import numpy as np
import cv2
import math

from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution
from scipy.optimize import minimize_scalar

from premiere.functionsMoge import computeInitialRotation
from premiere.functionsCommon import loadPkl2, keypointsBBox3D

# Description: This script is used to convert the poses from the camera frame to the floor frame.
def main():
    if len(sys.argv) != 4:
        print("Usage: python cameraFloorCompensation.py <input_pkl> <output_pkl> <moge_pkl>")
        sys.exit(1)

    input_pkl = sys.argv[1]
    outputpklName = sys.argv[2]
    moge_pkl = sys.argv[3]

    # Charger les données
    dataPKL, mogePKL = loadPkl2(input_pkl, moge_pkl)
    allFrameHumans = dataPKL['allFrameHumans']
    
    pivot = np.array([0, 0, 0])
    rotationCameraDegrees = mogePKL['rotation_camera_degrees']
    R_np = computeInitialRotation(rotationCameraDegrees)

    video_width = dataPKL.get('video_width', 640)
    video_height = dataPKL.get('video_height', 480)

    # Then apply camera transformations for each frame
    for i in range(len(allFrameHumans)):
        
        for j in range(len(allFrameHumans[i])):
            human = allFrameHumans[i][j]
            human['j3d_smplx'] = (R_np @ (human['j3d_smplx'] - pivot).T).T + pivot
            global_rot_vec_np = human['rotvec'][0]
            global_rot_mat, _ = cv2.Rodrigues(global_rot_vec_np)
            corrected_global_rot_mat = R_np @ global_rot_mat
            corrected_global_rot_vec, _ = cv2.Rodrigues(corrected_global_rot_mat)
            human['rotvec'][0] = corrected_global_rot_vec.reshape(1, 3)
            
            # Update translation to match head position after all transforms
            human['transl'] = human['j3d_smplx'][15]

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
    min_y = np.min(foot_y_coords) if len(foot_y_coords) > 0 else 0
 
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
        cameraPosition = np.zeros(3, dtype=np.float32)
        cameraRotation = np.zeros(3, dtype=np.float32)
        cameraPosition[1] = cameraPosition[1]-min_y
        globalCameraRotations.append(cameraRotation)
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

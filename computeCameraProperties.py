"""
Compute camera properties from pickle files and save to JSON.

This script reads VGGT, MoGe, and data pickle files, extracts camera parameters,
computes necessary transformations, and saves the results to a JSON file.

Usage:
  python computeCameraProperties.py <vggt_pkl> <moge_pkl> <input_pkl> <fov> <final_json>

:param vggt_pkl: Path to VGGT pickle file with camera parameters
:param moge_pkl: Path to MoGe pickle file with rotation information
:param input_pkl: Path to data pickle file with human detections
:param fov: Field of view in degrees (will be overridden by VGGT value)
:param final_json: Output JSON file path
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Import common functions
from premiere.functionsCommon import loadPkl, loadPkl3
from premiere.functionsMoge import computeInitialRotation

def main():
    """Main function to process command-line arguments and compute camera properties."""
    if len(sys.argv) != 6:
        print("Usage: python computeCameraProperties.py <vggt_pkl> <moge_pkl> <input_pkl> <fov> <final_json>")
        sys.exit(1)
        
    # Parse command line arguments
    vggtPklName = sys.argv[1]
    mogePklName = sys.argv[2]
    inputPklName = sys.argv[3]
    fov_x_degrees = float(sys.argv[4])  # Note: This gets overridden by vggtPKL value
    finalJSONName = sys.argv[5]
    
    display = False
    
    # Load pickle files using functionsCommon.loadPkl3
    print(f"Reading pickle files...")
    vggtPKL, mogePKL, dataPKL = loadPkl3(vggtPklName, mogePklName, inputPklName)
    
    # Extract camera parameters
    fov_x_degrees = vggtPKL['fov_x_degrees'][0] 
    fov_x_radians = fov_x_degrees * np.pi / 180
    dynamicFov = not vggtPKL["fov_fixed"]
    cameraFixed = vggtPKL["camera_fixed"]
    cameraMotionCoherent = vggtPKL["camera_motion_coherent"]
    
    # Extract rotation information
    angle = mogePKL['rotation_x_degrees']
    angle_rad = angle * np.pi / 180
    rotationCameraDegrees = mogePKL['rotation_camera_degrees']
    floorDetected = mogePKL['floor_detected']
    
    # Extract all 3D joints from detected humans
    allFrameHumans = dataPKL['allFrameHumans']
    all3DJoints = []
    
    for i in range(len(allFrameHumans)):
        humans = allFrameHumans[i]
        if len(humans) == 0:
            continue
        for h in humans:
            all3DJoints.append(h['j3d_smplx'])
    
    # Process 3D joints if available
    min_y = 0
    all3DJoints = np.array(all3DJoints)
    all3DJoints = all3DJoints.reshape(-1, 3)
    
    print("all3DJoints:", all3DJoints.shape)
    if len(all3DJoints) != 0:
        # Apply camera rotation to 3D joints
        R_camera = computeInitialRotation(rotationCameraDegrees)
        all3DJoints = (R_camera @ all3DJoints.T).T
        
        # Extract X and Y coordinates for visualization
        x_coords = all3DJoints[:, 2].flatten()
        y_coords = -all3DJoints[:, 1].flatten()
        
        # Calculate minimum Y coordinate (floor level)
        min_y = np.min(y_coords)
        print("min_y:", min_y)
        
        # Optionally display point cloud visualization
        if display:
            plt.scatter(x_coords, y_coords, s=2)
            plt.xlabel("X")
            plt.ylabel("Z")
            plt.title("2D Projection (X,Z)")
            plt.show()
    
    # Prepare output data
    allData = {
        'video': dataPKL['video'],
        'fov_x_deg': float(fov_x_degrees),
        'fov_x_rad': float(fov_x_radians),
        'video_width': dataPKL['video_width'],
        'video_height': dataPKL['video_height'],
        'video_fps': dataPKL['video_fps'],
        'floor_angle_deg': angle,
        'floor_angle_rad': angle_rad,
        'camera_fixed': cameraFixed,
        'dynamic_fov': dynamicFov,
        'camera_motion_coherent': cameraMotionCoherent,
        'camera_rotation_deg': rotationCameraDegrees,
        'camera_rotation_rad': np.radians(rotationCameraDegrees).tolist(),
        'floor_detected': floorDetected,
        'floor_Zoffset': float(min_y)
    }
    print(allData)
    
    # Write output JSON
    print("Writing JSON:", finalJSONName)
    with open(finalJSONName, 'w') as file:
        json.dump(allData, file)

if __name__ == "__main__":
    main()








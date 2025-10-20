#!/usr/bin/env python3
"""
Script to load SMPL-X parameters from a pickle file, optionally apply a floor
compensation (rotation and offset), and then either save the result as GLB/OBJ
models or process 3D joints. The script uses smplx to build the SMPL-X model
and either saves the meshes or just processes the data, depending on the
arguments. This script also 
"""

import os
import math
import pickle
import torch
import smplx
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import json
import sys
# Functions from a local module that handle saving OBJ/GLB files
# and estimating the 3D keypoints from SMPL-X parameters with optional floor compensation.

from extract3DFileFromPkl import (
    save_obj, 
    save_glb, 
    estimateFromSMPLXJ3DWithFloor
)

# Read SMPL-X model path from environment variable
models_path = os.environ["MODELS_PATH"]

# Apply the offset based on the sign of the values
def apply_offset(value, offset):
    if value >= 0:
        return value + offset
    else:
        return value - offset
def main():
    """
    Main function to parse arguments, load a PKL file with SMPL-X parameters,
    create a SMPL-X model, apply optional floor transformation, and export
    the resulting meshes in .obj or .glb format (or none) if needed. Also compute the minmax dimension of the scene and dump
    """
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default='',
                        help="name of the input PKL file containing SMPL-X parameters which is to be preocessed. Example: nlf-final.pkl")
    parser.add_argument("--input_dir", type=str, default='', help="Path to the scenes of the video")
    parser.add_argument("--output_3dformat", type=str, choices=['obj', 'glb', 'none'], default='glb',
                        help="Output format for 3D files: obj, glb, or none.")
    parser.add_argument("--use_floor", action="store_true",
                        help="Apply floor compensation using stored or specified angle/offset.")
    parser.add_argument("--floor_angle_deg", type=float, default=0,
                        help="Floor angle in degrees to be applied to all frames if --use_floor is set.")
    parser.add_argument("--floor_Zoffset", type=float, default=0,
                        help="Floor Z offset applied to all frames if --use_floor is set.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run SMPL-X model on. Default='cpu'.")
    parser.add_argument("--save_indiv", action="store_true",
                        help="Save individual obj for each human")
    parser.add_argument("--process_3Dmesh", action="store_true",
                        help="Process the 3D mesh saving")
    
    
    args = parser.parse_args()

    inputPkl = args.input_pkl
    
    PROCESS_MESH=args.process_3Dmesh
    if PROCESS_MESH:
        output3DFormat = args.output_3dformat

    inputdirectory=args.input_dir
    
    floorAngle = math.radians(args.floor_angle_deg)
    floorZoffset = args.floor_Zoffset
    scenes=sorted([folder for folder in os.listdir(inputdirectory) if folder.startswith("scene_")], key=lambda x: int(x.split("_")[1]))
      # Create SMPL-X model on the chosen device
    device = torch.device(args.device)
    modelSMPLX = smplx.create(
        models_path, 
        model_type='smplx',
        gender='neutral',
        use_pca=False, 
        flat_hand_mean=True,
        num_betas=10,
        ext='npz'
    ).to(device)
    scenes_area_data = []
   
    for scene_idx,scene in enumerate(scenes):
        print("Processing scene:", scene)
        inputPklName = os.path.join(inputdirectory,scene,f'{inputPkl}')
        print("Reading input PKL:", inputPklName)
        outputDir = os.path.join(inputdirectory,scene,f'corrected_3d_poses')
        scene_area_path=os.path.join(inputdirectory)
      
        

        with open(inputPklName, 'rb') as f:
            allDataPKL = pickle.load(f)

        # 'allFrameHumans' contains SMPL-X parameters for each frame
        dataPKL = allDataPKL['allFrameHumans']
        print("Number of frames:", len(dataPKL))
        frameNumber = len(dataPKL)


      

        print("Model name:", allDataPKL['model_name'])
        
        # Some data might not include facial expression parameters
        useExpression = True
        if allDataPKL['model_type'] in ["hmr2", "nlf"]:
            useExpression = False

        # --------------------------------------------------------------------------
        # JSON or other data processing placeholder
        # Here we run the "estimateFromSMPLXJ3DWithFloor" function to get 3D joints
        # --------------------------------------------------------------------------
        print("Processing frames to estimate 3D joints (with optional floor compensation).")
        # pbar = tqdm(total=len(dataPKL), unit=' frames', dynamic_ncols=True, position=0, leave=True)
        # for i in range(len(dataPKL)):
        #     for j in range(len(dataPKL[i])):
        #         # j3d: floor-compensated 3D keypoints
        #         # j3dOri: the original 3D keypoints
        #         j3d, t_id = estimateFromSMPLXJ3DWithFloor(
        #             dataPKL[i][j], 
        #             modelSMPLX, 
        #             floorAngles[i], 
        #             floorZoffsets[i],
        #             useExpression, 
        #             device
        #         )
        #         j3dOri = dataPKL[i][j]['j3d_smplx']
        #         final_j3d= {'id': t_id, 'j3d': j3d}
        #         #save j3d as .pkl file containing t_id and j3d data
        #         outputModelName = os.path.join(outputDir, f"{i:06d}_{j:02d}.pkl")
        #         with open(outputModelName, 'wb') as f:
        #             pickle.dump(final_j3d, f)
        #         # You can handle j3d or j3dOri here or in your further logic...
        #     pbar.update(1)
        # pbar.close() 
       
        if not PROCESS_MESH:
            all_human_pelvis=[]
            all_human_ypoint=[]
            os.makedirs(outputDir, exist_ok=True)
            pbar = tqdm(total=len(dataPKL), unit=' frames', dynamic_ncols=True, position=0, leave=True)
            if all(len(i) == 0 for i in dataPKL):
                final_j3d = dataPKL
                outputModelName = os.path.join(outputDir, f"{scene}.pkl")
                #print("outputModelName:", outputModelName)
                with open(outputModelName, 'wb') as f:
                    pickle.dump(final_j3d, f)
            else:
                for i in range(len(dataPKL)):
                    if len(dataPKL[i]) == 0:
                        # Save empty file if dataPKL[i] is empty

                        final_j3d = {'id': None, 'BBox': [], 'j3d': []}
                        outputModelName = os.path.join(outputDir, f"{i:06d}_00.pkl")
                        with open(outputModelName, 'wb') as f:
                            pickle.dump(final_j3d, f)
                        #print(f"Saved empty file: {outputModelName}")
                    for j in range(len(dataPKL[i])):
                        # j3d: floor-compensated 3D keypoints
                        # j3dOri: the original 3D keypoints
                     
                        j3d = estimateFromSMPLXJ3DWithFloor(
                            dataPKL[i][j], 
                            modelSMPLX,
                            useExpression, 
                            device
                        )
                       
                        
                        t_id = dataPKL[i][j]['id']
                          # take min x, min y, min z and max x, max y, max z from j3d
                        # to calculate bounding box
                        min_x = min(j3d[:,0])
                        min_y = min(j3d[:,1])
                        min_z = min(j3d[:,2])
                        max_x = max(j3d[:,0])
                        max_y = max(j3d[:,1])
                        max_z = max(j3d[:,2])
                        # calculate bounding box dimensions
                        BBox = [min_x, min_y, min_z, max_x, max_y, max_z]
                        final_j3d = {'id': t_id, 'BBox':BBox, 'j3d': j3d}
                        # Save j3d as .pkl file containing t_id and j3d data
                        all_human_pelvis.append(j3d[0,:])
                        all_human_ypoint.append(j3d[:,1])
                        outputModelName = os.path.join(outputDir, f"{i:06d}_{j:02d}.pkl")
                        with open(outputModelName, 'wb') as f:
                            pickle.dump(final_j3d, f)
                            
                    pbar.update(1)
                pbar.close()
            all_human_pelvis = np.array(all_human_pelvis)

            # Find the min max dimension of the scene to save a .json
            if np.any(all_human_pelvis):
                
                # Extract min values for x (index 0) and z (index 2) axes
                min_x = float(np.min(all_human_pelvis[:, 0]))  # x-axis
                min_z = float(np.min(all_human_pelvis[:, 2])) # z-axis
                max_x = float(np.max(all_human_pelvis[:, 0]))  # x-axis
                max_z = float(np.max(all_human_pelvis[:, 2]))  # z-axis
                all_y_values = np.concatenate(all_human_ypoint)
                min_y = float(np.min(all_y_values))
                max_y = float(np.max(all_y_values))
                offset=1
        
   
                min_x -= offset  # Subtract offset from min values
                max_x += offset  # Add offset to max values
                #min_y -= offset  # Subtract offset from min values
                max_y += offset  # Add offset to max values
                min_z -= offset  # Subtract offset from min values
                max_z += offset  # Add offset to max values
            else:
                min_x = 0  # Subtract offset from min values
                max_x = 0  # Add offset to max values
                min_y = 0  # Subtract offset from min values
                max_y = 0  # Add offset to max values
                min_z = 0  # Subtract offset from min values
                max_z = 0  # Add offset to max values

            scene_data = {"scene": f"{scene_idx}","min_x": min_x,"max_x": max_x,"min_y": min_y,"max_y": max_y,"min_z": min_z,"max_z": max_z}
            print("Scnee data is: ",scene_data)
            scenes_area_data.append(scene_data)
   
       
     # Save to JSON
   
    json_file_scene_Area=os.path.join(scene_area_path,"area_scene_minmax.json")
   
    with open(json_file_scene_Area, "w") as json_file:
        json.dump(scenes_area_data, json_file, indent=4)






if __name__ == "__main__":
    main()

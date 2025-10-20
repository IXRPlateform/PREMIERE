#!/usr/bin/env python3
"""
Script to load SMPL-X parameters from a pickle file, optionally apply a floor
compensation (rotation and offset), and then either save the result as GLB/OBJ
models or process 3D joints. By default we save it as a glb file. The script uses smplx to build the SMPL-X model
and either saves the meshes or just processes the data, depending on the
arguments.
"""

import os
import math
import pickle
import torch
import smplx
import shutil
from tqdm import tqdm
from argparse import ArgumentParser

# Functions from a local module that handle saving OBJ/GLB files
# and estimating the 3D keypoints from SMPL-X parameters with optional floor compensation.
from Save3DFile import (
    save_obj, 
    save_glb, 
)

# Read SMPL-X model path from environment variable
models_path = os.environ["MODELS_PATH"]
folder3D_name="3Dmesh"



def main(args):
    
    inputPkl = args.input_pkl
    output3DFormat = args.output_3dformat
    inputdirectory=args.input_dir
    floorAngle = math.radians(args.floor_angle_deg)
    floorZoffset = args.floor_Zoffset
    obj_resolution= args.resolution_thresh

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

    # Sort all the scenes folder
    scenes = sorted([folder for folder in os.listdir(inputdirectory) if folder.startswith("scene_")], key=lambda x: int(x.split("_")[1]))
    

    scene_frame_counts = {}  # Store the frame count of each scene for tracking

    # First loop to compute total_frames correctly
    for scene in scenes:
        inputPklName = os.path.join(inputdirectory, scene, f'{inputPkl}')
        with open(inputPklName, 'rb') as f:
            allDataPKL = pickle.load(f)
        scene_number = int(scene.split("_")[1])
        scene_frame_counts[scene_number] = len(allDataPKL["allFrameHumans"])

    #  If a specific scene number is given, filter the scenes list
    if args.scene_number is not None:
        specific_scenes = [f"scene_{str(num).zfill(3)}" for num in args.scene_number]
        valid_scenes = [scene for scene in specific_scenes if scene in scenes]
        print("The valid scenes are: ",valid_scenes)

        if valid_scenes:
            scenes = valid_scenes  # Keep only the valid scenes
            print("Final scene list:", scenes)
        else:
            print(f"None of the specified scenes were found: {specific_scenes}")
    else:
        print("No specific scene numbers provided, using all available scenes.")

        total_frames=0
    
    parent_directory = os.path.dirname(os.path.dirname(inputdirectory))
   
    # Start to loop inside each scenes to process and extract the mesh
    for scene in scenes:
        scene_number = int(scene.split("_")[1])
        print(scene_number)  
        if args.scene_number is not None:
            # Store the total number of frames before current scene for global frame consistency
            total_frame_before_current_scene=sum(frame_counts for scene_numbers_data, frame_counts in scene_frame_counts.items() if scene_numbers_data < scene_number) 
            total_frames=total_frame_before_current_scene

       
        print("Processing scene:", scene)
        inputPklName = os.path.join(inputdirectory,scene,f'{inputPkl}')
        print("Reading input PKL:", inputPklName)
        outputDir = os.path.join(inputdirectory,scene,f'corrected_3d_poses')
        print(outputDir)
        

        with open(inputPklName, 'rb') as f:
            allDataPKL = pickle.load(f)

        # 'allFrameHumans' contains SMPL-X parameters for each frame
        dataPKL = allDataPKL['allFrameHumans']
        print("Number of frames:", len(dataPKL))
        frameNumber = len(dataPKL)
        frame_data=allDataPKL["allFrameHumans"]
        scene_frames_count=len(frame_data)
        

        

        

        print("Model name:", allDataPKL['model_name'])
        
        # Some data might not include facial expression parameters
        useExpression = True
        if allDataPKL['model_type'] in ["hmr2", "nlf"]:
            useExpression = False

        # Create a temporary directory inside each scenes to store the mesh before moving to body folder
        meshdir3D = os.path.join(inputdirectory,scene,folder3D_name)
        if not os.path.exists(meshdir3D):
            os.makedirs(meshdir3D)
        
        print(f"Created directory: {meshdir3D}")
        
        if output3DFormat != 'none':
            ext = output3DFormat.lower()
            print(f"Saving {ext.upper()} files to {meshdir3D}")
            pbar = tqdm(total=frameNumber, unit=' frames', dynamic_ncols=True, position=0, leave=True)

            if ext == "obj":
                for i in range(frameNumber):
                    f_num=total_frames+i
                    outputModelName = os.path.join(meshdir3D, f"{f_num:06d}.{ext}")
                    save_obj(
                        humans_in_frame=dataPKL[i],
                        filename=outputModelName,
                        model=modelSMPLX,
                        obj_res=obj_resolution,
                        useExpression=useExpression,
                        device=args.device,
                        save_individal_obj=args.save_indiv
                    )
                    pbar.update(1)

            elif ext == "glb":
                for i in range(frameNumber):
                    f_num=total_frames+i
                    outputModelName = os.path.join(meshdir3D, f"{f_num:06d}.{ext}")
                    save_glb(
                        humans_in_frame=dataPKL[i],
                        filename=outputModelName,
                        model=modelSMPLX,
                        useExpression=useExpression,
                        device=args.device,
                        save_individal_glb=args.save_indiv
                    )
        
                    pbar.update(1)
            pbar.close()
        if args.scene_number==None:
            total_frames+=scene_frames_count


def copy_files(args):
    inputdirectory=args.input_dir
    parent_directory = os.path.dirname(os.path.dirname(inputdirectory))
    result_scenes = sorted([folder for folder in os.listdir(inputdirectory) if folder.startswith("scene_")], key=lambda x: int(x.split("_")[1]))
    if args.scene_number is not None:
        specific_scenes = [f"scene_{str(num).zfill(3)}" for num in args.scene_number]
        valid_scenes = [scene for scene in specific_scenes if scene in result_scenes]
       
        if valid_scenes:
            result_scenes = valid_scenes  # Keep only the valid scenes
            
    mesh_dir_name="body"
    main_output_dir=os.path.join(parent_directory,mesh_dir_name)
    if not os.path.exists(main_output_dir):
            os.makedirs(main_output_dir) 
  

    for scene_folder in result_scenes:
        full_scene_path = os.path.join(inputdirectory, scene_folder)
        full_obj_folder_path = os.path.join(full_scene_path, folder3D_name)
        
        
        # Ensure the scene's objfolder exists
        if os.path.exists(full_obj_folder_path):
            
            
            # Create destination folder for the scene if it doesn't exist
            full_destination_scene_path = os.path.join(main_output_dir, scene_folder)
            
            # Create the full destination path for this scene folder if it doesn't exist
            if not os.path.exists(full_destination_scene_path):
                os.makedirs(full_destination_scene_path)
            # Get all files in the obj folder (e.g., .obj, .mtl)
            files_in_objfolder = os.listdir(full_obj_folder_path)

            if not files_in_objfolder:
                print(f"No files found in {full_obj_folder_path}")
            
            
            # Move all files
            print(f"Moving mesh files to {mesh_dir_name} directory")
            pbar = tqdm(total=len(files_in_objfolder), unit=' mesh', dynamic_ncols=True, position=0, leave=True)
            
            for file_name in files_in_objfolder:
                old_file_path = os.path.join(full_obj_folder_path, file_name)
                
                if os.path.exists(old_file_path):
                    # Ensure the destination directory exists before copying
                    new_file_path = os.path.join(full_destination_scene_path, file_name)

                    # Move the file
                    shutil.move(old_file_path, new_file_path)
                    #print(f"Copied: {old_file_path} -> {new_file_path}")
                    
                else:
                    print(f"File does not exist: {old_file_path}")  # Log if file does not exist
                pbar.update(1)  # Update progress for each scene
            shutil.rmtree(full_obj_folder_path)  # Removes directory and all its contents
        else:
            print(f"obj folder path does not exist for {scene_folder}: {full_obj_folder_path}")
            

if __name__ == "__main__":
      # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default='',
                        help="name of the input PKL file containing SMPL-X parameters which is to be preocessed. Example: nlf-final.pkl")
    parser.add_argument("--input_dir", type=str, default='', help="Path to the scenes of the video")
    parser.add_argument("--output_3dformat", type=str, choices=['obj', 'glb', 'none'], default='glb',
                        help="Output format for 3D files: obj, glb, or none.")
    parser.add_argument("--use_floor", action="store_true",default=True,
                        help="Apply floor compensation using stored or specified angle/offset.")
    parser.add_argument("--floor_angle_deg", type=float, default=0,
                        help="Floor angle in degrees to be applied to all frames if --use_floor is set.")
    parser.add_argument("--floor_Zoffset", type=float, default=0,
                        help="Floor Z offset applied to all frames if --use_floor is set.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run SMPL-X model on. Default='cpu'.")
    parser.add_argument("--save_indiv", action="store_true",default=True,
                        help="Save individual obj for each human")
    parser.add_argument("--scene_number",type=int,nargs="+",default=None,help="Scene numbers to create GLB files for only those scenes (provide space-separated values)")
    parser.add_argument("--resolution_thresh", type=float, default=1,
                        help="Decrease the resolution of the obj mesh as required by the user")


    args = parser.parse_args()
    main(args)
    copy_files(args)

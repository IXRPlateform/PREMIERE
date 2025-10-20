import os
import pickle
import json
import numpy as np
import shutil
from argparse import ArgumentParser
import logging

# Initialize global frame index
global_frame_index = 0

def combine_correct_3d_poses(directory):
    print("The directory is: ",directory)
    corrected_3d_poses = os.path.join(directory, 'corrected_3d_poses')
    pkl_files = [f for f in os.listdir(corrected_3d_poses) if f.endswith('.pkl')]
    pkl_files.sort()

    # Dictionary to store all frame data
    final_data = {'allFrameHumans': []}
    frame_data = {}

    if len(pkl_files) == 1:
        file_path = os.path.join(corrected_3d_poses, pkl_files[0])
        with open(file_path, 'rb') as infile:
            data = pickle.load(infile)
        final_data['allFrameHumans'] = data
    else:
        for fname in pkl_files:
            frame_id = int(fname.split('_')[0])  # Extract frame identifier as an integer
            file_path = os.path.join(corrected_3d_poses, fname)

            with open(file_path, 'rb') as infile:
                data = pickle.load(infile)
            if frame_id not in frame_data:
                frame_data[frame_id] = []
            frame_data[frame_id].append(data)  # Each frame stores multiple persons

        # Convert frame_data dictionary into a sorted list based on frame order
        final_data['allFrameHumans'] = [frame_data[key] for key in sorted(frame_data.keys())]
# save the combined data into a single pickle file
    output_path = os.path.join(directory, 'corrected_3d_poses.pkl')
    with open(output_path, 'wb') as outfile:
        pickle.dump(final_data, outfile)

    shutil.rmtree(corrected_3d_poses)

# Helper function: recursively convert ndarray and NumPy types to standard Python types
def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):  # If the object is a NumPy array, convert it
        return obj.tolist()
    elif isinstance(obj, list):  # If it's a list, recursively process its elements
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, dict):  # If it's a dictionary, process its values
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (np.float32, np.float64)):  # Convert NumPy floats to Python floats
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):  # Convert NumPy integers to Python integers
        return int(obj)
    else:
        return obj  # If it's neither, return as is

def pkl_to_json(scene_path, scene):
    global global_frame_index

    pkl_file_path1 = os.path.join(scene_path, input_pkl_type)
    pkl_file_path2 = os.path.join(scene_path, 'corrected_3d_poses.pkl')
    
    with open(pkl_file_path1, 'rb') as file:
        data1 = pickle.load(file)
    
    with open(pkl_file_path2, 'rb') as file2:
        data2 = pickle.load(file2)
  
    frames_data1 = data1.get('allFrameHumans', [])
    frames_data2 = data2.get('allFrameHumans', [])

    # Initialize lists for 2D and 3D data
    list_2d = []
    list_3d_corrected = []

    # Process each frame
    for frame_idx, (frame1, frame2) in enumerate(zip(frames_data1, frames_data2)):
        # Global frame numbering
        global_frame_no = global_frame_index
        
        # Create 2D and 3D frame data
        frame_2d = {"frame_no": global_frame_no, "no_person": len(frame1), "poses": []}
        frame_3d_corrected = {"frame_no": global_frame_no, "no_person": len(frame2), "poses": []}
        
        # Process each person in the frame
        for person1 in frame1:
            # Process 2D and 3D poses from `frame1`
            bbox2d = person1.get('bbox', [None, None])
            bbox2d_min = convert_ndarray_to_list(bbox2d[0])  # [xmin, ymin, zmin]
            bbox2d_max = convert_ndarray_to_list(bbox2d[1])  # [xmax, ymax, zmax]
            bbox2d_combined = bbox2d_min + bbox2d_max
            pose_2d = {"id": person1.get('id', None), "BBox": bbox2d_combined, "keypoints": [
                {"x": kp[0], "y": kp[1]} for kp in convert_ndarray_to_list(person1.get('j2d_smplx', []))],
                "score": convert_ndarray_to_list(person1.get('score', None))}
            frame_2d["poses"].append(pose_2d)

        for person2 in frame2:
            BBox=convert_ndarray_to_list(person2.get('BBox', [None, None]))
            pose_3d_corrected = {"id": person2.get('id', None), "BBox": BBox, "keypoints": [
                {"x": kp[0], "y": kp[1], "z": kp[2]} for kp in convert_ndarray_to_list(person2.get('j3d', []))]}
            frame_3d_corrected["poses"].append(pose_3d_corrected)
        
        # Add the processed frame to the lists
        list_2d.append(frame_2d)
        list_3d_corrected.append(frame_3d_corrected)

        # Increment global frame index for the next frame
        global_frame_index += 1
    # Save the processed data as JSON files
    output_dir = os.path.join(scene_path)
    
    with open(os.path.join(output_dir, f'{scene}_poses.json'), 'w') as f:
        json.dump(list_2d, f, indent=4)
    
    with open(os.path.join(output_dir, f'{scene}_poses3d_corrected.json'), 'w') as f:
        json.dump(list_3d_corrected, f, indent=4)

    # move the JSON files to the respective directories
    move_json_files(base_directory, scene, scene_path)

# Function to move JSON files to the appropriate directories
def move_json_files(base_directory, scene, scene_path):
    # Define the target directories
    target_2d_dir = f'{base_directory}/{output_dirname}/2d_poses'
    target_3d_corrected_dir=f'{base_directory}/{output_dirname}/3d_poses'

    # Create target directories if they don't exist
    os.makedirs(target_2d_dir, exist_ok=True)
    os.makedirs(target_3d_corrected_dir, exist_ok=True) 

    # Path to the 2D and 3D JSON files
    json_2d_path = os.path.join(scene_path, f'{scene}_poses.json')
    json_3d_corrected_path = os.path.join(scene_path, f'{scene}_poses3d_corrected.json')

    shutil.move(json_2d_path, os.path.join(target_2d_dir))
    shutil.move(json_3d_corrected_path, os.path.join(target_3d_corrected_dir))

def rename_json_files(directory):
    """
    Rename JSON files to start from 0 and increment sequentially.
    """
    # Get all JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])

    # Rename files to start from 0
    for idx, file_name in enumerate(json_files):
        if '3d' in file_name:
            new_name = f'scene_{idx:03}_poses3d.json'
        elif '3d_corrected' in file_name:
            new_name = f'scene_{idx:03}_poses3d_corrected.json'
        else:
            new_name = f'scene_{idx:03}_poses.json'
        
        old_file_path = os.path.join(directory, file_name)
        new_file_path = os.path.join(directory, new_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)

def update_json_structure(directory):
    """
    Update the structure of each renamed JSON file.
    """
    # Get all JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])

    for file_name in json_files:
        # Construct the full file path
        file_path = os.path.join(directory, file_name)
        
        # Open the existing JSON file and load its contents
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Modify the structure
        scene_no = int(file_name.split('_')[1])
        start_frame = data[0]["frame_no"]  # Start frame is the frame_no of the first frame
        end_frame = data[-1]["frame_no"]   # End frame is the frame_no of the last frame

        # Create the new structure
        updated_structure = {
            "scene_no": scene_no,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": data
        }

        # Save the updated structure back to the file
        with open(file_path, 'w') as f:
            json.dump(updated_structure, f, indent=4)

def process_files_and_generate_bboxes(directory, target_dir, is_3d=False):
    """
    Process 2D or 3D pose files, remove keypoints, and generate new files with combined bounding boxes.
    """
    # Get all JSON files that start with 'scene' and end with '.json'
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json') and f.startswith('scene')])

    # List to hold new data for index.json
    index_data = []

    for file_name in json_files:
        file_path = os.path.join(directory, file_name)

        # Read the scene JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Process the data to remove keypoints
        for frame in data["frames"]:
            for pose in frame["poses"]:
                # Remove keypoints from the pose data
                pose.pop('keypoints', None)

        # Define the new file path for the updated JSON
        if "3d" in file_name:
            new_file_name = file_name.replace('poses', '') 
        else:
            new_file_name = file_name.replace('_poses', '') 

        new_file_path = os.path.join(target_dir, new_file_name)

        # Save the updated JSON file
        with open(new_file_path, 'w') as f:
            json.dump(data, f, indent=4)

def combine_scenes(directory,is_poses=False):
    """
    Combine all the scenes in a given directory (2d or 3d) into a single JSON file.
    """
    scenes = []
    # Get all JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])

    for file_name in json_files:
        file_path = os.path.join(directory, file_name)
        
        # Read the data from each scene JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract the scene data
        scene_no = data['scene_no']
        start_frame = data['start_frame']
        end_frame = data['end_frame']
        frames = data['frames']

        # Create a scene object with the desired structure
        scene = {
            "scene_no": scene_no,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": frames
        }

        scenes.append(scene)

    # Define the output filename based on whether it's 2D or 3D
    output_filename = 'BBs.json' if not is_poses else 'poses.json'
    output_path = os.path.join(directory, output_filename)
    final_structure = {"scenes": scenes}

    # Save the combined JSON to the output file
    with open(output_path, 'w') as f:
        json.dump(final_structure, f, indent=4)

def generate_index_json(directory):
    """
    Generate an index.json file for the given directory (2D or 3D poses).
    The index will contain scene info including scene_no, start_frame, end_frame, and the path.
    """
    # Get all JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json') and f.startswith('scene')])

    index_data = []
    

    for file_name in json_files:
        file_path = os.path.join(directory, file_name)

        # Read the scene JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract scene details
        scene_no = data["scene_no"]
        start_frame = data["start_frame"]
        end_frame = data["end_frame"]

        # Define the path to the file (relative to the base directory)
        relative_path = os.path.join(directory, file_name)
        
        # Create a scene object for the index
        scene_info = {
            "scene_no": scene_no,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "path": relative_path.replace(base_directory, '') 
        }

        index_data.append(scene_info)

    # Define the output path for the index.json file
    output_index_path = os.path.join(directory, 'index.json')

    # Save the index data to the index.json file
    with open(output_index_path, 'w') as f:
        json.dump(index_data, f, indent=4)

def generate_3Dindex_json(directory):
    """
    Generate an index.json file for the given directory (2D or 3D poses).
    The index will contain scene info including scene_no, start_frame, end_frame, and the path.
    """
    # Get all JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json') and f.startswith('scene')])

    index_data = []
    additional_scene_dimension=os.path.join(base_directory, 'area_scene_minmax.json')
    # Load Stage area min max values
    with open(additional_scene_dimension, 'r') as f:
        additional_info = json.load(f)
    for file_name in json_files:
        file_path = os.path.join(directory, file_name)

        # Read the scene JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract scene details
        scene_no = data["scene_no"]
        start_frame = data["start_frame"]
        end_frame = data["end_frame"]

        # Define the path to the file (relative to the base directory)
        relative_path = os.path.join(directory, file_name)
        
        scene_dimension_info = next(
            (info for info in additional_info if info["scene"] == str(scene_no)), None
        )
        # Create a scene object for the index
        scene_info = {
            "scene_no": scene_no,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "path": relative_path.replace(base_directory, '') 
        }
        if scene_dimension_info:
            scene_info.update({
                "min_x": scene_dimension_info["min_x"],
                "max_x": scene_dimension_info["max_x"],
                "min_y": scene_dimension_info["min_y"],
                "max_y": scene_dimension_info["max_y"],
                "min_z": scene_dimension_info["min_z"],
                "max_z": scene_dimension_info["max_z"]
            })
        index_data.append(scene_info)

    # Define the output path for the index.json file
    output_index_path = os.path.join(directory, 'index.json')

    # Save the index data to the index.json file
    with open(output_index_path, 'w') as f:
        json.dump(index_data, f, indent=4)

def copy_scenespkl(args_parser):
    base_directory=args_parser.directory
   
    result_scenes = sorted([folder for folder in os.listdir(base_directory) if folder.startswith("scene_")], key=lambda x: int(x.split("_")[1]))
    scene_folder_name="scenepkl"

    scene_pkl_folder = os.path.join(base_directory, output_dirname,scene_folder_name)
    if not os.path.exists(scene_pkl_folder):
        os.makedirs(scene_pkl_folder) 
    for scene_folder in result_scenes:
        # Extract the numeric part of the scene name
        scene_number = int(scene_folder.split("_")[1]) - 1  # Subtract 1
    
        # Format it back to the same naming convention
        new_scene_folder_name = f"scene_{scene_number:03d}"
    
        # Create the new folder path
        new_scene_folder = os.path.join(scene_pkl_folder, new_scene_folder_name)
        if not os.path.exists(new_scene_folder):
            os.makedirs(new_scene_folder) 
        old_scene_folder=os.path.join(base_directory,scene_folder)
        old_full_pickle_path=os.path.join(old_scene_folder,args_parser.input_pkl)

        if os.path.isfile(old_full_pickle_path):

            
            new_scene_pkle_file=os.path.join(new_scene_folder,args_parser.input_pkl)
           

            shutil.copy(old_full_pickle_path, new_scene_pkle_file)
        else:
            print(f"No pickle file for: {scene_folder}")
    


    
def process_scene(base_directory):
    scenes = sorted([folder for folder in os.listdir(base_directory) if folder.startswith("scene_")], key=lambda x: int(x.split("_")[1]))
    logging.info(f"Processing {len(scenes)} scenes")
    logging.info("Converting pkl data into json")
    for scene in scenes:
        scene_path = os.path.join(base_directory, scene)
        combine_correct_3d_poses(scene_path)
        pkl_to_json(scene_path, scene)
    logging.info("2d data converted and moved in directory")
    logging.info("3d data converted and moved in directory")

    dir_2d=f'{base_directory}/{output_dirname}/2d_poses'
    dir_3d_corrected=f'{base_directory}/{output_dirname}/3d_poses'
    logging.info("Renaming json files for 0-based scene numbering and updating the structure")
    rename_json_files(dir_2d)
    rename_json_files(dir_3d_corrected)
    update_json_structure(dir_2d)
    update_json_structure(dir_3d_corrected)
    
    logging.info("Extracting bounding boxes for 2D and 3D")
    # # # Extract bounding boxes from 2D and 3D poses
    dir_2d_bb = f'{base_directory}/{output_dirname}/2d_BoundingBoxes'
    dir_3d_bb = f'{base_directory}/{output_dirname}/3d_BoundingBoxes'
    os.makedirs(dir_2d_bb, exist_ok=True)
    os.makedirs(dir_3d_bb, exist_ok=True)
    process_files_and_generate_bboxes(dir_2d, dir_2d_bb, is_3d=False)
    process_files_and_generate_bboxes(dir_3d_corrected, dir_3d_bb, is_3d=True)
    
    logging.info("Combining 2D and 3D bounding boxes as well as poses")
    # Combine 2D and 3D bounding boxes as well as poses
    combine_scenes(dir_2d_bb, is_poses=False)
    combine_scenes(dir_3d_bb, is_poses=False) 
    combine_scenes(dir_2d, is_poses=True)
    combine_scenes(dir_3d_corrected, is_poses=True)

    logging.info("Generating index.json for 2D and 3D")
    # Generate index.json for both 2D and 3D
    generate_index_json(dir_2d)
    generate_index_json(dir_2d_bb)
    generate_3Dindex_json(dir_3d_bb) 
    generate_3Dindex_json(dir_3d_corrected)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Processing the data')
    parser = ArgumentParser()
    parser.add_argument("--directory", type=str, default=None, help="Directory to store the processed files")
    parser.add_argument("--input_pkl", type=str, default="nlf-final-filtered-floorc.pkl", help="input pickle file name to generate the annotation data")
    parser.add_argument("--output_dirname", type=str, default="annotationdata", help="Directory name that saves all the annotation data")
    args = parser.parse_args()
    base_directory=args.directory
    output_dirname=args.output_dirname
    input_pkl_type=args.input_pkl
    process_scene(base_directory)
    copy_scenespkl(args)

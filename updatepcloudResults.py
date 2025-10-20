import os
import json
import subprocess
import shutil
from gql.transport.aiohttp_websockets import AIOHTTPWebsocketsTransport

import os
import requests
from fastapi import HTTPException
import argparse
import sys
import concurrent.futures
from functools import partial
import time
"""Usage of the script
    
    python updatepcloudResults.py --folder_name <folder_name> --directory <directory> --folder_id <folder_id>
    
    Example:
    python updatepcloudResults.py --folder_name cm8ek9jj2003oqe0cfeqcou7s --directory /home/user/PREMIERETOOLV2 --folder_id 12345678
    
    """

# Define the subscription 
PCLOUD_ACCESS_TOKEN = os.getenv('PCLOUD_ACCESS_TOKEN',"I3khZnXis494Ylt0ZUOWkkkZdH0seO9mHdmaoAMLpWUwQV1kD1uX")
# pCloud API credentials and endpoints
pCloud_headers = {"Authorization": f"Bearer {PCLOUD_ACCESS_TOKEN}"}  
UPLOAD_DIRECTORY = "uploads"


# Définir les variables d'environnement du proxy
os.environ['HTTP_PROXY'] = 'http://cache.univ-st-etienne.fr:3128'
os.environ['HTTPS_PROXY'] = 'http://cache.univ-st-etienne.fr:3128'

# Proxy configuration
proxy = {
    "http": "http://cache.univ-st-etienne.fr:3128",
    "https": "http://cache.univ-st-etienne.fr:3128"
}

# Get the directory where mesh extraction is located
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "dimitri"))
def update_results(directory, folder_name):
    results_directory = os.path.join(directory, f'pcloudresults/{folder_name}')
    
    # Rename annotationdata folder if annotationdata_old does not exist
    if "annotationdata_old" not in os.listdir(results_directory):
        annotation_folder = os.path.join(results_directory, "annotationdata")
        annotation_folder_old = os.path.join(results_directory, "annotationdata_old")
        if os.path.exists(annotation_folder):  # Ensure the folder exists before renaming
            os.rename(annotation_folder, annotation_folder_old)

    #Load the video file path
    video_file = os.path.join(directory, f'pcloudvideos/{folder_name}.mp4')
    
    # Set the environment variable
    os.environ["MODELS_PATH"] = os.path.join(directory, "models")

    # Prepare the command to run the external script
    command = ["python","premiereFullVideoProcessing.py", "--video", video_file, "--directory", results_directory, "--step", "4", "--removeshadows"]
    try:
        subprocess.run(command, check=True)
        print("Video processing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during video processing: {e}")

    command=["python", "generatel3DfromPkl.py", "--input_dir", results_directory, "--use_floor", "--input_pkl", "nlf-final-filtered.pkl"]
    try:
        subprocess.run(command, check=True)
        print("Video processing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during video processing: {e}")
    
    command=["python", "arrange_data.py", "--directory", results_directory]
    try:
        subprocess.run(command, check=True)
        print("Video processing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during video processing: {e}")
    
    meshdir=os.path.join(results_directory,'annotationdata/scenepkl')
    #print("meshdir:",meshdir)
    #meshdata
    extract_mesh_script = os.path.join(current_dir, "dimitri", "extract3Dmesh.py")
    command=["python", extract_mesh_script, "--input_pkl", "nlf-final-filtered.pkl", "--input_dir", meshdir, "--output_3dformat", "glb"]
    try:
        subprocess.run(command, check=True)
        print("Video processing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during video processing: {e}")
    
    #move body to annotationdata
    shutil.move(os.path.join(results_directory, 'body'), os.path.join(results_directory, 'annotationdata'))
    shutil.rmtree(os.path.join(results_directory, 'annotationdata', 'scenepkl'))
     
     #Load JSON data
    json_file = os.path.join(directory, f'pcloudresults/{folder_name}/pcloud_folders.json')
    with open(json_file, "r") as file:
        data = json.load(file)
    return data






def upload_batch(upload_url, params, headers, proxy, file_batch):
    """Upload a batch of files with proper file handling"""
    files = {}
    batch_results = []
    
    try:
        # Open all files in this batch
        for i, file_path in enumerate(file_batch):
            filename = os.path.basename(file_path)
            files[f"file{i}"] = (filename, open(file_path, 'rb'))
        
        # Upload the batch
        response = requests.post(
            upload_url,
            params=params,
            headers=headers,
            files=files,
            proxies=proxy
        )
        
        # Close all files immediately after upload
        for f in files.values():
            f[1].close()
        
        if response.status_code == 200:
            result = response.json()
            if result.get("result") == 0:
                return [(True, os.path.basename(fp)) for fp in file_batch]
            else:
                return [(False, os.path.basename(fp), f"API error: {result}") for fp in file_batch]
        else:
            return [(False, os.path.basename(fp), f"HTTP {response.status_code}") for fp in file_batch]
            
    except Exception as e:
        # Ensure files are closed if error occurs
        for f in files.values():
            if not f[1].closed:
                f[1].close()
        return [(False, os.path.basename(fp), str(e)) for fp in file_batch]

def upload_files_to_pcloud_parallel_batch(folder_id, local_file_paths, max_workers, batch_size):
    """Combined parallel and batch upload function"""
    upload_url = "https://eapi.pcloud.com/uploadfile"
    params = {"folderid": folder_id}
    
    #logging.info(f"Starting parallel batch upload of {len(local_file_paths)} files (batch size: {batch_size}, workers: {max_workers})")
    
    # Create batches of files
    batches = [local_file_paths[i:i + batch_size] for i in range(0, len(local_file_paths), batch_size)]
    
    # Create partial function with fixed parameters
    upload_func = partial(upload_batch, upload_url, params, pCloud_headers, proxy)
    
    success_count = 0
    failed_files = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_func, batch): batch for batch in batches}
        
        for future in concurrent.futures.as_completed(futures):
            batch = futures[future]
            try:
                results = future.result()
                for success, filename, *error in results:
                    if success:
                        success_count += 1
                    else:
                        failed_files.append((filename, error[0] if error else "Unknown error"))
            except Exception as e:
                #logging.error(f"Batch failed completely: {str(e)}")
                failed_files.extend((os.path.basename(fp), "Batch failed") for fp in batch)
    
    total_time = time.time() - start_time
    #logging.info(f"Upload completed: {success_count}/{len(local_file_paths)} files in {total_time:.2f} seconds")
    print(f"Successfully uploaded {success_count}/{len(local_file_paths)} files to folder {folder_id}")
    
    if failed_files:
        #logging.warning(f"Failed to upload {len(failed_files)} files")
        print(f"Failed files: {failed_files}")
    
    return success_count == len(local_file_paths)
def multiple_upload_file_to_pcloud(data, local_folder_path, max_category_workers, file_batch_size, upload_workers):
    """Main upload function with parallel category processing and batch uploads"""
    # Create folder structure
    subfolder_ids = data['annotationDataFolderIds']

    # File categorization
    annotation_withoutbody_folders = {k: v for k, v in subfolder_ids.items() if k != "body"}
    files_by_category = {category: [] for category in annotation_withoutbody_folders.keys()}
    
    start_time = time.time()
    for root, _, files in os.walk(local_folder_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            is_json_file = file_name.endswith(".json")
            is_obj_file = file_name.endswith(".obj")
            is_glb = file_name.endswith(".glb")
            
            if is_json_file and "body" not in local_file_path:
                matched_category = next((c for c in files_by_category.keys() if c in local_file_path), None)
                if matched_category:
                    files_by_category[matched_category].append(local_file_path)
            elif is_obj_file or is_glb:
                matched_category = next((c for c in files_by_category.keys() if c in local_file_path), None)
                if matched_category:
                    files_by_category[matched_category].append(local_file_path)

    # Parallel upload by category with batch processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_category_workers) as executor:
        futures = []
        for category, file_paths in files_by_category.items():
            if file_paths:
                futures.append(executor.submit(
                    upload_files_to_pcloud_parallel_batch,
                    subfolder_ids[category],
                    file_paths,
                    upload_workers,
                    file_batch_size
                ))
        
        # Wait for all uploads to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                #logging.error(f"Error in category upload: {str(e)}")
                print(f"Error in category upload: {str(e)}")

    total_time = time.time() - start_time
    #logging.info(f"Total upload time: {total_time:.2f} seconds")
    print(f"Total upload time: {total_time:.2f} seconds")




# def _upload_file_to_pcloud(folder_id, local_file_path):
#     upload_url = "https://eapi.pcloud.com/uploadfile"
#     filename = os.path.basename(local_file_path)

#     params = {"folderid": folder_id, "filename": filename}
    
#     try:
#         with open(local_file_path, 'rb') as file:
#             # print(f"Uploading {filename} to pCloud (Folder ID: {folder_id})...")

#             response = requests.post(upload_url, params=params, headers=pCloud_headers, files={'file': file}, proxies=proxy)
#             # print("Response Status:", response.status_code)

#             if response.status_code == 200:
#                 result = response.json()
                
#                 if result.get("result") == 0:
#                     result = response.json()
#                     # print(f"Successfully uploaded {filename} to pCloud!")
#                 else:
#                     raise HTTPException(status_code=400, detail=f"Failed to upload {filename}")
#             else:
#                 raise HTTPException(status_code=response.status_code, detail=response.text)
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error uploading {filename}: {str(e)}")

# # Function to upload all files while maintaining folder structure
# def upload_file_to_pcloud(folder_id, local_folder_path, sub_folder, file_extension_filter=None):
#     local_subfolder_path = os.path.join(local_folder_path, sub_folder)
    
#     if not os.path.exists(local_subfolder_path):
#         print(f"Local folder does not exist: {local_subfolder_path}")
#         return
    
#     for root, _, files in os.walk(local_subfolder_path):
#         for file_name in files:
#             if file_extension_filter and not file_name.lower().endswith(file_extension_filter):
#                 continue  # Skip files not matching the filter

#             local_file_path = os.path.join(root, file_name)
#             #print(f"Uploading {file_name} from {sub_folder}...")
#             _upload_file_to_pcloud(folder_id, local_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload results to pCloud')
    parser.add_argument('--folder_name', type=str, help='local folder name inside pcloudresults fo which results are being updated') #localFolderName from pcloud_folders.json
    parser.add_argument('--directory', type=str, help='Main PREMIERETOOLV2 directory')  #main premiere: ./PREMIERETOOLV2
    parser.add_argument('--folder_id', type=int, help='The ID of the folder to upload to') #pCloudAnnotationsFolderId from pcloud_folders.json
    args = parser.parse_args()
    folder_name = args.folder_name
    directory = args.directory if args.directory else os.getcwd()
    folder_id = args.folder_id

    #data=update_results(directory,folder_name)
    
    json_file = os.path.join(directory, f'pcloudresults/{folder_name}/pcloud_folders.json')
    with open(json_file, "r") as file:
        data = json.load(file)

    local_folder_path=os.path.join(directory, f'pcloudresults/{folder_name}/annotationdata')
    max_category_workers=3 # Number of parallel category uploads
    file_batch_size=2 # Number of files to upload per batch
    upload_workers=3 # Number of parallel uploads per category. More than 3 can cause upload congestion 
    multiple_upload_file_to_pcloud(data, local_folder_path,max_category_workers,file_batch_size,upload_workers)
    # print("Uploading files to pCloud...")
    # for sub_folder, folder_id in data['annotationDataFolderIds'].items():
    #     # print(f"Name: {sub_folder}, ID: {folder_id}")

    #     if sub_folder.startswith("scene_"):
    #         local_scene_path = os.path.join(local_folder_path, "body", sub_folder)
    #         upload_file_to_pcloud(folder_id, os.path.join(local_folder_path, "body"), sub_folder, file_extension_filter=".glb")
    #     elif sub_folder != "body":
    #         upload_file_to_pcloud(folder_id, local_folder_path, sub_folder)

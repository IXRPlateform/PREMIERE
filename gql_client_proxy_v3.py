import asyncio 
from gql import Client, gql 
from gql.transport.aiohttp_websockets import AIOHTTPWebsocketsTransport

import os
import requests
from fastapi import FastAPI, HTTPException
#from master_offline import asr
import librosa
import json
import logging
import websockets
import aiohttp
import sys
import subprocess
import smtplib
import cv2
import time
import fcntl
from collections import deque
from datetime import datetime, timezone
import yaml
import fcntl
import shutil
import concurrent.futures
from functools import partial

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import time
from boto3.s3.transfer import TransferConfig
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


#---------------------------------------------Digital Ocean S3 bucket configuration START-----------------------------------
access_key_id = os.getenv('DO_SPACES_KEY')
secret_access_key =os.getenv('DO_SPACES_SECRET')
region =os.getenv('DO_SPACES_REGION')
bucket_name = os.getenv('DO_SPACES_BUCKET')
endpoint =os.getenv('DO_SPACES_ENDPOINT')

url="https://ams3.digitaloceanspaces.com"
if not all([access_key_id, secret_access_key, region, bucket_name, endpoint]):
    raise ValueError("Missing necessary S3 configuration from environment variables")

s3 = boto3.client(
    's3',
    region_name=region,
    endpoint_url=endpoint,
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key
)


#---------------------------------------------EMD-----------------------------------


# Get the directory where mesh extraction is located
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "meshextract3D"))

def retry_request(func, *args, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        print(f"Attempt: {attempt+1}")
        try:
            return func(*args, **kwargs)  # Call the function
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                raise 

def parse_iso_datetime(dt_str):
    return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)

# Get current UTC time
script_start_time = datetime.now(timezone.utc)

def get_pcloud_file_link(fileid: str) -> str:
    return retry_request(_get_pcloud_file_link, fileid)  # Call the actual function with retry logic

def _get_pcloud_file_link(fileid: str) -> str:
    response = requests.get("https://eapi.pcloud.com/getfilelink", params={"fileid": fileid}, headers=pCloud_headers, proxies=proxy)
    if response.status_code == 200:
        result = response.json()
        if result.get("result") == 0:
            hosts = result.get("hosts", [])
            path = result.get("path", "")
            if hosts and path:
                return f"https://{hosts[0]}{path}"
        else:
            raise HTTPException(status_code=400, detail="Failed to retrieve the file link")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

def download_and_save_file(download_url: str, local_filename: str) -> tuple:
    return retry_request(_download_and_save_file, download_url, local_filename)

def _download_and_save_file(download_url: str, local_filename: str) -> tuple:
    response = requests.get(download_url, stream=True)

    if response.status_code == 200:
        # Define directories and save file
        video_dir = os.path.join(os.getcwd(), "pcloudvideos")
        results_dir = os.path.join(os.getcwd(), "pcloudresults")

        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        video_save_path = os.path.join(video_dir, local_filename)
        local_filename_no_ext = os.path.splitext(local_filename)[0]
        results_save_path = os.path.join(results_dir, local_filename_no_ext)
        os.makedirs(results_save_path, exist_ok=True)

        with open(video_save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return video_save_path, results_save_path
    else:
        raise HTTPException(status_code=response.status_code, detail="Error downloading the file")

def create_folder_on_pcloud(folder_id, folder_name):
    create_folder_url = "https://eapi.pcloud.com/createfolderifnotexists"
    params = {"folderid": folder_id, "name": folder_name}
    
    response = requests.get(create_folder_url, params=params, headers=pCloud_headers, proxies=proxy)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("result") == 0:
            return result.get("metadata", {}).get("folderid")
        else:
            raise HTTPException(status_code=400, detail=f"Failed to create folder: {folder_name}")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

# Function to upload a file to pCloud
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
# async def upload_file_to_pcloud(folder_id, local_folder_path):

#     # Create subfolders only once
#     subfolder_ids = {
#         "2d_poses": create_folder_on_pcloud(folder_id, "2d_poses"),
#         "3d_poses": create_folder_on_pcloud(folder_id, "3d_poses"),
#         "2d_BoundingBoxes": create_folder_on_pcloud(folder_id, "2d_Bounding_Boxes"),
#         "3d_BoundingBoxes": create_folder_on_pcloud(folder_id, "3d_Bounding_Boxes"),
#         "body": create_folder_on_pcloud(folder_id, "body"),
#     }
    
#     # List subfolders inside scenepkl
#     scenepkl_path = os.path.join(local_folder_path, "body")
#     scenepkl_scenes = os.listdir(scenepkl_path)
    
#     # Create subfolders inside scenepkl on pCloud and get their folder IDs
#     for scene in scenepkl_scenes:
#         scene_path = os.path.join(scenepkl_path, scene)
#         if os.path.isdir(scene_path):  # Ensure it is a directory
#             subfolder_ids[scene] = create_folder_on_pcloud(subfolder_ids["body"], scene)

#     # Prepare data for JSON output with a clear parent folder description
#     folder_structure = {
#         "pCloudAnnotationsFolderId": folder_id,
#         "annotationDataFolderIds": {folder_name: folder_id for folder_name, folder_id in subfolder_ids.items()}
#     }
    
#     # Save the folder structure as a JSON file
#     parent_directory = os.path.dirname(local_folder_path)
#     json_file_path = os.path.join(parent_directory, "pcloud_folders.json")
#     with open(json_file_path, "w") as json_file:
#         json.dump(folder_structure, json_file, indent=4)
    
#     print(f"Folder structure saved to: {json_file_path}")
    
#     # Walk through files and upload them
#     for root, _, files in os.walk(local_folder_path):
#         for file_name in files:
#             local_file_path = os.path.join(root, file_name)

#             # Determine file type
#             is_json_file = file_name.endswith(".json")

#             for category, folderid in subfolder_ids.items():
#                 # Upload JSON files **only** to annotation categories, NOT to scene-specific folders
#                 if is_json_file and category in local_file_path and "body" not in local_file_path:
#                     _upload_file_to_pcloud(folderid, local_file_path)
#                     break  # Prevent further scene-based uploads for this file

#                 # Ensure non-JSON files (like `.pkl`) are uploaded correctly under scenepkl
#                 elif "body" in category and category in local_file_path and not is_json_file:
#                     for scene in scenepkl_scenes:
#                         if scene in local_file_path:
#                             subfolder_id = subfolder_ids.get(scene)
#                             _upload_file_to_pcloud(subfolder_id, local_file_path)
#                             break






def get_video_info(video_path):
    """Extract FPS, width, height, and file size from the video."""
    if not os.path.exists(video_path):
        return None, None, None, None  

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, None  

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    file_size = round(os.path.getsize(video_path) / (1000 * 1000), 2)  # Convert bytes to decimal MB

    return fps, width, height, file_size

def get_scene_count(results_dir):
    """Count subdirectories starting with 's' in the processing results directory."""
    if not os.path.exists(results_dir):
        return 0

    scene_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("s")]
    return len(scene_dirs)


# Load email config from YAML file
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
async def send_email(id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processing_results, error_message=None):
    config = load_config()

    # Extract email settings
    email_settings = config["email"]
    from_addr = email_settings["from_addr"]
    to_addrs = email_settings["to_addrs"]
    smtp_server = email_settings["smtp_server"]
    smtp_port = email_settings["smtp_port"]
    password = email_settings["password"]

    subject = f"Video Processed"

    # Get video properties
    fps, width, height, file_size = get_video_info(saved_file_path)
    scene_count = get_scene_count(processing_results)

    # Email body content
    body = f"""
    Hello,

    A new video has been found.

    **Video Details:**
    - Filename/ID: {id}
    - pCloud File ID: {pCloudFileId}
    - pCloud Annotations Folder ID: {pCloudAnnotationsFolderId}
    - File Size: {file_size} MB
    - FPS: {fps if fps else 'N/A'}
    - Resolution (widthxheight): {width if width else 'N/A'} x {height if height else 'N/A'}
    - Number of Scenes Detected: {scene_count}

    {"Error Details: " + error_message if error_message else "No errors detected. Video processed successfully and annotation data uploaded on pcloud."}

    Have a great day!

    Best regards,  
    UJM Premiere Team
    """
    
    msg = f"From: {from_addr}\r\n"
    msg += f"To: {', '.join(to_addrs)}\r\n"
    msg += f"Subject: {subject}\r\n"
    msg += "\r\n"
    msg += body

    # SMTP server configuration
    smtpserver = smtp_server
    server = smtplib.SMTP(smtpserver, smtp_port)
    server.set_debuglevel(1)

    server.starttls()
    server.login(from_addr, password)
    server.sendmail(from_addr, to_addrs, msg)
    server.quit()


async def multiple_upload_file_to_pcloud(folder_id, local_folder_path):
    print("the folder id is---------", folder_id)
    print("the local folder path is-------", local_folder_path)
    # Function to upload an entire folder to a Space
    s3_prefix='annotations/'
    start_time = time.time()
    print("uploading....")
    for root, dirs, files in os.walk(local_folder_path):
        
        for file in files:
            local_path = os.path.join(root, file)
            
            
            # Compute the relative path and convert to S3 key
            relative_path = os.path.relpath(local_path, local_folder_path)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            #print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            #print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3.upload_file(local_path, bucket_name, s3_key)
    end_time = time.time()  # End timer
    total_time = end_time - start_time
    
    logging.info(f"Total upload time: {total_time:.2f} seconds")
    print(f"Total upload time: {total_time:.2f} seconds")


#convert video file to audio
def convert_video_to_audio(video_file_path: str, audio_file_path: str):
    command = f"ffmpeg -i {video_file_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_file_path}"
    os.system(command)

async def run_subprocess(command, error_message, id=None, pCloudFileId=None, pCloudAnnotationsFolderId=None, saved_file_path=None, processing_results=None):
    """Run a subprocess with error handling and retries."""
    try:

        print("Running subprocess asynchronously...")
        # disable print statements in log
        sys.stdout = open(os.devnull, 'w')
        # Use asyncio.create_subprocess_exec for non-blocking execution
        process = await asyncio.create_subprocess_exec(
            *command, 
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Wait for the process to complete and capture output
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logging.info(f"Command succeeded: {' '.join(command)}")
            logging.info(f"Output: {stdout.decode().strip()}")
            return stdout.decode().strip()
        else:
            logging.info(f"{error_message}: {stderr.decode().strip()}")
            # Send error email
            await send_email(id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processing_results, error_message=f"{error_message}\n{stderr.decode().strip()}")
            raise Exception(f"Error occurred in subprocess: {error_message}\n{stderr.decode().strip()}")
    
    
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
        await send_email(id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processing_results, error_message=f"{error_message}\n{str(e)}")
        raise Exception(f"Unexpected error occurred: {str(e)}")

# GraphQL Queries
fetch_videos_query = gql("""
    query GetUploadedVideos {
      videoObjects {
        id
        pCloudAnnotationsFolderId
        pCloudFileId
        uploadedAt
      }
    }
""")

subscription_query = gql(""" 
    subscription VideoObjectCreated {
      videoObjectCreated {
        id
        pCloudAnnotationsFolderId
        pCloudFileId
        uploadedAt
      }
    }
""")


# async def download_and_process(video):
#     id = video['id']
#     pCloudFileId = video['pCloudFileId']
#     pCloudAnnotationsFolderId = video['pCloudAnnotationsFolderId']

#     print(f"Processing video ID: {id}")

#     if not pCloudFileId:
#         logging.info(f"No pCloudFileId found for video ID: {id}")
#         return
#     download_url = get_pcloud_file_link(pCloudFileId)
#     # Step 1: Get download link
#     print(f"Fetching download link for file ID: {pCloudFileId}")
#     saved_file_path, processsing_results = download_and_save_file(download_url, f"{id}.mp4")
#     print("Video file saved to:", saved_file_path)
#     logging.info(f"Downloaded video {id} and saved to {saved_file_path}")

#     # Get the current working directory
#     directory = os.getcwd()

#     #Step 4: Set the environment variable MODELS_PATH
#     os.environ["MODELS_PATH"] = f"{directory}/models"
#     print("Environment variable MODELS_PATH set to:", os.environ["MODELS_PATH"])

#     #Step 5: Process Video
#     print("Processing video")
#     #await run_subprocess(["python", "premiereFullVideoProcessing.py", "--video", saved_file_path, "--directory", processsing_results], 
#     #                "Video processing failed", id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processsing_results)

#     #Step 6: Generate corrected poses & keypoints
#     print("Generating corrected poses & keypoints")
#     #await run_subprocess(["python", "generatel3DfromPkl.py", "--input_dir", processsing_results, "--use_floor", "--input_pkl", "nlf-final-filtered-floorc.pkl"], 
#     #                "Pose generation failed", id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processsing_results)

#     #Step 7: Arrange Data
#     print("Arranging data")
#     #await run_subprocess(["python", "arrange_data.py", "--directory", processsing_results], 
#     #            "Data arrangement failed", id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processsing_results)
#     meshdir=os.path.join(processsing_results,'annotationdata/scenepkl')
#     print("meshdir:",meshdir)
#     #meshdata
#     #extract_mesh_script = os.path.join(current_dir, "meshextract3D", "extract3Dmesh.py")
#     #await run_subprocess(["python", extract_mesh_script, "--input_pkl", "nlf-final-filtered-floorc.pkl", "--input_dir", meshdir, "--output_3dformat", "glb"],
#     #                     "Mesh extraction failed", id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processsing_results)
#     #move body to annotationdata
    
#     #shutil.move(os.path.join(processsing_results, 'body'), os.path.join(processsing_results, 'annotationdata'))
#     #shutil.rmtree(os.path.join(processsing_results, 'annotationdata', 'scenepkl')
#     data_folder=os.path.join(processsing_results,'annotationdata')
#     random_data_generator.generate_random_data(data_folder)
#     new_dir_with_pcloudannotationID=os.path.join(processsing_results,pCloudAnnotationsFolderId)
#     os.makedirs(new_dir_with_pcloudannotationID, exist_ok=True)
#     for item in os.listdir(data_folder):
#         src = os.path.join(data_folder, item)
#         dst = os.path.join(new_dir_with_pcloudannotationID, item)
        

        
#         shutil.move(src, dst)
#     shutil.move(new_dir_with_pcloudannotationID, data_folder)

   
   
   
   
#     pose_folder=data_folder
#     print("accessing pose folder", pose_folder) 
#     print("pCloudAnnotationsFolderId", pCloudAnnotationsFolderId)
#     logging.info("uploading pose folder to pCloud")
#     await multiple_upload_file_to_pcloud(pCloudAnnotationsFolderId, pose_folder)
#     #Step 8: Send email
#     #logging.info("Sending email notification")
#     #await send_email(id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processsing_results)
    
async def download_and_process(video):
    id = video['id']
    pCloudFileId = video['pCloudFileId']
    pCloudAnnotationsFolderId = video['pCloudAnnotationsFolderId']

    print(f"Processing video ID: {id}")

    if not pCloudFileId:
        logging.info(f"No pCloudFileId found for video ID: {id}")
        return
    download_url = get_pcloud_file_link(pCloudFileId)
    # Step 1: Get download link
    print(f"Fetching download link for file ID: {pCloudFileId}")
    saved_file_path, processsing_results = download_and_save_file(download_url, f"{id}.mp4")
    print("Video file saved to:", saved_file_path)
    logging.info(f"Downloaded video {id} and saved to {saved_file_path}")

    # Get the current working directory
    directory = os.getcwd()

    #Step 4: Set the environment variable MODELS_PATH
    os.environ["MODELS_PATH"] = f"{directory}/models"
    print("Environment variable MODELS_PATH set to:", os.environ["MODELS_PATH"])

    #Step 5: Process Video
    print("Processing video")
    await run_subprocess(["python", "premiereFullVideoProcessing.py", "--video", saved_file_path, "--directory", processsing_results], 
                    "Video processing failed", id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processsing_results)

    #Step 6: Generate corrected poses & keypoints
    print("Generating corrected poses & keypoints")
    await run_subprocess(["python", "generatel3DfromPkl.py", "--input_dir", processsing_results, "--use_floor", "--input_pkl", "nlf-final-filtered-floorc.pkl"], 
                    "Pose generation failed", id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processsing_results)

    #Step 7: Arrange Data
    print("Arranging data")
    await run_subprocess(["python", "arrange_data.py", "--directory", processsing_results], 
                "Data arrangement failed", id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processsing_results)
    meshdir=os.path.join(processsing_results,'annotationdata/scenepkl')
    print("meshdir:",meshdir)
    #meshdata
    extract_mesh_script = os.path.join(current_dir, "meshextract3D", "extract3Dmesh.py")
    await run_subprocess(["python", extract_mesh_script, "--input_pkl", "nlf-final-filtered-floorc.pkl", "--input_dir", meshdir, "--output_3dformat", "glb"],
                         "Mesh extraction failed", id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processsing_results)
    #move body to annotationdata
    shutil.move(os.path.join(processsing_results, 'body'), os.path.join(processsing_results, 'annotationdata'))
    shutil.rmtree(os.path.join(processsing_results, 'annotationdata', 'scenepkl'))


    # Organize the data ---------
    data_folder=os.path.join(processsing_results,'annotationdata')
    new_dir_with_pcloudannotationID=os.path.join(processsing_results,pCloudAnnotationsFolderId)
    os.makedirs(new_dir_with_pcloudannotationID, exist_ok=True)
    for item in os.listdir(data_folder):
        src = os.path.join(data_folder, item)
        dst = os.path.join(new_dir_with_pcloudannotationID, item)
        

        
        shutil.move(src, dst)
    shutil.move(new_dir_with_pcloudannotationID, data_folder)
    pose_folder=data_folder
    print("accessing pose folder", pose_folder) 
    print("pCloudAnnotationsFolderId", pCloudAnnotationsFolderId)
    logging.info("uploading pose folder to pCloud")
    await multiple_upload_file_to_pcloud(pCloudAnnotationsFolderId, pose_folder)
    #Step 8: Send email
    logging.info("Sending email notification")
    await send_email(id, pCloudFileId, pCloudAnnotationsFolderId, saved_file_path, processsing_results)

async def subscribe_to_new_videos(session,video_queue,processed_ids):
    async for result in session.subscribe(subscription_query):
        new_video = result['videoObjectCreated']
        if new_video['id'] not in processed_ids:
            print(f"New video uploaded: {new_video}.............Appending to queues")
            video_queue.append(new_video)


async def main():
    
    logging.basicConfig(filename="gql_proxy.log",format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO,force=True)
    # Set specific log levels for unnecessary logs
    logging.getLogger("gql").setLevel(logging.WARNING)  # Change to WARNING
    logging.getLogger("asyncio").setLevel(logging.WARNING)  # Change to WARNING
    logging.getLogger("gql.transport.aiohttp_websockets").setLevel(logging.WARNING)  # Suppress redundant logs
    while True:
        try:
            proxy = 'http://cache.univ-st-etienne.fr:3128'

            transport = AIOHTTPWebsocketsTransport(
                url='wss://premierecmsdev.medidata.pt/api/graphql',
                proxy=proxy
            )
            
            # Create a GraphQL client using the WebSocket transport 
            async with Client(transport=transport, fetch_schema_from_transport=True) as session: 
                print("Connected to server")
                logging.info(f"Connected to server")
                # Fetch Existing Videos
                existing_videos = await session.execute(fetch_videos_query)
                #print("The existing videos are: ", existing_videos['videoObjects'])
                for video in existing_videos['videoObjects']:
                    video_uploaded_time=video["uploadedAt"]
                    video_uploaded_time = parse_iso_datetime(video_uploaded_time)
                    if video_uploaded_time > script_start_time:
                        #print(f"{video_uploaded_time} is in the future.")
                        if video['id'] not in processed_ids:
                            #append videos in queue by adding threshold time
                            pass#print("will append here")
                            #video_queue.append(video)
                    elif video_uploaded_time < script_start_time:
                        pass
                        #print(f"{video_uploaded_time} is in the past.")
                
                # Subscribe to New Videos
                asyncio.create_task(subscribe_to_new_videos(session, video_queue, processed_ids))
                    
                if video_queue:
                    print("video queue: ",video_queue)
                    current_video = video_queue.popleft()
                    processed_ids.add(current_video['id'])
                    await download_and_process(current_video)
                await asyncio.sleep(5)

                    
        except (websockets.ConnectionClosedError, asyncio.IncompleteReadError) as e:
            logging.error("WebSocket connection closed: %s", e)
            await asyncio.sleep(5)  # Wait before trying to reconnect
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            await asyncio.sleep(5)  # Wait before trying to reconnect




if __name__ == '__main__':
    LOCK_FILE = "/tmp/my_script.lock"

    # Open the lock file
    lock_file = open(LOCK_FILE, 'w')

    try:
        # Try to acquire an exclusive non-blocking lock
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Script is already running. Exiting...")
        sys.exit(0)
    try:
        logger = logging.getLogger(__name__)
        # Run the main function in an asyncio event loop
        video_queue = deque()  # Queue to track unprocessed videos
        processed_ids = set()
        loop = asyncio.get_event_loop() 
        loop.run_until_complete(main()) 
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
        os.remove(LOCK_FILE)

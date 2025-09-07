"""
SMPL-X 3D Human Pose Visualization Web Application

This Flask application loads SMPL-X human pose data from a pickle file and provides
endpoints to visualize it in a 3D web interface. The script processes the pose data,
computes body meshes, and serves them to a frontend visualization tool.

Usage:
    python appVisualization.py <input_pkl>
"""
import os
import pickle
import json
import sys
import logging
import socket
import numpy as np
import torch
import roma
import smplx
from flask import Flask, render_template, request
from json import JSONEncoder
from scipy.spatial.transform import Rotation as R
from logging import getLogger

# Set up environment
os.environ['DATA_ROOT'] = os.path.join(os.environ["MODELS_PATH"], 'smplfitter')
models_path = os.environ["MODELS_PATH"]

class NumpyArrayEncoder(JSONEncoder):
    """Custom JSON encoder to handle NumPy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Validate command-line arguments
if len(sys.argv) != 2:
    print("Usage: python appVisualization.py <input_pkl>")
    sys.exit(1)

# Set up Flask app
dir_path = os.path.abspath("templates-Visualization")
print(dir_path)
app = Flask(__name__, root_path=dir_path)

# Load data from pickle file
try:
    with open(sys.argv[1], 'rb') as file:
        allDataPKL = pickle.load(file)
    dataPKL = allDataPKL['allFrameHumans']
except (FileNotFoundError, KeyError) as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Process and prepare data
allFramesNumber = len(dataPKL)
print("Video length: ", allFramesNumber, " frames")
print('FPS: ', allDataPKL.get("video_fps", 30))

# Find maximum number of humans in any frame
maxHumans = 0
for i in range(allFramesNumber):
    maxHumans = max(maxHumans, len(dataPKL[i]))
print("Maximum number of people in a frame:", maxHumans)

# Check if we should use expression parameters based on model type
useExpression = True
print("model_name: ", allDataPKL['model_name'])
if (allDataPKL['model_type'] == "hmr2") or (allDataPKL['model_type'] == "nlf"):
    useExpression = False

# Set number of keypoints to use
keypointsNumber = 127

# Find bounding box of all keypoints across all frames
min_values = np.array([float('inf'), float('inf'), float('inf')])
max_values = np.array([float('-inf'), float('-inf'), float('-inf')])

# Process each frame and track IDs
nbTrackId = 0
for i in range(allFramesNumber):
    for j in range(len(dataPKL[i])):
        # Normalize keypoints
        for k in range(len(dataPKL[i][j]['j3d_smplx'][0:keypointsNumber])):
            dataPKL[i][j]['j3d_smplx'][k] /= 1.0
            
        # Update min/max bounds
        keypoints = np.array(dataPKL[i][j]['j3d_smplx'][0:keypointsNumber])
        min_values = np.minimum(min_values, keypoints.min(axis=0))
        max_values = np.maximum(max_values, keypoints.max(axis=0))
        
        # Track maximum person ID
        nbTrackId = max(nbTrackId, dataPKL[i][j]['id'])
nbTrackId = int(nbTrackId) + 1

print("Number of tracks: ", nbTrackId)
print("Min keypoints: ", min_values)
print("Max keypoints: ", max_values)

# Create tracks for each person ID
allTracks = []
allTracksSegments = []
for i in range(nbTrackId):
    allTracks.append([])
    allTracksSegments.append([])

# Populate tracks with frame data
for i in range(allFramesNumber):
    for j in range(len(dataPKL[i])):
        if dataPKL[i][j]['id'] != -1:
            element = (i, j, dataPKL[i][j]['j3d_smplx'][0].tolist(), 
                       np.squeeze(dataPKL[i][j]['transl_pelvis'], axis=0).tolist())
            allTracks[dataPKL[i][j]['id']].append(element)

# Initialize container for vertices
currentVertices = []

# Find first frame with human data
notempty = 0
while notempty < len(dataPKL) and len(dataPKL[notempty]) == 0:
    notempty += 1

posesDetected = True
# Safety check if no humans are found
if notempty >= len(dataPKL):
    print("Warning: No humans detected in any frame!")
    notempty = 0
    posesDetected = False
if posesDetected:
    print("First frame with humans: ", notempty)
else:
    print("No humans detected in any frame, starting from frame 0")
    notempty = 0

# Initialize SMPL-X model
model = smplx.create(
    models_path, 'smplx',
    gender='neutral',
    use_pca=False, flat_hand_mean=True,
    num_betas=10,
    ext='npz').cuda()

# Zero translation tensor for model initialization
t = torch.zeros(1, 3).cuda()

@app.route("/getInfos")
def getInfos():
    """Return metadata about the loaded data"""
    fps = allDataPKL.get("video_fps", 30)
    floor_Zoffset = allDataPKL.get("floor_Zoffset", 0)
    floor_angle_deg = allDataPKL.get("floor_angle_deg", 0)
    
    # Transform allTracks into objects with a poseId and data field
    newTracks = []
    for i, track in enumerate(allTracks):
        if len(track) > 0:
            newTracks.append({"poseId": i, "data": track})
    
    # Get camera parameters with sensible defaults
    camera_rotation_deg = allDataPKL.get("camera_rotation_deg", [floor_angle_deg, 0, 0])
    camera_fixed = allDataPKL.get("camera_fixed", True)
    dynamic_fov = allDataPKL.get("dynamic_fov", False)

    camera_fovs = allDataPKL.get("camera_fovs", np.full(allFramesNumber, 60.0))
    if not isinstance(camera_fovs, np.ndarray):
        camera_fovs = np.array(camera_fovs)
    
    # Default camera transforms if not provided
    default_rotation = np.zeros((allFramesNumber, 3))
    default_translation = np.zeros((allFramesNumber, 3))
    
    camera_rotations = allDataPKL.get("camera_rotations", default_rotation)
    camera_positions = allDataPKL.get("camera_translations", default_translation)

    # Prepare response with all metadata
    allInfos = {
        "totalFrameNumber": allFramesNumber,
        "notempty": notempty,
        "maxHumans": maxHumans,
        "nbKeyPoints": keypointsNumber,
        "allTracks": newTracks, 
        "video_width": allDataPKL.get("video_width", 1920),
        "video_height": allDataPKL.get("video_height", 1080),
        "video_fps": fps, 
        "floor_angle_deg": floor_angle_deg,
        "floor_Zoffset": floor_Zoffset,
        "camera_rotation_deg": camera_rotation_deg,
        "camera_fixed": camera_fixed,
        "dynamic_fov": dynamic_fov,
        "fileName": sys.argv[1],
        "camera_fovs": camera_fovs.tolist(),
        "camera_rotations": camera_rotations.tolist(),
        "camera_positions": camera_positions.tolist(),
        "smplx_faces": model.faces.flatten().tolist(),
    }
    return json.dumps(allInfos)

@app.route("/getVertices")
def getVertices():
    """Generate and return SMPL-X mesh vertices for the requested frame"""
    frame = int(request.args.get('frame'))
    if len(dataPKL[frame]) == 0:
        return json.loads('{}')
    
    allsmplx = []
    allId = []
    allKeypoints = []
    
    for humanID in range(len(dataPKL[frame])):
        # Get human parameters from data
        betas = torch.from_numpy(np.expand_dims(dataPKL[frame][humanID]['shape'], axis=0)).cuda()
        if useExpression:
            expression = torch.from_numpy(np.expand_dims(dataPKL[frame][humanID]['expression'], axis=0)).cuda()
        pose = torch.from_numpy(np.expand_dims(dataPKL[frame][humanID]['rotvec'], axis=0)).cuda()           
        
        # Prepare arguments for SMPL-X model
        bs = pose.shape[0]
        kwargs_pose = {
            'betas': betas,
            'return_verts': True,
            'pose2rot': True 
        }
        kwargs_pose['global_orient'] = t.repeat(bs, 1)
        kwargs_pose['body_pose'] = pose[:, 1:22].flatten(1)
        
        # Handle model-specific parameters
        if useExpression:
            kwargs_pose['left_hand_pose'] = pose[:, 22:37].flatten(1)
            kwargs_pose['right_hand_pose'] = pose[:, 37:52].flatten(1)
            kwargs_pose['expression'] = expression.flatten(1)
            kwargs_pose['jaw_pose'] = pose[:, 52:53].flatten(1)
            kwargs_pose['leye_pose'] = t.repeat(bs, 1)
            kwargs_pose['reye_pose'] = t.repeat(bs, 1)      
        else:
            kwargs_pose['jaw_pose'] = pose[:, 22:23].flatten(1)
            kwargs_pose['leye_pose'] = pose[:, 23:24].flatten(1)
            kwargs_pose['reye_pose'] = pose[:, 24:25].flatten(1)
            kwargs_pose['left_hand_pose'] = pose[:, 25:40].flatten(1)
            kwargs_pose['right_hand_pose'] = pose[:, 40:55].flatten(1)
        
        # Forward pass through model
        output = model(**kwargs_pose)
        verts = output.vertices
        j3d = output.joints
        
        # Apply transformations
        Rmat = roma.rotvec_to_rotmat(pose[:, 0])
        pelvis = j3d[:, [0]]
        j3d = (Rmat.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)
        person_center = j3d[:, [15]]
        vertices = (Rmat.unsqueeze(1) @ (verts - pelvis).unsqueeze(-1)).squeeze(-1)
        
        # Apply translation
        trans = torch.from_numpy(dataPKL[frame][humanID]['transl']).cuda()
        trans = trans - person_center
        vertices = vertices + trans 
        vertices = vertices.detach().cpu().numpy().squeeze().flatten().tolist()
        currentVertices.append(vertices)
        j3d = j3d + trans
        j3d = j3d.detach().cpu().numpy().squeeze().tolist()
        
        # Store results
        allKeypoints.append(j3d)
        allId.append(int(dataPKL[frame][humanID]['id']))
        allsmplx.append(vertices)
 
    # Return data as JSON
    allData = (allKeypoints, allId, allsmplx)
    return json.dumps(allData, cls=NumpyArrayEncoder)

@app.route("/getMeshes")
def getMeshes():
    """Return SMPL-X keypoints for the requested frame"""
    frame = int(request.args.get('frame'))
    if len(dataPKL[frame]) == 0:
        return json.loads('{}')
    
    allsmplx = []
    allId = []
    allKeypoints = []
    
    for humanID in range(len(dataPKL[frame])):
        allKeypoints.append(dataPKL[frame][humanID]['j3d_smplx'][0:keypointsNumber])
        allId.append(int(dataPKL[frame][humanID]['id']))
        
    allData = (allKeypoints, allId, allsmplx)
    return json.dumps(allData, cls=NumpyArrayEncoder)

@app.route("/get")
def get():
    """Legacy endpoint returning keypoints data for the requested frame"""
    frame = int(request.args.get('frame'))
    if len(dataPKL[frame]) == 0:
        return json.loads('{}')
    
    allsmpl3DKeyPoints = []
    allId = []
    for humanID in range(len(dataPKL[frame])):
        smpl3DKeyPoints = dataPKL[frame][humanID]['j3d_smplx'][0:keypointsNumber]
        allsmpl3DKeyPoints.append(smpl3DKeyPoints)
        allId.append(int(dataPKL[frame][humanID]['id']))
        
    allData = (allsmpl3DKeyPoints, allId)
    return json.dumps(allData, cls=NumpyArrayEncoder)

@app.route("/")
def index():
    """Render the main visualization page"""
    return render_template("index.html")
 
def find_open_port():
    """Find and return an available port for the server"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    _, port = sock.getsockname()
    sock.close()
    return port
 
if __name__ == "__main__":
    print("Starting visualization server...")
    dynamic_port = find_open_port()
    print(f"Server running on port {dynamic_port}")
    print(f"Open in your browser: http://127.0.0.1:{dynamic_port}")

    # Disable Flask's default logger
    log = getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(debug=False, port=dynamic_port)

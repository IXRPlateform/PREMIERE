import math
import numpy as np
import pickle

def projectPoints3dTo2d_focal(points_3d, focal, image_width, image_height):
    """
    Projects 3D points to 2D using a focal length instead of a field of view.
    """
    cx = image_width / 2.0
    cy = image_height / 2.0
    projected_points = []
    for (x, y, z) in points_3d:
        if z != 0:
            x_2d = (focal * x / z) + cx
            y_2d = (focal * y / z) + cy
            projected_points.append((x_2d, y_2d))
        else:
            projected_points.append((None, None))  # or handle differently
    return projected_points

def projectPoints3dTo2d(points_3d, fov, width, height):
    """
    points_3d : np.array of shape (N, 3) (X, Y, Z) in the camera coordinate system
    fov       : field of view (in degrees)
    width     : width of the image in pixels
    height    : height of the image in pixels
    
    Returns an np.array (N, 2) with the projected (u, v) coordinates.
    """
    # 1. Compute the focal length in pixels
    f = (width / 2) / math.tan(math.radians(fov) / 2)
    
    # 2. Principal point (image center)
    cx = width / 2.0
    cy = height / 2.0

    # 3. Separate X, Y, Z components
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]
    
    # 4. Projection (assuming Z > 0)
    #    NB: If Z < 0, the point is behind the camera.
    u = f * (X / Z) + cx
    v = f * (Y / Z) + cy
    
    # 5. Stack into (N, 2)
    projected_2d = np.vstack((u, v)).T
    return projected_2d

def buildTracks(allFrameHumans, maxId):
    # Calculate the size of each track
    tracksSize = np.zeros(maxId+1, dtype=int)
    for i in range(len(allFrameHumans)):
        for j in range(len(allFrameHumans[i])):
            if (allFrameHumans[i][j]['id'] != -1):
                tracksSize[allFrameHumans[i][j]['id']] += 1
    print('tracksSize: ', tracksSize)

    # Create the tracks
    tracks = []
    for i in range(maxId+1):
        tracks.append(np.zeros((tracksSize[i],3), dtype=int))

    # Create the tracksCurrentPosition
    tracksCurrentPosition = np.zeros(maxId+1, dtype=int)

    for i in range(len(allFrameHumans)):
        for j in range(len(allFrameHumans[i])):
            if(allFrameHumans[i][j]['id'] != -1):
                idToProcess = allFrameHumans[i][j]['id']
                tracks[idToProcess][tracksCurrentPosition[idToProcess]] = [i, j, idToProcess]
                tracksCurrentPosition[idToProcess] += 1
    return tracks, tracksSize

def buildTracksForFusion(allFrameHumans, maxId):
    # Calculate the size of each track
    tracksSize = np.zeros(maxId+1, dtype=int)
    for i in range(len(allFrameHumans)):
        for j in range(len(allFrameHumans[i])):
            if (allFrameHumans[i][j]['id'] != -1):
                tracksSize[allFrameHumans[i][j]['id']] += 1
    print('tracksSize: ', tracksSize)

    # Create the tracks
    tracks = []
    for i in range(maxId+1):
        tracks.append(np.zeros((tracksSize[i],3), dtype=int))

    # Create the tracksCurrentPosition
    tracksCurrentPosition = np.zeros(maxId+1, dtype=int)

    for i in range(len(allFrameHumans)):
        for j in range(len(allFrameHumans[i])):
            if(allFrameHumans[i][j]['id'] != -1):
                idToProcess = allFrameHumans[i][j]['id']
                idSam = allFrameHumans[i][j]['idsam']
                tracks[idToProcess][tracksCurrentPosition[idToProcess]] = [i, j, idSam]
                tracksCurrentPosition[idToProcess] += 1
    return tracks, tracksSize

def computeMaxId(allFrameHumans):
    # Calculate the maximum number of humans and the maximum id
    maxHumans = 0
    maxId = 0
    maxIndex = -1
    for i in range(len(allFrameHumans)):
        currentHumans = len(allFrameHumans[i])
        if currentHumans > maxHumans:
            maxHumans = currentHumans
            maxIndex = i
        for j in range(len(allFrameHumans[i])):
            maxId = max(maxId, allFrameHumans[i][j]['id'])
    #maxHumans = 4
    print('maxHumans: ', maxHumans)
    print('maxId: ', maxId)
    print('maxIndex: ', maxIndex)
    return maxId

def extractDetections(detection):
    """
    Extract bounding box information from a single ultralytics Results object.
    """
    allHumans = []
    boxes = detection.boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        width = x2 - x1
        height = y2 - y1
        conf = float(box.conf[0].item())
        if width <= 0 or height <= 0:
            continue
        # Dummy id for now
        obj_id = -1
        allHumans.append({
            'conf': conf,
            'id': obj_id,
            'x': x1,
            'y': y1,
            'xtop': x2,
            'ytop': y2,
            'width': width,
            'height': height
        })
    return allHumans

def keypointsBBox3D(keypoints):
    """
    Compute the 3D bounding box of a set of keypoints.

    This function returns the minimum and maximum coordinates among
    all keypoints provided.

    :param keypoints: 3D coordinates of keypoints (N x 3).
    :type keypoints: np.ndarray
    :return: A tuple (min_coords, max_coords), each with shape (3,).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    min_coords = np.min(keypoints, axis=0)
    max_coords = np.max(keypoints, axis=0)
    return min_coords, max_coords

def keypointsBBox2D(keypoints):
    """
    Compute the 2D bounding box of a set of keypoints.

    This function returns the minimum and maximum coordinates among
    all keypoints provided.

    :param keypoints: 2D coordinates of keypoints (N x 2).
    :type keypoints: np.ndarray
    :return: A tuple (min_coords, max_coords), each with shape (2,).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    min_coords = np.min(keypoints, axis=0)
    max_coords = np.max(keypoints, axis=0)
    return min_coords, max_coords

def loadPkl(input_pkl):
    """
    Load data from a pickle file.
    
    :param input_pkl: Path to the input pickle file.
    :type input_pkl: str
    :return: Loaded data.
    :rtype: dict
    """
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)
    return data

def loadPkl2(input1_pkl, input2_pkl):
    """
    Load data from a pickle files.
    """
    with open(input1_pkl, 'rb') as f:
        data1 = pickle.load(f)
    with open(input2_pkl, 'rb') as f:
        data2 = pickle.load(f)
    return data1, data2

def loadPkl3(input1_pkl, input2_pkl, input3_pkl):
    """
    Load data from a pickle files
    """
    with open(input1_pkl, 'rb') as f:
        data1 = pickle.load(f)
    with open(input2_pkl, 'rb') as f:
        data2 = pickle.load(f)
    with open(input3_pkl, 'rb') as f:
        data3 = pickle.load(f)
    return data1, data2, data3

def savePkl(output_pkl, data):
    """
    Save data to a pickle file.
    
    :param output_pkl: Path to the output pickle file.
    :type output_pkl: str
    :param data: Data to save.
    :type data: dict
    """
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)
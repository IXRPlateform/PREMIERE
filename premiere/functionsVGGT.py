import math
import numpy as np

def calculate_mean_without_outliers(data, k=1.5):
    """
    Calculate the mean of the data excluding outliers using IQR method.
    
    Args:
        data: List or array of numeric values
        k: Multiplier for IQR to define outliers (default: 1.5)
    
    Returns:
        Mean value after removing outliers
    """
    if len(data) == 0:
        return None
        
    data_array = np.array(data)
    
    # Calculate Q1, Q3 and IQR
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    
    # Define bounds for outliers
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    # Filter out outliers
    filtered_data = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]
    
    # Calculate statistics
    if len(filtered_data) > 0:
        mean_without_outliers = np.mean(filtered_data)
        return mean_without_outliers
    else:
        return np.mean(data_array)  # Return regular mean if all are outliers


def computeIntrinsic(video_width, video_height, fov_x):
    """
    Calcule la matrice intrinsèque (K) d'une caméra pinhole à partir du FOV horizontal
    et de la résolution (on suppose ici des pixels carrés et le centre de l'image au milieu).
    """
    fx = (video_width / 2) / math.tan(fov_x / 2)
    fy = fx  # pixels carrés
    cx = video_width / 2
    cy = video_height / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    return K

def projectPoints(points, K):
    """
    Projette une liste de points 3D (shape : [N, 3]) dans le plan image via la matrice K.
    Si z <= 0, on retourne NaN pour éviter la division par zéro.
    """
    projected = []
    for pt in points:
        x, y, z = pt
        if z <= 0:
            projected.append([np.nan, np.nan])
        else:
            u = K[0, 0] * (x / z) + K[0, 2]
            v = K[1, 1] * (y / z) + K[1, 2]
            projected.append([u, v])
    return np.array(projected)

def estimanteVGGTScaleFactor ( posePkl, vggtPkl ):
    allFrameHumans = posePkl['allFrameHumans']
    
    # Récupérer la résolution vidéo
    video_width = posePkl.get('video_width', 640)
    video_height = posePkl.get('video_height', 480)
    
    all_frame_numbers = vggtPkl['frames']
    all_depth_maps = vggtPkl['depth_maps']
    ratios = []

    for i in range(len(all_depth_maps)):
        for human in allFrameHumans[all_frame_numbers[i]]:
            if (not human['occluded']):
                # Check if the head position is valid
                headPosition = human['j2d_smplx'][15]
                headPosition[0] = headPosition[0] / video_width
                headPosition[1] = headPosition[1] / video_height
                if headPosition[0] >= 0 and headPosition[1] < 1:
                    # Calculate indices
                    x_idx = int(headPosition[0] * all_depth_maps[i].shape[1])
                    y_idx = int(headPosition[1] * all_depth_maps[i].shape[0])
                    
                    # Check if indices are within bounds
                    if (0 <= x_idx < all_depth_maps[i].shape[1] and 
                        0 <= y_idx < all_depth_maps[i].shape[0]):
                        headPositionInDepthmap = (x_idx, y_idx)
                        cameraDistanceInDepthmap = all_depth_maps[i, headPositionInDepthmap[1], headPositionInDepthmap[0]]
                        cameraDistance = np.linalg.norm(human['transl'])
                        ratios.append(cameraDistance/cameraDistanceInDepthmap)
                    else:
                        print(f"Warning: Head position ({x_idx}, {y_idx}) is out of depth map bounds " 
                                f"({all_depth_maps[i].shape[1]}, {all_depth_maps[i].shape[0]}) for frame {all_frame_numbers[i]}")

    if len(ratios) == 0:
        print("Warning: No valid ratios found.")
        return 1
    else:
        print(f"Found {len(ratios)} valid ratios.")
        # Calculate the mean ratio without outliers
        mean_ratio = calculate_mean_without_outliers(ratios)
        return mean_ratio
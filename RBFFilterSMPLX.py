import sys
import pickle
import torch
import copy
import os
import roma
import smplx
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from premiere.functionsSMPLX import updateHumanFromSMPLX
from premiere.functionsCommon import (
    projectPoints3dTo2d, 
    keypointsBBox2D, 
    keypointsBBox3D,
    buildTracks,
    computeMaxId
)

os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def buildInterpolator(Xm, Ym, rbfkernel='linear', rbfepsilon=1.0, rbfsmooth=.01, neighbors=400):
    """
    Build an interpolator for the given data points, with outlier handling.
    
    Args:
        Xm: Input coordinates
        Ym: Output values
        rbfkernel: Kernel type for RBF interpolation or 'univariatespline'
        rbfepsilon: Epsilon parameter for RBF kernel
        rbfsmooth: Smoothing parameter
        neighbors: Number of neighbors to consider
        
    Returns:
        An interpolator object
    """
    # Detect outliers using z-scores
    mean_Ym = np.mean(Ym)
    std_Ym = np.std(Ym)
    if std_Ym == 0:
        z_scores = np.zeros_like(Ym)
    else:
        z_scores = np.abs((Ym - mean_Ym) / std_Ym)
    threshold = 4  # Threshold for outlier detection

    # Filter out outliers
    inliers = z_scores < threshold
    Xm_filtered = Xm[inliers]
    Ym_filtered = Ym[inliers]

    # Create appropriate interpolator based on kernel type
    if rbfkernel == 'univariatespline':
        interpolator = UnivariateSpline(Xm_filtered, Ym_filtered, k=3, s=rbfsmooth)
    else:
        # Build points for the interpolator in 1D (extension to 2D for compatibility)
        points = np.column_stack((Xm_filtered, np.zeros_like(Xm_filtered)))
        values = Ym_filtered
        interpolator = RBFInterpolator(points, values, kernel=rbfkernel, 
                                       epsilon=rbfepsilon, smoothing=rbfsmooth)
    
    return interpolator

def RBFFilterTrackRotvec(k, dataPKL, track, trackSize, rbfkernel, rbfsmooth, rbfepsilon, ava):
    """
    Interpolate rotation vectors for a track using quaternion interpolation.
    
    Args:
        k: Index of the rotation vector to process
        dataPKL: Dictionary containing human data
        track: Track data
        trackSize: Size of the track
        rbfkernel: Kernel type for RBF interpolation
        rbfsmooth: Smoothing parameter
        rbfepsilon: Epsilon parameter for RBF kernel
        ava: Availability array indicating which frames have data
    """
    # Calculate real size (including missing frames)
    realSize = track[trackSize - 1][0] - track[0][0] + 1
    denom = realSize - 1 if realSize > 1 else 1

    # Create arrays for time and quaternions
    times = np.zeros(trackSize, dtype=float)
    quats = np.zeros((trackSize, 4), dtype=float)
    
    # Fill arrays with existing data
    for i in range(trackSize):
        t = float(track[i][0] - track[0][0]) / denom
        times[i] = t
        
        # Convert rotation vector to quaternion
        rotvec = dataPKL[track[i][0]][track[i][1]]['rotvec'][k]
        quat = R.from_rotvec(rotvec).as_quat()
        
        # Normalize quaternion
        norm = np.linalg.norm(quat)
        if norm != 0:
            quat = quat / norm
            
        # Ensure quaternion continuity
        if i > 0 and np.dot(quats[i-1], quat) < 0:
            quat = -quat
        quats[i] = quat

    # Prepare input for interpolator
    X = np.column_stack((times, np.zeros_like(times)))
    # Build interpolator for quaternion values (4D)

    if rbfkernel == 'univariatespline':
        rbfi = UnivariateSpline(X, quats, k=3, s=rbfsmooth)
    else:
        rbfi = RBFInterpolator(X, quats, kernel=rbfkernel, 
                              epsilon=rbfepsilon, smoothing=rbfsmooth)

    # Update existing frames with interpolated values
    for i in range(trackSize):
        t = times[i]
        x_eval = np.array([[t, 0]])
        interp_quat = rbfi(x_eval)[0]
        
        # Normalize interpolated quaternion
        norm = np.linalg.norm(interp_quat)
        if norm != 0:
            interp_quat /= norm
            
        # Convert back to rotation vector
        dataPKL[track[i][0]][track[i][1]]['rotvec'][k] = R.from_quat(interp_quat).as_rotvec()

    # Fill missing frames with interpolated data
    for i in range(realSize):
        if ava[i] == 0:
            t = float(i) / denom
            x_eval = np.array([[t, 0]])
            interp_quat = rbfi(x_eval)[0]
            
            # Normalize interpolated quaternion
            norm = np.linalg.norm(interp_quat)
            if norm != 0:
                interp_quat /= norm
                
            # Store interpolated rotation vector
            dataPKL[track[0][0] + i][-1]['rotvec'][k] = R.from_quat(interp_quat).as_rotvec()
            
    return k

def RBFFilterTrackArray(key, dim, dataPKL, track, trackSize, rbfkernel, rbfsmooth, rbfepsilon, ava):
    """
    Interpolate array data (shape, expression, transl) for a track.
    
    Args:
        key: Key of the array to interpolate ('shape', 'expression', 'transl')
        dim: Dimension of the array
        dataPKL: Dictionary containing human data
        track: Track data
        trackSize: Size of the track
        rbfkernel: Kernel type for RBF interpolation
        rbfsmooth: Smoothing parameter
        rbfepsilon: Epsilon parameter for RBF kernel
        ava: Availability array indicating which frames have data
    """
    realSize = track[trackSize-1][0] - track[0][0] + 1
    size = realSize - trackSize
    
    # Prepare arrays for missing frames if needed
    if size != 0:
        Xmt = np.zeros(size, dtype=float)
        
    # Process each dimension separately
    for c in range(dim):
        # Extract data for this dimension
        Xm = np.zeros(trackSize, dtype=float)
        Ym = np.zeros(trackSize, dtype=float)
        for i in range(trackSize):
            trackPositions = dataPKL[track[i][0]][track[i][1]][key]
            Xm[i] = float(track[i][0] - track[0][0]) / realSize
            Ym[i] = trackPositions[c]
            
        # Build interpolator
        rbfi = buildInterpolator(Xm, Ym, rbfkernel, rbfepsilon, rbfsmooth)
        
        # Interpolate values for existing frames
        if rbfkernel == 'univariatespline':
            di = rbfi(Xm)
        else:
            newXm = np.column_stack((Xm, np.zeros_like(Xm)))
            di = rbfi(newXm)
            
        # Update existing frames
        for i in range(trackSize):
            dataPKL[track[i][0]][track[i][1]][key][c] = di[i]
            
        # Handle missing frames if any
        if size != 0:
            count = 0
            for i in range(realSize):
                if ava[i] == 0:
                    Xmt[count] = float(i) / realSize
                    count += 1

            # Interpolate values for missing frames
            if rbfkernel == 'univariatespline':
                dit = rbfi(Xmt)
            else:
                newXmt = np.column_stack((Xmt, np.zeros_like(Xmt)))
                dit = rbfi(newXmt)
                
            # Update missing frames
            count = 0
            for i in range(realSize):
                if ava[i] == 0:
                    dataPKL[track[0][0] + i][-1][key][c] = dit[count]
                    count += 1
                    
    return

def fill_track_gaps(allFrameHumans, track, trackSize):
    """
    Fill gaps in a track by copying and marking humans as occluded.
    
    Args:
        allFrameHumans: List of frames with human data
        track: Track data
        trackSize: Size of the track
        
    Returns:
        count: Number of filled frames
        ava: Availability array indicating which frames have data
    """
    # Calculate the real size (including missing frames)
    realSize = track[trackSize - 1][0] - track[0][0] + 1
    
    # Create availability array
    ava = np.zeros(realSize, dtype=int)
    for i in range(trackSize):
        ava[track[i][0] - track[0][0]] = 1
        
    # Fill gaps with copies of the first human in the track
    count = 0
    for i in range(realSize):
        if ava[i] == 0:
            human = copy.deepcopy(allFrameHumans[track[0][0]][track[0][1]])
            human['occluded'] = True
            allFrameHumans[track[0][0] + i].append(human)
            count += 1
            
    return count, ava

def process_track(allFrameHumans, track, trackSize, nbRotvec, kernel, smooth, epsilon, useExpression):
    """
    Process a single track: fill gaps and interpolate data.
    
    Args:
        allFrameHumans: List of frames with human data
        track: Track data
        trackSize: Size of the track
        nbRotvec: Number of rotation vectors
        kernel: Kernel type for RBF interpolation
        smooth: Smoothing parameter
        epsilon: Epsilon parameter for RBF kernel
        useExpression: Whether to interpolate expression data
        
    Returns:
        None (modifies allFrameHumans in-place)
    """
    print(f"    trackSize: {trackSize}")
    print(f"    trackStart: {track[0][0]}")
    print(f"    trackEnd: {track[trackSize - 1][0]}")
    realSize = track[trackSize - 1][0] - track[0][0] + 1
    print(f"    realSize: {realSize}")
    
    # Fill gaps in the track
    count, ava = fill_track_gaps(allFrameHumans, track, trackSize)
    print(f"    count: {count}")
    
    # Interpolate rotation vectors
    pbar = tqdm(total=nbRotvec, unit=' Rotation vector', dynamic_ncols=True, position=0, leave=True)
    for k in range(nbRotvec):
        RBFFilterTrackRotvec(k, allFrameHumans, track, trackSize, kernel, smooth, epsilon, ava)
        pbar.update(1)
    pbar.close()
    print("    rotvec done")
    
    # Interpolate shape parameters
    RBFFilterTrackArray('shape', 10, allFrameHumans, track, trackSize, kernel, smooth, epsilon, ava)
    print("    shape done")
    
    # Interpolate expression parameters if needed
    if useExpression:
        RBFFilterTrackArray('expression', 10, allFrameHumans, track, trackSize, kernel, smooth, epsilon, ava)
        print("    expression done")
        
    # Interpolate translation
    RBFFilterTrackArray('transl', 3, allFrameHumans, track, trackSize, kernel, smooth, epsilon, ava)

def update_smplx_data(allFrameHumans, modelSMPLX, dataPKL, useExpression):
    """
    Update SMPLX data for all humans in all frames.
    
    Args:
        allFrameHumans: List of frames with human data
        modelSMPLX: SMPLX model
        dataPKL: Dictionary containing human data
        useExpression: Whether to use expression data
        
    Returns:
        None (modifies allFrameHumans in-place)
    """
    print("Update SMPLX")
    pbar = tqdm(total=len(allFrameHumans), unit=' frames', dynamic_ncols=True, position=0, leave=True)
    
    for i in range(len(allFrameHumans)):
        # Filter out humans with id -1
        allFrameHumans[i] = [human for human in allFrameHumans[i] if human['id'] != -1]
        
        for j in range(len(allFrameHumans[i])):
            # Update human from SMPLX
            allFrameHumans[i][j] = updateHumanFromSMPLX(
                allFrameHumans[i][j], modelSMPLX, useExpression)
            
            # Project 3D keypoints to 2D
            proj_2d = projectPoints3dTo2d(
                allFrameHumans[i][j]['j3d_smplx'],
                fov=dataPKL['camera_fovs'][i],
                width=dataPKL['video_width'],
                height=dataPKL['video_height']
            )
            allFrameHumans[i][j]['j2d_smplx'] = proj_2d
            
            # Calculate bounding boxes
            min_coords_3d, max_coords_3d = keypointsBBox3D(allFrameHumans[i][j]['j3d_smplx'])
            allFrameHumans[i][j]['bbox3d'] = [min_coords_3d, max_coords_3d]
            
            min_coords_2d, max_coords_2d = keypointsBBox2D(allFrameHumans[i][j]['j2d_smplx'])
            allFrameHumans[i][j]['bbox'] = [min_coords_2d, max_coords_2d]
            
        pbar.update(1)
    pbar.close()

def load_smplx_model(models_path, useExpression):
    """
    Load SMPLX model.
    
    Args:
        models_path: Path to SMPLX models
        useExpression: Whether to use expression data
        
    Returns:
        modelSMPLX: Loaded SMPLX model
    """
    print('[INFO] Loading SMPLX model')
    gender = 'neutral'
    modelSMPLX = smplx.create(
        models_path, 'smplx',
        gender=gender,
        use_pca=False, flat_hand_mean=True,
        num_betas=10,
        ext='npz').cuda()
    print('[INFO] SMPLX model loaded')
    return modelSMPLX

def main():
    if len(sys.argv) != 6:
        print("Usage: python RBFFilterSMPLX.py <input_pkl> <output_pkl> <kernel> <smooth> <epsilon>")
        sys.exit(1)

    # Parse command line arguments
    fileName = sys.argv[1]
    outputFile = sys.argv[2]
    kernel = sys.argv[3]
    smooth = float(sys.argv[4])
    epsilon = float(sys.argv[5])

    # Load data
    print("read pkl file: ", fileName)
    with open(fileName, 'rb') as file:
        dataPKL = pickle.load(file)

    allFrameHumans = dataPKL['allFrameHumans']

    # Remove humans with id -1
    for frame in allFrameHumans:
        frame[:] = [human for human in frame if human['id'] != -1]

    # Calculate maximum ID and build tracks
    maxId = computeMaxId(allFrameHumans)
    
    # Calculate track sizes
    tracksSize = np.zeros(maxId + 1, dtype=int)
    for i in range(len(allFrameHumans)):
        for j in range(len(allFrameHumans[i])):
            if allFrameHumans[i][j]['id'] != -1:
                tracksSize[allFrameHumans[i][j]['id']] += 1
    print('tracksSize: ', tracksSize)
    print('total frames: ', len(allFrameHumans))

    # Build tracks
    tracks = []
    for i in range(maxId + 1):
        tracks.append(np.zeros((tracksSize[i], 2), dtype=int))

    tracksCurrentPosition = np.zeros(maxId + 1, dtype=int)

    for i in range(len(allFrameHumans)):
        for j in range(len(allFrameHumans[i])):
            if allFrameHumans[i][j]['id'] != -1:
                idToProcess = allFrameHumans[i][j]['id']
                tracks[idToProcess][tracksCurrentPosition[idToProcess]] = [i, j]
                tracksCurrentPosition[idToProcess] += 1

    # Find first non-empty frame
    notempty = 0
    while notempty < len(allFrameHumans) and (len(allFrameHumans[notempty]) == 0):
        notempty += 1
    if notempty == len(allFrameHumans):
        notempty = -1
    print("notempty: ", notempty)

    if notempty != -1:
        # Load SMPLX model
        models_path = os.environ["MODELS_PATH"]
        
        # Check if we should use expression parameters
        useExpression = True
        if dataPKL['model_type'] == "hmr2" or dataPKL['model_type'] == "nlf":
            useExpression = False
            
        # Load SMPLX model
        modelSMPLX = load_smplx_model(models_path, useExpression)

        # Get number of rotation vectors
        nbRotvec = allFrameHumans[notempty][0]['rotvec'].shape[0]
        print("nbRotvec: ", nbRotvec)
        print()

        # Process each track
        for t in range(len(tracks)):
            if tracksSize[t] == 0:
                continue
            print(f"Track: {t}/{len(tracks) - 1}")
            process_track(
                allFrameHumans, tracks[t], tracksSize[t], 
                nbRotvec, kernel, smooth, epsilon, useExpression
            )
            print()
            
        # Update SMPLX data after all tracks are processed
        update_smplx_data(allFrameHumans, modelSMPLX, dataPKL, useExpression)

    # Save output
    with open(outputFile, 'wb') as handle:
        pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()

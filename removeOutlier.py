import sys
import pickle
import numpy as np
import os
import smplx

from premiere.functionsCommon import buildTracks, computeMaxId
from premiere.functionsSMPLX import updateHumanFromSMPLX

def check_and_straighten_legs(human, video_width, video_height, modelSMPLX=None):
    """
    Check if leg keypoints are out of frame and straighten legs if needed.
    Also detects and corrects inverted feet positions.
    
    :param human: Dictionary containing human data
    :param video_width: Width of the video frame
    :param video_height: Height of the video frame
    :param modelSMPLX: Optional SMPLX model for recalculating joint positions
    :return: Human dictionary with potentially updated pose, boolean indicating if changes were made, and correction type
    """
    # Define leg joint indices in SMPLX j2d_smplx
    leg_joints = [1, 2, 4, 5, 7, 8, 10, 11]  # Hips, knees, ankles, toes
    foot_joints = [7, 8, 10, 11]  # Ankles and toes
    
    # Check if these keypoints are outside the frame
    is_outside = False
    for idx in leg_joints:
        x, y = human['j2d_smplx'][idx]
        if x < 0 or x >= video_width or y < 0 or y >= video_height:
            is_outside = True
            break
    
    # Check if feet appear to be inverted
    is_feet_inverted = False
    if not is_outside and 'j2d_smplx' in human:
        # Left/right ankle positions (indices 7, 8)
        left_ankle_x = human['j2d_smplx'][7][0]
        right_ankle_x = human['j2d_smplx'][8][0]
        
        # Left/right toe positions (indices 10, 11)
        left_toe_x = human['j2d_smplx'][10][0]
        right_toe_x = human['j2d_smplx'][11][0]
        
        # Check if left foot is on the right side and right foot is on the left side
        # This simple check assumes the person is mostly facing the camera
        if (left_ankle_x > right_ankle_x and left_toe_x > right_toe_x):
            is_feet_inverted = True
    
    # Modify pose if legs are outside frame or feet are inverted
    if is_outside or is_feet_inverted:
        # Create a copy of the rotational vectors
        modified_rotvec = human['rotvec'].copy()
        
        # Set rotation vectors for hip and knee joints to zero (straight)
        # Hip joints (indices 1 and 2 in the pose vector)
        modified_rotvec[1] = np.zeros(3)
        modified_rotvec[2] = np.zeros(3)
        # Knee joints (indices 4 and 5 in the pose vector)
        modified_rotvec[4] = np.zeros(3)
        modified_rotvec[5] = np.zeros(3)
        
        # Reset ankle joints to have feet parallel to the ground
        # Ankle joints (indices 7 and 8 in the pose vector)
        modified_rotvec[7] = np.array([1.3, 0, 0])  # Left ankle: pitch rotation to point foot forward
        modified_rotvec[8] = np.array([1.3, 0, 0])  # Right ankle: pitch rotation to point foot forward
        
        # If feet are inverted or legs are outside, apply outward rotation for better foot placement
        hip_outward_angle = 0.15  # Slightly stronger than before
        modified_rotvec[1][1] = -hip_outward_angle  # Left hip: negative Y rotation (outward)
        modified_rotvec[2][1] = hip_outward_angle   # Right hip: positive Y rotation (outward)
        
        # Update the human's rotational vectors
        human['rotvec'] = modified_rotvec
        
        # If a SMPLX model is provided, recalculate the 3D and 2D joint positions
        if modelSMPLX is not None:
            useExpression = 'expression' in human and human['expression'] is not None
            human = updateHumanFromSMPLX(human, modelSMPLX, useExpression)
            
            # Update 2D projections
            from premiere.functionsCommon import projectPoints3dTo2d
            human['j2d_smplx'] = projectPoints3dTo2d(
                human['j3d_smplx'], 
                fov=human.get('fov', 70),
                width=video_width,
                height=video_height
            )
            
            # Additional check - ensure feet have appropriate Y position (height from ground)
            # This assumes Y is the vertical axis in your coordinate system
            if 'j3d_smplx' in human:
                # Check feet height - adjust if needed
                left_ankle_y = human['j3d_smplx'][7][1]
                right_ankle_y = human['j3d_smplx'][8][1]
                left_toe_y = human['j3d_smplx'][10][1]
                right_toe_y = human['j3d_smplx'][11][1]
                
                # If any foot joint is significantly below others, adjust the translation to align them
                min_y = min(left_ankle_y, right_ankle_y, left_toe_y, right_toe_y)
                if min_y < -0.05:  # If feet are below the ground plane
                    human['transl'][1] -= min_y  # Adjust the Y translation to bring feet to ground
                    
                    # Recalculate projections with adjusted translation
                    human = updateHumanFromSMPLX(human, modelSMPLX, useExpression)
                    human['j2d_smplx'] = projectPoints3dTo2d(
                        human['j3d_smplx'], 
                        fov=human.get('fov', 70),
                        width=video_width,
                        height=video_height
                    )
        
        # Return reason for modification in debug message
        correction_type = "out-of-frame legs" if is_outside else "inverted feet"
        return human, True, correction_type
    
    return human, False, None

def removeOutliersInTrack(track, trackSize, allFrameHumans, fps, velocity_thresh_factor=4.0, straighten_legs=False):
    """
    Remove outliers from a track based on velocity thresholds.

    :param track: list of tuples (frame_index, person_index)
    :param trackSize: number of points in the track
    :param allFrameHumans: data structure holding human data per frame
    :param fps: frames per second of the video
    :param velocity_thresh_factor: multiplicative threshold on the MAD-based speed outlier detection
    :param straighten_legs: whether to straighten legs when they're not visible in frame
    """
    times = np.zeros(trackSize, dtype=int)
    points3d = np.zeros((trackSize,3), dtype=float)

    # Collect frames (times) and 3D points
    for i in range(trackSize):
        times[i] = track[i][0]
        points3d[i] = allFrameHumans[track[i][0]][track[i][1]]['j3d_smplx'][0]

    # If the track is too short, skip outlier detection
    if trackSize < 3:
        return

    # Compute time deltas in seconds
    # dt[i] = (times[i+1] - times[i]) / fps
    dt_frames = np.diff(times)  
    dt_seconds = dt_frames / fps

    # Handle any zero time deltas if they occur
    valid_dt_mask = dt_frames != 0
    if not np.all(valid_dt_mask):
        # If some dt are zero, you may handle them differently
        # e.g., skip those segments, or set them to a very small non-zero
        # For simplicity here, let's just skip them in velocity calculation
        pass

    # Compute velocities (points3d are in "units" -> e.g., meters if scaled; otherwise the unit is your 3D coordinate system)
    # velocity[i] = (points3d[i+1] - points3d[i]) / (times[i+1] - times[i] in seconds)
    velocities = (points3d[1:] - points3d[:-1]) / dt_seconds[:, None]  # shape = (trackSize-1, 3)
    speed = np.linalg.norm(velocities, axis=1)

    # Compute median and MAD of speed
    median_speed = np.median(speed)
    abs_dev = np.abs(speed - median_speed)
    mad_speed = np.median(abs_dev) + 1e-9  # add a small eps to avoid division by zero

    # Compute a z-score-like metric for outliers
    z_score_like = abs_dev / mad_speed

    # Identify segments (between i and i+1) with large jumps
    large_jumps = z_score_like > velocity_thresh_factor

    # We will mark outliers for the track points
    outlier_mask = np.zeros(trackSize, dtype=bool)

    # Mark outliers in both points around the large jump (or only i+1, depending on preference)
    for i, jump in enumerate(large_jumps):
        if jump:
            # Mark i+1 (and optionally i) as outlier
            outlier_mask[i+1] = True

    # Remove outliers from allFrameHumans
    count = 0
    for i in range(trackSize):
        if outlier_mask[i]:
            frame_idx = track[i][0]
            person_idx = track[i][1]
            allFrameHumans[frame_idx][person_idx]['id'] = -1
            count += 1

    # Check and straighten legs if enabled
    if straighten_legs:
        # Load SMPLX model once if needed
        modelSMPLX = None
        legs_fixed = 0
        
        for i in range(trackSize):
            if outlier_mask[i]:
                continue  # Skip outliers we've already marked
                
            frame_idx = track[i][0]
            person_idx = track[i][1]
            human = allFrameHumans[frame_idx][person_idx]
            
            # Get video dimensions from the human data (needs to be added to dataPKL)
            video_width = dataPKL.get('video_width', 1920)  # Default if not found
            video_height = dataPKL.get('video_height', 1080)  # Default if not found
            
            # Load SMPLX model if needed and not already loaded
            if modelSMPLX is None and 'j2d_smplx' in human:
                try:
                    models_path = os.environ.get("MODELS_PATH")
                    if models_path:
                        modelSMPLX = smplx.create(
                            models_path, 'smplx',
                            gender='neutral',
                            use_pca=False, flat_hand_mean=True,
                            num_betas=10,
                            ext='npz').cuda()
                except Exception as e:
                    print(f"Could not load SMPLX model: {e}")
            
            # Check and straighten legs if needed
            if modelSMPLX is not None and 'j2d_smplx' in human:
                updated_human, was_modified, correction_type = check_and_straighten_legs(
                    human, video_width, video_height, modelSMPLX
                )
                if was_modified:
                    allFrameHumans[frame_idx][person_idx] = updated_human
                    legs_fixed += 1
                    if correction_type == "inverted feet":
                        print(f"  Frame {frame_idx}: Fixed inverted feet")
        
        if legs_fixed > 0:
            print(f"Fixed {legs_fixed} frames with out-of-frame legs in track {track[0][1]}")

    print(f"Removed {count} outliers from track {track[0][1]}")
    return


if len(sys.argv) < 3:
    print("Usage: python removeOutlier.py <input_pkl_path> <output_pkl_path> [straighten_legs=0]")
    sys.exit(1)

pkl_path = sys.argv[1]
output_path = sys.argv[2]
straighten_legs = False
if len(sys.argv) > 3:
    straighten_legs = int(sys.argv[3]) > 0

print("Read pkl:", pkl_path)
with open(pkl_path, 'rb') as f:
    dataPKL = pickle.load(f)

video_fps = dataPKL['video_fps']
allFrameHumans = dataPKL['allFrameHumans']

maxId = computeMaxId(allFrameHumans)
tracks, tracksSize = buildTracks(allFrameHumans, maxId)

print("Number of tracks:", len(tracks))
print("Number of humans:", maxId)
print("Processing...")

# Remove outliers in each track taking into account the fps
for i, track in enumerate(tracks):
    if tracksSize[i] > 0:
        removeOutliersInTrack(track, tracksSize[i], allFrameHumans, video_fps, 
                              straighten_legs=straighten_legs)

# Save the updated pickle
with open(output_path, 'wb') as handle:
    pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done! Output saved to:", output_path)

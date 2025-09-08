"""
Track Fusion Script
------------------
This script consolidates multiple tracking IDs based on their frequency
and cleans up tracking data by removing duplicates.
"""

import sys
import pickle
import numpy as np

from premiere.functionsCommon import buildTracksForFusion, computeMaxId


def analyzeTrack(tracks, trackId):
    """
    Analyzes a track to determine its most frequent SAM ID
    
    Args:
        tracks: Dictionary of all track data
        trackId: ID of the track to analyze
        
    Returns:
        int: Most frequent SAM ID in the track, or -1 if track is empty
    """
    # Extract SAM IDs for the given trackId
    idSAMs = tracks[trackId][:, 2]
    
    # Filter out -1 values (which represent unassigned IDs)
    idSAMs = idSAMs[idSAMs != -1]
    
    # Return -1 if no valid SAM IDs in this track
    if len(idSAMs) == 0:
        return -1
        
    # Count occurrences of each SAM ID value
    counts = np.bincount(idSAMs)
    
    # Find the most frequent SAM ID
    most_frequent_idSAM = np.argmax(counts)
    return most_frequent_idSAM


def main():
    """Main execution function"""
    # Check for correct command line arguments
    if len(sys.argv) < 4:
        print("Usage: python tracksFusion.py <input_pkl_path> <output_pkl_path> <trackSizeMin>")
        sys.exit(1)

    # Get input file path from command line
    pkl_path = sys.argv[1]
    output_path = sys.argv[2]
    trackSizeMin = int(sys.argv[3])

    # Load data from pickle file
    print(f"Reading pickle file: {pkl_path}")
    with open(pkl_path, 'rb') as file:
        dataPKL = pickle.load(file)

    # Extract human detection data
    allFrameHumans = dataPKL['allFrameHumans']

    # Compute the maximum ID and build tracks for fusion
    maxId = computeMaxId(allFrameHumans)
    tracks, tracksSize = buildTracksForFusion(allFrameHumans, maxId)

    # Process each track
    for i in range(len(tracksSize)):
        most_frequent = analyzeTrack(tracks, i)
        
        # If track has a valid ID and meets minimum size requirements
        if (most_frequent != -1) and (tracksSize[i] > trackSizeMin):
            # Assign the most frequent ID to all detections in this track
            for j in range(tracksSize[i]):
                frame_idx = tracks[i][j][0]
                human_idx = tracks[i][j][1]
                allFrameHumans[frame_idx][human_idx]['id'] = most_frequent
        else:
            # Mark all detections in this track as invalid (-1)
            for j in range(tracksSize[i]):
                frame_idx = tracks[i][j][0]
                human_idx = tracks[i][j][1]
                allFrameHumans[frame_idx][human_idx]['id'] = -1

    # Remove duplicate detections (humans with same ID in the same frame)
    for i in range(len(allFrameHumans)):
        unique_humans = []
        seen_ids = set()
        
        for human in allFrameHumans[i]:
            if human['id'] not in seen_ids:
                seen_ids.add(human['id'])
                unique_humans.append(human)
        
        allFrameHumans[i] = unique_humans
    
    # Save the processed data back to pickle file
    print(f"Writing results to: {output_path}")
    with open(output_path, 'wb') as handle:
        pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()


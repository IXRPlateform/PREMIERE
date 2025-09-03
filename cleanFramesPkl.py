"""
Clean and cluster humans in frames from a pickled dataset.

This script reads a pickle file containing a dictionary with 
key 'allFrameHumans', where each frame stores a list of detected humans.
Depending on the 'nbMaxHumans' parameter:
  - If nbMaxHumans == 1, only the highest-scoring human is kept per frame.
  - Otherwise, hierarchical clustering is applied to group humans 
    by their 3D pelvis translation. Then the top-scoring human per cluster 
    is kept, optionally limiting the total number of clusters to 'nbMaxHumans'
    if nbMaxHumans != -1.

Usage:
  python cleanFramesPkl.py <input_pkl> <output_pkl> <nbMaxHumans> <distance_threshold>

:param input_pkl: Path to the input pickle file.
:param output_pkl: Path for saving the cleaned pickle file.
:param nbMaxHumans: Maximum number of humans to keep per frame. 
                   If -1, keep all clusters.
:param distance_threshold: Distance threshold used by hierarchical clustering 
                          (ward linkage).
"""

import os
import sys
import copy
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# Import common functions
from premiere.functionsCommon import loadPkl, savePkl


def keep_best_human_per_frame(humans):
    """
    Keep only the best (highest-scoring) human in a frame.
    
    :param humans: List of human detections in a frame
    :return: List containing only the best human, or empty list if no humans
    """
    if len(humans) == 0:
        return []
    
    scores = np.array([h['score'] for h in humans])
    max_index = np.argmax(scores)
    return [humans[max_index]]


def cluster_humans_in_frame(humans, distance_threshold, max_humans):
    """
    Apply hierarchical clustering to humans based on 3D pelvis position.
    
    :param humans: List of human detections in a frame
    :param distance_threshold: Distance threshold for clustering
    :param max_humans: Maximum number of humans to keep (-1 for unlimited)
    :return: Tuple of (processed_humans, num_clusters)
    """
    # Handle simple cases
    if len(humans) == 0:
        return [], 0
    if len(humans) == 1:
        return humans, 1
        
    # Prepare points array for hierarchical clustering (pelvis translation)
    points = np.empty([len(humans), 3], dtype=float)
    for j, human in enumerate(humans):
        # Extract the 3D pelvis translation for each human
        points[j, 0] = human['transl_pelvis'][0][0]
        points[j, 1] = human['transl_pelvis'][0][1]
        points[j, 2] = human['transl_pelvis'][0][2]

    # Perform hierarchical clustering with 'ward'
    Z = linkage(points, method='ward')
    clusters_by_distance = fcluster(Z, distance_threshold, criterion='distance')
    
    # Gather the best-scoring human from each cluster
    cluster_best_list = []
    clusters_by_id = {}
    for index, cluster_id in enumerate(clusters_by_distance):
        if (cluster_id - 1) not in clusters_by_id:
            clusters_by_id[cluster_id - 1] = []
        clusters_by_id[cluster_id - 1].append(index)
    
    # Find the best human per cluster
    for cluster_id, indices in clusters_by_id.items():
        if len(indices) > 1:
            # More than one human in this cluster
            scores = np.array([humans[idx]['score'] for idx in indices])
            max_index = np.argmax(scores)
            best_human = humans[indices[max_index]]
        else:
            # Only one human in this cluster
            best_human = humans[indices[0]]
        
        cluster_best_list.append((best_human['score'], best_human, cluster_id))

    # Sort clusters by best score in descending order
    cluster_best_list.sort(key=lambda x: x[0], reverse=True)
    
    # If max_humans == -1, do not limit number of humans
    if max_humans == -1:
        final_cluster_best_list = cluster_best_list
    else:
        # Otherwise, keep only the top max_humans clusters
        final_cluster_best_list = cluster_best_list[:max_humans]

    # Extract the best humans from chosen clusters
    processed_humans = [item[1] for item in final_cluster_best_list]
    
    return processed_humans, len(final_cluster_best_list)


def main():
    # Command line argument validation
    if len(sys.argv) != 5:
        print("Usage: python cleanFramesPkl.py <input_pkl> <output_pkl> <nbMaxHumans> <distance_threshold>")
        sys.exit(1)

    # Read arguments
    input_pkl = sys.argv[1]
    output_pkl = sys.argv[2]
    nb_max_humans = int(sys.argv[3])
    distance_threshold = float(sys.argv[4])

    print("read pkl:", input_pkl)
    data_pkl = loadPkl(input_pkl)

    # Deep copy to avoid modifying the original data
    new_data_pkl = copy.deepcopy(data_pkl)

    # Preallocate arrays for clusters per frame and humans per frame
    nb_frames = len(data_pkl['allFrameHumans'])
    nb_clusters_per_frame = np.zeros(nb_frames, dtype=int)
    humans_per_frame = np.empty(nb_frames, dtype=int)

    # Count how many humans are initially in each frame
    for i in range(nb_frames):
        humans_per_frame[i] = len(data_pkl['allFrameHumans'][i])

    max_nb_humans_in = max(humans_per_frame)
    min_nb_humans_in = min(humans_per_frame)

    print('maxNbHumans (before cleanup):', max_nb_humans_in)
    print('minNbHumans (before cleanup):', min_nb_humans_in)

    # Process frames based on the requested number of humans
    for i in range(nb_frames):
        humans = data_pkl['allFrameHumans'][i]
        
        if nb_max_humans == 1:
            # Case 1: If nb_max_humans == 1, pick only the highest-scoring human per frame
            new_data_pkl['allFrameHumans'][i] = keep_best_human_per_frame(humans)
            nb_clusters_per_frame[i] = len(new_data_pkl['allFrameHumans'][i])
        else:
            # Case 2: Apply clustering to group nearby humans
            new_data_pkl['allFrameHumans'][i], nb_clusters_per_frame[i] = cluster_humans_in_frame(
                humans, distance_threshold, nb_max_humans)

    # Print cluster stats
    max_clusters = max(nb_clusters_per_frame)
    min_clusters = min(nb_clusters_per_frame)
    print('max_clusters:', max_clusters)
    print('min_clusters:', min_clusters)

    # Recompute the number of humans in each frame after cleanup
    for i in range(nb_frames):
        humans_per_frame[i] = len(new_data_pkl['allFrameHumans'][i])

    max_nb_humans_out = max(humans_per_frame)
    min_nb_humans_out = min(humans_per_frame)

    print('maxNbHumans (after cleanup):', max_nb_humans_out)
    print('maxNbHumansIndices:', np.argmax(humans_per_frame))
    print('minNbHumans (after cleanup):', min_nb_humans_out)

    # Finally, save the cleaned PKL file
    savePkl(output_pkl, new_data_pkl)


if __name__ == "__main__":
    main()

import os
import sys
import cv2
import pickle
import warnings

import numpy as np

# Treat warnings as errors
warnings.filterwarnings('error')

def calculate_average_euclidean_distance2(dataPKL, i, j, k, nbKeyPoints):
    # Extract keypoints for frames i and i+1
    keypoints_frame_i = dataPKL['allFrameHumans'][i][k]['j3d_smplx'][0:nbKeyPoints]
    keypoints_frame_i_plus_1 = dataPKL['allFrameHumans'][i+1][j]['j3d_smplx'][0:nbKeyPoints]
    # Compute Euclidean distance for each keypoint
    distances = np.linalg.norm(keypoints_frame_i - keypoints_frame_i_plus_1)
    # Compute average Euclidean distance
    average_distance = np.mean(distances)
    
    return average_distance

def calculate_average_euclidean_distance(dataPKL, i, j, k, nbKeyPoints):
    # Extract pelvis translation for frames i and i+1
    keypoints_frame_i = dataPKL['allFrameHumans'][i][k]['transl_pelvis']
    keypoints_frame_i_plus_1 = dataPKL['allFrameHumans'][i+1][j]['transl_pelvis']
    
    # Compute Euclidean distance between pelvis translations
    distance = np.linalg.norm(keypoints_frame_i - keypoints_frame_i_plus_1)
    # Return the computed distance
    #print("distance: ",distance)
    return distance

def main():
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python cleanFramesPkl.py <input_pkl> <output_pkl> <distance_threshold>")
        sys.exit(1)

    # Load the input pickle file
    print ("read pkl: ",sys.argv[1])
    file = open(sys.argv[1], 'rb')
    dataPKL = pickle.load(file) 
    file.close()

    threshold = float(sys.argv[3])
    nbKeyPoints = 10

    # Find the maximum number of humans detected in any frame
    maxHumans = 0
    for i in range(len(dataPKL['allFrameHumans'])):
        maxHumans = max(maxHumans, len(dataPKL['allFrameHumans'][i]))
    print('maxHumans: ', maxHumans)

    nextId = 0
    notempty = 0
    # Find the first non-empty frame
    while len(dataPKL['allFrameHumans'][notempty])==0:
        notempty += 1
    print("notempty: ",notempty)

    # Assign an ID to each person in the first non-empty frame
    for i in range(len(dataPKL['allFrameHumans'][notempty])):
        dataPKL['allFrameHumans'][notempty][i]['id'] = nextId
        nextId += 1

    print ('start processing')
    # Iterate through frames and assign IDs to detected humans
    for i in range(notempty,len(dataPKL['allFrameHumans'])-1):
        size = len(dataPKL['allFrameHumans'][i])
        sizeplusone = len(dataPKL['allFrameHumans'][i+1])
        if (size!=0 and sizeplusone!=0):   
            # Create a distance matrix between humans in consecutive frames
            distances = np.empty([sizeplusone, size],dtype=float)
            for j in range(sizeplusone):
                for k in range(size):
                    distances[j,k] = calculate_average_euclidean_distance(dataPKL, i, j, k, nbKeyPoints)
            # Assign IDs based on minimum distance if below threshold
            for j in range(sizeplusone):
                minIndex = np.argmin(distances)  # Index of minimum distance
                row = minIndex//size
                col = minIndex%size
                minDistance = distances[row,col]
                if(minDistance<threshold):
                    dataPKL['allFrameHumans'][i+1][row]['id'] = dataPKL['allFrameHumans'][i][col]['id']
                    # Prevent assigning the same ID to multiple humans
                    for k in range(sizeplusone):
                        distances[k,col] = np.inf 
            # Assign new IDs to humans not matched by distance
            for j in range(sizeplusone):
                if (dataPKL['allFrameHumans'][i+1][j]['id']==-1):
                    dataPKL['allFrameHumans'][i+1][j]['id'] = nextId
                    nextId += 1                
        else:
            # If current or next frame is empty, assign new IDs
            for j in range(sizeplusone):
                if(dataPKL['allFrameHumans'][i+1][j]['id']==-1):
                    dataPKL['allFrameHumans'][i+1][j]['id'] = nextId
                    nextId += 1

    # Collect tracking results for each ID
    allTracks = []
    for i in range(nextId):
        allTracks.append([])
    for i in range(len(dataPKL['allFrameHumans'])):
        for j in range(len(dataPKL['allFrameHumans'][i])):
            element = (i,j,np.squeeze(dataPKL['allFrameHumans'][i][j]['transl_pelvis'], axis=0).tolist())
            allTracks[dataPKL['allFrameHumans'][i][j]['id']].append(element)

    print ('done')
    print ("Last Id: ", nextId)

    # Save the updated pickle file
    with open(sys.argv[2], 'wb') as handle:
        pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL) 


if __name__ == "__main__":
    main()

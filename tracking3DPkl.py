import os
import sys
import cv2
import pickle
import warnings

import numpy as np

warnings.filterwarnings('error')

def calculate_average_euclidean_distance2(dataPKL, i, j, k, nbKeyPoints):
    # Extraire les keypoints pour les frames i et i+1
    keypoints_frame_i = dataPKL['allFrameHumans'][i][k]['j3d_smplx'][0:nbKeyPoints]
    keypoints_frame_i_plus_1 = dataPKL['allFrameHumans'][i+1][j]['j3d_smplx'][0:nbKeyPoints]
    # Calculer la distance euclidienne pour chaque keypoint
    distances = np.linalg.norm(keypoints_frame_i - keypoints_frame_i_plus_1)
    # Calculer la distance euclidienne moyenne
    average_distance = np.mean(distances)
    
    return average_distance

def calculate_average_euclidean_distance(dataPKL, i, j, k, nbKeyPoints):
    # Extraire les keypoints pour les frames i et i+1
    keypoints_frame_i = dataPKL['allFrameHumans'][i][k]['transl_pelvis']
    keypoints_frame_i_plus_1 = dataPKL['allFrameHumans'][i+1][j]['transl_pelvis']
    
    # Calculer la distance euclidienne pour chaque keypoint
    distance = np.linalg.norm(keypoints_frame_i - keypoints_frame_i_plus_1)
    # Calculer la distance euclidienne moyenne
    #print("distance: ",distance)
    return distance

if len(sys.argv) != 4:
    print("Usage: python cleanFramesPkl.py <input_pkl> <output_pkl> <distance_threshold>")
    sys.exit(1)

# Open the pkl file
print ("read pkl: ",sys.argv[1])
file = open(sys.argv[1], 'rb')
dataPKL = pickle.load(file) 
file.close()

threshold = float(sys.argv[3])
nbKeyPoints = 10

maxHumans = 0
for i in range(len(dataPKL['allFrameHumans'])):
    maxHumans = max(maxHumans, len(dataPKL['allFrameHumans'][i]))
print('maxHumans: ', maxHumans)

nextId = 0
notempty = 0
while len(dataPKL['allFrameHumans'][notempty])==0:
    notempty += 1
print("notempty: ",notempty)

# Assign an ID to each person in the first frame
for i in range(len(dataPKL['allFrameHumans'][notempty])):
    dataPKL['allFrameHumans'][notempty][i]['id'] = nextId
    nextId += 1

print ('start processing')
for i in range(notempty,len(dataPKL['allFrameHumans'])-1):
    size = len(dataPKL['allFrameHumans'][i])
    sizeplusone = len(dataPKL['allFrameHumans'][i+1])
    if (size!=0 and sizeplusone!=0):   
        distances = np.empty([sizeplusone, size],dtype=float)
        for j in range(sizeplusone):
            for k in range(size):
                distances[j,k] = calculate_average_euclidean_distance(dataPKL, i, j, k, nbKeyPoints)
        for j in range(sizeplusone):
            minIndex = np.argmin(distances)  # Index du minimum
            row = minIndex//size
            col = minIndex%size
            minDistance = distances[row,col]
            if(minDistance<threshold):
                dataPKL['allFrameHumans'][i+1][row]['id'] = dataPKL['allFrameHumans'][i][col]['id']
                for k in range(sizeplusone):
                    distances[k,col] = np.inf 
        for j in range(sizeplusone):
            if (dataPKL['allFrameHumans'][i+1][j]['id']==-1):
                dataPKL['allFrameHumans'][i+1][j]['id'] = nextId
                nextId += 1                
    else:
        for j in range(sizeplusone):
            if(dataPKL['allFrameHumans'][i+1][j]['id']==-1):
                dataPKL['allFrameHumans'][i+1][j]['id'] = nextId
                nextId += 1

# test results
allTracks = []
for i in range(nextId):
    allTracks.append([])
for i in range(len(dataPKL['allFrameHumans'])):
    for j in range(len(dataPKL['allFrameHumans'][i])):
        element = (i,j,np.squeeze(dataPKL['allFrameHumans'][i][j]['transl_pelvis'], axis=0).tolist())
        allTracks[dataPKL['allFrameHumans'][i][j]['id']].append(element)

print ('done')
print ("Last Id: ", nextId)
#print(allCountPointsInBoxTracking)

with open(sys.argv[2], 'wb') as handle:
    pickle.dump(dataPKL, handle, protocol=pickle.HIGHEST_PROTOCOL) 

"""
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
""" 

# python .\processFramesPkl.py .\pkl\D0-talawa_technique_intro-Scene-015_896L_Clean.pkl .\pkl\D0-talawa_technique_intro-Scene-015_896L_Clean_Track.pkl 0.4
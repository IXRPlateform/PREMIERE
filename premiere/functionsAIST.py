import os
import re
import json

def getDanceGenres ():
    return [ [ "gJB", "Ballet Jazz" ], 
             [ "gBR", "Break" ], 
             [ "gHO", "House" ], 
             [ "gKR", "Krump" ], 
             [ "gLH", "LA style Hip-hop" ], 
             [ "gLO", "Lock" ], 
             [ "gMH", "Middle Hip-hop" ], 
             [ "gPO", "Pop" ], 
             [ "gJS", "Street Jazz" ], 
             [ "gWA", "Waack" ] ]

def readAllCamerasSettings ( aistFileDirectory, settingsFilenames):
    allSettings = {}
    for settingFilename in settingsFilenames:
        finalSettingFilename = settingFilename +".json"
        finalSettingFilename = os.path.join(aistFileDirectory, "cameras", finalSettingFilename)
        with open( finalSettingFilename, 'r', encoding='utf-8') as f:
            setting = json.load(f)
        allSettings[settingFilename] = setting
    return allSettings

def getVideoSettings( videoFilename, settingDances):
    basename = videoFilename[:-4]
    key = re.sub(r'_c\d{2}', '_cAll', basename)
    return settingDances[key]

def getDanceGenreVideos ( allVideoFileNames, camera, danceGenre ):
    # Function to get the video of the dance genre
    # Input: camera, danceGenre
    # Output: video
    # Dependencies: None
    matchingVideos = []
    for videoFileName in allVideoFileNames:
        if camera in videoFileName and danceGenre in videoFileName:
            matchingVideos.append(videoFileName)
    return matchingVideos

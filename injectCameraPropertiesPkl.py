"""
Inject camera properties from a JSON file into a pickle file.

This script reads camera properties from a JSON file and injects them into
a pickle file containing human detection data.

Usage:
  python injectCameraPropertiesPkl.py <final_json> <input_pkl> <output_pkl>

:param final_json: Path to the JSON file with camera properties
:param input_pkl: Path to the input pickle file with human detections
:param output_pkl: Path for the output pickle file with injected properties
"""

import json
import sys

# Import common functions
from premiere.functionsCommon import loadPkl, savePkl

def main():
    """Main function to process command-line arguments and inject camera properties."""
    if len(sys.argv) != 4:
        print("Usage: python injectCameraPropertiesPkl.py <final_json> <input_pkl> <output_pkl>")
        sys.exit(1)

    # Parse command line arguments
    finalJSONName = sys.argv[1]
    inputPklName = sys.argv[2]
    outputPklName = sys.argv[3]
    
    # Load pickle data using functionsCommon.loadPkl
    print("Reading PKL:", inputPklName)
    dataPKL = loadPkl(inputPklName)
    
    # Load JSON data
    print("Reading JSON:", finalJSONName)
    with open(finalJSONName, 'r') as file:
        finalJSON = json.load(file)

    # Inject camera properties
    dataPKL['floor_angle_deg'] = finalJSON['floor_angle_deg']
    dataPKL['floor_Zoffset'] = finalJSON['floor_Zoffset']
    dataPKL['camera_rotation_deg'] = finalJSON['camera_rotation_deg']
    dataPKL['camera_fixed'] = finalJSON['camera_fixed']
    dataPKL['dynamic_fov'] = finalJSON['dynamic_fov']
    dataPKL['fov_x_deg'] = finalJSON['fov_x_deg']

    # Save updated pickle data using functionsCommon.savePkl
    print("Writing PKL:", outputPklName)
    savePkl(outputPklName, dataPKL)

if __name__ == "__main__":
    main()

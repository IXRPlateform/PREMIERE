"""
Remove shadow-like detections from a pickle file based on a threshold.

This script processes a pickle file containing human detections and removes entries 
that have a low computed score. The score is calculated as:
    score = homogeneity * detection_score * (1 / depth_translation_z)

Entries with a computed score below the threshold are considered shadows and are removed.

**Usage**:
    python removeShadowsPkl.py <input_pkl> <output_pkl> <threshold>

:param input_pkl: Path to the input pickle file containing detections (e.g., with 'homogeneity').
:param output_pkl: Path to save the filtered pickle file with shadow-like detections removed.
:param threshold: The score threshold below which detections are considered shadows and removed.
"""

import sys
import copy
import numpy as np

# Import common functions
from premiere.functionsCommon import loadPkl, savePkl

def main():
    """Main function to remove shadow-like detections from a pickle file."""
    # Validate command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python removeShadowsPkl.py <input_pkl> <output_pkl> <threshold>")
        sys.exit(1)

    # Parse arguments
    inputPklName = sys.argv[1]
    outputPklName = sys.argv[2]
    threshold = float(sys.argv[3])

    # Load input pickle file using functionsCommon.loadPkl
    print("Reading input pickle:", inputPklName)
    dataPKL = loadPkl(inputPklName)

    # Extract the frame-wise human detection data
    allFrameHumans = dataPKL['allFrameHumans']

    # Create a deep copy of the original data to store filtered results
    newDataPKL = copy.deepcopy(dataPKL)
    
    # Track statistics for reporting
    total_detections = 0
    shadows_removed = 0

    try:
        # Iterate over all frames and process detections
        for i in range(len(dataPKL['allFrameHumans'])):
            humans = dataPKL['allFrameHumans'][i]
            # Reset the newDataPKL list for this frame
            newDataPKL['allFrameHumans'][i] = []
            
            total_detections += len(humans)

            # Process each human detection in the frame
            for j in range(len(humans)):
                if 'homogeneity' in humans[j]:
                    # Compute the filtering score
                    score = humans[j]['homogeneity'] * humans[j]['score'] * (1 / humans[j]['transl'][2])
                    if score > threshold:
                        # Keep the detection if the score exceeds the threshold
                        newDataPKL['allFrameHumans'][i].append(humans[j])
                    else:
                        # Log shadow removal for debugging purposes
                        print(f"Shadow removed: Frame {i}, Detection {j}")
                        shadows_removed += 1
                else:
                    # If homogeneity isn't available, keep the detection
                    newDataPKL['allFrameHumans'][i].append(humans[j])

        # Save the filtered results to the output pickle file using functionsCommon.savePkl
        print(f"\nSaving filtered results to: {outputPklName}")
        savePkl(outputPklName, newDataPKL)
        
        # Report summary statistics
        print(f"\nProcessing complete:")
        print(f"  - Total detections: {total_detections}")
        print(f"  - Shadows removed: {shadows_removed} ({shadows_removed/total_detections*100:.1f}%)")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

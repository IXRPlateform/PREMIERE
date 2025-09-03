"""
Track humans with ByteTrack and update IDs in a pickle file of detections.

This script reads:
1. An input pickle file (with a dictionary containing 'allFrameHumans' per frame).
2. A video file.
3. A flag indicating whether to display the video and its bounding boxes.
4. An output pickle file path.

The script:
- Initializes a ByteTrack tracker.
- Iterates through each frame of the input video.
- Retrieves human detections (bounding boxes, scores, etc.) from the pickle file.
- Runs ByteTrack tracking to assign or update IDs.
- Optionally displays each frame with bounding boxes and IDs overlaid.
- Updates the 'id' field for each human in the pickle file and saves the result.

**Usage**:
    python trackBTPkl.py <input_pkl> <video> <output_pkl> <display: 0 No, 1 Yes>

:param input_pkl: Path to the input pickle file with pre-detected humans (list of boxes).
:param video: Path to the input video file.
:param output_pkl: Path to the output pickle file where updated results are stored.
:param display: 0 or 1 (if 1, displays the video with bounding boxes during processing).
"""

import sys
import numpy as np
import cv2

from tqdm import tqdm
from bytetrack.nets import nn

# Import common functions
from premiere.functionsCommon import loadPkl, savePkl

def draw_line(image, x1, y1, x2, y2, index):
    """
    Draw a bounding box with stylized corners and an ID label on `image`.

    :param image: The image where the box is drawn.
    :type image: np.ndarray
    :param x1: X-coordinate of the top-left corner.
    :type x1: int
    :param y1: Y-coordinate of the top-left corner.
    :type y1: int
    :param x2: X-coordinate of the bottom-right corner.
    :type x2: int
    :param y2: Y-coordinate of the bottom-right corner.
    :type y2: int
    :param index: The ID to display inside the bounding box.
    :type index: int
    """
    # Box corner length
    w = 10
    h = 10
    # Corner color
    color = (200, 0, 0)
    # Draw the bounding box rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + w, y1), color, 3)
    cv2.line(image, (x1, y1), (x1, y1 + h), color, 3)

    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - w, y1), color, 3)
    cv2.line(image, (x2, y1), (x2, y1 + h), color, 3)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2 - w, y2), color, 3)
    cv2.line(image, (x2, y2), (x2, y2 - h), color, 3)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1 + w, y2), color, 3)
    cv2.line(image, (x1, y2), (x1, y2 - h), color, 3)

    text = f'ID:{str(index)}'
    cv2.putText(image, text, (x1, y1 + 50),
                0, 0.5, (0, 255, 0),
                thickness=1, lineType=cv2.FILLED)

def main():
    """Main function to track humans with ByteTrack and update IDs in a pickle file."""
    # Check command line arguments
    if len(sys.argv) != 5:
        print("Usage: python trackBTPkl.py <input_pkl> <video> <output_pkl> <display: 0 No, 1 Yes>")
        sys.exit(1)

    # Parse arguments
    input_pkl = sys.argv[1]
    videoFilename = sys.argv[2]
    output_pkl = sys.argv[3]
    display = (int(sys.argv[4]) == 1)

    # Open the video
    video = cv2.VideoCapture(videoFilename)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load input pickle file using functionsCommon.loadPkl
    print("Reading pkl:", input_pkl)
    dataPKL = loadPkl(input_pkl)

    allFrameHumans = dataPKL['allFrameHumans']

    # ByteTrack parameters (these may need adjusting for your specific dataset)
    frame_rate = dataPKL['video_fps']   # or use fps from the video
    tracker = nn.BYTETracker(frame_rate)

    print("Tracking...")
    pbar = tqdm(total=total_frames, unit=' frames', dynamic_ncols=True, position=0, leave=True)
    count = 0

    try:
        while video.isOpened():
            ret, frame = video.read()
            if not ret or count >= len(allFrameHumans):
                break

            humans = allFrameHumans[count]
            # Extract bounding boxes and scores from each human
            boxes = [human['bbox'] for human in humans]
            scores = [human['score'] for human in humans]
            initialIdentities = [human['id'] for human in humans]
            classes = [0 for _ in humans]  # 0 = person class (COCO)

            # Convert (x, y, w, h) -> (x1, y1, x2, y2)
            for i in range(len(boxes)):
                x, y, w_, h_ = boxes[i]
                boxes[i] = [x, y, x + w_, y + h_]

            # Convert to numpy arrays
            boxes = np.array(boxes).astype(int)
            scores = np.array(scores)
            classes = np.array(classes).astype(float)

            # Sort by score descending
            indices = np.argsort(scores)[::-1]
            scores = scores[indices]
            boxes = boxes[indices]
            
            # If we have any bounding boxes, run ByteTrack
            if len(boxes) > 0:
                outputs = tracker.update(boxes, scores, classes)
                if len(outputs) > 0:
                    # ByteTrack output format: [x1, y1, x2, y2, track_id, score, class, index]
                    boxes_output = outputs[:, :4]
                    identities = outputs[:, 4]
                    object_classes = outputs[:, 6]
                    idx = outputs[:, 7]

                    # Draw results on frame and update IDs
                    for i, box in enumerate(boxes_output):
                        # Filter out non-person classes (only 0 is person in COCO)
                        if object_classes[i] != 0:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        tid = int(identities[i]) if identities is not None else 0
                        # Draw bounding box with ID
                        draw_line(frame, x1, y1, x2, y2, tid)
                        index = int(idx[i])
                        # The 'index' in ByteTrack output refers back to the sorted order
                        # Update the 'id' of the corresponding human in allFrameHumans
                        # We subtract 1 from tid so that IDs are consistent with some zero-based indexing
                        humans[indices[index]]['id'] = tid - 1

                if display:
                    cv2.imshow('Frame', frame)
                    # Press Q to exit early
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            count += 1
            pbar.update(1)

    except KeyboardInterrupt:
        print("\nTracking interrupted by user")
    except Exception as e:
        print(f"\nError during tracking: {e}")
    finally:
        pbar.close()
        video.release()
        if display:
            cv2.destroyAllWindows()

        # Calculate tracking statistics
        all_ids = set()
        for frame_humans in allFrameHumans:
            for human in frame_humans:
                if human['id'] >= 0:  # Only count valid IDs
                    all_ids.add(human['id'])
        
        # Save the updated pickle data using functionsCommon.savePkl
        print(f"Saving tracking results to: {output_pkl}")
        print(f"Total unique tracked IDs: {len(all_ids)}")
        savePkl(output_pkl, dataPKL)

if __name__ == "__main__":
    main()

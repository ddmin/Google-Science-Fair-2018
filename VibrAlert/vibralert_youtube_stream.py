# Import Computer Vision packages
from imutils.video import VideoStream as stream
import imutils
from PIL import ImageGrab
import cv2

# Import other helper libraries
import argparse
import time
import numpy as np

# User arguments
parser = argparse.ArgumentParser(description='Detect moving objects from webcam.')
parser.add_argument('--area', type=float, default=5000,
                    help='minimum area required for motion detection')
parser.add_argument('--frame', type=float, default=5,
                    help='how often the comparison frame should be changed')
parser.add_argument('--path', type=str, default=None,
                    help='video file path for output to be saved')

args = parser.parse_args()

# MINIMUM AREA - minimum area required for motion to be detected (in pixels)
# FRAME_RESET_RATE - the number of frames to be compared to the comparison frame before the
#                    comparison frame to be set to the current frame
# VIDEO_PATH - a file path for the recording to be stored
MINIMUM_AREA = args.area
FRAME_RESET_RATE = args.frame
VIDEO_PATH = args.path

# If the recording file path is provided, create Video Writer object
record = False
if VIDEO_PATH is not None:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VIDEO_PATH, fourcc, 20, (860, 400))
    record = True


comparisonFrame = None

frameReset = 0
print("Setting up feed.")
print("Live")
while True:
    
    frame = np.array(ImageGrab.grab(bbox=(40,300,900,700)))
    cv2.imshow("Feed", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text = "No Movement Detected"

    # Resize frame to 860 x 400
    frame = imutils.resize(frame, width=860, height=400)

    # Convert the frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Smooth the intensity of pixels across 21 x 21 pixel area
    # in order to make the algorithm
    # less sensitive to small pixel changes
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Set the comparison frame to the current frame if the comparison frame
    # has not been initialized or if the comparison frame 
    # has been compared FRAME_RESET_RATE number of times
    if comparisonFrame is None or frameReset == FRAME_RESET_RATE:
        frameReset = 0
        comparisonFrame = gray
        continue

    # Create a frame of the absolute difference between the 
    # comparison frame in the pixel values to see what moved
    delta = cv2.absdiff(comparisonFrame, gray)

    # Ignore the pixel if it has a pixel value of less than 25
    ret, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)

    # dilate the threshold frame in order to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # contour detection to find the edges of the threshold frame
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    direction = 0
    for c in cnts:
        # If contour is smaller than minimum area required to detect
        # move on to the next contour
        if cv2.contourArea(c) < MINIMUM_AREA:
            continue
        
        # x - x coordinate of object
        # y - y coordinate of object
        # w - approximate width of object
        # h - approximate height of object
        x, y, w, h = cv2.boundingRect(c)

        # draw a rectangle around the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = "Movement Detected"

        # Calculate which side the movement is in
        # Right = +, Left = -
        # x ranges from 0 to 500
        
        if (2*x + w)/2 <= 250:
            direction += 1
        else:
            direction -= 1

    cv2.putText(frame, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Check the position of the boxes
    # print out results
    dirtext = ""
    coordinate = (10, 290)
    if text == "Movement Detected":
        if direction > 0:
            dirtext = "Right"
            coordinate = (10, 100)
        elif direction < 0:
            dirtext = "Left"
            coordinate = (420, 100)
        else:
            dirtext = "Center"
            coordinate = (200, 200)

    cv2.putText(frame, dirtext, coordinate,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Writing to video at file path
    if record is True:
        out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    cv2.imshow("VibrAlert v0.6", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    frameReset += 1
    key = cv2.waitKey(1) & 0xFF

    print(dirtext)
        
    # Exit after user presses "Esc"
    if key == 27:
        break
    
cv2.destroyAllWindows()

if record is True:
    out.release()

print('End Feed')

# TODO: Finish commenting code
#       Think about what to do with y, h values
#       Depth perception?
#       callibrate size with camera lens field of view
#       Get rid of boxes too small? too large?

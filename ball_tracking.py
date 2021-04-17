from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time

# empty func
def nothing(x):
    pass

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())

# define the default trackbar to find the right color range
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 91, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 82, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

camera = cv2.VideoCapture(0)

pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

while True:
    # adding trackbar to find the right color range
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
	
    # assign values from trackbar to blueLower/ blueUpper
    blueLower = (l_h, l_s, l_v)
    blueUpper = (u_h, u_s, u_v)

    (grabbed, frame) = camera.read()

    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        #center = (int(x),int(y))
        radius = int(radius)
        M = cv2.moments(cnt)
        center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
        # draw circle around subject OI
        if radius > 10:
            cv2.circle(frame, (int(x),int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)
    # loop over the set of tracked points
    for i in np.arange(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i-1] is None or pts[i] is None:
            continue
        # check to see if enough points have been accumulated in the buffer
        if counter >= 10 and i == 1 and pts[-1] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables 
            dX = pts[-1][0] - pts[i][0]
            dY = pts[-1][1] - pts[i][1]
            (dirX, dirY) = ("", "")
            # ensure there is significant movement in the x-direction
            if np.abs(dX) > 20:
                dirX = "East" if np.sign(dX) == 1 else "West"
            # ensure there is significant movement in the y-direction
            if np.abs(dY) > 20:
                dirY = "North" if np.sign(dY) == 1 else "South"
            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)
            # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)
    # show the movement deltas and the direction of movement on the frame
    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)
    # show the frame to our screen and increment the frame counter
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1
	# if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    # as output here there will be only the subject OI in color
    result = cv2.bitwise_and(frame, frame, mask=mask)

    #cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

camera.release()
cv2.destroyAllWindows()


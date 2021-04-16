from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# empty func
def nothing(x):
    pass

# define the default trackbar to find the right color range
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 91, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 164, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 135, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

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

    if args.get("video") and not grabbed:
        print("something went wrong!")
        break

    frame = imutils.resize(frame, width=600)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)

        # draw circle arout subject OI
        if radius > 5:
            cv2.circle(frame, center, radius, (0, 255, 255), 2)

    # as output here there will be only the subject OIin color
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()


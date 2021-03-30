import numpy as np
import cv2
import time

# read the webcam
cap = cv2.VideoCapture(0)
time.sleep(2.0)

# initialize the background frame to empty
background = None

# read frames
while True:
	(grabbed, frame) = cap.read()
	
	if not grabbed:
		print("Couldn't read the camera.")
		break
	# convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# apply smoothing to remove noise
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	
	# initialize  the background with the first frame
	if background is None:
		background = gray
		continue
	
	# calculate the difference between the background and the current frame
	difference = cv2.absdiff(background, gray)

	# apply threshold
	threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate threshold image to fill in holes
	threshold = cv2.dilate(threshold, None, iterations = 2)

	# find contours on the threshold image
	im, contours, hierarchy = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# loop over contours
	for c in contours:
		# remove the smallest contour
		if cv2.contourArea(c) < 500:
			continue
		
		# calculate the bounds of the contour and draw it on the frame
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imshow("Camera", frame)
		cv2.imshow("Thresh", threshold)
		cv2.imshow("Diff", difference)

		key = cv2.waitKey(1) & 0xFF

		time.sleep(0.015)
		if key == ord('q'):
			break	
		
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

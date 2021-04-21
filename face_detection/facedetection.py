###
#  The code written below mostly follows the steps from the following datacamp tutorial: https://www.datacamp.com/community/tutorials/face-detection-python-opencv
###


import numpy as np
import cv2 
import matplotlib.pyplot as plt


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
def detect_faces(cascade, test_image, scaleFactor = 1.1):
    image_copy = test_image.copy()
    
    #convert the test image to gray scale as opencv expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    
    #applying the classifier 
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
    
    #drawing the rectangle around the face(s)
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
        
    return image_copy
	
#loading the image
test_image2 = cv2.imread('test_image.jpg')

#loading the classifier
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#calling the function to detect faces
faces = detect_faces(haar_cascade_face, test_image2)

#convert to RGB and display image
plt.imshow(convertToRGB(faces))

#saving the new picture
cv2.imwrite('faces_detected.png', faces)
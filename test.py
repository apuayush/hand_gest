import cv2
import numpy as np
import matplotlib.pyplot as plt

hand_cascade = cv2.CascadeClassifier('hand_haarcascade.xml')
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hand = hand_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in hand:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0),2)
    
    cv2.imshow('hand',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

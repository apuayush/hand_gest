from training.module import *
from training.datahandler import *

from Network import Network

if __name__ == "__main__":

    camera = cv2.VideoCapture(0)


    while True:
        ret_value, frame = camera.read()
        frame = imutils.resize(frame, width=700)

        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        (height, width) = frame.shape[:2]

        # for right now
        roi = frame

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
        fingers = count_fingers(thresholded, segmented)
        cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Thesholded", thresholded)
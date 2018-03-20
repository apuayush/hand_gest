from utils import detector_utils as detector_utils
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime

img = cv2.imread("test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detection_graph, sess = detector_utils.load_inference_graph()
cam = cv2.VideoCapture(0)
c=0
while True:
    c += 1
    _, frame = cam.read()
    if c%3 != 0:
        continue

    fram_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, scores = detector_utils.detect_objects(fram_rgb, detection_graph, sess)

    detector_utils.draw_box_on_image(
        5, 0.3, scores, boxes, 640, 380, fram_rgb
    )


    print(boxes, scores)
    cv2.imshow("Hand detection", cv2.cvtColor(fram_rgb, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

from utils import detector_utils as detector_utils
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime

img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detection_graph, sess = detector_utils.load_inference_graph()

boxes, scores = detector_utils.detect_objects(img, detection_graph, sess)

detector_utils.draw_box_on_image(
    5, 0.3, scores, boxes, 640, 380, img
)


print(boxes, scores)
cv2.imshow("Hand detection", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if cv2.waitKey(125) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

options = {
    'model': '/home/apurvnit/Projects/darkflow/cfg/tiny-yolo-voc-1c.cfg',
    'load': 9625,
    'threshold': 0.1,
    'gpu': 0.8
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    if ret:
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 5)
            frame = cv2.putText(
                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))

    keypress = cv2.waitKey(1) & 0xFF

    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
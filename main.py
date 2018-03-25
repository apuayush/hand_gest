import utils.detector_utils as detector_utils
import cv2
from predict_gesture import *
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
import datetime

frame_processed = 0

def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        print("> ===== in worker loop, frame ", frame_processed)
        frame = input_q.get()
        if frame is not None:
            # actual detection
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # draw bounding boxes
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"], scores, boxes, cap_params['width']
                , cap_params['height'], frame)
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


if __name__ == "__main__":

    cap_params = {'width': 640, 'height': 380}
    frame_processed = 0

    # Video configs
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, cap_params['width'])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_params['height'])

    input_q = Queue(maxsize=10)
    output_q = Queue(maxsize=10)

    # cap_params['im_width'], cap_params['im_height'] = cam.get()
    cap_params['score_thresh'] = 0.3

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = 2
    c = 0

    # num_workers = 5

    # To parallelize detection
    pool = Pool(
        4, worker, (input_q, output_q, cap_params, frame_processed)
    )

    while True:
        c += 1
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        fram_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_q.put(fram_rgb)
        output_frame = output_q.get()

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("hand tracking", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pool.terminate()
    cam.release()
    cv2.destroyAllWindows()

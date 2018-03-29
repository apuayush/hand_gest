import utils.detector_utils as detector_utils
import cv2
from keras.models import model_from_json
import numpy as np
from predict_gesture import GestureClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing

from multiprocessing import Queue, Pool
import time
import datetime

frame_processed = 0

gesture = GestureClassifier()

print("comeback")
called = False
c = 0
ges, score = "None", 0.0


def worker(input_q, output_q, cap_params, frame_processed):
    global called, c, ges, score
    if not called:
        gesture.load_model()
        called =True

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
            details = {
                'boxes': boxes,
                'scores': scores
            }
            # draw bounding boxes
            cropped_image = detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"], scores, boxes, cap_params['width']
                , cap_params['height'], frame)

            if cropped_image is not None and c == 0:
                cropped_image = cv2.flip(cropped_image, 1)
                ges, score = gesture.predict(cropped_image/255)
            print(ges, score)
            details['frame'] = frame
            details['cropped_image'] = cropped_image
            details['ges'] = ges
            details['score'] = score

            output_q.put(details)
            frame_processed += 1
        else:   
            output_q.put({
                'boxes': [],
                'frame': frame,
                'ges': ges,
                'score': score
            })
    c = (c+1) % 10
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
    cap_params['num_hands_detect'] = 1

    # num_workers = 5

    # To parallelize detection
    pool = Pool(
        4, worker, (input_q, output_q, cap_params, frame_processed)
    )

    ges, score = 'None', 0.0

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        fram_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_q.put(fram_rgb)
        output_details = output_q.get()

        output_frame = cv2.cvtColor(output_details['frame'], cv2.COLOR_RGB2BGR)

        try:
            print(output_details['ges'],output_details['score'])
            cv2.imshow("cropped image", output_details['cropped_image'])
        except:
            pass
        print(c)
        c = (c+1)%10

        # cv2.imwrite("test/img"+str(c)+".jpg", output_frame)
        # print(str(c)+'\t'+str(output_details['boxes']))

        cv2.putText(output_frame, output_details['ges'] + " - " + str(output_details['score']), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("hand tracking", output_frame)

        c = (c + 1) % 10
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pool.terminate()
    cam.release()
    cv2.destroyAllWindows()

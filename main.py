import utils.detector_utils as detector_utils
import cv2
from keras.models import model_from_json
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing

from multiprocessing import Queue, Pool
import time
import datetime

frame_processed = 0

# gesture = GestureClassifier()

print("comeback")
called = False

def preprocess(img):
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    # print(img.shape)
    return img

def worker(input_q, output_q, cap_params, frame_processed):

    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    # gesture.load_model()
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

            details['frame'] = frame
            details['cropped_image'] = cropped_image

            # gesture.predict()

            output_q.put(details)
            frame_processed += 1
        else:   
            output_q.put({
                'boxes': [],
                'scores': 0.0,
                'frame': frame
            })
    sess.close()


if __name__ == "__main__":
    json_file = open('training/model.json', 'r')
    loaded_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('training/hand_detection_weights.h5')
    print("completed loading model")


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

    ges, score = 'None', 0.0

    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        fram_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_q.put(fram_rgb)
        output_details = output_q.get()

        output_frame = cv2.cvtColor(output_details['frame'], cv2.COLOR_RGB2BGR)

        if c == 0 and output_details['cropped_image'] is not None:
            cropped = preprocess(output_details['cropped_image'])
            k = model.predict(np.array([cropped]))
            score = max(k[0])
            ges = chr(65 + list(k[0]).index(score))

            # ges, score = output_details['gesture'].predict(output_details['cropped_image'])

        try:
            print(ges,score)
            cv2.imshow("cropped image", output_details['cropped_image'])
        except:
            pass
        print(c)
        c = (c+1)%10

        # cv2.imwrite("test/img"+str(c)+".jpg", output_frame)
        # print(str(c)+'\t'+str(output_details['boxes']))

        cv2.putText(output_frame, str(ges) + " - " + str(score), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("hand tracking", output_frame)

        c = (c + 1) % 10
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pool.terminate()
    cam.release()
    cv2.destroyAllWindows()

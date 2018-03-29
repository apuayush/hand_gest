from keras.models import model_from_json
import cv2
import numpy as np


def preprocess(img):
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    # print(img.shape)
    return img

class GestureClassifier:
    def __init__(self):
        self.model = None

    def load_model(self):
        json_file = open('training/model.json', 'r')
        loaded_model = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.load_weights('training/hand_detection_weights.h5')
        print("completed loading model")

    def predict(self, img):
        # print(img.shape)
        img1 = img.copy()
        img1 = preprocess(img1)
        k = self.model.predict(np.array([img1]))
        score = max(k[0])
        res = list(k[0]).index(score)

        return chr(65 + res), score

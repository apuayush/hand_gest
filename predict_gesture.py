from keras.models import model_from_json
import cv2
import numpy as np

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

    def predict(self, img, annotations):
        pass

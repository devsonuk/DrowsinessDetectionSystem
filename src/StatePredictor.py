import cv2
import numpy as np
from tensorflow import keras


class StatePredictor:
    def __init__(self, eye_model, yawn_model, img, face_landmarks):
        self.model = eye_model
        self.yawn_model = yawn_model
        self.img = img
        self.face_landmarks = face_landmarks
        self.h = face_landmarks[27][1] - face_landmarks[23][1]
        self.left_eye_start = face_landmarks[130]
        self.left_eye_end = face_landmarks[133]
        self.right_eye_start = face_landmarks[362]
        self.right_eye_end = face_landmarks[359]
        self.mouth_openness = 0.0
        self.eye_openness = 0.0
        self.consciousness = 0.0

    def predict_eye_state(self, image):
        image = cv2.resize(image, (20, 10))
        image = image.astype(dtype=np.float32)
        image_batch = np.reshape(image, (1, 10, 20, 1))
        image_batch = keras.applications.mobilenet.preprocess_input(image_batch)
        pred = self.model.predict(image_batch)[0][1]
        # print(pred[0][0]*100)
        # return np.argmax(self.model.predict(image_batch)[0])
        return pred

    def get_left_eye_state(self, draw=True):
        x1, y1 = self.left_eye_start
        x2, y2 = self.left_eye_end
        y_min = y2 - self.h
        y_max = y1 + self.h
        left_eye = self.img[y_max:y_min, x1:x2]
        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Eye", left_eye)
        pred = self.predict_eye_state(image=left_eye)
        if draw:
            cv2.rectangle(self.img, (x1, y_max), (x2, y_min), (0, 255, 0), 2)
        return pred

    def get_right_eye_state(self, draw=True):
        x1, y1 = self.right_eye_start
        x2, y2 = self.right_eye_end
        y_min = y2 - self.h
        y_max = y1 + self.h
        right_eye = self.img[y_max:y_min, x1:x2]
        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        pred = self.predict_eye_state(image=right_eye)
        if draw:
            cv2.rectangle(self.img, (x1, y_max), (x2, y_min), (0, 255, 0), 2)
        return pred

    def get_eye_openness(self):
        # self.eye_openness = (self.get_right_eye_state(draw=False) + self.get_right_eye_state(draw=False)) / 2
        self.eye_openness = self.get_left_eye_state(draw=False)
        return round(self.eye_openness, 2)

    def get_mouth_sate(self):
        lips = (self.face_landmarks[13][1] - self.face_landmarks[0][1]) + (
                self.face_landmarks[17][1] - self.face_landmarks[14][1])
        mouth_height = self.face_landmarks[17][1] - self.face_landmarks[0][1]
        # print(mouth_height, lips)
        self.mouth_openness = (mouth_height - lips) / 45
        return round(self.mouth_openness, 2)

    def get_consciousness(self):
        self.consciousness
        try:
            # self.consciousness = (0.6 * self.eye_openness) * (0.4 * (1 / self.mouth_openness))
            self.consciousness = self.eye_openness*(1-self.mouth_openness)
        except:
            self.consciousness = round(self.eye_openness, 2)
        finally:
            self.consciousness
            return round(self.consciousness, 2)

    def get_mouth_sates(self, draw=True):
        (x, y, w, h) = cv2.boundingRect(np.array([self.face_landmarks]))
        # print('h', x, y, w, h)
        img_size = (160, 160, 3)
        image = self.img[y:y + h, x:x + w]
        image = cv2.resize(image, img_size[:2])
        image = image.astype(dtype=np.float32)
        # image_batch = np.reshape(image, (1, 224, 224, 3))
        image_batch = np.expand_dims(image, axis=0)
        # image_batch = keras.applications.mobilenet.preprocess_input(image_batch)
        # print('Mouth')
        pred = self.yawn_model.predict_classes(image_batch)[0]
        if pred == 1:
            self.mouth_openness = 'Open'
        else:
            self.mouth_openness = 'Close'
        # self.mouth_state = 'Open' if self.yawn_model.predict_classes(image_batch)[0] else 'Close'
        # print(pred)
        return self.mouth_openness

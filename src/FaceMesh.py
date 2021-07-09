import time
import cv2
import imutils
import mediapipe as mp
from keras.models import load_model

from src.StatePredictor import StatePredictor


class FaceMeshDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces,
                                                 self.min_detection_confidence, self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(img_rgb)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec,
                                               self.drawSpec)
                face = []
                for lm in faceLms.landmark:
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def test():
    print('Loading model...')
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = FaceMeshDetector()
    eye_model = load_model("../models/eye_state_detector.hdf5")
    # yawn_model = load_model("../models/yawn_detector.h5")
    print('Loading model done')
    while True:
        print('Hi')
        success, img = cap.read()
        img = imutils.resize(img, width=800, height=680)
        img = cv2.flip(img, 1)
        img, faces = detector.findFaceMesh(img, draw=False)

        if len(faces) != 0:
            print(len(faces))
            for face_landmarks in faces:
                predictor = StatePredictor(eye_model, 'yawn_model', img, face_landmarks)
                left_eye_state = predictor.get_left_eye_state()
                cv2.putText(img, f'Left Eye : {left_eye_state}', (10, 100), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
                right_eye_state = predictor.get_right_eye_state()
                cv2.putText(img, f'Right Eye : {right_eye_state}', (10, 125), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == "__main__":
    test()

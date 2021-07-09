import datetime
import time
from enum import Enum
import cv2
import imutils
import mediapipe as mp
import numpy as np
from keras.models import load_model
from StatePredictor import StatePredictor
from src.User import User



class Environment(Enum):
    Development = 1
    Production = 2
    Test = 3


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


def fancyDraw(img, bbox, l=30, t=5, rt=1):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    cv2.rectangle(img, bbox, (255, 0, 255), rt)
    # Top Left  x,y
    cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
    cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
    # Top Right  x1,y
    cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
    cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
    # Bottom Left  x,y1
    cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
    cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
    # Bottom Right  x1,y1
    cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
    cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
    return img


def draw_triangle(img, bbox, thickness=5):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    c = int(x + (w // 2))
    pts = np.array([[x, y1], [c, y], [x1, y1]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 0, 255), thickness, cv2.LINE_4)
    print(w)
    print_text(img, 'ALERT', '', (int(x * 1.1), int(0.9 * y1)), color=(0, 0, 255), thickness=2, rd=False,
               fontScale=(3 - 224 / w))
    return img


def print_text(img, label, value, pos,
               font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
               fontScale=1,
               color=(15, 5, 5),
               thickness=1,
               rd=True,
               ):
    if rd:
        value = '{:.2f}'.format(value)
    cv2.putText(img, f'{label}{value}', pos, font, fontScale, color, thickness, cv2.LINE_4)


def get_predicted_state(b_rate, y_rate):
    return b_rate >= 20 or y_rate >= 10


def main():
    mode = Environment.Development
    user = User(time.time(), 0, 0)
    print('Environment:', mode.name)
    print('Loading models...')
    detector = FaceMeshDetector()
    eye_model = load_model("../models/eye_state_detector.hdf5")
    # yawn_model = load_model("../models/yawn_detector.h5")
    print('Model loaded successfully.')
    print('Warming video sensors...')
    cap = cv2.VideoCapture(0)
    print('Live detection starting...')
    p_time = 0
    draw = True
    while True:
        success, img = cap.read()
        img = imutils.resize(img, width=1080, height=720)
        img = cv2.flip(img, 1)
        frame, faces = detector.findFaceMesh(img.copy(), draw=True)
        if mode == Environment.Development or mode == Environment.Test:
            cv2.imshow("Model View", frame)
        try:
            if len(faces) == 0:
                print_text(img, "No Face Detected", '', (500, 50), color=(0, 0, 255), rd=False, thickness=2)
                continue
            else:
                for face_landmarks in faces:
                    predictor = StatePredictor(eye_model, 'yawn_model', img, face_landmarks)
                    print_text(img, "Login At: ", time.ctime(user.login_time), (10, 85), rd=False)
                    print_text(img, "Driving Duration: ", datetime.timedelta(seconds=int(user.get_total_duration())),
                               (10, 120), rd=False)
                    print_text(img, "Eye Openness: ", predictor.get_eye_openness(), (10, 155))
                    print_text(img, "Mouth Openness: ", predictor.get_mouth_sate(), (10, 190))
                    print_text(img, "Conciseness Level: ", predictor.get_consciousness(), (10, 225))
                    if predictor.eye_openness < 0.50:
                        user.blink_count += 1
                        print_text(img, "EYES CLOSED", '', (900, 75), color=(0, 0, 255), rd=False, thickness=2)
                    if predictor.mouth_openness > 0.35:
                        user.yawn_count += 1
                        print_text(img, "YAWNING DETECTED", '', (825, 125), color=(0, 0, 255), rd=False, thickness=2)
                    bbox = cv2.cv2.boundingRect(np.array([face_landmarks]))
                    if(draw):
                        img = fancyDraw(img, bbox)

                    try:
                        # b_rate = user.get_blink_rate() // int(fps)
                        # y_rate = user.get_yawn_rate() // int(fps)
                        b_rate = user.get_blink_rate() // 10
                        y_rate = user.get_yawn_rate() // 10
                    except:
                        b_rate = user.get_blink_rate()
                        y_rate = user.get_yawn_rate()
                    finally:
                        print_text(img, "Blink Rate(blink/min): ", b_rate, (10, 260))
                        print_text(img, "Yawn Rate(yawn/min): ", y_rate, (10, 295))
                        print_text(img, "Gender: ", 'Male', (10, 330), rd=False)
                        print_text(img, "Age: ", '22-25 years', (10, 365), rd=False )

                        if get_predicted_state(b_rate, y_rate):
                            draw = False
                            img = draw_triangle(img, bbox)
                            print_text(img, "DROWSINESS DETECTED", '', (800, 175), color=(0, 0, 255), rd=False,
                                       thickness=2)
                    print('Live detecting')
        except:
            print_text(img, "No Face Detected", '', (500, 50), color=(0, 0, 255), rd=False, thickness=2)
        finally:
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            print_text(img, 'Frame Rate(per/sec): ', int(fps), (10, 50))
            cv2.imshow("Drowsiness Detection System", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == "__main__":
    main()

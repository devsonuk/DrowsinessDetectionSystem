import cv2
import mediapipe as mp
import time

import numpy as np


class FaceDetector:
    def __init__(self, minDetectionConfidence=0.5):
        self.minDetectionConfidence = minDetectionConfidence
        self.mpFaceDetector = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetector.FaceDetection(self.minDetectionConfidence)

    def findFaces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(img_rgb)
        bboxes = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                # print(detection.location_data.relative_bounding_box)
                bbox_class = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bbox_class.xmin * iw), int(bbox_class.ymin * ih), \
                       int(bbox_class.width * iw), int(bbox_class.height * ih)
                bboxes.append([bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20),
                                cv2.FONT_ITALIC, 2,
                                (255, 0, 255), 2)
        return img, bboxes

    @staticmethod
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


def draw_triangle(img, thickness=10):
    pts = np.array([[400, 250], [540, 50], [680, 250]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 0, 255), thickness, cv2.LINE_4)
    return img


def main():
    # cap = cv2.VideoCapture("../videos/video1.mp4")
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img)
        img = draw_triangle(img)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()



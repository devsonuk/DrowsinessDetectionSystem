import matplotlib.pyplot as plt
import os
import cv2

from FaceDetection import FaceDetector


class DataPreProcessor:
    def __init__(self, src, dest, img_size=224):
        self.detector = FaceDetector()
        self.src = src
        self.dest = dest
        self.img_size = img_size

    def process(self):
        categories = ["yawn", "no_yawn"]
        print("Processing...")
        for category in categories:
            path_link = os.path.join(self.src, category)
            os.mkdir(os.path.join(self.dest, category))
            tag = 1
            for image in os.listdir(path_link):
                img = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
                img, bboxes = self.detector.findFaces(img, draw=False)
                for face in bboxes:
                    # print(face[0])
                    (x, y, w, h) = face[0]
                    roi_color = img[y:y + h, x:x + w]
                    resized_array = cv2.resize(roi_color, (self.img_size, self.img_size))
                    resized_array = cv2.cvtColor(resized_array, cv2.COLOR_BGR2RGB)
                    plt.imsave(f'{os.path.join(self.dest, category, str(tag))}.jpeg', resized_array)
                    tag += 1
        print("Process successfully completed")


def main():
    os.mkdir("../data/yawn_data")
    dpp = DataPreProcessor("../data/dataset1", "../data/yawn_data")
    dpp.process()


if __name__ == "__main__":
    main()

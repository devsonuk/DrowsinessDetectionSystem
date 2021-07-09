from tkinter import *
import imutils
from PIL import Image, ImageTk
from ProcessImages import *

# GUI Section
root = Tk()
root.title("Drowsiness Detection System")
root.config(background="#FFFFFF")


# Navbar Section
navbar_frame = Frame(root, bg='black', width=900, height=200)
navbar_frame.grid(row=0, column=0)
logo = Label(navbar_frame, text='Guard')
logo.grid(row=0, column=0)

login_btn = Button(navbar_frame, text='Login')
login_btn.grid(row=0, column=1, padx=10, pady=10)





# Graphics window
img_frame = LabelFrame(root, text="Live Monitor", width=900, height=600)
img_frame.grid(row=1, column=0, padx=10, pady=10)



# Image viewer
my_img = ImageTk.PhotoImage(Image.open("../assets/img1.jpg"))
img_label = Label(img_frame, image=my_img)
img_label.pack()
# cap = cv2.VideoCapture(0)


def capture_video(detector, eye_model, user, mode):
    # Cv2 Section
    success, frame = cap.read()
    frame = imutils.resize(frame, width=900, height=600)
    frame = cv2.flip(frame, 1)
    cv2image = process(frame, detector, eye_model, user, mode)
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)

    # Tkinter Section
    img = Image.fromarray(cv2image)
    img_tk = ImageTk.PhotoImage(image=img)
    img_label.image = img_tk
    img_label.configure(image=img_tk)
    img_label.after(10, lambda: capture_video(detector, eye_model, user, mode))


def initiate_process():
    print('Loading models...')
    detector = FaceMeshDetector()
    eye_model = load_model("../models/eye_state_detector.hdf5")
    # yawn_model = load_model("../models/yawn_detector.h5")
    print('Models loaded successfully...')
    mode = Environment.Production
    user = User(time.time(), 0, 0)
    print('Warming camera sensor...')
    capture_video(detector, eye_model, user, mode)
    print('Starting live detection...')


def main():
    # initiate_process()
    root.mainloop()


if __name__ == "__main__":
    main()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from tkinter import messagebox
import numpy as np
from camera_utils import create_video_capture, read_frame


detector = MTCNN()

emotion_model = "./data/_mini_XCEPTION.106-0.65.hdf5"
ageProto="./data/age_deploy.prototxt"
ageModel="./data/age_net.caffemodel"
genderProto="./data/gender_deploy.prototxt"
genderModel="./data/gender_net.caffemodel"

def get_camera():
    cap = create_video_capture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open webcam or fallback video.")
        return None
    return cap

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
Emotions = ["angry","disgust","scared","happy","sad","surprised","neutral"]

face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model,compile=False)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


def ageAndgender():
    cap = get_camera()
    if cap is None:
        return

    while True:
        ret, img = read_frame(cap)
        if not ret or img is None:
            messagebox.showerror("Camera Error", "Failed to read from webcam or fallback video.")
            break
        default_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = face_cascade.detectMultiScale(image=default_img, scaleFactor=1.3, minNeighbors=5)
        for x, y, w, h in face:
            roi = default_img[y:y + h, x:x + w]
            blob = cv2.dnn.blobFromImage(roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f"{gender}, {age} year", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

        cv2.imshow("Gender and Age Prediction", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def emotion():
    cap = get_camera()
    if cap is None:
        return

    while True:
        ret, img = read_frame(cap)
        if not ret or img is None:
            messagebox.showerror("Camera Error", "Failed to read from webcam or fallback video.")
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)

        for x, y, w, h in face:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = Emotions[preds.argmax()]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))

        cv2.imshow("Gender and Age Prediction", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

 

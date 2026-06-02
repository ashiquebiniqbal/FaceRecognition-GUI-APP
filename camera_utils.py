import cv2
import os

FALLBACK_VIDEO = os.path.join(os.path.dirname(__file__), "data", "WIN_20230920_07_56_11_Pro.mp4")


def create_video_capture(source=0):
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap

    if os.path.exists(FALLBACK_VIDEO):
        fallback = cv2.VideoCapture(FALLBACK_VIDEO)
        if fallback.isOpened():
            return fallback

    return cap


def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        # try looping a fallback video
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if total > 0 and pos >= total:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        except Exception:
            pass
    return ret, frame

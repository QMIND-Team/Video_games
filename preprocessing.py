from cv2 import COLOR_RGB2GRAY, cvtColor, resize


def preprocess(frame):
    frame = cvtColor(frame, COLOR_RGB2GRAY)
    frame = resize(frame, (100, 128))
    return frame

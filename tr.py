import cv2

class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, img = self.video.read()
        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

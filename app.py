from flask import Flask, render_template, Response
from tr import Video
import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 50

folderPath = "Resources/HEADER"
myList = os.listdir(folderPath)
myList.sort()
print(myList)
overlayList = []

# for imPath in myList:
#     image = cv2.imread(f'{folderPath}/{imPath}')
#     overlayList.append(image)
# print(len(overlayList))
#
# header = overlayList[0]
# drawColor = (171, 71, 0)

# video = cv2.VideoCapture(0)
# video.set(3, 1280)
# video.set(4, 720)
#
# detector = htm.handDetector(detectionCon=0.85)
# xp, yp = 0, 0
# imgCanvas = np.zeros((720, 1280, 3), np.uint8)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/index')
def get_ses():
    return render_template('index.html')
# def gen(tr):
#     while True:
#         frame = tr.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame +
#                b'\r\n\r\n')
def xyz():
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)
    print(len(overlayList))

    header = overlayList[0]
    drawColor = (171, 71, 0)
    video = cv2.VideoCapture(0)
    video.set(3, 1280)
    video.set(4, 720)

    detector = htm.handDetector(detectionCon=0.85)
    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    while True:
        # 1. Import image
        ret, img = video.read()
        img = cv2.flip(img, 1)

        # 2. Find Hand Landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # print(lmList)

            # tip of index and middle finger
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # 3. Check which fingers are up

            fingers = detector.fingersUp()
            # print(fingers)

            # 4. If selection mode - Two fingers are up

            if fingers[1] and fingers[2]:
                xp, yp = 0, 0

                print("Selection Mode")
                # checking for the click
                if y1 < 125:
                    if 264 < x1 < 424:
                        header = overlayList[0]
                        drawColor = (171, 71, 0)
                    elif 424 < x1 < 574:
                        header = overlayList[1]
                        drawColor = (180, 105, 255)
                    elif 574 < x1 < 730:
                        header = overlayList[2]
                        drawColor = (51, 51, 255)
                    elif 730 < x1 < 884:
                        header = overlayList[3]
                        drawColor = (0, 102, 51)
                    elif 884 < x1 < 1050:
                        header = overlayList[4]
                        drawColor = (0, 128, 255)
                    elif 1050 < x1 < 1250:
                        header = overlayList[5]
                        drawColor = (0, 0, 0)

                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            # 5. If drawing mode - Index finger is up

            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                print("Drawing Mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        print(img.shape, imgInv.shape, imgCanvas.shape)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # setting the header image
        img[0:125, 0:1280] = header
        # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
        cv2.imshow("Image", img)
        # cv2.imshow("Canvas", imgCanvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        _, buffer = cv2.imencode('.jpg',img)
        frame2= buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')



@app.route('/video')
def video():
    return Response(xyz(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)

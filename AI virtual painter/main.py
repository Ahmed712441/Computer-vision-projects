import cv2
from mediapipe_modules import Handtracking
import time
import os
import numpy as np

thickness = [6,50]
t_index = 0
board = np.zeros((720,1200,3),dtype=np.uint8)
cap = cv2.VideoCapture(0)
cap.set(3,1300)
cap.set(4,1000)
hands = Handtracking(max_hands=1,min_detection_confidence=0.65)
ctime = 0
ptime = 0
photos = os.listdir("photos")
index = 0
color = (0,0,255)
old_point_x,old_point_y = 0,0
cx,cy= 0,0
first_frame = True

while True:
    success , img = cap.read()
    img = cv2.flip(img,1)
    img[0:100,0:1200] = cv2.imread(os.path.join(os.getcwd(),"photos",photos[index]))
    hands.findhands(img,draw=True)
    hands.fingersup()
    if(hands.up[1] == 1 and hands.up[2] == 1):
        first_frame = True
        _ ,cx , cy = hands.lmlist[8]
        cv2.circle(img,(cx,cy),6,color,-1)
        if(cy<200):
            if(0<cx<200):
                index = 0
                t_index = 0
                color = (0,0,255)
            elif(200<cx<400):
                index = 1
                t_index = 0
                color = (255,0,0)
            elif(400<cx<600):
                index = 2
                t_index = 0
                color = (0,255,0)
            elif(600<cx<800):
                index = 3
                t_index = 0
                color = (255,0,255)
            else:
                index = 4
                t_index = 1
                color = (0,0,0)
    elif(hands.up[1] == 1 and hands.up[2] == 0 ):
        _ ,cx , cy = hands.lmlist[8]
        if first_frame:
            old_point_x,old_point_y = cx,cy
            first_frame = False
        cv2.circle(img,(cx,cy),6,color,-1)
        cv2.line(board,(int(old_point_x),int(old_point_y)),(int(cx),int(cy)),color,thickness[t_index])
        old_point_x,old_point_y = cx,cy
    gray_img = cv2.cvtColor(board,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY_INV)
    img2 = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img[0:720,0:1200],img2)
    img = cv2.bitwise_or(img,board)
    ctime = time.time()
    fps=1/(ctime-ptime)
    cv2.putText(img,str(int(fps)),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),1,cv2.LINE_AA)
    ptime = ctime
    cv2.imshow('board',board)
    cv2.imshow('my cam',img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

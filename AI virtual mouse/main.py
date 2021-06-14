import cv2
from mediapipe_modules import Handtracking
import time
import os
import numpy as np
import autopy
import math

cap = cv2.VideoCapture(0)
hands = Handtracking(max_hands=1,min_detection_confidence=0.85)
padding = 60
smoothen = 5
ctime = 0
ptime = 0
old_point_x,old_point_y = 0,0
cx,cy= 0,0
screen_width , screen_height = autopy.screen.size()
first_frame = True
screen_width , screen_height = int(screen_width) , int(screen_height)

while True:
    success , img = cap.read()
    img = cv2.flip(img,1)
    cv2.rectangle(img,(padding,padding),(img.shape[1]-padding,img.shape[0]-padding),(0,255,90),3)
    hands.findhands(img,draw=False)
    hands.fingersup()
    if(len( hands.lmlist)!=0):
        _ ,cx , cy = hands.lmlist[8]
        if(padding<cx<img.shape[1]-padding and padding<cy<img.shape[0]-padding):
            if(hands.up[1] == 1 and hands.up[2] == 1):
                _ ,px , py = hands.lmlist[12]
                cv2.line(img,(cx,cy),(px,py),(0,255,0),5)
                cv2.circle(img,(cx,cy),10,(255,0,0),-1)
                cv2.circle(img,(px,py),10,(255,0,0),-1)
                length = math.hypot(cx-px,cy-py)
                if(length < 20):
                    autopy.mouse.click()
            elif(hands.up[1] == 1 and hands.up[2] == 0 ):
                cv2.circle(img,(cx,cy),6,(255,0,0),-1)
                mousex =np.interp(cx,[padding,img.shape[1]-padding],[0,screen_width])
                mousey =np.interp(cy,[padding,img.shape[0]-padding],[0,screen_height])
                mousex = old_point_x + (mousex-old_point_x)/smoothen
                mousey = old_point_y + (mousey-old_point_y)/smoothen
                autopy.mouse.move(mousex, mousey)
                old_point_x,old_point_y = mousex, mousey

    ctime = time.time()
    fps=1/(ctime-ptime)
    cv2.putText(img,str(int(fps)),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),1,cv2.LINE_AA)
    ptime = ctime
    cv2.imshow('my cam',img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

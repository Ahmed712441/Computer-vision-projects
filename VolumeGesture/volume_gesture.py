import cv2
from mediapipe_modules import Handtracking
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
min_vol,max_vol, _ = volume.GetVolumeRange()
min_length , max_length = 0 , 120
# volume.SetMasterVolumeLevel(-20.0, None)
cap = cv2.VideoCapture(0)
hands = Handtracking(max_hands=1)
ctime = 0
ptime = 0
length = 0
old_length = 0
smoothen = 5

while True:
    success , img = cap.read()
    img = cv2.flip(img,1)
    hands.findhands(img,draw=True)
    hands.fingersup()
    if(hands.up[0] == 1 and hands.up[1] == 1 and hands.up[2] == 0):
        x1,y1,x2,y2=hands.lmlist[4][1],hands.lmlist[4][2],hands.lmlist[8][1],hands.lmlist[8][2]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
        length = math.hypot(x2-x1,y2-y1)
        length = old_length +(length-old_length)/smoothen
        cv2.rectangle(img,(40,30),(90,30+max_length),(0,255,0),10)
        bar = max_length - int(length)
        if bar < 0:
            bar = 0
        cv2.rectangle(img,(40,30+max_length),(90,30+bar),(0,255,0),-1)
        old_length = length
        v = np.interp(length,[min_length , max_length],[min_vol,max_vol])
        volume.SetMasterVolumeLevel(v, None)
    ctime = time.time()
    fps=1/(ctime-ptime)
    cv2.putText(img,str(int(fps)),(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),1,cv2.LINE_AA)
    ptime = ctime
    cv2.imshow('my cam',img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

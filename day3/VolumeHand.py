import math

import cv2
import time
import numpy as np
from HandTrackingModule import handDetector
cap=cv2.VideoCapture(0)

wCam,hCam=640,480
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

detector=handDetector()

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange=volume.GetVolumeRange()

minVol=volRange[0]
maxVol=volRange[1]
vol=0
volBar=400
volPer=0
while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList,_=detector.findPosition(img,draw=False)

    if lmList:
        x1,y1=lmList[4][1],lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2

        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED) # 绘制点

        length=math.hypot(x2-x1,y2-y1) # 计算三角形斜边

        vol=np.interp(length,[50,300],[minVol,maxVol])  # 线性，length值[50,300]区间改变，[minVol,maxVol]相对应改变
        volBar = np.interp(length, [50, 300], [400, 150])
        # volPer = np.interp(length, [50, 300], [0, 100])
        volPer = np.interp(vol, [minVol,maxVol], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
    cv2.rectangle(img,(50,150),(85,400),(0,255,0),3) #(50xmin,150ymin),(85xmax,400ymax)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'{int(volPer)}%',(40,450),cv2.FONT_HERSHEY_COMPLEX,
                1,(0,255,0),3)
    cv2.putText(img, f'FPS:{int(fps)}%', (40, 70), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0,0), 3)
    cv2.imshow('Image',img)
    cv2.waitKey(1)

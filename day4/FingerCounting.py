import cv2
import time
import os
from HandTrackingModule import handDetector
wCam,hCam=640,480

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath='Image'
myList=os.listdir(folderPath)
overlayList=[]

for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    image=cv2.resize(image,(200,200),interpolation=cv2.INTER_CUBIC)
    overlayList.append(image)

pTime=0

detector=handDetector(detectionCon=0.75)

tipIds=[4,8,12,16,20]

while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList,_=detector.findPosition(img,draw=False)


    if len(lmList):
        fingers=[]
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,len(tipIds)):
            if lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers=fingers.count(1)
        img[0:200,0:200]=overlayList[totalFingers-1]

        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,
                    10,(255,0,0),25)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{int(fps)}',(400,70),cv2.FONT_HERSHEY_COMPLEX,
                1,(255,0,0),3)

    cv2.imshow('Image',img)
    cv2.waitKey(1)
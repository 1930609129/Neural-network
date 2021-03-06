import cv2
import time
from day.PoseModule import poseDetector
cap = cv2.VideoCapture('./1.mp4')
pTime = 0
detector=poseDetector()
while True:
    success, img = cap.read()
    img=detector.findPose(img)
    lmList=detector.getPosition(img,draw=False)
    if len(lmList)!=0:
        print(lmList[6])
        cv2.circle(img,(lmList[6][1],lmList[6][2]),15,(0,0,255),cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
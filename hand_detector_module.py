import cv2
import mediapipe as mp
import time
import math
from mediapipe.python.solutions.drawing_utils import draw_landmarks
import numpy as np

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if(self.results.multi_hand_landmarks):
            for handsLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handsLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosisition(self, img, handNo=0, draw=True):
        self.lmList = []
        if(self.results.multi_hand_landmarks):
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []
        # Fix for left and right hand
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
            
        return fingers
        

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosisition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
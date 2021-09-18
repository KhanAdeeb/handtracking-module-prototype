import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils
        

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
                for handsLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, handsLms, self.mphands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, HandNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            currentHand = self.results.multi_hand_landmarks[HandNo]
            for id, lm in enumerate(currentHand.landmark):
                        print(id, lm)
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        #print(id, cx, cy)
                        lmList.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList
    
    def display_fps(self, img, cTime=0, pTime=0):
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255), 3)


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    Detector = handDetector()
    while True:
        success, img = cap.read()
        img = Detector.findHands(img)
        lmlist = Detector.findPosition(img)
        Detector.display_fps(img, cTime, pTime)
    
        cv2.imshow("Image", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()
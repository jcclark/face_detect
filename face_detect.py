# -*- coding: UTF-8 -*-
import cv2
from matplotlib import pyplot as plt

plt.figure()
plt.rcParams['figure.figsize'] = (224, 224)
face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture("video.mp4")

i = 0
j = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram_first = cv2.calcHist([gray],[0],None,[256],[0,256])
        plt.hist(histogram_first)
        j += 1
        plt.savefig("Hist/hist-00{}.png".format(j))
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x,y,w,h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            i += 1            
            cv2.imwrite("Pessoas/pessoa-00{}.jpg".format(i), crop_img)
            gray_new = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            hist_secundary = cv2.calcHist([gray_new],[0],None,[256],[0,256])
            plt.hist(hist_secundary)            
            plt.savefig("HistFace/hist-face-00{}.png".format(i))  
    else: 
        break

cap.release()
cv2.destroyAllWindows()
print("♩♪♫♬♭This is the end ♩♪♫♬♭")
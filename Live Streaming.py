#!/usr/bin/env python
# coding: utf-8

# ## Live Streaming

import cv2
import numpy as np
import webbrowser
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (244, 224))
    return img, roi


model = load_model('vgg_face_recog.h5')
# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    face = np.expand_dims(face, axis=0)
    
    try:
        res = np.argmax(model.predict(face, 1, verbose = 0), axis=1)
        print(res)
        if res[0] == 0:
            cv2.putText(image, "Hey Aftab", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
        elif res[0] == 1:
            cv2.putText(image, "Hey Kushagra", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )        
       

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()   


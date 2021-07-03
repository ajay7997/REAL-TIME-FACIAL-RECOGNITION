import cv2
import numpy as np 
import sqlite3
import os
from twilio.rest import Client
index = 0
def SMS(name):
  account_sid="ACd506f1f0c3b5009fec4603e85c13aba9"
  auth_token="77464d09990a7a5175b089c23d502052"
  client=Client(account_sid, auth_token)
          
  message = client.messages \
              .create(
              body=name+" is recognized at hyderabad ",
              from_='+15098225628',
              to='+918919724433'
              )
  print(message.sid)

          
conn = sqlite3.connect('database.db')
c = conn.cursor()
fname = "recognizer/trainingData.yml"
if not os.path.isfile(fname):
  print("Please train the data first")
  exit(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)
while True:
  ret, img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
    c.execute("select name from users where id = (?);", (ids,))
    result = c.fetchall()
    name = result[0][0]
    
    if conf < 50:
      if(index==0):
        SMS(name)
        index+=1
      #cv2.putText(img, name, (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)
      
    else:
      #cv2.putText(img, 'No Match', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
      print("no match")
  cv2.imshow('Face Recognizer',img)
      
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break
cap.release()
cv2.destroyAllWindows()

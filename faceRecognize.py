
###     ATUL RAJARAM PUNDE
###     MSC(IMCA) SEM-III
###     ROLL NO 1655
###     PYTHON
###     FACE RECOGNITION PROJECT USING LBPH ALGORITHM
###     TWO FILES: (1)faceRecognize.py and (2)faceTrain.py

import cv2  #use to read image, camera operations
import os   #operating system dependent functionality. Ex- We can access folders
import pickle   #use to convert python object into byte stream to store it in Database

#face_cascade = cv2.CascadeClassifier('C:\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('E:\Final Project\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()   #inbuilt algorith to recognize faces
#eye_cascade = cv2.CascadeClassifier('C:\Python\Python37\Lib\site-packages\cv2\data\haarcascade_eye.xml')
eye_cascade = cv2.CascadeClassifier('E:\Final Project\haarcascade_eye.xml')
#recognizer.read("D:\Python\Trainner.yml")   # Trained faces Reading from file 
recognizer.read("E:\Final Project\Trainner.yml")   # Trained faces Reading from file 

with open("lablels.pickle",'rb') as f:
    og_labels = pickle.load(f)  #gives folder name and labels
    labels = {v:k for k,v in og_labels.items()} #gives folder name and labels

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    #opening Camera*..

def face_extractor(img):    #function to check face is available or not in front of camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #Converting image to gray scale
    faces = face_cascade.detectMultiScale(gray, 1.5, 5) #syntax is detectMultiScale(image, rejectLevels, levelWeights) 
    
    if faces is():  #if face is not found 
        return None
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]   #roi = region of interest
            roi_color = img[y:y+h, x:x+w] 
    return roi_gray

flag = True
while True:
    ret, img = cap.read()   #continuously reading from camera
    if face_extractor(img) is not None: #Execute this only when face is available in front of camera
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #Converting image to gray scale

        faces = face_cascade.detectMultiScale(gray, 1.5, 5) #syntax is detectMultiScale(image, rejectLevels, levelWeights) 
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]   #roi = region of interest
            roi_color = img[y:y+h, x:x+w] 
            id_,obj_dist = recognizer.predict(roi_gray) #predict image based on roi_gray
        
            #Writing text on recangle (name of perticular person in the dataset)
            if obj_dist>=45:
                cv2.putText(img,labels[id_],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2) 
            
            img_name = "RecognizedImage.jpg"    #Last Image Stored with this name
            path = "E:\Project"      #Last image store in this location
            cv2.imwrite(os.path.join(path,img_name), img)   #Function to store data from program to drive
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)   #drawing rectangle around face of colour blue

        eyes = eye_cascade.detectMultiScale(roi_gray,1.5,5) #scale factor = 1.5#syntax is detectMultiScale(image, rejectLevels, levelWeights) 
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)  #Drawing rectangle around eyes
        flag = False
    else:
        print("Image not found..")  #if face is not available in front of camera
        pass
    
    cv2.imshow('To Close Camera: Esc/Backspace',img)
    k = cv2.waitKey(30) & 0xff  #0xFF is a hexadecimal constant which is 11111111 in binary. 
    if k == 27 or k==8: #askii values Esc:27 and Backspace:8
        break   #if Esc or Backspace hit camera will be closed

if flag == False:
    print("\nRecognized Student Is : ",labels[id_],"\n")    #printing the name of person 
else:
    print("\nNo Face is Detected...\n")
    
cap.release()   #Close the camera before terminating program
cv2.destroyAllWindows   #closes all opened windows
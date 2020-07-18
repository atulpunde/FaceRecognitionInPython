###_____________________________________________________________________###_________________________________________
###     ATUL RAJARAM PUNDE                                              ###
###     FERGUSSON COLLEGE (SELF-FINANCE) PUNE-411004                    ###
###     MSC(IMCA) SEM-III                                               ###
###     ROLL NUMBER : 1655                                              ###
###     SUBJECT : PYTHON                                                ###
###     PROJECT NAME : FACE RECOGNITION USING LBPH ALGORITHM            ###
###     TWO PROJECT FILES : (1)faceRecognize.py and (2)faceTrain.py     ###
###_____________________________________________________________________###_________________________________________

import cv2             #use to read image, camera operations
import os              #operating system dependent functionality. Ex- We can access folders
import numpy as np     #NumPy contains a multi-dimentional array and matrix data structures.
import pickle          #use to convert python object into byte stream to store it in Database
from PIL import Image  #pillow: supports for opening,manipulating,saving IMAGE file 

image_dir = os.path.join("E:\Final Project\Dataset")   #path of DATABASE
face_cascade = cv2.CascadeClassifier('E:\Final Project\haarcascade_frontalface_alt2.xml')

current_id = 1  #contains folder number
label_ids = {}  #contains folder names and ids
Y_labels = []   #temp. arrays
X_train = []

print("\nProcessing Data...\n")

for root,dirs,files in os.walk(image_dir):
    for file in files:  #traversing into drive,folders,files
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)  #path - path of each image in folder
            label = os.path.basename(root)  #label contains folder name

            if not label in label_ids:
                label_ids[label] = current_id
                current_id+=1
            id_ = label_ids[label]  #folder no assingning to ids
            
            #PIL package have Image.py file 
            pil_image = Image.open(path).convert("L")   #info of image , size and convert to gray Scale
           
            image_array = np.array(pil_image)

            #syntax is detectMultiScale(image, rejectLevels, levelWeights) 
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors = 3)

            for (x,y,w,h) in faces:
                crop_image = image_array[y:y+h,x:x+w]
                X_train.append(crop_image)  #number array
                Y_labels.append(id_)#id

with open("lablels.pickle",'wb') as f:
    pickle.dump(label_ids,f)        #use to store object into byte

recognizer = cv2.face.LBPHFaceRecognizer_create()   #creating recognizer
recognizer.train(X_train,np.array(Y_labels))    #train recognizer
recognizer.save("E:\Final Project\Trainner.yml")   #save recognizer in file
print("Dataset Trained Successfully...\n")

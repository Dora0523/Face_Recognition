import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime 

#path to image file
path = input("Please enter the path to the file containing known images\n")

images = []  #list of image names
classNames = [] #list of image names without .jpg
myList = os.listdir(path) #list of images contained in Images folder

########### classNames = list of names of images ##############
for cl in myList: #cl = name of each image
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


######################### find encodings #########################

def findEncodings(images):
    encodeList = []
    for img in images:
        #convert to rgb
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


###################  write name & time arrived ##################
def markAttendance(name):    
    #create file if file not exist
    with open('Attendance.csv','a+') as fl:
        fl.seek(0) #first line of file
        c = fl.read(1)
        if not c: #if file is empty
         fl.writelines(f'Attendance:\nName,\tTime\n')

    #open file, read and write
    with open('Attendance.csv','r+') as f: 
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0]) #append name to nameList

        if name not in nameList:
            now = datetime.now() #date&time
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


##################### clear contents of csv file #################      
def clearContents():
 with open('Attendance.csv','r+') as f: 
  f.seek(0)
  f.truncate()

##################################################################

encodeListKnown = findEncodings(images)
print('Encodings Complete')

################### video capture ###############################
cap = cv2.VideoCapture(0)
while True:
    success, img=cap.read()
    #reduce size of image to 1/4
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #lowest distance == best match
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex]<0.45:
            name = classNames[matchIndex].upper()
        else: name = 'Unknown'

        y1,x2,y2,x1 = faceLoc
        #resize back size of image
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        # add box 
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        # add name
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        #write to csv file
        markAttendance(name) 

################## show webcam ###################################

    #cv2.imshow('Webcam',img)
    #cv2.waitKey(1)


    
import cv2
import numpy as np
import face_recognition
import os
# Import the datetime Class:
from datetime import datetime

#  Define the Directory Path:
path = 'ImagesAttendance'
# Initialize Lists:
images = []
classNames = []
# List Files in the Directory:
myList = os.listdir(path)
print(myList)
# Load Images and Extract Class Names:
for cl in myList:
    # Constructing the full path to the image using f-string
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


#  Function Definition:
def findEncodings(images):
    # Initialize an Empty List:
    encodeList = []
    # Iterate Over the Images:
    for img in images:
        # Convert Image to RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Compute Face Encoding:
        encode = face_recognition.face_encodings(img)[0]
        # Append Face Encoding to the List:
        encodeList.append(encode)
    #     Return the List of Face Encodings:
    return encodeList


def markAttendance(name):
    # Open the CSV File:
    with open('Attendance.csv', 'r+') as f:
        # Read Existing Data:
        myDataList = f.readlines()
        # Extract Existing Names:
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        #     Check for Duplicate Name:
        if name not in nameList:
            # Get Current Time:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            # Write New Entry to the File:
            f.writelines(f'\n{name},{dtString}')


# Load Known Face Encodings:
encodeListKnown = findEncodings(images)
print('Encoding Complete')
#  Open Webcam Capture:
cap = cv2.VideoCapture(0)
#  Capture and Process Frames in a Loop:
while True:
    success, img = cap.read()
    # Resize and Convert Image to RGB:
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect Faces and Compute Face Encodings in the Current Frame:
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # Loop Through Detected Faces and Compare with Known Faces:
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        # Update Display and Mark Attendance:
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    # Display Webcam Feed and Wait for a Key Press:
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

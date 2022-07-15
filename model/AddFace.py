import encodings
import face_recognition
import imutils
import pickle
import time
import cv2
import os

def save_photo(video_capture,frame):
    cv2.imshow('img1',frame)
    if cv2.waitKey(0) & 0xFF == ord('y'): 
        cv2.imwrite('images/c1.png',frame)
        cv2.destroyAllWindows()
        video_capture.release()
    elif cv2.waitKey(0) & 0xFF == ord("r"):
        ret, new_frame = video_capture.read()
        save_photo(video_capture,new_frame)
        

def addFace():
    data = pickle.loads(open('images/face_enc', "rb").read())
    video_capture =  cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    save_photo(video_capture,frame)
    photo = cv2.imread("images/c1.png")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model = "hog")
    encodings = face_recognition.face_encodings(rgb,boxes)
    for encoding in encodings:
        data["encodings"].append(encoding)
        name = input()
        data["names"].append(name)
    f = open("images/face_enc","wb")
    f.write(pickle.dumps(data))
    f.close()


addFace()
        
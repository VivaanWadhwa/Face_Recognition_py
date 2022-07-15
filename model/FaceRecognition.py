import face_recognition
import pickle
import cv2
import os
 
 
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# print(cascPathface)
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('images/face_enc', "rb").read())
 
print("Streaming started")
video_capture = cv2.VideoCapture(0)
# loop over frames from the video file stream
while True:
    ret, frame = video_capture.read() # ret tells if image is returned, frame is picture itself
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts image to grayscale
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)  # Detects faces in image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converts image to rgb
    encodings = face_recognition.face_encodings(rgb)  #returns 128 bit image encoding for each face
    names = []
    for encoding in encodings:  #Identifies if the image belongs to someone
        matches = face_recognition.compare_faces(data["encodings"],
         encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
 
 
        names.append(name)
        for ((x, y, w, h), name) in zip(faces, names): #Draw arectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #pressing q exits
        break
video_capture.release()
cv2.destroyAllWindows()
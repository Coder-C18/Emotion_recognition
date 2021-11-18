import cv2
import  face_recognition
from tensorflow.keras import models
import  matplotlib.pyplot as plt
import  numpy as np
Class=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
color=(255,0,0)
thiness=3
# define a video capture object
vid = cv2.VideoCapture(0)
model = models.load_model('Model/emotion_Classifi_ver2.h5')
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    face_loc=face_recognition.face_locations(frame)
    if len(face_loc)>0:
        fl=face_loc[0]
        y1,x2,y2,x1=fl
        #predict
        crop_img = frame[x1+20:x2+20,y1+20:20+y2]
        image = cv2.resize(  crop_img, (48, 48))
        im = image.reshape((1, 48, 48, 3)) / 255
        emotion_classifi = Class[np.argmax(model.predict(im))]
        # cv2.imshow(crop_img)
        #show
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,1)
        cv2.putText(frame,emotion_classifi,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
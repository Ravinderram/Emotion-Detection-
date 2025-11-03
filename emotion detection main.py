import cv2 as cv
import numpy as np
from keras import models

# Load the pre-trained emotion detection model
model = models.load_model("C:/Users/singh/Downloads/FER_Model.h5")

# Define the emotion map
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Start video capture from the default camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera!")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    classifier = cv.CascadeClassifier("D:/Semester 7/Computer Vision/emotion detection project/haarcascade_frontalface_default.xml")
    faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    color = (0, 0, 255)

    if len(faceRects):  # If there are faces detected
        for faceRect in faceRects:  # Draw rectangles around each face
            x, y, w, h = faceRect
            # Draw rectangle around face
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Extract face region
            src = gray[y:y + w, x:x + h]
            # Resize to 48x48
            img = cv.resize(src, (48, 48))
            # Normalize pixel values
            img = img / 255.0
            # Expand dimensions to match model input
            img = np.expand_dims(img, axis=0)
            img = np.array(img, dtype='float32').reshape(-1, 48, 48, 1)
            # Predict emotion
            y_pred = model.predict(img)
            output_class = np.argmax(y_pred[0])
            # Display the emotion on the frame
            cv.putText(frame, emotion_map[output_class], (x, y - 10), cv.FONT_HERSHEY_COMPLEX,
                       1.0, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv.imshow("frame", frame)
    # Exit on 'q' key press
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv.destroyAllWindows()
# done here 
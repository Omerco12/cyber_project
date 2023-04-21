import cv2
import json
import numpy as np
from keras.models import load_model

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained face recognition model
model = load_model('model.h5')

# Load the class labels
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)
    print(class_labels)

def get_key_by_value(dict_obj, value):
    for k, v in dict_obj.items():
        if v == value:
            return k
    return None



# Define a function to recognize a face
def recognize_face(frame, face_coords):
    # Crop the face from the frame
    x, y, w, h = face_coords
    face = frame[y:y + h, x:x + w]

    # Preprocess the face image
    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face, axis=0)
    face = face / 255.0

    # Use the pre-trained model to make a prediction
    prediction = model.predict(face)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index] * 100

    # Draw a rectangle around the face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Write the class name and confidence on the face, or "unknown" if confidence is less than 50%
    if confidence >= 99:
        class_name = get_key_by_value(class_labels, class_index)
        cv2.putText(frame, f'{class_name} ({confidence:.2f}%)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    2)
    else:
        cv2.putText(frame, 'unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Perform face detection on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for face_coords in faces:
        recognize_face(frame, face_coords)

    # Display the annotated frame
    cv2.imshow('Face Recognition', frame)

    # Check for a key press to exit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

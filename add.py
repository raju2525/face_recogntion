import pickle
import cv2
import numpy as np
import os

# Set up directories and variablesRR
face_data = []
names = []
num_photos = 500  # Number of photos per face

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

name = input("Enter Your Name: ").strip()
photo_count = 0

while photo_count < num_photos:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Capture the face and resize for consistency
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten()
        
        # Save the image data and label
        face_data.append(resized_img)
        names.append(name)
        photo_count += 1

        # Display the capture count on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturing {photo_count}/{num_photos}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Face Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Convert face data to NumPy array
face_data = np.array(face_data)

# Save the face data and names
os.makedirs('data', exist_ok=True)
with open('data/names.pkl', 'wb') as f:
    pickle.dump(names, f)
with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump(face_data, f)

print("Data saved successfully.")

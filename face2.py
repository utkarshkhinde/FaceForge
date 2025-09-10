import cv2
import os

# Create a directory to store captured faces
save_dir = "faces"
os.makedirs(save_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Press 'c' to Capture Face, 'q' to Quit")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image_count = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Press 'c' to Capture Face, 'q' to Quit", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c') and len(faces) > 0:
        # Save the first detected face
        (x, y, w, h) = faces[0]
        face_img = frame[y:y+h, x:x+w]
        file_path = os.path.join(save_dir, f"face_{image_count}.jpg")
        cv2.imwrite(file_path, face_img)
        print(f"✅ Face saved to: {file_path}")
        image_count += 1

cap.release()
cv2.destroyAllWindows()

import face_recognition
import os
import cv2

# Path to stored faces
faces_dir = "faces"

# Load known faces
known_faces = []
known_names = []

for filename in os.listdir(faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(faces_dir, filename)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)

        if encoding:  # Only if a face is detected
            known_faces.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])

# Capture a photo using webcam
cap = cv2.VideoCapture(0)
print("Capturing face...")

ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture image.")
    exit()

# Convert to RGB for face_recognition
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
encodings = face_recognition.face_encodings(rgb_frame)

if not encodings:
    print("No face found in the captured image.")
    exit()

# Use the first face found in the frame
input_encoding = encodings[0]

# Compare with known faces
results = face_recognition.compare_faces(known_faces, input_encoding)
matched = False

for i, match in enumerate(results):
    if match:
        print(f"✅ Face matched with: {known_names[i]}")
        matched = True
        break

if not matched:
    print("❌ No match found.")

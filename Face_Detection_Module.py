import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Process each detected face
    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop face
        face = frame[y:y + h, x:x + w]

        # Resize for MobileNetV2
        face_img = cv2.resize(face, (224, 224))

        # Convert to array
        face_array = np.array(face_img, dtype=np.float32)

        # Add batch dimension
        face_array = np.expand_dims(face_array, axis=0)

        # Preprocess input
        face_array = preprocess_input(face_array)

        # Generate embedding
        embedding = model.predict(face_array, verbose=0)

        # Display embedding size
        cv2.putText(
            frame,
            f"Embedding: {embedding.shape}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Print first 5 values
        print("Face Embedding:", embedding[0][:5])

    # Show webcam feed
    cv2.imshow("Real-Time Face Detection + Embedding", frame)

    # Quit with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
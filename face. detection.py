import cv2

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
web_cam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, img = web_cam.read()
    if not ret:  # If no frame is captured, break the loop
        break

    # Convert the image to grayscale (required for face detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the output
    cv2.imshow("Face Detection", img)

    # Break the loop if the 'Esc' key is pressed
    key = cv2.waitKey(10)
    if key == 27:  # ASCII value of 'Esc' key
        break

# Release the webcam and close windows
web_cam.release()
cv2.destroyAllWindows()

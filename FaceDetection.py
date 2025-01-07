import cv2
import time

person_detected = False

cap = cv2.VideoCapture(0)
on = 0
off = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection (you can use other methods like motion detection as well)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:  #face detected
        on += 1
        person_detected = True
    else:               #face not detected
        person_detected = False
        off += 1

        
    stop_t = time.time()
    per = (on/(on+off)) * 100
    print(f"Engagement is {per} percent")    
    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

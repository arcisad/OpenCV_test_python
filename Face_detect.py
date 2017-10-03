import cv2
import os


def detect_dnd_display(frame_in, cascade):
    faces = cascade.detectMultiScale(
        frame_in,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_in, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x_out = x
        y_out = y
    cv2.imshow("Faces found", frame_in)


cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    cas = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cas)
    # Display the resulting frame
    detect_dnd_display(frame, faceCascade)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

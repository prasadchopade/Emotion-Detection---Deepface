from deepface import DeepFace
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Get the result from deepface
    result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)

    # Handle multiple face detection
    if isinstance(result, list):
        for res in result:
            emotions = res["emotion"]
            
            # Get the emotions separately
            angry = "Angry: " + str(round(emotions["angry"], 2))
            happy = "Happy: " + str(round(emotions["happy"], 2))
            sad = "Sad: " + str(round(emotions["sad"], 2))
            surprise = "Surprise: " + str(round(emotions["surprise"], 2))
            neutral = "Neutral: " + str(round(emotions["neutral"], 2))

            # Apply the Haar-future to get the face coordinates
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw the rectangle around the face and print the emotions
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, angry, (x + w + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, happy, (x + w + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, sad, (x + w + 10, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, surprise, (x + w + 10, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, neutral, (x + w + 10, y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(25)
    if key == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

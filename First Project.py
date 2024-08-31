import cv2
import os
import csv
from datetime import datetime

KNOWN_FACES_DIR = "dataset/known_faces"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the attendance file
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])

def capture_new_face(name):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Capture New Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_image = gray[y:y+h, x:x+w]
                cv2.imwrite(os.path.join(KNOWN_FACES_DIR, f"{name}.jpg"), face_image)
            break
    cap.release()
    cv2.destroyAllWindows()

def log_attendance(name):
    with open(ATTENDANCE_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        writer.writerow([name, date, time])

def recognize_faces():
    known_faces = {}
    for filename in os.listdir(KNOWN_FACES_DIR):
        img = cv2.imread(os.path.join(KNOWN_FACES_DIR, filename), cv2.IMREAD_GRAYSCALE)
        known_faces[filename.split(".")[0]] = img

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_image = gray[y:y+h, x:x+w]
            name = "Unknown"
            for known_name, known_face in known_faces.items():
                known_face_resized = cv2.resize(known_face, (w, h))
                result = cv2.matchTemplate(face_image, known_face_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > 0.6:
                    name = known_name
                    break
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            if name != "Unknown":
                log_attendance(name)
        cv2.imshow('Recognize Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("1. Capture New Face")
        print("2. Recognize Faces")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            name = input("Enter the name of the person: ")
            capture_new_face(name)
        elif choice == '2':
            recognize_faces()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")



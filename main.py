import cv2
import dlib
from scipy.spatial import distance
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

@app.route('/')
def index():
    return render_template('index.html')

def detect_eyes():
    cap = cv2.VideoCapture(0)
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = hog_face_detector(gray)

        # Calculate brightness
        brightness = np.mean(gray)  # Average pixel intensity
        brightness_state = "Bright" if brightness > 60 else "Dark"

        state = "Open"
        
        if len(faces) == 0:
            state = "No Faces Found"
        
        for face in faces:
            face_landmarks = dlib_facelandmark(gray, face)
            leftEye, rightEye = [], []

            for n in range(36, 42):
                leftEye.append((face_landmarks.part(n).x, face_landmarks.part(n).y))

            for n in range(42, 48):
                rightEye.append((face_landmarks.part(n).x, face_landmarks.part(n).y))

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)
            EAR = (left_ear + right_ear) / 2
            EAR = round(EAR, 2)

            if EAR < 0.16:
                state = "Closed"
            print(EAR, state)

        socketio.emit('eye_state', {'state': state, 'brightness': brightness_state})

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == 27:  # Exit on ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

def run_flask():
    socketio.run(app, debug=True, use_reloader=False)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    detect_eyes()

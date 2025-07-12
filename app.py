from flask import Flask, render_template, Response, redirect, url_for
import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime
import smtplib
import threading
from playsound import playsound
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

app = Flask(__name__)

# Define parameters
project_folder = "dataset"
unknown_folder = "unknown_persons"
confidence_threshold = 0.5
EMAIL = "finalprojectc123@gmail.com"
app_password = os.getenv("EMAIL_APP_PASSWORD")

camera = None
known_face_encodings = []
known_face_names = []

# Load known faces from the dataset
def load_known_faces(dataset_folder):
    encodings = []
    names = []
    for person_name in os.listdir(dataset_folder):
        person_folder = os.path.join(dataset_folder, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_image)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    names.append(person_name)
    print("Loaded known persons:", names)
    return encodings, names

known_face_encodings, known_face_names = load_known_faces(project_folder)

# Save and send unknown person's photo via email
def send_alert(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(unknown_folder, f"unknown_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)

    if not app_password:
        print("App password is not set.")
        return

    subject = "Unknown Person Alert"
    body = f"""
    Alert: An unknown person has been detected at your door.
    Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Location: Your Home Entrance
    """

    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = EMAIL
        msg["To"] = EMAIL
        msg.attach(MIMEText(body))

        with open(image_path, "rb") as file:
            attachment = MIMEApplication(file.read(), _subtype="jpg")
            attachment.add_header("Content-Disposition", "attachment", filename=os.path.basename(image_path))
            msg.attach(attachment)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL, app_password)
            server.send_message(msg)

        print("Email alert sent for unknown person.")

        os.remove(image_path)
    except Exception as e:
        print("Failed to send email:", str(e))

# Video capture and face recognition
def generate_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=confidence_threshold)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"
            color = (0, 0, 255)

            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]
                color = (0, 255, 0)

                # Ring the bell when a known person is detected
                threading.Thread(target=playsound, args=("ring_sound.mp3",), daemon=True).start()
                print(f"Bell ringing for {name}")

            else:
                # Send email alert for unknown person
                threading.Thread(target=send_alert, args=(frame.copy(),), daemon=True).start()

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return "Camera started"

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
    return "Camera stopped"

@app.route('/logout')
def logout():
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)

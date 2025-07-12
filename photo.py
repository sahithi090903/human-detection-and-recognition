import cv2
import os
import numpy as np
from datetime import datetime
import face_recognition
from playsound import playsound
import smtplib
import threading
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import decode_header

# Define parameters
project_folder = "dataset"
unknown_folder = "unknown_persons"
confidence_threshold = 0.5
IMAP_SERVER = "imap.gmail.com"
EMAIL = "finalprojectc123@gmail.com"
app_password = os.getenv("EMAIL_APP_PASSWORD")

# Create unknown folder if not exists
if not os.path.exists(unknown_folder):
    os.makedirs(unknown_folder)

# Load known faces from the dataset
def load_known_faces(dataset_folder):
    known_face_encodings = []
    known_face_names = []
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
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)
    print("Loaded known persons:", known_face_names)
    return known_face_encodings, known_face_names

# Save and send unknown person's photo via email
def handle_unknown_person(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(unknown_folder, f"unknown_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Unknown person's photo saved: {image_path}")
    send_alert(image_path)

# Send email alert with the photo and check for commands
def send_alert(image_path):
    sender_email = EMAIL
    receiver_email = EMAIL

    if not app_password:
        print("App password is not set.")
        return

    subject = "Unknown Person Alert"
    body = f"""
    Alert: An unknown person has been detected at your door.
    Time of Detection: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Location: Your Home Entrance
    """

    def send_email():
        try:
            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"] = sender_email
            msg["To"] = receiver_email
            msg.attach(MIMEText(body))
            with open(image_path, "rb") as file:
                img_attachment = MIMEApplication(file.read(), _subtype="jpg")
                img_attachment.add_header("Content-Disposition", "attachment", filename=os.path.basename(image_path))
                msg.attach(img_attachment)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, app_password)
                server.send_message(msg)
            print("Alert email with photo sent.")
        except Exception as e:
            print("Failed to send email:", str(e))
        finally:
            try:
                os.remove(image_path)
                print("Unknown person's photo deleted after email sent.")
            except Exception as e:
                print("Failed to delete photo:", str(e))

    def check_email_for_commands():
        try:
            mail = imaplib.IMAP4_SSL(IMAP_SERVER)
            mail.login(EMAIL, app_password)
            mail.select("inbox")

            status, messages = mail.search(None, '(UNSEEN)')
            if status == "OK":
                for num in messages[0].split():
                    status, msg = mail.fetch(num, '(RFC822)')
                    if status == "OK":
                        for response in msg:
                            if isinstance(response, tuple):
                                msg = email.message_from_bytes(response[1])
                                subject, encoding = decode_header(msg["subject"])[0]
                                if isinstance(subject, bytes):
                                    subject = subject.decode(encoding if encoding else "utf-8")
                                if subject.lower() == "open":
                                    print("Command received: Door Open")
                                elif subject.lower() == "close":
                                    print("Command received: Door Close")
            mail.logout()
        except Exception as e:
            print(f"Error checking email: {e}")

    threading.Thread(target=send_email, daemon=True).start()
    threading.Thread(target=check_email_for_commands, daemon=True).start()

# Recognize faces and trigger alerts
def recognize_faces(known_face_encodings, known_face_names):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=confidence_threshold)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]
                print(f"Known person detected: {name}")
                playsound("ring_sound.mp3")
                color = (0, 255, 0)
            else:
                name = "Unknown"
                print("Unknown person detected! Sending email...")
                handle_unknown_person(frame)
                color = (0, 0, 255)

            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Webcam - Recognition Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    known_face_encodings, known_face_names = load_known_faces(project_folder)
    recognize_faces(known_face_encodings, known_face_names)

# take_attendance.py — detect faces from webcam, match to registered students, save attendance
# Uses facenet-pytorch (MTCNN multi-face detector + InceptionResnetV1 embeddings).
# Model weights (.pt format) are downloaded automatically on first run — no h5py needed.

import json
from datetime import date, datetime
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

# --- Configuration ---
STUDENTS_FILE = "./students.json"
ATTENDANCE_FILE = "./attendance.json"
COSINE_THRESHOLD = 0.40   # cosine distance threshold (lower = stricter match)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# keep_all=True so MTCNN returns ALL faces in the frame
mtcnn = MTCNN(keep_all=True, device=DEVICE)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def load_students():
    with open(STUDENTS_FILE, "r") as f:
        return json.load(f)["students"]


def load_attendance():
    with open(ATTENDANCE_FILE, "r") as f:
        return json.load(f)


def save_attendance(data):
    with open(ATTENDANCE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def find_best_match(embedding, students):
    """Compare embedding against all students; return (student, distance) or (None, dist)."""
    best_student = None
    best_dist = float("inf")
    for student in students:
        dist = cosine(embedding, student["embedding"])
        if dist < best_dist:
            best_dist = dist
            best_student = student
    if best_dist < COSINE_THRESHOLD:
        return best_student, best_dist
    return None, best_dist


def main():
    students = load_students()
    if not students:
        print("No students registered. Run register_student.py first.")
        return

    attendance_data = load_attendance()
    today = date.today().isoformat()  # "YYYY-MM-DD"

    # IDs already marked today — avoid duplicates
    already_marked = set()
    if today in attendance_data:
        already_marked = {entry["id"] for entry in attendance_data[today]}

    new_entries = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    print("Attendance mode. Press 'q' to stop and save.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Detect bounding boxes for all faces
        boxes, _ = mtcnn.detect(pil_img)

        if boxes is not None:
            # Get aligned face tensors for all detected faces
            face_tensors = mtcnn(pil_img)   # shape: (N, 3, 160, 160) or None
            if face_tensors is not None:
                # Ensure batch dimension exists even for a single face
                if face_tensors.ndim == 3:
                    face_tensors = face_tensors.unsqueeze(0)

                with torch.no_grad():
                    embeddings = resnet(face_tensors.to(DEVICE)).cpu().numpy()

                # Loop through every detected face
                for box, embedding in zip(boxes, embeddings):
                    matched_student, dist = find_best_match(embedding, students)

                    if matched_student and matched_student["id"] not in already_marked:
                        now_time = datetime.now().strftime("%H:%M:%S")
                        entry = {
                            "id": matched_student["id"],
                            "name": matched_student["name"],
                            "time": now_time,
                        }
                        new_entries.append(entry)
                        already_marked.add(matched_student["id"])
                        print(f"  Marked present: {matched_student['name']} (ID: {matched_student['id']}) dist={dist:.4f}")

                    # Draw bounding box and label on the live frame
                    x1, y1, x2, y2 = (int(c) for c in box)
                    label = matched_student["name"] if matched_student else "Unknown"
                    color = (0, 255, 0) if matched_student else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Attendance - press 'q' to stop", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Merge new entries into attendance.json under today's date key
    if today not in attendance_data:
        attendance_data[today] = []
    attendance_data[today].extend(new_entries)
    save_attendance(attendance_data)

    print(f"\n--- Attendance for {today} ---")
    if attendance_data[today]:
        for e in attendance_data[today]:
            print(f"  {e['name']} (ID: {e['id']}) at {e['time']}")
    else:
        print("  No students marked present.")
    print("Done.")


if __name__ == "__main__":
    main()

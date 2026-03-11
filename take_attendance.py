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


def perform_liveness_check(cap, face_bbox, max_checks=12, motion_threshold=0.015,
                           area_var_threshold=0.008, wait_ms=80):
    """
    Checks small real movements (blinks/head shifts) over a short video window —
    photos/screens remain nearly static and therefore fail.

    Returns True if the face is live, False if it is likely a static photo/screen.
    """
    x, y, w, h = (int(c) for c in face_bbox)
    diffs       = []  # normalized frame-to-frame pixel motion values
    area_ratios = []  # normalized face-bbox area per frame (detects size jitter)
    prev_gray = None

    for _ in range(max_checks):
        ret, frame = cap.read()
        if not ret:                   # camera read failed — fail safe
            return False
        fh, fw = frame.shape[:2]
        frame_area = fw * fh or 1

        # Expand bbox by 10% margin so minor head shifts stay in crop
        margin_x, margin_y = int(w * 0.10), int(h * 0.10)
        x1 = max(x - margin_x, 0)
        y1 = max(y - margin_y, 0)
        x2 = min(x + w + margin_x, fw)
        y2 = min(y + h + margin_y, fh)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Resize to fixed 64x64 so comparisons are stable across distances
        gray = cv2.cvtColor(cv2.resize(roi, (64, 64)), cv2.COLOR_BGR2GRAY)

        # Bbox area variance: a real held face wobbles slightly; a photo stays perfectly still
        area_ratios.append((x2 - x1) * (y2 - y1) / frame_area)

        if prev_gray is not None:
            # Motion signal: photos/screens produce near-zero diff; real faces blink/move
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0
            diffs.append(diff)

        prev_gray = gray
        cv2.waitKey(wait_ms)

    if not diffs:                     # not enough frames — fail safe
        return False

    avg_motion = float(np.mean(diffs))
    area_var   = float(np.var(area_ratios))

    # CALL PERFORM_LIVENESS_CHECK HERE (in main() loop, before find_best_match):
    # face_bbox = (box[0], box[1], box[2]-box[0], box[3]-box[1])
    # if not perform_liveness_check(cap, face_bbox): print("  Liveness failed"); continue

    # Live if motion OR bbox-area variance exceeds threshold
    return avg_motion > motion_threshold or area_var > area_var_threshold


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

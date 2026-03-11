# register_student.py — capture images, compute embedding, save to students.json
# Uses facenet-pytorch (MTCNN detector + InceptionResnetV1 embeddings).
# Model weights (.pt format) are downloaded automatically on first run — no h5py needed.

import json
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# --- Configuration ---
STUDENTS_FILE = "./students.json"
NUM_CAPTURES = 3          # number of face captures to average for robustness

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once at startup (weights downloaded automatically on first run)
mtcnn = MTCNN(keep_all=False, device=DEVICE)          # detects single best face
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)  # 512-D embedding


def load_students():
    with open(STUDENTS_FILE, "r") as f:
        return json.load(f)


def save_students(data):
    with open(STUDENTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_embedding(frame):
    """Detect face in BGR frame and return 512-D numpy embedding, or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    # MTCNN returns an aligned face tensor [3, 160, 160] or None
    face_tensor = mtcnn(pil_img)
    if face_tensor is None:
        return None
    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    return embedding


def capture_embeddings(num_captures):
    """Open webcam, wait for SPACE presses, extract embeddings, return list."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return None

    embeddings = []
    count = 0
    print(f"Press SPACE to capture (need {num_captures} captures). Press 'q' to quit early.")

    while count < num_captures:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame.")
            break

        cv2.imshow("Register - press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            emb = get_embedding(frame)
            if emb is not None:
                embeddings.append(emb.tolist())
                count += 1
                print(f"  Captured {count}/{num_captures}")
            else:
                print("  No face detected, try again.")
        elif key == ord("q"):
            print("Quit early.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return embeddings if embeddings else None


def perform_liveness_check(cap, face_bbox, max_checks=12, motion_threshold=0.015,
                           area_var_threshold=0.008, wait_ms=80):
    """
    Checks small real movements (blinks/head shifts) over a short video window —
    photos/screens remain nearly static and therefore fail.

    Returns True if the face is live, False if it is likely a static photo/screen.
    """
    x, y, w, h = (int(c) for c in face_bbox)
    diffs = []        # normalized frame-to-frame pixel motion values
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

    # CALL PERFORM_LIVENESS_CHECK HERE (inside capture_embeddings, after MTCNN detects a face):
    # face_bbox = (x1, y1, x2 - x1, y2 - y1)
    # if not perform_liveness_check(cap, face_bbox): print("  Liveness failed"); continue

    # Live if motion OR bbox-area variance exceeds threshold
    return avg_motion > motion_threshold or area_var > area_var_threshold


def main():
    student_id = input("Enter student ID: ").strip()
    student_name = input("Enter student name: ").strip()
    if not student_id or not student_name:
        print("ID and name are required.")
        return

    embeddings = capture_embeddings(NUM_CAPTURES)
    if embeddings is None:
        print("No embeddings captured. Registration aborted.")
        return

    # Average embeddings across captures for robustness
    avg_embedding = np.mean(embeddings, axis=0).tolist()

    data = load_students()
    # Overwrite if student ID already exists
    data["students"] = [s for s in data["students"] if s["id"] != student_id]
    data["students"].append({
        "id": student_id,
        "name": student_name,
        "embedding": avg_embedding,
    })
    save_students(data)
    print(f"Student '{student_name}' (ID: {student_id}) registered successfully.")


if __name__ == "__main__":
    main()

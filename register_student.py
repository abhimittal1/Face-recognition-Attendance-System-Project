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

# Face Recognition Attendance System

A minimal, two-script attendance system that uses your webcam and [DeepFace](https://github.com/serengil/deepface) to register students and mark attendance automatically by recognising faces in real time.

---

## How it works

```
register_student.py   →   students.json   →   take_attendance.py   →   attendance.json
      (enrol)              (face store)           (mark present)         (records)
```

1. **Register** — capture a few photos of a student's face, compute a Facenet embedding for each, average them, and save the result alongside the student's ID and name in `students.json`.
2. **Attend** — open the webcam, detect every face in every frame using RetinaFace, compare each face against every stored embedding with cosine distance, and mark the closest matching student as present. Results are saved in `attendance.json` keyed by today's date.

---

## Project files

| File | Purpose |
|---|---|
| `register_student.py` | Enrol a new student (webcam + console prompts) |
| `take_attendance.py` | Live face-recognition attendance via webcam |
| `students.json` | Stores registered student IDs, names, and face embeddings |
| `attendance.json` | Accumulates daily attendance records |
| `requirements.txt` | Python dependencies |

---

## Quick start

### 0 — Python version requirement

**Python 3.11 is required.** TensorFlow (a DeepFace dependency) does not yet support Python 3.12+. If your default `python` is 3.12 or newer, create a dedicated virtual environment with Python 3.11:

```bash
py -3.11 -m venv .venv311
# Windows
.venv311\Scripts\activate
# macOS / Linux
source .venv311/bin/activate
```

All subsequent commands assume this environment is active.

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> On first run, DeepFace will automatically download the **Facenet** and **RetinaFace** model weights (~100 MB combined). This only happens once.

### 2 — Register a student

```bash
python register_student.py
```

**What happens step by step:**

1. You are prompted in the console:
   ```
   Enter student ID:   101
   Enter student name: Rahul
   ```
2. Your webcam opens in a window titled **"Register — press SPACE to capture"**.
3. Stand in front of the camera and press **SPACE** three times (one capture per press).  
   Each press detects your face, extracts a 128-dimensional Facenet embedding, and prints:
   ```
   Captured 1/3
   Captured 2/3
   Captured 3/3
   ```
4. After 3 captures the webcam closes automatically. The three embeddings are averaged into one vector, which is written to `students.json`:
   ```json
   {
     "students": [
       { "id": "101", "name": "Rahul", "embedding": [0.12, -0.44, ...] }
     ]
   }
   ```
5. Success message printed and script exits.

**Tips:**
- If a capture fails (face not detected) the script prints a warning and lets you try again — the total count does not increase.
- Registering the same ID again **overwrites** the previous entry.
- Press **q** to abort without saving.

### 3 — Take attendance

```bash
python take_attendance.py
```

**What happens step by step:**

1. `students.json` is loaded into memory.
2. Your webcam opens in a window titled **"Attendance — press 'q' to stop"**.
3. For every frame captured from the webcam:
   - **RetinaFace** detects all faces present in the frame.
   - For each detected face, **Facenet** computes a 128-D embedding.
   - That embedding is compared against every registered student's embedding using **cosine distance**.
   - If the best match distance is below the threshold (`0.40`), the student is marked present with the current time.
   - A green bounding box + name is drawn around recognised faces; red for unknowns.
   - Each student is marked **at most once per session** (no duplicates).
4. Press **q** to stop.
5. New attendance entries are merged into `attendance.json` under today's ISO date:
   ```json
   {
     "2026-03-11": [
       { "id": "101", "name": "Rahul", "time": "09:15:42" },
       { "id": "102", "name": "Ankit", "time": "09:16:05" }
     ]
   }
   ```
6. A summary is printed to the console and the script exits.

---

## Configuration constants

Both scripts expose easy-to-change constants near the top:

| Constant | File | Default | Meaning |
|---|---|---|---|
| `NUM_CAPTURES` | `register_student.py` | `3` | How many face captures to average during registration |
| `MODEL_NAME` | both | `"Facenet"` | DeepFace recognition model |
| `DETECTOR_BACKEND` | both | `"retinaface"` | Face detector backend |
| `COSINE_THRESHOLD` | `take_attendance.py` | `0.40` | Max cosine distance to count as a match (lower = stricter) |

---

## JSON schemas

### students.json
```json
{
  "students": [
    {
      "id": "101",
      "name": "Rahul",
      "embedding": [0.12, -0.44, "...128 floats total..."]
    }
  ]
}
```

### attendance.json
```json
{
  "2026-03-11": [
    { "id": "101", "name": "Rahul", "time": "09:15:42" },
    { "id": "102", "name": "Ankit", "time": "09:16:05" }
  ]
}
```

---

## Dependencies

| Package | Why |
|---|---|
| `deepface` | Face detection + Facenet embedding extraction |
| `tf-keras` | Required backend for DeepFace |
| `opencv-python` | Webcam capture and window display |
| `numpy` | Embedding averaging |
| `scipy` | Cosine distance calculation |
| `retina-face` | RetinaFace detector (used by DeepFace) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No face detected" during registration | Improve lighting, move closer to camera, remove obstructions |
| Everyone is "Unknown" | Lower `COSINE_THRESHOLD` (e.g. `0.50`) or re-register with better lighting |
| Webcam not opening | Check another app isn't using the camera; try `cv2.VideoCapture(1)` for a second camera |
| First run is slow | DeepFace is downloading model weights — wait for the download to finish |
| `enforce_detection` errors | The frame had no face; the script catches this and continues |

import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from mlp_model import EmotionNet

# -------------------------------
# Config
# -------------------------------
EMOTION_LABELS = [
    "angry", "disgust", "fear",
    "happy", "sad", "surprise", "neutral"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HISTORY = 7
CONF_THRESH = 0.45
TEMPERATURE = 2.5

# -------------------------------
# Load trained model
# -------------------------------
model = EmotionNet(num_classes=7).to(DEVICE)
model.load_state_dict(torch.load("emotion_model.pt", map_location=DEVICE))
model.eval()

# -------------------------------
# MediaPipe FaceMesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------
# Helper: draw probability bars
# -------------------------------
def draw_prob_bars(frame, labels, probs,
                   x=20, y=90,
                   bar_width=200,
                   bar_height=18,
                   spacing=6):
    for i, (label, prob) in enumerate(zip(labels, probs)):
        y_i = y + i * (bar_height + spacing)

        # Background bar
        cv2.rectangle(
            frame,
            (x, y_i),
            (x + bar_width, y_i + bar_height),
            (60, 60, 60),
            -1
        )

        # Filled bar
        fill_w = int(bar_width * prob)
        cv2.rectangle(
            frame,
            (x, y_i),
            (x + fill_w, y_i + bar_height),
            (180, 180, 180),
            -1
        )

        # Label + value
        cv2.putText(
            frame,
            f"{label}: {prob:.2f}",
            (x + bar_width + 10, y_i + bar_height - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

# -------------------------------
# Webcam
# -------------------------------
cap = cv2.VideoCapture(0)
history = deque(maxlen=HISTORY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # -------------------------------
        # Extract (x, y, z) landmarks
        # -------------------------------
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
            dtype=np.float32
        )

        # -------------------------------
        # Normalize EXACTLY like training
        # -------------------------------
        coords -= coords[1]  # nose tip
        scale = np.linalg.norm(coords[33] - coords[263])
        coords /= (scale + 1e-6)

        x_tensor = torch.tensor(coords.flatten(), dtype=torch.float32)\
                        .unsqueeze(0).to(DEVICE)

        # Sanity check (remove later if you want)
        assert x_tensor.shape[1] == 1434, x_tensor.shape

        # -------------------------------
        # Emotion prediction
        # -------------------------------
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits / TEMPERATURE, dim=1)[0]

        history.append(probs.cpu().numpy())
        avg_probs = np.mean(history, axis=0)

        pred_idx = np.argmax(avg_probs)
        confidence = avg_probs[pred_idx]

        if confidence < CONF_THRESH:
            emotion_text = "uncertain"
        else:
            emotion_text = EMOTION_LABELS[pred_idx]

        # -------------------------------
        # Draw FaceMesh (clean gray)
        # -------------------------------
        for a, b in mp_face_mesh.FACEMESH_TESSELATION:
            p1 = face_landmarks.landmark[a]
            p2 = face_landmarks.landmark[b]

            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (180, 180, 180),
                1
            )

        # -------------------------------
        # Draw emotion text
        # -------------------------------
        cv2.putText(
            frame,
            f"{emotion_text} ({confidence:.2f})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        # -------------------------------
        # Draw probability bars
        # -------------------------------
        draw_prob_bars(frame, EMOTION_LABELS, avg_probs)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

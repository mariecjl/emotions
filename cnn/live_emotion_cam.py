import cv2
import torch
import torch.nn.functional as F
import numpy as np

from cnnmodel import EmotionCNN

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "emotion_cnn_fer2013.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# -----------------------
# LOAD MODEL
# -----------------------
model = EmotionCNN(num_classes=7).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded on", DEVICE)

# -----------------------
# FACE DETECTOR
# -----------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------
# PREPROCESS (MATCH TRAINING)
# -----------------------
def preprocess_face(face_img):
    """
    face_img: grayscale face image (H,W)
    returns: torch tensor [1,1,48,48]
    """
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype(np.float32) / 255.0
    face_img = torch.tensor(face_img).unsqueeze(0).unsqueeze(0)
    return face_img

# -----------------------
# DRAW EMOTION BARS
# -----------------------
def draw_emotion_bars(frame, probs, x=10, y=20):
    bar_w = 200
    bar_h = 18

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        bar_len = int(bar_w * prob)

        cv2.rectangle(
            frame,
            (x, y + i*25),
            (x + bar_len, y + i*25 + bar_h),
            (0, 255, 0),
            -1
        )

        cv2.rectangle(
            frame,
            (x, y + i*25),
            (x + bar_w, y + i*25 + bar_h),
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"{emotion}: {prob*100:.1f}%",
            (x + bar_w + 10, y + i*25 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

# -----------------------
# LIVE CAMERA LOOP
# -----------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        input_tensor = preprocess_face(face).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        emotion_idx = probs.argmax()
        emotion_label = EMOTIONS[emotion_idx]

        # Face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            emotion_label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

        draw_emotion_bars(frame, probs)

    cv2.imshow("Live Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

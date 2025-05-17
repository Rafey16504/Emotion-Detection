from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class EmotionCNN(nn.Module):
    def __init__(self, version=1):
        super(EmotionCNN, self).__init__()
        if version == 3:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),

                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),

                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),

                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 3 * 3, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, 7)
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),

                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),

                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 3 * 3, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(64, 7)
            )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_configs = [("emotionDetector_1.pth", 1), ("emotionDetector_2.pth", 3)]
models = []
for path, version in model_configs:
    model = EmotionCNN(version).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    models.append(model)
    
dummy = torch.randn(1, 1, 48, 48).to(device)
with torch.no_grad():
    for m in models:
        m(dummy)
        
# Define transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Labels
labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam (0 = default camera), CAP_DSHOW = faster init on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera_active = False  # Track if the camera is active

def detect_emotion():
    global camera_active
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize frame to 640x480 for faster processing
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            emotion_data = {"emotion": "No face detected", "confidence": 0.0}
        else:
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                roi = gray[y:y+h, x:x+w]

                try:
                    img = cv2.resize(roi, (48, 48))
                except:
                    continue

                img_tensor = transform(img).unsqueeze(0).to(device)

                # Predict using ensemble
                with torch.no_grad():
                    probs_sum = torch.zeros(7, device=device)
                    for model in models:
                        logits = model(img_tensor)
                        probs = F.softmax(logits, dim=1).squeeze()
                        probs_sum += probs
                    avg_probs = probs_sum / len(models)
                    pred_idx = torch.argmax(avg_probs).item()
                    confidence = avg_probs[pred_idx].item()
                    emotion = labels[pred_idx]

                emotion_data = {"emotion": emotion, "confidence": round(confidence, 2)}
                break  # Process only the first face detected

        # Encode frame as Base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Emit both the frame and emotion data
        socketio.emit('update', {"frame": frame_base64, "emotion": emotion_data})

@socketio.on('start_camera')
def handle_start_camera():
    global camera_active
    if not camera_active:
        camera_active = True
        socketio.start_background_task(detect_emotion)

@socketio.on('stop_camera')
def handle_stop_camera():
    global camera_active
    camera_active = False

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
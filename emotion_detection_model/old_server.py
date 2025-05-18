from flask import Flask, Response, render_template
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from threading import Thread, Lock
import time

app = Flask(__name__)

#--------------------------------------------------------------------------------------------------------

# CNN class for model 1 and 2
class EmotionCNN(nn.Module):
    def __init__(self, version=1):
        super(EmotionCNN, self).__init__()
        if version == 2:
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
        elif version == 3:
            self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1)
        )

            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 1 * 1, 256),
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
    
#--------------------------------------------------------------------------------------------------------

# Models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_configs = [("emotionDetector_1.pth", 1), ("emotionDetector_2.pth", 2), ("emotionDetector_3.pth", 3)]
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

#--------------------------------------------------------------------------------------------------------

# Define transform

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

#--------------------------------------------------------------------------------------------------------

# Labels

labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#--------------------------------------------------------------------------------------------------------

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_lock = Lock()
current_frame = None
processing_frame = None
skip_frames = 2  # Process every 3rd frame
frame_counter = 0

#--------------------------------------------------------------------------------------------------------

# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

#--------------------------------------------------------------------------------------------------------

def capture_frames():
    global current_frame, frame_counter
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
            frame_counter += 1
        time.sleep(0.01)

Thread(target=capture_frames, daemon=True).start()

#--------------------------------------------------------------------------------------------------------

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
        
        img_tensor = transform(roi).unsqueeze(0).to(device)
        
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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255), 2)
    
    return frame

#--------------------------------------------------------------------------------------------------------

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#--------------------------------------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

#--------------------------------------------------------------------------------------------------------

def generate_frames():
    global processing_frame, frame_counter
    last_processed = frame_counter
    
    while True:
        with frame_lock:
            if current_frame is None:
                continue
                
            if (frame_counter - last_processed) < skip_frames:
                continue
                
            processed_frame = process_frame(current_frame.copy())
            last_processed = frame_counter
            
            ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

#--------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
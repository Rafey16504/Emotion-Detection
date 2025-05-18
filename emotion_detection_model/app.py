from flask import Flask, jsonify
from flask_socketio import SocketIO
import cv2
import base64
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    max_http_buffer_size=10 * 1024 * 1024
)

#--------------------------------------------------------------------------------------------------------

# Model class for emotion detection

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

# Load models

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

# Emotion labels
  
labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#--------------------------------------------------------------------------------------------------------

# Transformations

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

#--------------------------------------------------------------------------------------------------------

# Load Haar Cascade for face detection

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#--------------------------------------------------------------------------------------------------------

# SocketIO event handlers

@socketio.on('process_frame')
def handle_frame(data):
    try:
        img_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(img_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))
        
        results = []
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi_pil = Image.fromarray(roi).convert('L')
            img_tensor = transform(roi_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                probs_sum = torch.zeros(7, device=device)
                for model in models:
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze()
                    probs_sum += probs
                avg_probs = probs_sum / len(models)
                pred_idx = torch.argmax(avg_probs).item()
                confidence = avg_probs[pred_idx].item()
            
            results.append({
                'emotion': labels[pred_idx],
                'confidence': round(confidence * 100, 1),
                'bbox': [int(x), int(y), int(w), int(h)]
            })
        
        socketio.emit('predictions', results)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        socketio.emit('error', {'message': 'Processing failed'})

#--------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
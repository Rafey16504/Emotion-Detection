import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import numpy as np

#--------------------------------------------------------------------------------------------------------

# CNN class for model 1 and 2

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
    
#--------------------------------------------------------------------------------------------------------

# Load models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_configs = [("emotionDetector_1.pth", 1), ("emotionDetector_2.pth", 3)]
models = []
for path, version in model_configs:
    model = EmotionCNN(version).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    models.append(model)
    
#--------------------------------------------------------------------------------------------------------

# Warm up models with dummy input

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

# Face detector

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#--------------------------------------------------------------------------------------------------------

# Webcam (0 = default camera), CAP_DSHOW = faster init on Windows

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#--------------------------------------------------------------------------------------------------------

#Process every nth frame (skip frames to increase FPS)

frame_skip = 1
frame_count = 0

#--------------------------------------------------------------------------------------------------------

# Cache last face position and result

last_prediction = None
last_face_coords = None

#--------------------------------------------------------------------------------------------------------

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 640x480 for faster processing
    
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
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

        # Box around face
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
        # Emotion label
        
        label_text = emotion

        # filled box behind text
        
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_TRIPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 5, y), (0, 0, 255), -1)

        # Draw text
        
        cv2.putText(frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)

    # Show frame
    
    cv2.imshow("Optimized Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

#--------------------------------------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()

from flask import Flask, Response, render_template
import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

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

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Preprocess and predict
            img = cv2.resize(roi, (48, 48))
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                probs_sum = torch.zeros(7, device=device)
                for model in models:
                    logits = model(img_tensor)
                    probs = F.softmax(logits, dim=1).squeeze()
                    probs_sum += probs
                avg_probs = probs_sum / len(models)
                pred_idx = torch.argmax(avg_probs).item()
                confidence = avg_probs[pred_idx].item()
            
            # Draw annotations
            emotion = labels[pred_idx]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 0, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
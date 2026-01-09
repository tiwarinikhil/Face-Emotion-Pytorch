import torch
import cv2
import numpy as np
from utility.model import EmotionNet
from torchvision import transforms

# Load the trained model
model = EmotionNet().cuda()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Emotion dictionary
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Image transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = transform(cropped_img).unsqueeze(0).unsqueeze(0).cuda()

        with torch.no_grad():
            output = model(cropped_img)
            _, prediction = torch.max(output, 1)
            emotion_label = emotion_dict[prediction.item()]

        cv2.putText(frame, emotion_label, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

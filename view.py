import torch
import cv2
import argparse
from utility.model import EmotionNet
from torchvision import transforms
from collections import Counter
from PIL import Image

# Command-line argument to specify input image
ap = argparse.ArgumentParser()
ap.add_argument("--image", help="Path to the input image", required=True)
args = ap.parse_args()

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

# Image transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load and process the image
image_path = args.image  # Get the image path from the command line argument
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Image not found!")
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

print("\nDetected Faces & Emotions:")
print("---------------------------")

emotion_list = []  # Store all emotion labels
face_id = 1

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi_pil = Image.fromarray(roi_gray)
    cropped_img = transform(roi_pil).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(cropped_img)
        _, prediction = torch.max(output, 1)
        emotion_label = emotion_dict[prediction.item()]

    emotion_list.append(emotion_label)
    print(f"Person {face_id}: {emotion_label}")

    cv2.putText(frame, f"ID:{face_id} {emotion_label}",
                (x, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2)

    face_id += 1

# -----------------------------
# EMOTION SUMMARY CALCULATIONS
# -----------------------------
total_faces = len(emotion_list)
emotion_counts = Counter(emotion_list)

sorted_counts = sorted(emotion_counts.items(), key=lambda x: x[1])

print("\nEmotion Summary (Ascending Order):")
print("----------------------------------")
for emotion, count in sorted_counts:
    percentage = (count / total_faces) * 100
    print(f"{emotion}: {count} ({percentage:.2f}%)")

# Resize image to 480p (854x480)
output_480p = cv2.resize(frame, (854, 480))
cv2.imshow("Emotion Detection (480p)", output_480p)
cv2.waitKey(0)
cv2.destroyAllWindows()
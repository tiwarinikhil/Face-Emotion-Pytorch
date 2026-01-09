import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from utility.emotion_dataset import EmotionDataset, plot_model_history
from utility.model import EmotionNet

# Load datasets and create DataLoader
train_dir = 'data/train'
val_dir = 'data/test'

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = EmotionDataset(train_dir, transform)
val_dataset = EmotionDataset(val_dir, transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, and optimizer
model = EmotionNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Transfer images and labels to GPU
            # print(f"images: {type(images)}, labels: {type(labels)}")  # Debugging line

            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                # Transfer images and labels to GPU
                images, labels = images.cuda(), labels.cuda()

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss.append(running_loss / len(val_loader))
        val_accuracy.append(100 * correct / total)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    plot_model_history(train_loss, val_loss, train_accuracy, val_accuracy)
    torch.save(model.state_dict(), 'model.pth')


train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)

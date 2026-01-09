from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np


class EmotionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Get all subdirectories (emotions)
        self.emotion_labels = os.listdir(image_dir)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.emotion_labels)}  # Map label to index
        
        for label in self.emotion_labels:
            emotion_folder = os.path.join(image_dir, label)
            if os.path.isdir(emotion_folder):  # Ensure it's a folder
                # Add images from this emotion folder
                for filename in os.listdir(emotion_folder):
                    file_path = os.path.join(emotion_folder, filename)
                    if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg')):
                        self.image_paths.append(file_path)
                        self.labels.append(self.label_to_idx[label])  # Store the integer index of the label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]  # This will be the integer label (index)

        # Load the image
        image = Image.open(image_path).convert('RGB')  # Ensure it is in RGB format

        if self.transform:
            image = self.transform(image)

        return image, label




def plot_model_history(train_loss, val_loss, train_accuracy, val_accuracy):
    """
    Plot Accuracy and Loss curves given the training and validation history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # plot accuracy
    axs[0].plot(train_accuracy, label="train")
    axs[0].plot(val_accuracy, label="validation")
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='best')

    # plot loss
    axs[1].plot(train_loss, label="train")
    axs[1].plot(val_loss, label="validation")
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='best')

    plt.savefig('plot.png')
    plt.show()

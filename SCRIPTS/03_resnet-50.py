import pandas as pd
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_DIR   = SCRIPT_DIR.parent / "DATA" / "dataset_split"
manifest_path = SPLIT_DIR / "manifest.csv"

df = pd.read_csv(manifest_path)
print(df.head())

class WeatherDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = int(row["label_int"])

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

added_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), transform=standard_transform)
val_dataset   = datasets.ImageFolder(os.path.join(SPLIT_DIR, "val"),   transform=standard_transform)
test_dataset  = datasets.ImageFolder(os.path.join(SPLIT_DIR, "test"),  transform=standard_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print class mapping for reference
print("Class to index mapping:")
print(train_dataset.class_to_idx)

# ------------------------------------------------------
# Training + Validation Loop
# ------------------------------------------------------
def train_model(model, train_load, val_load, criterion, optimizer, device, num_epochs=10):

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        # -------------------------
        # Training Phase
        # -------------------------
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in tqdm(train_load, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_train_loss = running_loss / len(train_load.dataset)
        epoch_train_acc = running_corrects.double() / len(train_load.dataset)

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")

        # -------------------------
        # Validation Phase
        # -------------------------
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for images, labels in tqdm(val_load, desc="Validation", leave=False):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels)

        epoch_val_loss = val_loss / len(val_load.dataset)
        epoch_val_acc = val_corrects.double() / len(val_load.dataset)

        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # -------------------------
        # Save Best Model
        # -------------------------
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), "best_resnet50.pth")
            print("✓ Best model saved.")

    print("\nTraining complete.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    return model



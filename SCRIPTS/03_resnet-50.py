import pandas as pd
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = SCRIPT_DIR.parent / "DATA" / "dataset_split"

standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), transform=standard_transform)
val_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "val"), transform=standard_transform)
test_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "test"), transform=standard_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print class mapping for reference
print("Class to index mapping:")
print(train_dataset.class_to_idx)


# ------------------------------------------------------
# Training + Validation Loop
# ------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # -------------------------
        # Training Phase
        # -------------------------
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
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

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")

        # -------------------------
        # Validation Phase
        # -------------------------
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects.double() / len(val_loader.dataset)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
for name, param in model.named_parameters():
    if "layer4" not in name:  # keep last block trainable
        param.requires_grad = False
num_features = model.fc.in_features
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(num_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=10
)

best_model = models.resnet50(pretrained=False)

# Replace the classifier to match your dataset
num_classes = len(train_dataset.classes)
best_model.fc = nn.Linear(best_model.fc.in_features, num_classes)

best_model.load_state_dict(torch.load("best_resnet50.pth"))

best_model = best_model.to(device)
best_model.eval()

def evaluate_testset(model, test_loader, criterion, device):
    model.eval()

    test_loss = 0.0
    test_corrects = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            test_loss += loss.item() * images.size(0)
            test_corrects += torch.sum(preds == labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = test_corrects.double() / len(test_loader.dataset)

    return test_loss, test_acc.item(), all_preds, all_labels

criterion = nn.CrossEntropyLoss()

test_loss, test_acc, test_preds, test_labels = evaluate_testset(
    model=best_model,
    test_loader=test_loader,
    criterion=criterion,
    device=device
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Acc:  {test_acc:.4f}")



cm = confusion_matrix(test_labels, test_preds)
print(classification_report(test_labels, test_preds, target_names=train_dataset.classes))

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
plt.savefig("evaluation_resnet50_best.png", dpi=300, bbox_inches="tight")





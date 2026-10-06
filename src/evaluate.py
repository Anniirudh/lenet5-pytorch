import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (confusion_matrix,classification_report)

from model import LeNet5

#Configuration

BATCH_SIZE=128

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH="models/lenet5_mnist.pth"
CLASS_NAMES=[
    "0","1","2","3","4","5","6","7","8","9"]

#Dataset

transform=transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

#Load Model

model=LeNet5().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

#Evaluation

all_predictions = []
all_labels = []

correct = 0
total = 0

criterion = nn.CrossEntropyLoss()

total_loss = 0.0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )
test_loss=total_loss/len(test_loader)

test_accuracy=(100* correct/total)

print("\n======= EVALUATION =======")

print(f"Test Loss: {test_loss:.4f}")

print(f"Test Accuracy: {test_accuracy:.2f}%")

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES
)

print("\n========== Classification Report ==========")

print(report)


with open(
    "results/classification_report.txt",
    "w"
) as file:

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    file.write(
        f"Test Accuracy: {test_accuracy:.2f}%\n\n"
    )

    file.write(report)


#Confusion Matrix

cm = confusion_matrix(
    all_labels,
    all_predictions
)


plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("LeNet-5 Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()

print(
    "\nConfusion matrix saved to "
    "results/confusion_matrix.png"
)

#Sample predictions

images, labels = next(iter(test_loader))

images_gpu = images.to(DEVICE)

with torch.no_grad():

    outputs = model(images_gpu)

    predictions = outputs.argmax(dim=1)


plt.figure(figsize=(12, 8))

for i in range(16):

    image = images[i].squeeze().numpy()

    # Undo normalization for visualization
    image = image * 0.3081 + 0.1307

    plt.subplot(4, 4, i + 1)

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(
        f"Pred: {predictions[i].item()} | "
        f"True: {labels[i].item()}"
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "results/sample_predictions.png",
    dpi=300
)

plt.close()

print(
    "Sample predictions saved to "
    "results/sample_predictions.png"
)
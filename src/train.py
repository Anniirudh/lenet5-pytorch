import torch 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import LeNet5

#Configuration

BATCH_SIZE=128
NUM_WORKERS=0
EPOCHS=10
LEARNING_RATE=0.001

DEVICE=torch.device(
    "cuda" if torch.cuda.is_available() else 'cpu'
)

#Data Transformation

transform=transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])


#Dataset

train_dataset=datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset=datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


#Data Loaders

train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

test_loader=DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

#Training

def train(model,loader,criterion,optimizer):
    model.train()

    total_loss=0
    correct=0
    total=0

    progress_bar=tqdm(
        loader,
        desc='Training'
    )

    for images,labels in progress_bar:

        images=images.to(DEVICE)
        labels=labels.to(DEVICE)

        optimizer.zero_grad()

        outputs=model(images)

        loss=criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

        predictions=outputs.argmax(dim=1)

        correct+=(predictions==labels).sum().item()
        total+=labels.size(0)
        accuracy=100* correct/total
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{accuracy:.2f}%"
        )

    epoch_loss=total_loss/len(loader)
    epoch_accuracy=100* correct/total

    return epoch_loss,epoch_accuracy

# Evaluation

def evaluate(model,loader,criterion):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

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

    loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return loss, accuracy


def main():
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    model = LeNet5().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        train_loss, train_accuracy = train(
            model,
            train_loader,
            criterion,
            optimizer
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )

        print(
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )

    torch.save(
        model.state_dict(),
        "models/lenet5_mnist.pth"
    )

    print("\nModel saved successfully.")


if __name__=='__main__':
    main()
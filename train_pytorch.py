import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from data_loader_pytorch import preparing_dataset
from model_pytorch import Transfer_learning_Cifar

def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

def main():

    TARGET_SIZE = 64
    BATCH_SIZE = 64
    NUM_WORKERS = 2
    EPOCHS = 20
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training running on device: {device}")


    print("📦 Loading CIFAR-10 dataset...")
    train_loader, test_loader = preparing_dataset(target_size=TARGET_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = Transfer_learning_Cifar(num_classes=10).to(device)
    summary(model, input_size=(BATCH_SIZE, 3, TARGET_SIZE, TARGET_SIZE))
    criterion = nn.CrossEntropyLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

    best_acc = 0.0
    print("\n🎬 Starting Training Pipeline...\n")


    for epoch in range(EPOCHS):

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}]"
              f"| Train Loss: {train_loss:.4f} - {train_acc:.2f}%"
              f"| Test Loss: {test_loss:.4f} - {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_vgg16_cfar10.pth")
            print(f"💾 New best model saved with accuracy: {best_acc:.2f}%")

    print("\n✅ Training Pipeline Terminated Cleanly!")

if __name__ == "__main__":
    main()





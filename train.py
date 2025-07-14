import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.vit import MiniViT
from utils import save_checkpoints, load_checkpoints


def main():
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50
    best_acc = 0.0

    # load data and data augmentation
    print("Loading data...")
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ]
    )

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    # define model and optimizer
    print("Building model...")
    model = MiniViT(img_size=32, patch_size=4, embed_dim=256, depth=8, heads=8).to(
        device
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # train model
    print("Training model...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = 100 * correct / total
            pbar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f} | Train Acc {acc}%")
        test_acc, test_loss = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}: Val Acc {test_acc:.2f}% | Val Loss {test_loss:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoints(model, path="best_model.pt")

        train_acc = acc
        train_loss_list.append(total_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(test_acc)
        val_loss_list.append(test_loss)

    import pandas as pd

    df = pd.DataFrame(
        {
            "train_loss": train_loss_list,
            "train_acc": train_acc_list,
            "val_loss": val_loss_list,
            "val_acc": val_acc_list,
        }
    )
    df.to_csv("train_log.csv", index=False)
    print("training_log.csv saved")

    from utils import plot_curves

    plot_curves(train_loss_list, val_loss_list, train_acc_list, val_acc_list)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    acc = 0

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = 100 * correct / total
        pbar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

    avg_loss = total_loss / len(dataloader)
    return acc, avg_loss


if __name__ == "__main__":
    main()

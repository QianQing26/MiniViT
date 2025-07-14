import torch
import matplotlib.pyplot as plt


def save_checkpoints(model, path="best_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"best model saved to {path}")


def load_checkpoints(model, path="best_model.pt"):
    model.load_state_dict(torch.load(path))
    print(f"best model loaded from {path}")
    return model


def plot_curves(train_loss, val_loss, train_acc, val_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("acc.png")
    plt.close()

    print("loss.png and acc.png saved!")

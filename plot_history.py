import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    history_path = "results/training_history.txt"

    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f"Could not find {history_path}. Run train.py first."
        )

    df = pd.read_csv(history_path)

    required_columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training history: {missing}")

    os.makedirs("results", exist_ok=True)

    # -----------------------------
    # Loss Curve
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/loss_curve.png", dpi=300)
    plt.show()
    plt.close()

    # -----------------------------
    # Accuracy Curve
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(df["epoch"], df["val_acc"], marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/accuracy_curve.png", dpi=300)
    plt.show()
    plt.close()

    print("Saved loss curve to: results/loss_curve.png")
    print("Saved accuracy curve to: results/accuracy_curve.png")


if __name__ == "__main__":
    main()
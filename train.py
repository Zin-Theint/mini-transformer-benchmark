import os
import time
import random
import numpy as np
import torch
import torch.nn as nn

from data import create_dataloader
from model import MiniTransformerClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(logits, labels):
    """
    logits: (batch_size,)
    labels: (batch_size,)
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # -----------------------------
    # Config
    # -----------------------------
    seed = 42
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 15

    vocab_size = 5
    max_seq_len = 20
    d_model = 64
    num_heads = 4
    d_ff = 128
    num_layers = 1
    dropout = 0.1
    use_positional_encoding = True

    train_path = "data/train.csv"
    val_path = "data/validation.csv"
    test_path = "data/test.csv"

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pt")

    # -----------------------------
    # Setup
    # -----------------------------
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # Data
    # -----------------------------
    print("Loading datasets...")

    _, train_loader = create_dataloader(
        csv_path=train_path,
        batch_size=batch_size,
        shuffle=True,
        verify_labels=False,
    )

    _, val_loader = create_dataloader(
        csv_path=val_path,
        batch_size=batch_size,
        shuffle=False,
        verify_labels=False,
    )

    _, test_loader = create_dataloader(
        csv_path=test_path,
        batch_size=batch_size,
        shuffle=False,
        verify_labels=False,
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = MiniTransformerClassifier(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
        use_positional_encoding=use_positional_encoding,
    ).to(device)

    print("Model parameter count:", count_parameters(model))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------
    # Training Loop
    # -----------------------------
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0

    start_time = time.time()

    print("\nStarting training...\n")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        #if val_acc > best_val_acc:
        #    best_val_acc = val_acc
        #    best_epoch = epoch
        #    torch.save(model.state_dict(), best_model_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    total_training_time = time.time() - start_time

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Total training time: {total_training_time:.2f} seconds")

    # -----------------------------
    # Load Best Model and Test
    # -----------------------------
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # -----------------------------
    # Save training history
    # -----------------------------
    history_path = os.path.join(save_dir, "training_history.txt")
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for i in range(num_epochs):
            f.write(
                f"{i+1},"
                f"{history['train_loss'][i]:.6f},"
                f"{history['train_acc'][i]:.6f},"
                f"{history['val_loss'][i]:.6f},"
                f"{history['val_acc'][i]:.6f}\n"
            )

    summary_path = os.path.join(save_dir, "train_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"d_model: {d_model}\n")
        f.write(f"num_heads: {num_heads}\n")
        f.write(f"d_ff: {d_ff}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"dropout: {dropout}\n")
        f.write(f"use_positional_encoding: {use_positional_encoding}\n")
        f.write(f"Parameter count: {count_parameters(model)}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Validation accuracy at best loss: {best_val_acc:.6f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Test loss: {test_loss:.6f}\n")
        f.write(f"Test accuracy: {test_acc:.6f}\n")
        f.write(f"Total training time (seconds): {total_training_time:.6f}\n")

    print(f"\nSaved best model to: {best_model_path}")
    print(f"Saved training history to: {history_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
import os
import time
import copy
import torch
import torch.nn as nn
import pandas as pd

from data import create_dataloader
from model import MiniTransformerClassifier
from train import train_one_epoch, evaluate, set_seed, count_parameters


# -----------------------------
# CONFIG (shared across all runs)
# -----------------------------
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20

TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/validation.csv"
TEST_PATH = "data/test.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# MODEL CONFIGS TO TEST
# -----------------------------
MODEL_CONFIGS = [
    {
        "name": "Model_A",
        "use_positional_encoding": True,
        "num_heads": 1,
        "num_layers": 1,
    },
    {
        "name": "Model_B",
        "use_positional_encoding": True,
        "num_heads": 4,
        "num_layers": 1,
    },
    {
        "name": "Model_C",
        "use_positional_encoding": False,
        "num_heads": 4,
        "num_layers": 1,
    },
    {
        "name": "Model_D",
        "use_positional_encoding": True,
        "num_heads": 4,
        "num_layers": 2,
    },
]


def run_experiment(config):
    print(f"\n=== Running {config['name']} ===")

    set_seed(SEED)

    # -----------------------------
    # Load data
    # -----------------------------
    _, train_loader = create_dataloader(
        csv_path=TRAIN_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verify_labels=False,
    )

    _, val_loader = create_dataloader(
        csv_path=VAL_PATH,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verify_labels=False,
    )

    _, test_loader = create_dataloader(
        csv_path=TEST_PATH,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verify_labels=False,
    )

    # -----------------------------
    # Build model
    # -----------------------------
    model = MiniTransformerClassifier(
        vocab_size=5,
        max_seq_len=20,
        d_model=64,
        num_heads=config["num_heads"],
        d_ff=128,
        num_layers=config["num_layers"],
        dropout=0.1,
        use_positional_encoding=config["use_positional_encoding"],
    ).to(DEVICE)

    param_count = count_parameters(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state_dict = None

    start_time = time.time()

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time

    if best_state_dict is None:
        raise RuntimeError(f"No best model state was saved for {config['name']}")

    # -----------------------------
    # Evaluate best model on test set
    # -----------------------------
    model.load_state_dict(best_state_dict)

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

    result = {
        "Model": config["name"],
        "PositionalEncoding": config["use_positional_encoding"],
        "Heads": config["num_heads"],
        "Layers": config["num_layers"],
        "Best_Epoch": best_epoch,
        "Val_Loss": round(best_val_loss, 4),
        "Val_Acc": round(best_val_acc, 4),
        "Test_Loss": round(test_loss, 4),
        "Test_Acc": round(test_acc, 4),
        "Params": param_count,
        #"Time_sec": round(total_time, 2),
        "Time_min": round(total_time / 60, 2),   
    }

    return result


def main():
    print("Starting Benchmark...\n")

    os.makedirs("results", exist_ok=True)

    results = []

    for config in MODEL_CONFIGS:
        result = run_experiment(config)
        results.append(result)

    df = pd.DataFrame(results)

    print("\n=== Benchmark Results ===")
    print(df)

    df.to_csv("results/benchmark_results.csv", index=False)

    print("\nResults saved to results/benchmark_results.csv")


if __name__ == "__main__":
    main()
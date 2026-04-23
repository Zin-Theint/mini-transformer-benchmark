import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    This ensures your results are consistent across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(logits, labels):
    """
    Compute binary classification accuracy.

    logits: (batch_size,)
    labels: (batch_size,)
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    correct = (preds == labels).sum().item()
    total = labels.size(0)

    return correct / total


def count_parameters(model):
    """
    Count number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """
    Convert time in seconds to a readable string.
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"
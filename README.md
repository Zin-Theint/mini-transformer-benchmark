# Mini Transformer Benchmark

## Project Overview

This project implements a **mini Transformer encoder from scratch using PyTorch** for a synthetic sequence classification task.

The goal is to analyze how different Transformer components affect performance, specifically:

- positional encoding
- number of attention heads
- number of encoder layers

---

## Task Description

Given a padded token sequence, the model must predict:

> Whether the **first non-padding token** appears again in the **second half** of the non-padding sequence.

### Example

Sequence:
[A, C, B, D, A, PAD, PAD]

- First token: A
- Non-padding sequence: [A, C, B, D, A]
- Second half: [D, A]
- Since A appears → label = 1

---

## Vocabulary

| Token | ID  |
| ----- | --- |
| PAD   | 0   |
| A     | 1   |
| B     | 2   |
| C     | 3   |
| D     | 4   |

---

## Sequence Format

- True sequence length: between 6 and 20
- All sequences are padded to a fixed length of 20
- Padding token = 0

---

## Project Structure

```
mini-transformer-benchmark/
├── data/
│ ├── test.csv
│ ├── train.csv
│ └── validation.csv
├── results/
│ ├── accuracy_curve.png
│ ├── benchmark_results.csv
│ ├── loss_curve.png
│ ├── train_summary.txt
│ └── training_history.txt
├── benchmark.py
├── data.py
├── model.py
├── train.py
├── plot_history.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Create a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Verify dataset loading

```bash
python data.py
```

---

### 2. Train a single model

```bash
python train.py
```

Outputs:

- Best model saved in `results/`
- Training history file
- Training summary file

---

### 3. Run benchmark experiments

```bash
python benchmark.py
```

Outputs:

- Comparison results across model configurations
- Saved to: `results/benchmark_results.csv`

---

## Model Configuration

The following configurations are evaluated:

| Model | Positional Encoding | Heads | Layers |
| ----- | ------------------- | ----- | ------ |
| A     | Yes                 | 1     | 1      |
| B     | Yes                 | 4     | 1      |
| C     | No                  | 4     | 1      |
| D     | Yes                 | 4     | 2      |

---

## Evaluation Metrics

- Validation Loss (used for model selection)
- Validation Accuracy
- Test Accuracy
- Training Time
- Parameter Count

---

## Model Architecture

The Transformer encoder is implemented manually using PyTorch:

- Token Embedding (`nn.Embedding`)
- Sinusoidal Positional Encoding
- Scaled Dot-Product Attention
- Multi-Head Self-Attention
- Feed-Forward Network
- Residual Connections + Layer Normalization
- Masked Mean Pooling
- Binary Classification Head

---

## Notes

- Padding tokens are handled using attention masks
- Masked mean pooling is used for sequence representation
- Model selection is based on validation loss

---

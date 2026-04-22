import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Vocabulary from assignment
VOCAB = {
    "PAD": 0,
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
}

PAD_ID = VOCAB["PAD"]
MAX_SEQ_LEN = 20


def get_token_columns():
    return [f"token_{i:02d}" for i in range(1, MAX_SEQ_LEN + 1)]


def get_mask_columns():
    return [f"mask_{i:02d}" for i in range(1, MAX_SEQ_LEN + 1)]


def compute_label_from_valid_tokens(valid_tokens):
    
    if len(valid_tokens) == 0:
        raise ValueError("Valid token sequence is empty.")

    length = len(valid_tokens)
    mid = length // 2

    first_token = valid_tokens[0]
    second_half = valid_tokens[mid:]

    return 1 if first_token in second_half else 0


def remove_padding(tokens, mask):
    return [tok for tok, m in zip(tokens, mask) if m == 1]

class MiniTransformerDataset(Dataset):
    def __init__(self, csv_path, verify_labels=False):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        self.token_cols = get_token_columns()
        self.mask_cols = get_mask_columns()

        required_columns = self.token_cols + self.mask_cols + ["label", "seq_len"]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        self.input_ids = torch.tensor(
            self.df[self.token_cols].values,
            dtype=torch.long
        )

        self.attention_mask = torch.tensor(
            self.df[self.mask_cols].values,
            dtype=torch.long
        )

        self.labels = torch.tensor(
            self.df["label"].values,
            dtype=torch.float32
        )

        self.seq_lens = torch.tensor(
            self.df["seq_len"].values,
            dtype=torch.long
        )

        if verify_labels:
            self.verify_labels()

    def verify_labels(self):
        
        mismatch_count = 0

        for idx in range(len(self.df)):
            tokens = self.input_ids[idx].tolist()
            mask = self.attention_mask[idx].tolist()
            actual_label = int(self.labels[idx].item())

            valid_tokens = remove_padding(tokens, mask)
            expected_label = compute_label_from_valid_tokens(valid_tokens)

            if expected_label != actual_label:
                mismatch_count += 1
                print(f"[Mismatch] Row {idx}: expected={expected_label}, actual={actual_label}")
                print(f"  tokens      = {tokens}")
                print(f"  mask        = {mask}")
                print(f"  valid_tokens= {valid_tokens}")

        if mismatch_count > 0:
            raise ValueError(f"Label verification failed: {mismatch_count} mismatches found.")

        print(f"Label verification passed for {self.csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],            # shape: (20,)
            "attention_mask": self.attention_mask[idx],  # shape: (20,)
            "label": self.labels[idx],                   # scalar float
            "seq_len": self.seq_lens[idx],               # scalar long
        }


def create_dataloader(csv_path, batch_size=32, shuffle=False, verify_labels=False):
    dataset = MiniTransformerDataset(
        csv_path=csv_path,
        verify_labels=verify_labels
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataset, loader


if __name__ == "__main__":
    train_path = "data/train.csv"
    val_path = "data/validation.csv"
    test_path = "data/test.csv"

    print("Loading train dataset...")
    train_dataset, train_loader = create_dataloader(
        train_path,
        batch_size=4,
        shuffle=True,
        verify_labels=True
    )

    print("Train dataset size:", len(train_dataset))

    first_sample = train_dataset[0]
    print("\nFirst sample:")
    print("input_ids:", first_sample["input_ids"])
    print("attention_mask:", first_sample["attention_mask"])
    print("label:", first_sample["label"])
    print("seq_len:", first_sample["seq_len"])

    first_batch = next(iter(train_loader))
    print("\nFirst batch shapes:")
    print("input_ids:", first_batch["input_ids"].shape)
    print("attention_mask:", first_batch["attention_mask"].shape)
    print("label:", first_batch["label"].shape)
    print("seq_len:", first_batch["seq_len"].shape)

    print("\nLoading validation dataset...")
    val_dataset, _ = create_dataloader(val_path, batch_size=32, shuffle=False, verify_labels=True)
    print("Validation dataset size:", len(val_dataset))

    print("\nLoading test dataset...")
    test_dataset, _ = create_dataloader(test_path, batch_size=32, shuffle=False, verify_labels=True)
    print("Test dataset size:", len(test_dataset))
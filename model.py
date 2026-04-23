import math
import torch
import torch.nn as nn

from data import create_dataloader

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Added to token embeddings so the model knows token order.
    """

    def __init__(self, d_model, max_seq_len=20, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)  # (seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """
    Computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, attention_mask=None):
        """
        q: (batch_size, num_heads, seq_len, head_dim)
        k: (batch_size, num_heads, seq_len, head_dim)
        v: (batch_size, num_heads, seq_len, head_dim)
        attention_mask: (batch_size, seq_len)
            1 = valid token
            0 = padding

        Returns:
            output: (batch_size, num_heads, seq_len, head_dim)
            attn_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)

        if attention_mask is not None:
            # Convert (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Manual implementation of multi-head self-attention.
    """

    def __init__(self, d_model=64, num_heads=4, dropout=0.1):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        x: (batch_size, seq_len, d_model)
        -> (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        x: (batch_size, num_heads, seq_len, head_dim)
        -> (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, x, attention_mask=None):
        """
        x: (batch_size, seq_len, d_model)
        attention_mask: (batch_size, seq_len)
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attn_output, attn_weights = self.attention(q, k, v, attention_mask)

        attn_output = self.combine_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output, attn_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network:
        Linear(d_model -> d_ff)
        ReLU
        Linear(d_ff -> d_model)
    """

    def __init__(self, d_model=64, d_ff=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    One Transformer encoder block:
        1) Multi-head self-attention + residual + layer norm
        2) Feed-forward + residual + layer norm
    """

    def __init__(self, d_model=64, num_heads=4, d_ff=128, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        attn_output, attn_weights = self.self_attention(x, attention_mask)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x, attn_weights


class MiniTransformerClassifier(nn.Module):
    """
    Mini Transformer classifier for binary sequence classification.

    Pipeline:
        input_ids -> embedding -> positional encoding (optional)
                 -> encoder blocks
                 -> masked mean pooling
                 -> classifier
    """

    def __init__(
        self,
        vocab_size=5,
        max_seq_len=20,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=1,
        dropout=0.1,
        use_positional_encoding=True,
    ):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0,
        )

        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=d_model,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Linear(d_model, 1)

    def masked_mean_pooling(self, x, attention_mask):
        """
        x: (batch_size, seq_len, d_model)
        attention_mask: (batch_size, seq_len)
        """
        mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        x = x * mask

        summed = x.sum(dim=1)  # (batch_size, d_model)
        lengths = mask.sum(dim=1).clamp(min=1e-9)  # avoid divide by zero

        pooled = summed / lengths
        return pooled

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len)

        Returns:
            logits: (batch_size,)
        """
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)

        if self.use_positional_encoding:
            x = self.positional_encoding(x)

        attn_weights_all = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, attention_mask)
            attn_weights_all.append(attn_weights)

        pooled = self.masked_mean_pooling(x, attention_mask)
        logits = self.classifier(pooled).squeeze(-1)

        return logits


if __name__ == "__main__":
    train_path = "data/train.csv"

    _, train_loader = create_dataloader(
        csv_path=train_path,
        batch_size=4,
        shuffle=True,
        verify_labels=False,
    )

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    model = MiniTransformerClassifier(
        vocab_size=5,
        max_seq_len=20,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=1,
        dropout=0.1,
        use_positional_encoding=True,
    )

    logits = model(input_ids, attention_mask)

    print("Input shape:", input_ids.shape)
    print("Attention mask shape:", attention_mask.shape)
    print("Logits shape:", logits.shape)
    print("Logits:", logits)
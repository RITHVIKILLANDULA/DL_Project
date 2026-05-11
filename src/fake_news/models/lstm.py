from __future__ import annotations



import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_id: int,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        _, (hidden, _) = self.encoder(emb)
        if self.encoder.bidirectional:
            x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            x = hidden[-1]
        x = self.dropout(x)
        return self.classifier(x)


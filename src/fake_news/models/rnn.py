from __future__ import annotations



import torch
from torch import nn


class RNNClassifier(nn.Module):
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
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            nonlinearity="tanh",
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        outputs, _ = self.encoder(emb)
        mask = (input_ids != self.pad_id).unsqueeze(-1).float()
        summed = (outputs * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        x = self.dropout(pooled)
        return self.classifier(x)

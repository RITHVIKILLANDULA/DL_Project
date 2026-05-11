from __future__ import annotations



from collections import Counter
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]


class SimpleTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


def build_vocab(texts: list[str], min_freq: int = 2) -> Vocab:
    counter: Counter[str] = Counter()
    tokenizer = SimpleTokenizer()
    for text in texts:
        counter.update(tokenizer.tokenize(text))

    tokens = [PAD_TOKEN, UNK_TOKEN]
    tokens.extend([tok for tok, cnt in counter.items() if cnt >= min_freq])
    stoi = {tok: idx for idx, tok in enumerate(tokens)}
    return Vocab(stoi=stoi, itos=tokens)


def encode_text(text: str, vocab: Vocab, max_len: int) -> list[int]:
    tokens = text.split()
    ids = [vocab.stoi.get(tok, vocab.unk_id) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab.pad_id] * (max_len - len(ids))
    return ids


class NewsDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len: int) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text_ids = encode_text(self.texts[idx], self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(text_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


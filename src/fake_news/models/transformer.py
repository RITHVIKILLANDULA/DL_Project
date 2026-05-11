from __future__ import annotations



from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_transformer(model_name: str = "distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model


# Fake News Detection (DL Project 2.5)

End-to-end deep learning pipeline for fake news detection using:
- LIAR (short political statements)
- FakeNewsNet (full news articles)

It includes preprocessing, train/val/test split, RNN/LSTM/Transformer training, evaluation, error analysis, and an optional Streamlit demo.

## 1) Project Structure

- `data/raw/`: input datasets
- `data/processed/`: generated train/val/test CSV files
- `models/`: trained checkpoints and metrics
- `reports/`: final evaluation artifacts
- `src/fake_news/`: library code (data, models, training, evaluation)
- `scripts/`: runnable CLI scripts
- `app/streamlit_app.py`: optional demo UI

## 2) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=%CD%\src
```

## 3) Data Preparation

Place files in `data/raw/` (or provide absolute paths):
- LIAR input can be either:
	- one TSV file, or
	- a folder containing `train.tsv`, `valid.tsv`, `test.tsv`
- FakeNewsNet input can be either:
	- one labeled CSV with label column (`label`/`target`/`class`/`is_fake`) and text-like column (`text`/`content`/`article`/`title`), or
	- a folder with the 4 split CSVs (`politifact_fake.csv`, `politifact_real.csv`, `gossipcop_fake.csv`, `gossipcop_real.csv`), or
	- a folder containing downloaded article files named `news content.json` (or `news_content.json`)

Run:

```bash
python scripts/prepare_data.py --liar-path liar_dataset --fakenewsnet-path data/raw/fakenewsnet --out-dir data/processed
```

Outputs:
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## 4) Train RNN/LSTM

RNN:

```bash
python scripts/train_classic.py --model rnn --train data/processed/train.csv --val data/processed/val.csv --epochs 8
```

LSTM:

```bash
python scripts/train_classic.py --model lstm --train data/processed/train.csv --val data/processed/val.csv --epochs 8
```

Artifacts:
- `models/best_rnn.pt` or `models/best_lstm.pt`
- `models/metrics_rnn.json` or `models/metrics_lstm.json`

## 5) Train Transformer

```bash
python scripts/train_transformer.py --train data/processed/train.csv --val data/processed/val.csv --model-name distilbert-base-uncased --epochs 3
```

Artifacts:
- `models/best_transformer/`
- `models/metrics_transformer.json`

## 6) Generate Predictions + Evaluate

For classic model checkpoint:

```bash
python scripts/predict_classic.py --checkpoint models/best_lstm.pt --input-csv data/processed/test.csv --out reports/predictions.csv
python scripts/evaluate_predictions.py --test data/processed/test.csv --predictions reports/predictions.csv --out-dir reports
```

Outputs:
- `reports/metrics_test.json`
- `reports/false_positives.csv`
- `reports/false_negatives.csv`

## 7) Optional Demo App

```bash
streamlit run app/streamlit_app.py
```

Use a trained checkpoint path (for example: `models/best_lstm.pt`) and paste text to classify.

## 8) Recommended Report Checklist

- Dataset summary + label mapping policy
- Preprocessing choices
- Model architectures and hyperparameters
- Metrics table (Accuracy, Precision, Recall, F1)
- Confusion matrix
- Error analysis (FP/FN examples and patterns)
- Comparison: RNN vs LSTM vs Transformer
- Limitations and next improvements

## 9) Build Final Report

Generate a consolidated comparison and error-analysis report from the saved predictions and metrics:

```bash
python scripts/build_final_report.py
```

Artifacts:
- `reports/final_report.md`
- `reports/final_report.json`
- `reports/false_positives_rnn.csv`
- `reports/false_negatives_rnn.csv`
- `reports/false_positives_lstm.csv`
- `reports/false_negatives_lstm.csv`
- `reports/false_positives_transformer.csv`
- `reports/false_negatives_transformer.csv`

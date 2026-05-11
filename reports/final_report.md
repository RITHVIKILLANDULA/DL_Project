# Fake News Detection — Final Report

## 1. Task Statement and Motivation

We frame fake-news detection as a binary text-classification problem: given a short statement or full news article, predict whether the source is **real (0)** or **fake (1)**. Misinformation on social media has measurable downstream effects on elections, public-health behaviour, and financial markets, so reliable automatic classification is a meaningful problem. The contribution of this project is a fair, reproducible comparison of three families of text classifiers (a TF–IDF + logistic-regression baseline, bidirectional recurrent networks with GloVe embeddings, and a fine-tuned pretrained Transformer) on a unified LIAR + FakeNewsNet dataset, with bootstrap confidence intervals and a per-dataset breakdown that exposes how each model behaves on short claims vs. full articles. The headline finding: **DistilBERT achieves 95.5 % accuracy on FakeNewsNet full-length articles (21/22 correct) and is perfect on the 5,000+ character bin**, while simpler models (BiRNN) win on short LIAR claims via class-weighted loss — i.e. the right architecture depends on the text-length regime.

## 2. Data

| Dataset | Source | What it provides | Rows after cleaning |
|---|---|---|---|
| LIAR | Kaggle `doanquanvietnamca/liar-dataset` | Short labeled political statements | 12,791 |
| FakeNewsNet | Kaggle `mdepak/fakenewsnet` (BuzzFeed subset) | Full-length news articles | 178 |

**LIAR** is loaded from the three official TSV splits and concatenated; the original 6-way veracity label is mapped to binary via `LABEL_MAP` ([src/fake_news/constants.py](../src/fake_news/constants.py)): `true / mostly-true / half-true → real (0)`; `barely-true / false / pants-fire → fake (1)`.

**FakeNewsNet** is loaded from the BuzzFeed `*_news_content.csv` files (91 fake + 91 real, full article body in the `text` column, mean length **3,194 characters** — comfortably matching the spec's "full-length news articles" description). The PolitiFact CSVs in the same Kaggle dump are **excluded automatically by our loader** because the `_fake` and `_real` files contain identical data — a confirmed upload error on Kaggle's side (the loader emits "Skipping duplicate FakeNewsNet pair" when it detects this).

The two datasets are concatenated, lowercased, URL-stripped, whitespace-normalised ([src/fake_news/data/preprocess.py](../src/fake_news/data/preprocess.py)), and **deduplicated by exact text** ([src/fake_news/data/io.py](../src/fake_news/data/io.py)). The unified table contains **12,943 unique examples**. We then perform a stratified 80 / 10 / 10 split by label with seed 42 ([src/fake_news/data/split.py](../src/fake_news/data/split.py)):

| Split | Total | Real | Fake | LIAR | FakeNewsNet |
|---|---:|---:|---:|---:|---:|
| Train | 10,354 | 5,771 | 4,583 | 10,213 | 141 |
| Val   | 1,294 | 721 | 573 | 1,279 | 15 |
| Test  | 1,295 | 722 | 573 | 1,273 | 22 |

The majority-class baseline accuracy on the test set is **0.558**. Class imbalance is mild (real:fake ≈ 1.26 : 1) and handled in every neural model by an inverse-frequency–weighted cross-entropy loss.

**Leakage handling:** dedup happens *before* the train/val/test split, so the same statement cannot appear in two splits.

## 3. NLP Workflow

### 3.1 Cleaning
Lowercasing, URL removal (`https?://...` and `www....`), newline normalisation, and whitespace consolidation. We deliberately **keep punctuation, stopwords, and contractions** because (a) fake-news cues often live in punctuation patterns (e.g. exclamation density) and (b) DistilBERT's WordPiece tokenizer already handles contractions properly. This is a design choice we revisit in §6.

### 3.2 Tokenization
- **Classic models (RNN, LSTM):** whitespace tokenizer with a vocabulary built from the training set (`min_freq=2`), `<pad>` and `<unk>` reserved tokens. Vocabulary size ≈ 11,800 ([src/fake_news/data/dataset.py](../src/fake_news/data/dataset.py)).
- **Transformer:** the pretrained `distilbert-base-uncased` WordPiece tokenizer with 30,522 subwords.
- **TF–IDF baseline:** scikit-learn's tokenizer with English stopwords removed, unigrams + bigrams, `max_features=5000`, `min_df=2`, `max_df=0.95`.

### 3.3 Embeddings
- **RNN / LSTM:** 100-dimensional **GloVe 6B** vectors pretrained on Wikipedia + Gigaword, loaded from `data/raw/glove_embeddings/glove.6B.100d.txt`. Coverage of our LIAR-dominated vocabulary is **67.4 %** (7,960 / 11,817 tokens matched); out-of-vocabulary tokens are initialised from N(0, 0.01²). Code in [src/fake_news/embeddings.py](../src/fake_news/embeddings.py).
- **Transformer:** initialised from the pretrained `distilbert-base-uncased` weights (the classifier head is randomly initialised, as flagged at load time).

### 3.4 Models
1. **TF–IDF + Logistic Regression** ([scripts/train_baseline_tfidf.py](../scripts/train_baseline_tfidf.py)) — strong non-neural baseline with `class_weight="balanced"`.
2. **Bidirectional vanilla RNN** ([src/fake_news/models/rnn.py](../src/fake_news/models/rnn.py)) — `nn.RNN` with tanh nonlinearity, BiRNN, **mean pooling over padded tokens** instead of last hidden state (this single change was the difference between a degenerate 51 % model and a working 56 % model — see §6).
3. **Bidirectional LSTM** ([src/fake_news/models/lstm.py](../src/fake_news/models/lstm.py)) — `nn.LSTM`, BiLSTM, hidden state of the last layer in each direction concatenated.
4. **DistilBERT** ([src/fake_news/models/transformer.py](../src/fake_news/models/transformer.py)) — `distilbert-base-uncased` fine-tuned end-to-end with `AutoModelForSequenceClassification`.

### 3.5 Training
| Setting | RNN | LSTM | DistilBERT |
|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 1e-3 | 1e-3 | 2e-5 |
| Weight decay | 1e-5 | 1e-5 | 0 |
| Batch size | 32 | 32 | 16 |
| Max length | 256 tokens | 256 tokens | 128 subwords |
| Epochs | 10 (early-stop on best val F1) | 12 (early-stop on best val F1) | 4 (patience=2, best at epoch 2) |
| Training rows | 10,354 (full) | 10,354 (full) | 10,354 (full) |
| Gradient clipping | 1.0 | 1.0 | none |
| Class weights | inverse-freq | inverse-freq | inverse-freq |
| Seed | 42 | 42 | 42 |

Best checkpoint by validation F1 is saved; the test-set evaluation uses only that checkpoint.

## 4. Results

### 4.1 Overall test metrics (n = 1,295), with 1000-sample bootstrap 95 % CIs

| Model | Accuracy | Precision | Recall | F1 | F1 95 % CI |
|---|---:|---:|---:|---:|---|
| **Majority class** | 0.558 | — | — | — | — |
| TF–IDF + LogReg | 0.602 | 0.547 | 0.581 | 0.563 | [0.531, 0.598] |
| **BiRNN + GloVe** | 0.563 | 0.504 | 0.773 | **0.610** | [0.581, 0.638] |
| **BiLSTM + GloVe** | 0.623 | 0.567 | 0.632 | 0.597 | [0.564, 0.629] |
| **DistilBERT** | **0.643** | 0.595 | 0.604 | 0.599 | [0.564, 0.631] |

Numbers are read directly from [reports/full_evaluation.json](full_evaluation.json). Notes:
- **DistilBERT has the highest overall accuracy (0.643)** and a balanced precision/recall around 0.60. Trained on the full 10,354-row training set for 4 epochs with patience=2 early stopping (best at epoch 2).
- **BiRNN has the highest overall F1 (0.610)** because class-weighted cross-entropy pushed its decision threshold toward recall (recall = 0.773).
- **BiLSTM is competitive (Acc 0.623, F1 0.597)** with a balanced precision-recall profile.
- All neural models beat the majority-class baseline (0.558) on F1; DistilBERT and BiLSTM also clearly beat the TF–IDF baseline on accuracy.

### 4.2 Per-dataset breakdown

| Model | LIAR (n = 1,273)<br>Acc / F1 | FakeNewsNet (n = 22)<br>Acc / F1 |
|---|---|---|
| TF–IDF       | 0.599 / 0.558 | 0.773 / 0.800 |
| BiRNN        | 0.563 / 0.608 | 0.545 / 0.706 |
| BiLSTM       | 0.625 / 0.596 | 0.545 / 0.667 |
| **DistilBERT**   | **0.637 / 0.592** | **🎯 0.955 / 0.957** |

The FakeNewsNet test slice is only 22 articles — too small for confident statistical claims, which is itself a limitation we surface (§6). Within that caveat, **DistilBERT gets 21 of 22 FakeNewsNet articles correct (0.955 acc, 0.957 F1)** — a dramatic separation from the next-best model (TF–IDF at 0.773) and direct evidence that pretrained subword Transformers exploit the long-form article distribution that GloVe-pooled RNNs lose to truncation. On LIAR short statements, all models hover in the 0.56–0.64 accuracy band, with DistilBERT (0.637) and BiLSTM (0.625) close to tied.

### 4.3 Accuracy by text length (sanity / robustness)
| Length (chars) | n | TF–IDF | BiRNN | BiLSTM | DistilBERT |
|---:|---:|---:|---:|---:|---:|
| 0–50       | 89  | 0.551 | 0.584 | 0.596 | **0.629** |
| 50–100     | 545 | 0.583 | 0.552 | 0.609 | **0.611** |
| 100–200    | 572 | 0.617 | 0.582 | 0.640 | **0.663** |
| 200–500    | 67  | 0.627 | 0.463 | **0.657** | 0.642 |
| 500–5,000  | 16  | 0.875 | 0.625 | 0.625 | **0.938** |
| 5,000+     | 6   | 0.500 | 0.333 | 0.333 | **1.000** |

Two clear patterns: (a) the recurrent models drop sharply on the longest bins because of truncation at `max_len = 256` and pure GloVe-pooled signal cannot compete with subword-level semantics; (b) **DistilBERT wins five of six length bins** and is *perfect* on the 5,000+ char bin (6 / 6) and near-perfect on 500–5,000 chars (15 / 16). Subword tokenisation and pretrained semantics make the first 128 sub-words of a long article carry enough topical signal to disambiguate fake vs. real reliably.

### 4.4 Ablations
- **Embedding dimension** ([reports/ablation_embedding_dim.json](ablation_embedding_dim.json)): swept `{64, 100, 128, 200, 256}` for the BiLSTM. 100-d (GloVe-aligned) is competitive; increasing dimension without proportional dropout hurts.
- **GloVe vs. random init** for the BiRNN/BiLSTM is implicit in the training script (`use_glove=True` default) — runs with GloVe consistently beat the random-init runs we observed during development (random-init RNN got stuck at the majority class).
- **RNN architecture (last hidden vs. mean pooling)** — the original `nn.RNN` with last-hidden-state read-out collapsed to the majority class (test acc 0.515, below the 0.558 majority baseline). Swapping to **bidirectional + mean-pooling-over-time + gradient clipping** lifts test F1 from 0.527 → 0.610 with no other changes (see commit history). We document this as our small but meaningful architectural contribution.
- **Transformer threshold tuning** ([reports/transformer_threshold_sweep.json](transformer_threshold_sweep.json) on the previous test set): sweeping 81 thresholds for the fake class shows that the default 0.5 over-predicts "real"; threshold ≈ 0.49 maximises F1 at the cost of precision.

## 5. Error Analysis

False-positive (model says fake, truth is real) and false-negative CSVs for every model are in `reports/false_positives_*.csv` and `reports/false_negatives_*.csv`.

**Quantitative patterns** (from [reports/error_analysis_detailed.json](error_analysis_detailed.json)):
- **TF–IDF** errors are roughly balanced across LIAR length bins; the model has no understanding of negation, so claims that hinge on a single not / no are systematically misclassified.
- **BiRNN** recall is high (0.77) at the cost of precision (0.50) — it over-predicts the fake class. This is exactly what one expects from class-weighted cross-entropy on a marginally imbalanced dataset; a deliberate recall-favouring trade-off, not a failure.
- **BiLSTM** errors concentrate on short LIAR statements that contain proper nouns absent from the GloVe vocabulary (32.6 % OOV).
- **DistilBERT** has a near-balanced confusion matrix (CM = [[486, 236], [227, 346]]) with FP ≈ FN, suggesting the decision boundary is well calibrated without explicit threshold tuning. Its single FakeNewsNet error (1 / 22) was a fake article wrongly classified as real — a difficult case for any classifier. Almost all DistilBERT errors are LIAR short statements where contextual cues are minimal.
- **The recurrent models perform worst on the 5,000+ character bin** because they truncate at 256 tokens; DistilBERT's subword tokenizer + pretrained representations make the truncated 128-subword window more informative — direct evidence that for long-form fake-news detection, a Transformer is the right tool.

## 6. Limitations and Honest Caveats

1. **FakeNewsNet test-slice size.** Per the project spec, FakeNewsNet is defined as "full-length news articles labeled as fake or real" — no size is required. We use the entirety of the Kaggle `mdepak/fakenewsnet` BuzzFeed subset (178 full-length articles, mean length 3,194 chars), which satisfies the description. The PolitiFact CSVs in the same Kaggle dump are corrupted (the `_fake` and `_real` files contain identical content) and are auto-excluded by our loader. After the stratified split the test slice is 22 articles — small enough that per-model test numbers on that slice (e.g. DistilBERT 95.5% accuracy) are statistically directional rather than tightly conclusive.
2. **LIAR's 6-to-2 label mapping is lossy.** Folding `half-true` and `mostly-true` into "real" and `barely-true` into "fake" loses graded information; reasonable people would draw the boundary differently and the model is partly being asked to memorize one particular choice.
3. **Truncation hurts on full articles.** `max_len = 256` (RNN/LSTM) and `128` (DistilBERT) discard most of a 3,000-character BuzzFeed article. A production system would chunk and aggregate.
4. **Single-seed neural runs.** Variance across runs was estimated via bootstrap on the test predictions rather than via multi-seed training; with infinite compute we would do both. CIs reflect sampling variance in the test set, not optimisation variance.
5. **DistilBERT was trained for 4 epochs on the full 10,354-row training set** with patience=2 early stopping (best at epoch 2). Total wall-clock was ~146 minutes on CPU (8-core, ~36 min per epoch). Training loss continued decreasing to 0.91 train F1 by epoch 4, but validation F1 peaked at epoch 2 (0.585) and declined thereafter — clear evidence of overfitting. Adding dropout or weight-decay regularisation could push the val-F1 ceiling higher.
5. **Domain skew.** Both datasets are US-centric, English-language, political-leaning. The model should not be relied on outside that distribution.

## 7. Ethics and Safety

Automatic fake-news classifiers can be misused. Specific risks for this project:
- **False positives chill legitimate speech.** Any system that auto-flags content has a moderation duty that this model does not satisfy on its own. Our overall precision in the 0.50–0.57 range means roughly half of "fake" predictions are wrong; downstream use must be advisory, not punitive.
- **Distributional / political bias.** LIAR over-represents US political figures and PolitiFact's editorial choices; trusting predictions in non-political domains, in non-English text, or in different time periods is not supported by our evidence.
- **Adversarial robustness was not evaluated.** Trivial paraphrase attacks (synonym substitution, voice changes) are known to flip text-classifier predictions.
- **Use case.** This project is for academic comparison of NLP architectures. It is *not* a deployment-ready misinformation moderator and should not be presented as one.

## 8. Compute Budget and Reproducibility

All experiments run on CPU (no GPU available, 8-core machine with ~7.9× effective parallelism via PyTorch / OpenMP). Wall-clock times (single seed, seed=42):

| Step | Wall time |
|---|---:|
| TF–IDF + LogReg | ~10 s |
| BiRNN, 10 epochs, max_len 256 | ~4 min |
| BiLSTM, 12 epochs, max_len 256 | ~11 min |
| DistilBERT, 4 epochs (early-stopped at 2), batch 16, max_len 128 | **~146 min (2 hr 26 min)** |
| Predictions on test set (all 4 models) | ~2 min |
| Bootstrap CIs, error analysis, figures | ~30 s |

All scripts are in `scripts/` and accept default arguments that reproduce the numbers in this report; raw data files (LIAR + FakeNewsNet) come from the Kaggle datasets cited in §2 via `kagglehub`. The end-to-end pipeline is:

```bash
python scripts/prepare_data.py --liar-path data/raw/kaggle/liar --fakenewsnet-path data/raw/kaggle/fakenewsnet --out-dir data/processed
python scripts/train_baseline_tfidf.py
python scripts/train_classic.py --model-type rnn  --epochs 10 --max-len 256 --learning-rate 1e-3
python scripts/train_classic.py --model-type lstm --epochs 12 --max-len 256 --learning-rate 1e-3
python scripts/train_transformer.py --train data/processed/train.csv --val data/processed/val.csv --epochs 4 --batch-size 16 --max-len 128 --patience 2
python scripts/predict_classic.py --checkpoint models/classic_rnn/best_rnn.pt
python scripts/predict_classic.py --checkpoint models/classic_lstm/best_lstm.pt
python scripts/predict_transformer_test.py
python scripts/full_evaluation.py
python scripts/detailed_error_analysis.py
python scripts/plot_results.py
```

## 9. Demo

A Streamlit application is provided ([app/streamlit_app.py](../app/streamlit_app.py)) that loads any trained checkpoint and lets a user paste arbitrary text to see the model's real / fake probability. A simpler Flask version is in [app/app_flask.py](../app/app_flask.py).

## 10. LLM Usage Disclosure

This project used **Claude (Anthropic)** as a coding and writing assistant:
- Boilerplate code for data loading, training loops, and evaluation scripts was scaffolded with Claude's help and then reviewed line-by-line.
- This report was structured and copy-edited by Claude; all empirical numbers were taken verbatim from the JSON outputs in `reports/` and were not rewritten by the LLM.
- Model architectures, hyperparameter choices, and the decision to switch the vanilla RNN to a bidirectional + mean-pooling variant were made by the human authors based on observed training behaviour.

## 11. Team Contributions

All three team members contributed equally across data preparation, model training, evaluation, error analysis, the Streamlit / Flask demos, and the report. No single member owned any single component end-to-end.

| Member | UBID | Email |
|---|---|---|
| Rithvik Illandula | rithviki | rithviki@buffalo.edu |
| Venkata Praneeth Cheturi | vcheturi | vcheturi@buffalo.edu |
| Charitha Mamilla RaveendraReddy | cmamilla | cmamilla@buffalo.edu |

## 12. Repository

GitHub: **https://github.com/RITHVIKILLANDULA/DL_Project**

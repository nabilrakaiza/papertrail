# papertrail

Comparing n-gram, RNN, LSTM, and Transformer language models on WikiText-2, plus an embedding ablation (random vs. self-trained vs. GloVe) and a downstream topic-classification probe on AG News.

Every model here is trained from scratch, following the trail of classic papers from Kneser-Ney smoothing to causal self-attention.

## Questions

1. **Model families (Part I):** On the same corpus, vocabulary, and test set, how do an n-gram, RNN, LSTM, and Transformer LM trade off prediction quality, training cost, inference cost, and generation quality?
2. **Embeddings (Part II):** With the LSTM fixed, does a frozen input embedding (self-trained skip-gram or public GloVe) help or hurt compared to learning it jointly?
3. **Representations (Part III):** Does better next-token prediction translate into better frozen features for topic classification?

## Status

| Part | Item | Status |
|---|---|---|
| I | Interpolated Kneser-Ney trigram | todo |
| I | RNN LM | todo |
| I | LSTM LM | todo |
| I | Transformer LM | todo |
| I | Timing benchmark + generations | todo |
| II | Skip-gram embeddings (own) | todo |
| II | LSTM + frozen skip-gram | todo |
| II | LSTM + frozen GloVe | todo |
| III | Feature extraction + logistic regression | todo |

## Repository layout

```
papertrail/
├── configs/
│   ├── part1/          # ngram, rnn, lstm, transformer
│   ├── part2/          # word2vec, lstm_w2v_frozen, lstm_glove_frozen
│   └── part3/          # probe (AG News)
├── papertrail/         # source package (data, models, train, evaluate, generate, ...)
├── scripts/            # one-off helpers (GloVe download)
├── results/            # metrics JSON/CSV and figures (tracked)
├── report/             # PDF report
├── data/               # downloaded datasets (gitignored)
├── embeddings/         # GloVe + own skip-gram vectors (gitignored)
└── runs/               # checkpoints and logs (gitignored)
```

## Setup

Tested with Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

All data is downloaded by the scripts. Nothing from the datasets is committed to this repo.

### WikiText-2 (Parts I and II)

- Source: Hugging Face `Salesforce/wikitext`, config `wikitext-2-v1` (the pre-tokenized version, not `-raw`).
- Splits: the official train / validation / test splits, unchanged.

Preprocessing, applied identically for every model:

1. Lowercase all text (needed to match GloVe, which is uncased).
2. Split on whitespace. The dataset is already tokenized; rare words (count < 3 in the original release) are already `<unk>`.
3. Keep section headings (e.g. `= = gameplay = =`) as ordinary text.
4. Drop empty lines. Append `<eos>` to the end of every non-empty line.
5. Build the vocabulary from the **train split only**. Any val/test token not in it maps to `<unk>`.
6. Concatenate each split into one token stream and cut it into contiguous chunks of `context_length` tokens for the neural models.

Special tokens: `<pad>`, `<unk>`, `<eos>`. The WikiText number/hyphen markers (`@-@`, `@,@`, `@.@`) are kept as ordinary tokens.

### AG News (Part III)

- Source: Hugging Face `fancyzhx/ag_news` (4 classes: World, Sports, Business, Sci/Tech).
- Subset: 10,000 training examples (2,500 per class, stratified), 2,000 validation examples carved from the remaining train split, and the full official test split (7,600).
- Text is lowercased and tokenized with the same rules as WikiText (punctuation split, hyphens and number separators mapped to `@-@`, `@,@`, `@.@`), then mapped to the **WikiText-2 vocabulary**. Unknown words become `<unk>`; the `<unk>` rate is reported.
- Sequences are truncated to 128 tokens and padded with `<pad>`.

### GloVe (Part II)

- `glove.6B.200d` (Wikipedia 2014 + Gigaword 5, 6B tokens, 400k uncased vocab).
- Download: `bash scripts/download_glove.sh` (saves to `embeddings/glove.6B.200d.txt`).

## Configurations

Each run is fully described by one YAML file. Shared settings live in `configs/base.yaml` and are overridden per run.

| Config | Part | What it trains |
|---|---|---|
| `configs/part1/ngram.yaml` | I | Interpolated Kneser-Ney trigram |
| `configs/part1/rnn.yaml` | I | Elman RNN LM, random trainable embeddings |
| `configs/part1/lstm.yaml` | I, II | LSTM LM, random trainable embeddings (also Part II setting 1) |
| `configs/part1/transformer.yaml` | I | Causal Transformer LM, random trainable embeddings |
| `configs/part2/word2vec.yaml` | II | Skip-gram vectors on the WikiText-2 train split |
| `configs/part2/lstm_w2v_frozen.yaml` | II | LSTM LM with frozen own skip-gram embeddings |
| `configs/part2/lstm_glove_frozen.yaml` | II | LSTM LM with frozen GloVe embeddings |
| `configs/part3/probe.yaml` | III | Frozen-LM features + logistic regression on AG News |

Design choices that keep the comparisons fair:

- **Same vocab, splits, and evaluation function** for all four Part I models.
- **Embedding dimension 200** for every neural model, chosen to match `glove.6B.200d`, so the Part I LSTM is reused as the "random init" setting in Part II.
- **Input and output embeddings are not tied.** Tying would freeze the output layer too in the frozen-embedding runs.
- In the frozen settings, the **entire** input embedding table is frozen, including rows for OOV and special tokens.

## Reproducing the results

Planned commands (the interface each script will expose):

```bash
# Part I
python -m papertrail.ngram     --config configs/part1/ngram.yaml
python -m papertrail.train     --config configs/part1/rnn.yaml
python -m papertrail.train     --config configs/part1/lstm.yaml
python -m papertrail.train     --config configs/part1/transformer.yaml
python -m papertrail.benchmark --configs configs/part1/*.yaml     # timing + model size
python -m papertrail.generate  --configs configs/part1/*.yaml     # matched-decoding samples

# Part II
bash scripts/download_glove.sh
python -m papertrail.word2vec  --config configs/part2/word2vec.yaml   # timed separately
python -m papertrail.train     --config configs/part2/lstm_w2v_frozen.yaml
python -m papertrail.train     --config configs/part2/lstm_glove_frozen.yaml

# Part III
python -m papertrail.probe     --config configs/part3/probe.yaml

# Figures and tables for the report
python -m papertrail.report_figs
```

Outputs:

- Checkpoints and per-epoch logs: `runs/<run_name>/`
- Final metrics: `results/<run_name>.json`
- Figures: `results/figures/`

## Evaluation protocol

**Prediction quality.** Test perplexity = `exp(mean next-token cross-entropy)` over every target token in the test stream, using the same vocabulary and the same token stream for all models. `<pad>` is excluded from the loss via `ignore_index` (with contiguous chunking there is no padding in the LM loss, but the guard is kept).

**Context.** Every model predicts each target from preceding tokens only. For a fair context comparison, RNN/LSTM hidden states are **reset at every chunk** during evaluation, so all neural models see at most `context_length` tokens of history. The n-gram sees 2 tokens of history.

**Training cost.** Wall-clock training time to the best validation checkpoint and in total, number of epochs, and trainable / total parameter counts (embedding parameters reported separately).

**Inference cost.** One common workload: a full forward pass over the test stream at batch size 32 and `context_length` 64. Measured after 10 warm-up batches, with `torch.cuda.synchronize()` around the timed region, reported as tokens per second (mean of 3 repeats).

**Learning behaviour.** Train and validation loss per epoch for every neural run, plotted on shared axes.

**Generation.** The same 5 prompts for every model, temperature 0.8, top-k 40, 50 new tokens, seed 42. The n-gram uses the same sampling procedure.

**Part III.** Two fixed-length representations from frozen LMs: mean of token hidden states excluding `<pad>`, and the last non-pad hidden state. Classifier: multinomial logistic regression (scikit-learn), regularization `C` chosen on the validation subset, test set evaluated once. Metrics: accuracy and macro-F1.

## Random seeds and determinism

- Global seed: `42` (Python, NumPy, PyTorch, CUDA), set in every entry point.
- `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
- Skip-gram: `gensim` with `workers=1` and `PYTHONHASHSEED=42` (gensim is only reproducible single-threaded).
- AG News subsampling and OOV embedding initialization use the same seed.
- All main results are from a single seed because of the compute budget. This is stated as a limitation in the report.

## Hardware

To be filled in with the exact GPU, CPU, and library versions used for the reported numbers.

| Item | Value |
|---|---|
| GPU | TBD |
| CPU | TBD |
| PyTorch / CUDA | TBD |

## Results

To be filled in from `results/`.

| Model | Params | Test PPL | Train time | Inference (tok/s) |
|---|---|---|---|---|
| KN trigram | | | | |
| RNN | | | | |
| LSTM | | | | |
| Transformer | | | | |

## References and attribution

- WikiText-2: Merity, Xiong, Bradbury, Socher. *Pointer Sentinel Mixture Models.* 2016. (CC BY-SA 3.0)
- AG News: Zhang, Zhao, LeCun. *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015.
- GloVe: Pennington, Socher, Manning. *GloVe: Global Vectors for Word Representation.* EMNLP 2014.
- Skip-gram: Mikolov, Chen, Corrado, Dean. *Efficient Estimation of Word Representations in Vector Space.* 2013. Implemented with gensim (Řehůřek and Sojka, 2010).
- Kneser-Ney smoothing: Kneser and Ney, 1995; Chen and Goodman, 1999.
- RNN LM: Mikolov et al., 2010. LSTM: Hochreiter and Schmidhuber, 1997. Transformer: Vaswani et al., 2017.
- Course notebooks adapted where noted in the source files.

## Tool use

AI coding assistants were used for repository scaffolding and code review. All experiments were run, and all analysis and interpretation in the report were written, by the author. Specific assisted components are noted in the report.

## License

Code: MIT (see `LICENSE`). Datasets and pretrained embeddings remain under their original licenses and are not redistributed here.

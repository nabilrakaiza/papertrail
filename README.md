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
| I | Data pipeline (WikiText-2, vocabulary, chunking) | done |
| I | Interpolated Kneser-Ney trigram | done, results in `results/ngram_kn3.json` |
| I | RNN LM | code done, not trained yet |
| I | LSTM LM | code done, not trained yet |
| I | Transformer LM | code done, not trained yet |
| I | Training loop and evaluation | done |
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
├── papertrail/         # source package
│   ├── config.py       # YAML loading with `inherit`
│   ├── data.py         # WikiText-2 -> tokens -> vocabulary -> chunks
│   ├── ngram.py        # interpolated Kneser-Ney trigram (fit, evaluate)
│   ├── rnn.py          # RNN and LSTM language models
│   ├── transformer.py  # decoder-only Transformer language model
│   ├── models.py       # build a neural model from a config, count parameters
│   ├── train.py        # training loop for the neural models
│   ├── evaluate.py     # loss / perplexity, shared by all neural models
│   └── utils.py        # seeding, device choice, timing helpers
├── scripts/            # one-off helpers (GloVe download)
├── results/            # metrics JSON/CSV and figures (tracked)
├── report/             # PDF report
├── data/               # downloaded datasets (gitignored)
├── embeddings/         # GloVe + own skip-gram vectors (gitignored)
└── runs/               # checkpoints and logs (gitignored)
```

## Setup

Developed and tested with Python 3.13 and PyTorch 2.9.

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

Resulting corpus (`python -m papertrail.data configs/base.yaml`), vocabulary 28,913 tokens including the three special tokens:

| Split | Tokens | Chunks of 64 | `<unk>` rate |
|---|---|---|---|
| train | 2,075,677 | 32,433 | 2.63% |
| validation | 216,347 | 3,381 | 5.42% |
| test | 244,102 | 3,815 | 6.23% |

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

### Models

| Model | Architecture | Parameters (of which input embedding) |
|---|---|---|
| KN trigram | Interpolated Kneser-Ney, order 3, one fixed discount of 0.75 | 1,377,988 distinct trigrams, 586,138 distinct bigrams |
| RNN | 2-layer Elman RNN (tanh), hidden size 512, dropout 0.3 | 21,505,849 (5,782,600) |
| LSTM | 2-layer LSTM, hidden size 512, dropout 0.3 | 24,178,489 (5,782,600) |
| Transformer | Decoder-only, 4 blocks, width 200, 4 heads, feed-forward 800, dropout 0.2 | 13,524,513 (5,782,600) |

The Transformer is post-LN with ReLU, fixed sinusoidal positions and the `sqrt(d_model)` embedding scale, as in Vaswani et al. (2017), and is initialized at `normal(std=0.02)` as in GPT-1. In every neural model most of the parameters are the output layer (hidden size x 28,913), because input and output embeddings are not tied.

Design choices that keep the comparisons fair:

- **Same vocab, splits, and evaluation function** for all four Part I models.
- **Embedding dimension 200** for every neural model, chosen to match `glove.6B.200d`, so the Part I LSTM is reused as the "random init" setting in Part II.
- **Input and output embeddings are not tied.** Tying would freeze the output layer too in the frozen-embedding runs.
- In the frozen settings, the **entire** input embedding table is frozen, including rows for OOV and special tokens.

## Reproducing the results

Run everything from the repository root. The first run downloads WikiText-2 into `data/`.

Working now:

```bash
# Part I
python -m papertrail.ngram     --config configs/part1/ngram.yaml
python -m papertrail.train     --config configs/part1/rnn.yaml
python -m papertrail.train     --config configs/part1/lstm.yaml
python -m papertrail.train     --config configs/part1/transformer.yaml

# Inspect a resolved config, or the corpus statistics
python -m papertrail.config    configs/part2/lstm_glove_frozen.yaml
python -m papertrail.data      configs/base.yaml
```

Planned, not written yet (the interface each script will expose):

```bash
# Part I
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

- Best checkpoint (lowest validation loss): `runs/<run_name>/best.pt`
- Train and validation loss and cumulative training time per epoch: `runs/<run_name>/log.json`
- Final metrics, measured with the best checkpoint: `results/<run_name>.json`
- Figures: `results/figures/`

## Evaluation protocol

**Prediction quality.** Test perplexity = `exp(mean next-token cross-entropy)` over every target token in the test stream, using the same vocabulary and the same token stream for all models. The targets are every token of a split except its first (244,101 in test), for the n-gram and the neural models alike. `<pad>` is excluded from the loss via `ignore_index`; the only padding is in the final chunk of each split, which is filled up to `context_length`.

**Context.** Every model predicts each target from preceding tokens only. For a fair context comparison, RNN/LSTM hidden states are **reset at every chunk** during evaluation, so all neural models see at most `context_length` tokens of history. The n-gram sees 2 tokens of history.

**Training procedure.** The same for every neural model: AdamW (weight decay 0.01), batch size 32, gradient-norm clipping at 1.0, at most 20 epochs, early stopping after 3 epochs without improvement in validation loss, and the checkpoint with the lowest validation loss is the one evaluated. Batches are read in corpus order without shuffling, so that the RNN and LSTM can carry their hidden state from one batch to the next (truncated backpropagation through time over 64 tokens). The RNN and LSTM use a constant learning rate of 1e-3; the Transformer uses 5e-4 with 1,000 warmup steps and cosine decay.

**Training cost.** Wall-clock training time to the best validation checkpoint and in total, number of epochs, and trainable / total parameter counts (embedding parameters reported separately). The time covers the training passes only, not the validation pass after each epoch, and the device is synchronized before each clock reading.

**Inference cost.** One common workload: a full forward pass over the test stream at batch size 32 and `context_length` 64. Measured after 10 warm-up batches, with `torch.cuda.synchronize()` around the timed region, reported as tokens per second (mean of 3 repeats).

**Learning behaviour.** Train and validation loss per epoch for every neural run, plotted on shared axes.

**Generation.** The same 5 prompts for every model, temperature 0.8, top-k 40, 50 new tokens, seed 42. The n-gram uses the same sampling procedure.

**Part III.** Two fixed-length representations from frozen LMs: mean of token hidden states excluding `<pad>`, and the last non-pad hidden state. Classifier: multinomial logistic regression (scikit-learn), regularization `C` chosen on the validation subset, test set evaluated once. Metrics: accuracy and macro-F1.

## Random seeds and determinism

- Global seed: `42` (Python, NumPy, PyTorch, CUDA), set in every entry point that uses randomness. The n-gram model uses none.
- `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
- Skip-gram: `gensim` with `workers=1` and `PYTHONHASHSEED=42` (gensim is only reproducible single-threaded).
- AG News subsampling and OOV embedding initialization use the same seed.
- Two CPU runs of the training loop with the same seed give identical losses. This has not been checked on a GPU, where some operations are not deterministic.
- All main results are from a single seed because of the compute budget. This is stated as a limitation in the report.

## Hardware

To be filled in with the exact GPU, CPU, and library versions used for the reported numbers. Each `results/<run_name>.json` from the training loop records the device and PyTorch version it ran on. The configs ask for `cuda`; on a machine without it the code falls back to Apple `mps`, then to `cpu`, and prints which one it used.

| Item | Value |
|---|---|
| GPU | TBD |
| CPU | TBD |
| PyTorch / CUDA | TBD |

## Results

Filled in from `results/` as runs finish. The neural models have not been trained yet.

| Model | Params | Test PPL | Train time | Inference (tok/s) |
|---|---|---|---|---|
| KN trigram | 1.38M trigrams | 226.00 | 10.2 s | |
| RNN | 21.5M | | | |
| LSTM | 24.2M | | | |
| Transformer | 13.5M | | | |

The KN trigram's validation perplexity is 240.66. Its "train time" is the time to count the n-grams, measured on an Apple M1 CPU.

## References and attribution

- WikiText-2: Merity, Xiong, Bradbury, Socher. *Pointer Sentinel Mixture Models.* 2016. (CC BY-SA 3.0)
- AG News: Zhang, Zhao, LeCun. *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015.
- GloVe: Pennington, Socher, Manning. *GloVe: Global Vectors for Word Representation.* EMNLP 2014.
- Skip-gram: Mikolov, Chen, Corrado, Dean. *Efficient Estimation of Word Representations in Vector Space.* 2013. Implemented with gensim (Řehůřek and Sojka, 2010).
- Kneser-Ney smoothing: Kneser and Ney, 1995; Chen and Goodman, 1999.
- RNN LM: Mikolov et al., 2010. LSTM: Hochreiter and Schmidhuber, 1997. Transformer: Vaswani et al., 2017.
- The Transformer in `papertrail/transformer.py` is ported from the author's earlier implementation, github.com/nabilrakaiza/transformer.
- Course notebooks adapted where noted in the source files.

## Tool use

An AI coding assistant (Claude Code, Anthropic) was used substantially in this project. Commits it produced carry a `Co-Authored-By: Claude` trailer.

- **Written by the assistant:** config loading (`config.py`), the data pipeline (`data.py`), the RNN and LSTM model (`rnn.py`), the training loop (`train.py`, `models.py`, `utils.py`), and the repository scaffolding (configs, checklist, this README).
- **Written jointly:**
  - `ngram.py`: the author wrote part of the counting, the trigram level of the probability, `distribution` and `evaluate`, with review from the assistant. The assistant wrote the file structure, the remaining count tables, the bigram and unigram levels, and `main`.
  - `evaluate.py`: the author wrote the structure of the function; the assistant corrected and completed it.
- **Written by the author:** the Transformer (`transformer.py`), ported from the author's earlier implementation. The assistant reformatted it and adapted it to this project's interface (untied output layer, loss moved out of the model).

The analysis and interpretation in the report are the author's own.

## License

Code: MIT (see `LICENSE`). Datasets and pretrained embeddings remain under their original licenses and are not redistributed here.

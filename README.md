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
| I | RNN LM | done, results in `results/rnn.json` |
| I | LSTM LM | done, results in `results/lstm.json` |
| I | Transformer LM | done, results in `results/transformer.json` |
| I | Training loop and evaluation | done |
| I | Timing benchmark | done, results in `results/benchmark.json` |
| I | Generations | done, samples in `results/generations.json` |
| II | Skip-gram embeddings (own) | done, cost in `results/word2vec_sg200.json` |
| II | LSTM + frozen skip-gram | done, results in `results/lstm_w2v_frozen.json` |
| II | LSTM + frozen GloVe | done, results in `results/lstm_glove_frozen.json` |
| III | Feature extraction + logistic regression | done, results in `results/probe_agnews.json` |

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
│   ├── word2vec.py     # skip-gram vectors on the train split (gensim)
│   ├── embeddings.py   # load skip-gram or GloVe vectors into a model, freeze them
│   ├── agnews.py       # AG News subsets, tokenized into the WikiText-2 vocabulary
│   ├── pooling.py      # mean and last-token pooling of hidden states
│   ├── probe.py        # frozen-LM features + logistic regression on AG News
│   ├── benchmark.py    # inference throughput and model size
│   ├── generate.py     # sampled continuations with matched decoding
│   ├── report_figs.py  # figures for the report
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
- Sequences are truncated to 64 tokens, the chunk length the language models were trained on, and padded at the end with `<pad>`.
- Resulting subsets (`python -m papertrail.agnews configs/part3/probe.yaml`):

| Subset | Items | Mean length (tokens) | `<unk>` rate | Longer than 64 tokens |
|---|---|---|---|---|
| train | 10,000 | 44.1 | 6.83% | 4.35% |
| validation | 2,000 | 43.9 | 6.94% | 4.90% |
| test | 7,600 | 43.7 | 6.96% | 3.79% |

The tokenizer approximates WikiText-2's conventions and is not the original tool. It also removes scraping debris that AG News carries (HTML tags, entities such as `#39;` and `amp;`). The most frequent unknown words are news vocabulary absent from WikiText-2, such as "corp", "afp", "oracle" and "darfur".

### GloVe (Part II)

- `glove.6B.200d` (Wikipedia 2014 + Gigaword 5, 6B tokens, 400k uncased vocab).
- Download: `bash scripts/download_glove.sh` (saves to `embeddings/glove.6B.200d.txt`). The script fetches `glove.6B.zip` from Stanford NLP's mirror on Hugging Face (`stanfordnlp/glove`); the original Stanford URL serves the same file but was too slow to use.
- Coverage: 27,810 of the 28,913 vocabulary tokens have a GloVe vector (96.19%), and those tokens make up 97.37% of the training text. The most frequent missing tokens are `<eos>` and the WikiText markers `@-@`, `@.@` and `@,@`; `<unk>` is in GloVe.
- The self-trained skip-gram vectors cover every vocabulary token except `<pad>`, which never occurs in the text. Training them took 233 s on the CPU (`results/word2vec_sg200.json`); this cost is separate from, and in addition to, the language-model training that uses them.
- Missing tokens get a row drawn from a normal distribution with mean 0 and the standard deviation of the vectors that were found, from a generator seeded with the global seed. The same rule is used for the self-trained skip-gram vectors.

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

Run everything from the repository root. The first run downloads WikiText-2 into `data/`. Once the datasets are cached, setting `HF_HUB_OFFLINE=1` skips the check against the Hugging Face Hub, which can otherwise stall for minutes when there is no network.

Working now:

```bash
# Part I
python -m papertrail.ngram     --config configs/part1/ngram.yaml
python -m papertrail.train     --config configs/part1/rnn.yaml
python -m papertrail.train     --config configs/part1/lstm.yaml
python -m papertrail.train     --config configs/part1/transformer.yaml
python -m papertrail.benchmark --configs configs/part1/*.yaml     # inference throughput + model size
python -m papertrail.generate  --configs configs/part1/*.yaml     # matched-decoding samples
python -m papertrail.report_figs                                  # loss curves

# Part II
bash scripts/download_glove.sh
python -m papertrail.word2vec  --config configs/part2/word2vec.yaml   # timed separately
python -m papertrail.train     --config configs/part2/lstm_w2v_frozen.yaml
python -m papertrail.train     --config configs/part2/lstm_glove_frozen.yaml
python -m papertrail.generate  --configs configs/part1/lstm.yaml configs/part2/lstm_w2v_frozen.yaml configs/part2/lstm_glove_frozen.yaml --output results/generations_part2.json

# Part III
python -m papertrail.probe     --config configs/part3/probe.yaml

# Inspect a resolved config, or the corpus statistics
python -m papertrail.config    configs/part2/lstm_glove_frozen.yaml
python -m papertrail.data      configs/base.yaml
python -m papertrail.agnews    configs/part3/probe.yaml
```

Outputs:

- Best checkpoint (lowest validation loss): `runs/<run_name>/best.pt`
- Train and validation loss and cumulative training time per epoch: `runs/<run_name>/log.json`
- Final metrics, measured with the best checkpoint: `results/<run_name>.json`
- Generated samples: `results/generations.json` (Part I) and `results/generations_part2.json` (Part II)
- Figures: `results/figures/`

## Evaluation protocol

**Prediction quality.** Test perplexity = `exp(mean next-token cross-entropy)` over every target token in the test stream, using the same vocabulary and the same token stream for all models. The targets are every token of a split except its first (244,101 in test), for the n-gram and the neural models alike. `<pad>` is excluded from the loss via `ignore_index`; the only padding is in the final chunk of each split, which is filled up to `context_length`.

**Context.** Every model predicts each target from preceding tokens only. For a fair context comparison, RNN/LSTM hidden states are **reset at every chunk** during evaluation, so all neural models see at most `context_length` tokens of history. The n-gram sees 2 tokens of history.

**Training procedure.** The same for every neural model: AdamW (weight decay 0.01), batch size 32, gradient-norm clipping at 1.0, at most 20 epochs, early stopping after 3 epochs without improvement in validation loss, and the checkpoint with the lowest validation loss is the one evaluated. Batches are read in corpus order without shuffling, so that the RNN and LSTM can carry their hidden state from one batch to the next (truncated backpropagation through time over 64 tokens). The RNN and LSTM use a constant learning rate of 1e-3; the Transformer uses 5e-4 with 1,000 warmup steps and cosine decay.

**Training cost.** Wall-clock training time to the best validation checkpoint and in total, number of epochs, and trainable / total parameter counts (embedding parameters reported separately). The time covers the training passes only, not the validation pass after each epoch, and the device is synchronized before each clock reading.

**Inference cost.** One common workload: a full forward pass over the test stream at batch size 32 and `context_length` 64, with the trained checkpoint in evaluation mode and no gradients. Measured after 10 warm-up batches and reported as test targets per second (mean of 3 repeats). The device is synchronized after every batch, so each batch's output is fully computed before the next is submitted. A neural forward pass produces a full distribution over the vocabulary at every position. The n-gram model does not need one to score a known next token, so it gets two numbers: scoring each actual target (all 244,101, a few dictionary lookups each), and producing the full distribution (timed on the first 256 positions only, because it takes about 0.07 s per position in pure Python).

**Learning behaviour.** Train and validation loss per epoch for every neural run, plotted on shared axes.

**Generation.** The same 5 prompts for every model, temperature 0.8, top-k 40, 50 new tokens. One sampling function serves all four models, and the random generator is re-seeded with 42 at the start of every continuation, so each model faces the same random draws for a given prompt. Generation continues past `<eos>` so that every sample has the same length.

**Part III.** Two fixed-length representations from frozen LMs, both built from the final layer's hidden states: their mean over the real tokens (excluding `<pad>`), and the hidden state at the last real token. Classifier: multinomial logistic regression (scikit-learn) on standardized features, with the standardization fitted on the training subset only. The regularization `C` is chosen by validation macro-F1 from the grid in the config, and the test set is evaluated once with that choice. Metrics: accuracy and macro-F1. The error examples saved for each probe are the 10 test items it got wrong with the highest confidence.

## Random seeds and determinism

- Global seed: `42` (Python, NumPy, PyTorch, CUDA), set in every entry point that uses randomness. The n-gram model uses none.
- `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
- Skip-gram: `gensim` with `workers=1` and `seed=42` (gensim is only reproducible single-threaded). Two runs gave bit-identical vectors, without setting `PYTHONHASHSEED`.
- AG News subsampling and OOV embedding initialization use the same seed.
- Two CPU runs of the training loop with the same seed give identical losses. This has not been checked on a GPU, where some operations are not deterministic.
- All main results are from a single seed because of the compute budget. This is stated as a limitation in the report.

## Hardware

All reported numbers come from one machine. Each `results/<run_name>.json` from the training loop records the device and PyTorch version it ran on. The configs ask for `cuda`; on a machine without it the code falls back to Apple `mps`, then to `cpu`, and prints which one it used.

| Item | Value |
|---|---|
| Machine | Apple M1 laptop, 8 cores, 8 GB unified memory, macOS 26.0 |
| GPU | Apple M1 integrated GPU, through PyTorch's `mps` backend. No CUDA. |
| CPU | Apple M1. Used for the n-gram model. |
| Software | Python 3.13.6, PyTorch 2.9.0 |
| Power state | Plugged in, with macOS Low Power Mode on. This laptop is set to use Low Power Mode whenever it is on mains power, in which a training step takes about 1.7 times as long as on battery. |

## Results

Part I, from `results/`. Perplexities are measured with the checkpoint that had the lowest validation loss.

| Model | Params | Validation PPL | Test PPL | Epochs run (best) | Time per epoch | Time to best | Inference (tok/s) |
|---|---|---|---|---|---|---|---|
| KN trigram | 1.38M trigrams | 240.66 | 226.00 | n/a | n/a | 10.2 s | 453,082 scoring targets; 9 full distribution |
| RNN | 21.5M | 268.92 | 245.14 | 16 (13) | 475 s | 103 min | 21,084 |
| LSTM | 24.2M | 226.13 | 210.84 | 10 (7) | 480 s | 56 min | 19,405 |
| Transformer | 13.5M | 160.41 | 149.52 | 9 (6) | 360 s | 36 min | 32,718 |

How the cost columns were measured:

- **All of them are for one power state:** plugged in with Low Power Mode on, the state the RNN, Transformer and Part II runs trained in.
- **Time per epoch** is the median time of a training step, measured for every model in a fresh process on an otherwise idle machine, times the 1,013 steps in an epoch. It is the mean of two rounds, which differed by at most 6%. Validation passes are not included.
- **Time to best** is the best epoch times the time per epoch. For the KN trigram it is the time to count the n-grams, on the CPU.
- **Inference** is from `results/benchmark.json`, on an idle machine. Repeats differed by at most 6%. The RNN and LSTM are close enough that their order is not reliable.
- **Why not the wall-clock times in the results files?** They are not comparable. The RNN and Transformer trained plugged in (recorded: 106.2 and 35.3 minutes to best, in line with the table). The LSTM's final run very likely started on battery, at full speed: its first eight epochs took about 288 s each, against 480 s when measured like-for-like, and its recorded 33.3 minutes to best reflects that. The RNN's recorded time also includes two slow early epochs (846 s and 623 s) from the laptop being in use.

**A PyTorch issue that affected the first LSTM run.** The LSTM was first trained with PyTorch's built-in two-layer module (`nn.LSTM(num_layers=2, dropout=0.3)`). In PyTorch 2.9 on Apple's `mps` device the dropout between that module's layers is broken: whichever mode the first call in the process uses, training or evaluation, is silently used for every later call, and in training mode the mask is the same on every call. That run was discarded. `papertrail/rnn.py` now stacks single-layer modules with an ordinary `nn.Dropout` between them, which computes the same model, and the LSTM was re-trained. The RNN (`nn.RNN`) and the Transformer were checked and are not affected. As a guard, every checkpoint is reloaded in a fresh process and must reproduce the loss recorded during training; all five do.

Part II, from `results/`: one architecture (the LSTM above), three input-embedding settings. Everything else, including the random seed and so the initial value of every other weight, is identical.

| Input embeddings | Trainable params | Vocabulary covered | Validation PPL | Test PPL | Epochs run (best) | Time per epoch |
|---|---|---|---|---|---|---|
| Random, trained with the model | 24.2M | n/a | 226.13 | 210.84 | 10 (7) | 480 s |
| Own skip-gram, frozen | 18.4M | 28,912 of 28,913 | 155.29 | 147.01 | 13 (10) | 452 s |
| GloVe 6B 200d, frozen | 18.4M | 27,810 of 28,913 | 178.42 | 167.55 | 13 (10) | 433 s |

- The frozen tables were checked after training and are bit-for-bit the tables the runs started with.
- Training the skip-gram vectors took a further 233 s on the CPU.
- The three tables differ in scale as well as in content: the random table starts with standard deviation 1.0 (PyTorch's default), the skip-gram vectors have 0.19 and GloVe's 0.43.
- Loss curves: `results/figures/part2_loss_curves.pdf`. Samples: `results/generations_part2.json`.

Part III, from `results/probe_agnews.json`: topic classification on AG News (4 balanced classes, so chance is 25%) from frozen language-model representations.

| Language model | LM test PPL | Pooling | Chosen C | Test accuracy | Test macro-F1 |
|---|---|---|---|---|---|
| LSTM | 210.84 | mean | 0.01 | 62.89% | 0.628 |
| LSTM | 210.84 | last token | 0.01 | 47.62% | 0.475 |
| Transformer | 149.52 | mean | 1.0 | 78.45% | 0.784 |
| Transformer | 149.52 | last token | 10.0 | 72.83% | 0.727 |

Three of the four chosen values of C sit at an end of the grid (0.01 to 10). The validation macro-F1 varies by at most 0.016 across the grid for any probe, so a wider grid would change little.

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

- **Written by the assistant:** config loading (`config.py`), the data pipeline (`data.py`), the RNN and LSTM model (`rnn.py`), the training loop (`train.py`, `models.py`, `utils.py`), the benchmark (`benchmark.py`), the figures (`report_figs.py`), the Part II embedding code (`word2vec.py`, `embeddings.py`), the AG News preparation (`agnews.py`), the probe (`probe.py`), and the repository scaffolding (configs, checklist, this README).
- **Written jointly:**
  - `ngram.py`: the author wrote part of the counting, the trigram level of the probability, `distribution` and `evaluate`, with review from the assistant. The assistant wrote the file structure, the remaining count tables, the bigram and unigram levels, and `main`.
  - `evaluate.py`: the author wrote the structure of the function; the assistant corrected and completed it.
  - `generate.py`: the author wrote the sampling function and the neural scoring function, with hints and review. The assistant wrote the file structure, the n-gram scoring function, the generation loop and `main`.
  - `pooling.py`: the author wrote both functions as loops, correctly; the assistant then replaced them with equivalent loop-free versions at the author's request.
- **Written by the author:** the Transformer (`transformer.py`), ported from the author's earlier implementation. The assistant reformatted it and adapted it to this project's interface (untied output layer, loss moved out of the model).

The analysis and interpretation in the report are the author's own.

## License

Code: MIT (see `LICENSE`). Datasets and pretrained embeddings remain under their original licenses and are not redistributed here.

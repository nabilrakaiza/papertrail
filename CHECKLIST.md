# Requirements checklist

Every required item from the brief, mapped to where it lives. Tick as you go. Deadline: Fri 16 Oct 2026, 23:59 SGT (aim to submit Thu 15 night).

## Part I (40)

- [x] n-gram LM with smoothing/interpolation: `configs/part1/ngram.yaml`, `papertrail/ngram.py`
- [x] RNN, LSTM, Transformer LMs with random trainable embeddings
- [x] All models predict from preceding tokens only (causal mask checked for Transformer)
- [x] Test perplexity on the common setup, padding excluded
- [x] Training time, model size, inference throughput on one common workload
- [x] Hardware, training budget, and timing procedure stated
- [x] Train/val loss curves for the 3 neural models
- [x] Generations: same prompts, matched decoding, same length
- [x] Discussion: repetition, local fluency, consistency
- [x] Tradeoffs, with observations kept separate from explanations

## Part II (30)

- [x] Same architecture (LSTM), emb dim, data, and training procedure across all 3 settings
- [x] Setting 1: random trainable (reused Part I LSTM)
- [x] Setting 2: own skip-gram, frozen; trained on train split only
- [x] Setting 3: GloVe 200d, frozen
- [x] Vocab coverage and OOV initialization documented
- [x] Entire input table frozen in settings 2 and 3 (`requires_grad=False` verified)
- [x] Skip-gram training cost reported separately from LM training
- [x] Compare: test PPL, convergence, stability, generations
- [x] Interpret trainability vs. extra data seen by GloVe

## Part III (20)

- [x] At least 2 representation choices on the same labeled data
- [x] How each input becomes a fixed-length vector is described
- [x] Same classifier family and evaluation procedure for both
- [x] Held-out accuracy and macro-F1
- [x] Representative errors examined
- [x] Does LM ranking (PPL) match downstream ranking?

## Report and repo (10)

- [x] PDF, at most 6 pages excluding appendices and references
- [x] Dataset and preprocessing stated
- [x] Models and variants, evaluation criteria
- [x] Main quantitative and qualitative observations
- [x] Limitations, failure cases, lessons learned
- [x] README: data access, preprocessing, dependencies, commands, configs, seeds
- [x] External code, datasets, pretrained models, and tool assistance cited
- [x] No data, keys, or checkpoints committed
- [x] Fresh clone reproduces main numbers (checked for the corpus, trigram, AG News subsets and skip-gram vectors; the neural models are covered by fixed seeds and the checkpoint-reload checks)
- [x] Repo accessible to the instructor (public on GitHub: github.com/nabilrakaiza/papertrail)

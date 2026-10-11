# Requirements checklist

Every required item from the brief, mapped to where it lives. Tick as you go. Deadline: Fri 16 Oct 2026, 23:59 SGT (aim to submit Thu 15 night).

## Part I (40)

- [x] n-gram LM with smoothing/interpolation: `configs/part1/ngram.yaml`, `papertrail/ngram.py`
- [x] RNN, LSTM, Transformer LMs with random trainable embeddings
- [x] All models predict from preceding tokens only (causal mask checked for Transformer)
- [x] Test perplexity on the common setup, padding excluded
- [ ] Training time, model size, inference throughput on one common workload
- [x] Hardware, training budget, and timing procedure stated
- [x] Train/val loss curves for the 3 neural models
- [x] Generations: same prompts, matched decoding, same length
- [ ] Discussion: repetition, local fluency, consistency
- [ ] Tradeoffs, with observations kept separate from explanations

## Part II (30)

- [ ] Same architecture (LSTM), emb dim, data, and training procedure across all 3 settings
- [x] Setting 1: random trainable (reused Part I LSTM)
- [ ] Setting 2: own skip-gram, frozen; trained on train split only
- [ ] Setting 3: GloVe 200d, frozen
- [x] Vocab coverage and OOV initialization documented
- [ ] Entire input table frozen in settings 2 and 3 (`requires_grad=False` verified)
- [ ] Skip-gram training cost reported separately from LM training
- [ ] Compare: test PPL, convergence, stability, generations
- [ ] Interpret trainability vs. extra data seen by GloVe

## Part III (20)

- [ ] At least 2 representation choices on the same labeled data
- [ ] How each input becomes a fixed-length vector is described
- [ ] Same classifier family and evaluation procedure for both
- [ ] Held-out accuracy and macro-F1
- [ ] Representative errors examined
- [ ] Does LM ranking (PPL) match downstream ranking?

## Report and repo (10)

- [ ] PDF, at most 6 pages excluding appendices and references
- [ ] Dataset and preprocessing stated
- [ ] Models and variants, evaluation criteria
- [ ] Main quantitative and qualitative observations
- [ ] Limitations, failure cases, lessons learned
- [ ] README: data access, preprocessing, dependencies, commands, configs, seeds
- [ ] External code, datasets, pretrained models, and tool assistance cited
- [ ] No data, keys, or checkpoints committed
- [ ] Fresh clone reproduces main numbers
- [ ] Repo accessible to the instructor (or zip prepared)

# Nano LLM Transformer

*A small language model written from scratch, and which of its parts earn their place.*

**Aniket Anil Naik** · Project 02 of the data science full-stack portfolio

## Abstract

**How good a storyteller can a 8.7-million-parameter transformer become on one laptop GPU, and which of its modern components actually earn their place?**

We study this question on *TinyStories V2 (GPT-4 stories), training split*: 0 rows as downloaded, used under Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0). Following the CRISP-DM process, we compare 3 approaches, 2 of them baselines, on 5,578,295 evaluation rows and rank them by validation perplexity. Evaluation split: official TinyStories V2 split; validation stories seen in training removed.

The best approach, **NanoLM, RoPE · SwiGLU · RMSNorm**, reaches 4.029 against 34.037 for the strongest baseline (Bigram, interpolated).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** next-token prediction (language modelling), RoPE, SwiGLU and RMSNorm from scratch, byte-level BPE tokenizer, KV-cache generation, verified, ablations against seed noise


## 1. Introduction

*Objective.* Train a small language model from scratch and measure it against baselines and ablations.

Modern language models share a recipe: rotary position embeddings, gated feed-forward layers and RMSNorm. This project implements that recipe from scratch, trains it on children's stories, and asks two questions a demo usually skips. How much better is it than simple word statistics? And does each piece of the recipe help at this scale, or is it carried along by habit?

| Finding | Value | Note |
|---|---|---|
| Measure | validation perplexity |  |
| Hardware | NVIDIA GeForce RTX 5070 Laptop GPU |  |

> Phase gate passed: Validation perplexity on deduplicated stories fixed as the measure before training.


## 2. Data

*Objective.* Check what a generated corpus repeats, and keep repeats out of the score.

2,717,495 training stories and 27,630 validation stories. Because the stories were generated, some repeat: 1 training stories are exact copies (after whitespace normalisation) of an earlier one, and **0 validation stories also appear in the training file**. Those are removed, leaving 27,630 validation stories the model has never seen.

| Finding | Value | Note |
|---|---|---|
| Training stories | 2,717,495 |  |
| Validation stories removed as seen | 0 |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | TinyStories V2 (GPT-4 stories), training split |
| Source | https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt |
| Licence | Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0) |
| Rows × columns (as downloaded) | 0 × 0 |
| SHA-256 | `6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443` |

> Phase gate passed: Validation-training overlap measured and removed by an executed assertion.


## 3. Method

*Objective.* Turn text into tokens with a vocabulary learned from the training stories only.

A byte-level BPE tokenizer with 4,096 tokens is trained on the first 200,000 training stories; byte-level means no character can ever be out of vocabulary. The model trains on the first 340M tokens of the training file and is scored on 5.6M tokens of unseen validation stories, averaging 3.98 characters per token.

| Finding | Value | Note |
|---|---|---|
| Vocabulary | 4,096 |  |
| Training tokens | 339,867,310 |  |
| Validation tokens | 5,578,295 |  |

> Phase gate passed: Tokenizer fitted on training stories only.


## 4. Models

*Objective.* Train the reference recipe, and each variant that removes one of its components.

The main model has 8,685,760 parameters: 6 layers, 8 heads, width 320, a 256-token context and tied input and output embeddings. It trained for 12,000 steps (197M tokens) in 26 minutes at 124k tokens per second.

Each ablation trains for 3,000 steps with one component swapped for its older equivalent. The reference was also trained with a second seed: the two differ by 0.0060 nats per token, which sets how large a difference must be before it means anything.

| Finding | Value | Note |
|---|---|---|
| Parameters | 8,685,760 |  |
| Training tokens seen | 197M |  |
| Seed-to-seed gap | 0.0060 nats/token |  |

**Table 1.** Tournament results.

| # | Model | Validation perplexity | Loss (nats/token) | Bits per character |
|---|---|---|---|---|
| 1 | **NanoLM, RoPE · SwiGLU · RMSNorm** 🏆 | 4.029 | 1.3934 | 0.506 |
| 2 | Bigram, interpolated *(baseline)* | 34.037 | 3.5275 | 1.280 |
| 3 | Unigram frequencies *(baseline)* | 357.962 | 5.8804 | 2.134 |

Ranked by validation perplexity (lower is better). Split: official TinyStories V2 split; validation stories seen in training removed; 5,578,295 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: Main model and 6 ablation runs completed.


## 5. Results

*Objective.* Compare against baselines, separate real ablation effects from seed noise, and verify the mechanics.

Validation perplexity is **4.03**, against 34.0 for an interpolated bigram and 358.0 for unigram frequencies.

Of the 4 ablations, 3 moved validation loss by more than the seed-to-seed gap: Learned positions instead of RoPE (+0.0522); GELU instead of SwiGLU (+0.0386); GPT-2 style: learned · GELU · LayerNorm (+0.0852). Every one of these is a single run at 3,000 steps, so the ranking is indicative rather than settled.

Two mechanics are checked by execution: changing future tokens leaves earlier logits unchanged, and generation with the KV cache produces exactly the tokens full recomputation does, at 4.7 ms per token against 4.2 ms.

| Finding | Value | Note |
|---|---|---|
| Validation perplexity | 4.03 |  |
| Bigram perplexity | 34.0 |  |
| KV cache speed-up | 0.9× |  |

> Phase gate passed: Baselines, seed-noise comparison and executed mechanics checks.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_validation_stories_unseen` | ✅ passed | 27,630 validation stories hashed against 2,717,495 training stories; 0 also appeared in training and were removed; 0 remain. |
| `assert_causal_mask` | ✅ passed | Tokens 40-63 of a 64-token sequence replaced; largest change in logits at positions 0-39 was 0.00e+00 (limit 1e-4). The model cannot see the future. |
| `assert_kv_cache_matches_recompute` | ✅ passed | Greedy 60-token continuations of 3 prompts generated with and without the KV cache; 3 of 3 identical. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- Ablations are single short runs, and seed noise is estimated from only two runs of the reference.
- Perplexity depends on the tokenizer, so it is only comparable between models that share this one.
- TinyStories was itself generated by GPT-4, so fluency here says little about modelling human-written text.
- Generation timings depend on the GPU and on what else was running.


## 7. Deployment

*Objective.* Serve generation, tokenisation and attention maps from the trained checkpoint.

The API loads `model.pt` and `tokenizer.json` and serves `/api/generate` with temperature, top-p and repetition penalty, `/api/tokenize`, and `/api/attention`, which returns every head's attention weights for a short prompt.

| Finding | Value | Note |
|---|---|---|
| Serving port | 8,002 |  |

> Phase gate passed: Checkpoint and tokenizer persisted together.

## 8. Conclusion

- Under half an hour of training on one laptop GPU took an 8.7-million-parameter model far beyond n-gram statistics on the same tokens.
- Replacing rotary position embeddings or SwiGLU with their older equivalents cost more than the seed-to-seed gap; replacing RMSNorm with LayerNorm did not.
- A second seed is the cheapest way to know whether an ablation result is a result at all.
- A correct KV cache is not automatically a faster one: at a short context on a GPU, recomputation was as quick.
- Measuring overlap between training and validation data is worth doing even when, as here, it finds none.

## Reproducibility

Trained 2026-09-17T15:51:08Z in 3810.0 s at commit `1e3bb26` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8002
npm run dev -w 02_nano_llm/client
```

## References

1. Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still speak coherent English? arXiv:2305.07759.
2. Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still speak coherent English? arXiv:2305.07759.
3. Vaswani, A. et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems 30.
4. Su, J. et al. (2021). RoFormer: Enhanced transformer with rotary position embedding. arXiv:2104.09864.
5. Shazeer, N. (2020). GLU variants improve transformer. arXiv:2002.05202.
6. Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. Advances in Neural Information Processing Systems 32.
7. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. Proceedings of ACL 2016.
8. Holtzman, A. et al. (2020). The curious case of neural text degeneration. Proceedings of ICLR 2020.

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*

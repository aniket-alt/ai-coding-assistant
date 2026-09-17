# CRISP-DM Report — Nano LLM Transformer

**Business question.** How good a storyteller can a 8.7-million-parameter transformer become on one laptop GPU, and which of its modern components actually earn their place?


## 1. Business Understanding

*Objective:* Train a small language model from scratch and measure it against baselines and ablations.


Modern language models share a recipe: rotary position embeddings, gated feed-forward layers and RMSNorm. This project implements that recipe from scratch, trains it on children's stories, and asks two questions a demo usually skips. How much better is it than simple word statistics? And does each piece of the recipe help at this scale, or is it carried along by habit?


| Finding | Value | Detail |
|---|---|---|
| Measure | `validation perplexity` |  |
| Hardware | `NVIDIA GeForce RTX 5070 Laptop GPU` |  |


*Code:* `ml/model.py:NanoLM`, `ml/train.py:main`


> **Phase gate: PASSED.** Validation perplexity on deduplicated stories fixed as the measure before training.


## 2. Data Understanding

*Objective:* Check what a generated corpus repeats, and keep repeats out of the score.


2,717,495 training stories and 27,630 validation stories. Because the stories were generated, some repeat: 1 training stories are exact copies (after whitespace normalisation) of an earlier one, and **0 validation stories also appear in the training file**. Those are removed, leaving 27,630 validation stories the model has never seen.


| Finding | Value | Detail |
|---|---|---|
| Training stories | `2,717,495` |  |
| Validation stories removed as seen | `0` |  |


*Code:* `ml/data.py:story_hash`, `ml/train.py:main`


> **Phase gate: PASSED.** Validation-training overlap measured and removed by an executed assertion.


## 3. Data Preparation

*Objective:* Turn text into tokens with a vocabulary learned from the training stories only.


A byte-level BPE tokenizer with 4,096 tokens is trained on the first 200,000 training stories; byte-level means no character can ever be out of vocabulary. The model trains on the first 340M tokens of the training file and is scored on 5.6M tokens of unseen validation stories, averaging 3.98 characters per token.


| Finding | Value | Detail |
|---|---|---|
| Vocabulary | `4,096` |  |
| Training tokens | `339,867,310` |  |
| Validation tokens | `5,578,295` |  |


*Code:* `ml/data.py:encode_to_file`


> **Phase gate: PASSED.** Tokenizer fitted on training stories only.


## 4. Modeling

*Objective:* Train the reference recipe, and each variant that removes one of its components.


The main model has 8,685,760 parameters: 6 layers, 8 heads, width 320, a 256-token context and tied input and output embeddings. It trained for 12,000 steps (197M tokens) in 26 minutes at 124k tokens per second.

Each ablation trains for 3,000 steps with one component swapped for its older equivalent. The reference was also trained with a second seed: the two differ by 0.0060 nats per token, which sets how large a difference must be before it means anything.


| Finding | Value | Detail |
|---|---|---|
| Parameters | `8,685,760` |  |
| Training tokens seen | `197M` |  |
| Seed-to-seed gap | `0.0060 nats/token` |  |


*Code:* `ml/model.py:NanoLM`, `ml/train.py:train_model`


> **Phase gate: PASSED.** Main model and 6 ablation runs completed.


## 5. Evaluation

*Objective:* Compare against baselines, separate real ablation effects from seed noise, and verify the mechanics.


Validation perplexity is **4.03**, against 34.0 for an interpolated bigram and 358.0 for unigram frequencies.

Of the 4 ablations, 3 moved validation loss by more than the seed-to-seed gap: Learned positions instead of RoPE (+0.0522); GELU instead of SwiGLU (+0.0386); GPT-2 style: learned · GELU · LayerNorm (+0.0852). Every one of these is a single run at 3,000 steps, so the ranking is indicative rather than settled.

Two mechanics are checked by execution: changing future tokens leaves earlier logits unchanged, and generation with the KV cache produces exactly the tokens full recomputation does, at 4.7 ms per token against 4.2 ms.


| Finding | Value | Detail |
|---|---|---|
| Validation perplexity | `4.03` |  |
| Bigram perplexity | `34.0` |  |
| KV cache speed-up | `0.9×` |  |


*Code:* `ml/train.py:ngram_cross_entropy`, `ml/model.py:NanoLM.generate`


> **Phase gate: PASSED.** Baselines, seed-noise comparison and executed mechanics checks.


## 6. Deployment

*Objective:* Serve generation, tokenisation and attention maps from the trained checkpoint.


The API loads `model.pt` and `tokenizer.json` and serves `/api/generate` with temperature, top-p and repetition penalty, `/api/tokenize`, and `/api/attention`, which returns every head's attention weights for a short prompt.


| Finding | Value | Detail |
|---|---|---|
| Serving port | `8,002` |  |


*Code:* `server/inference.py`, `server/main.py`


> **Phase gate: PASSED.** Checkpoint and tokenizer persisted together.

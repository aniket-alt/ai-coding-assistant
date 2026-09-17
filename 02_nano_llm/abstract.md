# Abstract — Nano LLM Transformer

**How good a storyteller can a 8.7-million-parameter transformer become on one laptop GPU, and which of its modern components actually earn their place?**

We study this question on *TinyStories V2 (GPT-4 stories), training split*: 0 rows as downloaded, used under Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0). Following the CRISP-DM process, we compare 3 approaches, 2 of them baselines, on 5,578,295 evaluation rows and rank them by validation perplexity. Evaluation split: official TinyStories V2 split; validation stories seen in training removed.

The best approach, **NanoLM, RoPE · SwiGLU · RMSNorm**, reaches 4.029 against 34.037 for the strongest baseline (Bigram, interpolated).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** next-token prediction (language modelling), RoPE, SwiGLU and RMSNorm from scratch, byte-level BPE tokenizer, KV-cache generation, verified, ablations against seed noise

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*

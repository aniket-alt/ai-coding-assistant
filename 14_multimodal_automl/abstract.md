# Abstract — Multimodal AutoML

**Can a catalogue classify products by type from their photo, name and attributes, and does combining them beat the best single source?**

We study this question on *Fashion Product Images (Small)*: 44,072 rows as downloaded, used under MIT, as declared by the Kaggle dataset; product photographs originate from a retailer's catalogue. Following the CRISP-DM process, we compare 10 approaches, 2 of them baselines, on 6,254 evaluation rows and rank them by macro-f1. Evaluation split: grouped by display name: 70% train, 15% validation, 15% test.

The best approach, **Late fusion: image + name + fields**, reaches 0.9740 against 0.5283 for the strongest baseline (CLIP zero-shot, image vs class names).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 1 warning findings.

**Keywords:** 45-class product classification from image, text and catalogue fields, CLIP image and text embeddings, late and early fusion, and AutoGluon, label-word masking to measure a text leak, text-to-image search

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*

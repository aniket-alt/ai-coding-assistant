# Multimodal AutoML

*What do a product's photo, name and catalogue fields each add, and does fusing them pay?*

**Aniket Anil Naik** · Project 14 of the data science full-stack portfolio

## Abstract

**Can a catalogue classify products by type from their photo, name and attributes, and does combining them beat the best single source?**

We study this question on *Fashion Product Images (Small)*: 44,072 rows as downloaded, used under MIT, as declared by the Kaggle dataset; product photographs originate from a retailer's catalogue. Following the CRISP-DM process, we compare 10 approaches, 2 of them baselines, on 6,254 evaluation rows and rank them by macro-f1. Evaluation split: grouped by display name: 70% train, 15% validation, 15% test.

The best approach, **Late fusion: image + name + fields**, reaches 0.9740 against 0.5283 for the strongest baseline (CLIP zero-shot, image vs class names).

3 of 3 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 1 warning findings.

**Keywords:** 45-class product classification from image, text and catalogue fields, CLIP image and text embeddings, late and early fusion, and AutoGluon, label-word masking to measure a text leak, text-to-image search


## 1. Introduction

*Objective.* Measure what each modality contributes to classifying products, and whether fusion pays.

A retailer receives products from many suppliers, and each has to be filed under the right article type. There are three kinds of evidence: a photograph, a free-text name and a few catalogue fields. Multimodal AutoML promises to combine them automatically. This project measures each source alone, fuses them several ways, and asks whether fusion beats the strongest single source by more than chance -- and how much of the text's apparent strength is simply the label written in the name.

| Finding | Value | Note |
|---|---|---|
| Target | article type |  |
| Modalities | image, name, catalogue fields |  |

> Phase gate passed: Target, modalities and the leak to be measured fixed before modelling.


## 2. Data

*Objective.* Find what in the catalogue would flatter a model.

44,072 products in 141 article types. The 45 types with at least 150 products are kept: 40,875 products.

**The name often states the answer.** 77.8% of names contain a word of their own article type. **Siblings share names.** 12,688 products repeat a display name already used by another.

| Finding | Value | Note |
|---|---|---|
| Products kept | 40,875 |  |
| Article types | 45 |  |
| Names containing their label | 77.8% |  |
| Repeated names | 12,688 |  |

### Provenance

✅ **Real, third-party data**

| Field | Value |
|---|---|
| Dataset | Fashion Product Images (Small) |
| Source | https://huggingface.co/datasets/ashraq/fashion-product-images-small/resolve/3859c76db2f6f3d3b9a3863345e3ccdbff75879d/data/train-00000-of-00002-6cff4c59f91661c3.parquet |
| Licence | MIT, as declared by the Kaggle dataset; product photographs originate from a retailer's catalogue |
| Rows × columns (as downloaded) | 44,072 × 10 |
| SHA-256 | `0f9b4042c6eea3c77aa0794630c956d1985de791c365abdb332f9d9cfa78d00a` |

> Phase gate passed: Label-in-name rate and duplicate names measured.


## 3. Method

*Objective.* Split so siblings never straddle train and test, and build each modality's features.

Products are grouped by display name before splitting: 28,510 train, 6,111 validation (used only to choose regularisation and fusion weights) and 6,254 test. Images and names are embedded once with openai/clip-vit-base-patch32. Names are also represented with TF-IDF, with a second copy in which every word of any article-type name is removed. Catalogue features are gender, baseColour, season, year, usage; the coarser category levels are excluded because they nearly determine the answer.

| Finding | Value | Note |
|---|---|---|
| Train / validation / test | 28,510 / 6,111 / 6,254 |  |

> Phase gate passed: Executed assertions: no shared names across splits, every class tested, hierarchy fields excluded.


## 4. Models

*Objective.* Model each modality alone, then fuse them late, early and with AutoGluon.

Single-modality models: logistic regression on CLIP image embeddings, logistic regression on TF-IDF names (full and masked), and LightGBM on catalogue fields, beside a majority-class baseline and CLIP zero-shot matching against the class names. Late fusion combines probabilities with weights chosen on validation; early fusion concatenates features into one logistic regression; AutoGluon trains on the same fused features.

| Finding | Value | Note |
|---|---|---|
| Contestants | 10 |  |

**Table 1.** Tournament results.

| # | Contestant | Macro-F1 | Accuracy | Top-3 accuracy |
|---|---|---|---|---|
| 1 | **Late fusion: image + name + fields** 🏆 | 0.9740 | 97.5% | 99.9% |
| 2 | Name only (TF-IDF) | 0.9694 | 95.6% | 99.7% |
| 3 | Early fusion: image + name + fields | 0.9554 | 96.9% | 99.7% |
| 4 | AutoGluon on fused features | 0.9465 | 96.5% | 99.6% |
| 5 | Late fusion: image + masked name + fields | 0.8585 | 91.3% | 99.4% |
| 6 | Image only (CLIP + logistic) | 0.8183 | 88.9% | 98.7% |
| 7 | Name only, label words removed (TF-IDF) | 0.5922 | 70.8% | 92.0% |
| 8 | CLIP zero-shot, image vs class names *(baseline)* | 0.5283 | 53.6% | 76.6% |
| 9 | Catalogue fields only (LightGBM) | 0.3023 | 49.2% | 78.1% |
| 10 | Most common article type *(baseline)* | 0.0060 | 15.5% | 31.3% |

Ranked by macro-f1 (higher is better). Split: grouped by display name: 70% train, 15% validation, 15% test; 6,254 evaluation rows. Generated from `artifacts/metrics.json`.

> Phase gate passed: 10 contestants scored on the same test products.


## 5. Results

*Objective.* Separate what fusion adds from what the name gives away.

**Late fusion: image + name + fields** has the highest macro-F1, 0.9740. The image alone reaches 0.8183; CLIP zero-shot, with no training, 0.5283.

**The name leak.** The name model's macro-F1 falls from 0.9694 to 0.5922 when label words are removed. On a random split, where 40.2% of test names also occur in training, it would have reported 0.9720.

**Does fusion pay?** Against the best honest single modality (Image only (CLIP + logistic)), late fusion of image, masked name and fields changes accuracy by +0.0249 (95% interval +0.0198 to +0.0301): **better**.

| Finding | Value | Note |
|---|---|---|
| Best macro-F1 | 0.9740 |  |
| Fusion vs best single | better |  |

> Phase gate passed: Paired bootstrap against the best single modality; leaks measured, not assumed.

## 6. Threats to validity

### Assertions executed during training

| Assertion | Result | What it measured |
|---|---|---|
| `assert_no_name_in_two_splits` | ✅ passed | 28,510 training, 6,111 validation and 6,254 test products grouped by display name; 0 names appear in more than one split. |
| `assert_every_class_in_test` | ✅ passed | All 45 article types have test products; missing: []. |
| `assert_hierarchy_fields_excluded` | ✅ passed | Catalogue features ['gender', 'baseColour', 'season', 'year', 'usage']; the coarser category levels, which determine most of the article type, are excluded. |

Recorded by the assertion ledger while training ran; generated from `artifacts/metrics.json`.

### Limitations

- The product photographs come from a retailer's catalogue; the MIT licence is the Kaggle dataset author's and may not cover the images themselves.
- Catalogue labels were entered by people and some are inconsistent, such as shirts labelled as tops.
- Grouping by exact display name does not catch near-identical names, so some siblings may still straddle the split.
- Fusion weights and regularisation were chosen on one validation split; a different split could favour different settings.


## 7. Deployment

*Objective.* Serve the gallery, per-modality predictions and text search.

The API serves test products with every contestant's prediction, classifies any typed name with the text models, and searches test images by description with CLIP.

| Finding | Value | Note |
|---|---|---|
| Serving port | 8,014 |  |

> Phase gate passed: Test-set predictions and CLIP search served from the training run's artifacts.

## 8. Conclusion

- A text field that contains the label can make a model look almost perfect; masking the label words is a cheap way to see what is left.
- With the label words masked, fusing the name and catalogue fields with the photo was significantly better than the photo alone.
- Grouping the split by shared names changed the measured score very little here, which is only knowable by measuring it.
- A frozen CLIP model gave a strong image classifier with a logistic regression on top, and a usable zero-shot one with none.
- Under a fifteen-minute limit, AutoGluon on fused features did not beat a simple late fusion whose weights were chosen on validation.

## Reproducibility

Trained 2026-09-17T17:29:22Z in 129.8 s at commit `9922732` with Python 3.11.6, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.1.

```bash
python ml/train.py
../.venv-automl/Scripts/python ml/autogluon_run.py
python ml/train.py
python -m uvicorn main:app --app-dir server --port 8014
npm run dev -w 14_multimodal_automl/client
```

## References

1. Aggarwal, P. (2019). Fashion Product Images (Small) [Dataset]. Kaggle. https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
2. Aggarwal, P. (2019). Fashion Product Images (Small) [Dataset]. Kaggle.
3. Radford, A. et al. (2021). Learning transferable visual models from natural language supervision. Proceedings of ICML 2021.
4. Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(2), 423–443.
5. Shi, X. et al. (2021). Benchmarking multimodal AutoML for tabular data with text fields. NeurIPS 2021 Datasets and Benchmarks Track.
6. Kaufman, S. et al. (2012). Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4).

---

*Generated from this project's artifacts by `scripts/generate_docs.py`.*

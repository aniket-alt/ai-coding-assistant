# CRISP-DM Report — Multimodal AutoML

**Business question.** Can a catalogue classify products by type from their photo, name and attributes, and does combining them beat the best single source?


## 1. Business Understanding

*Objective:* Measure what each modality contributes to classifying products, and whether fusion pays.


A retailer receives products from many suppliers, and each has to be filed under the right article type. There are three kinds of evidence: a photograph, a free-text name and a few catalogue fields. Multimodal AutoML promises to combine them automatically. This project measures each source alone, fuses them several ways, and asks whether fusion beats the strongest single source by more than chance -- and how much of the text's apparent strength is simply the label written in the name.


| Finding | Value | Detail |
|---|---|---|
| Target | `article type` |  |
| Modalities | `image, name, catalogue fields` |  |


*Code:* `ml/train.py:main`


> **Phase gate: PASSED.** Target, modalities and the leak to be measured fixed before modelling.


## 2. Data Understanding

*Objective:* Find what in the catalogue would flatter a model.


44,072 products in 141 article types. The 45 types with at least 150 products are kept: 40,875 products.

**The name often states the answer.** 77.8% of names contain a word of their own article type. **Siblings share names.** 12,688 products repeat a display name already used by another.


| Finding | Value | Detail |
|---|---|---|
| Products kept | `40,875` |  |
| Article types | `45` |  |
| Names containing their label | `77.8%` |  |
| Repeated names | `12,688` |  |


*Code:* `ml/data.py:load`, `ml/data.py:label_words`


> **Phase gate: PASSED.** Label-in-name rate and duplicate names measured.


## 3. Data Preparation

*Objective:* Split so siblings never straddle train and test, and build each modality's features.


Products are grouped by display name before splitting: 28,510 train, 6,111 validation (used only to choose regularisation and fusion weights) and 6,254 test. Images and names are embedded once with openai/clip-vit-base-patch32. Names are also represented with TF-IDF, with a second copy in which every word of any article-type name is removed. Catalogue features are gender, baseColour, season, year, usage; the coarser category levels are excluded because they nearly determine the answer.


| Finding | Value | Detail |
|---|---|---|
| Train / validation / test | `28,510 / 6,111 / 6,254` |  |


*Code:* `ml/train.py:main`, `ml/embed.py:cached`, `ml/data.py:mask_label_words`


> **Phase gate: PASSED.** Executed assertions: no shared names across splits, every class tested, hierarchy fields excluded.


## 4. Modeling

*Objective:* Model each modality alone, then fuse them late, early and with AutoGluon.


Single-modality models: logistic regression on CLIP image embeddings, logistic regression on TF-IDF names (full and masked), and LightGBM on catalogue fields, beside a majority-class baseline and CLIP zero-shot matching against the class names. Late fusion combines probabilities with weights chosen on validation; early fusion concatenates features into one logistic regression; AutoGluon trains on the same fused features.


| Finding | Value | Detail |
|---|---|---|
| Contestants | `10` |  |


*Code:* `ml/train.py:logistic`, `ml/train.py:main`


> **Phase gate: PASSED.** 10 contestants scored on the same test products.


## 5. Evaluation

*Objective:* Separate what fusion adds from what the name gives away.


**Late fusion: image + name + fields** has the highest macro-F1, 0.9740. The image alone reaches 0.8183; CLIP zero-shot, with no training, 0.5283.

**The name leak.** The name model's macro-F1 falls from 0.9694 to 0.5922 when label words are removed. On a random split, where 40.2% of test names also occur in training, it would have reported 0.9720.

**Does fusion pay?** Against the best honest single modality (Image only (CLIP + logistic)), late fusion of image, masked name and fields changes accuracy by +0.0249 (95% interval +0.0198 to +0.0301): **better**.


| Finding | Value | Detail |
|---|---|---|
| Best macro-F1 | `0.9740` |  |
| Fusion vs best single | `better` |  |


*Code:* `ml/train.py:bootstrap`, `ml/train.py:scores`


> **Phase gate: PASSED.** Paired bootstrap against the best single modality; leaks measured, not assumed.


## 6. Deployment

*Objective:* Serve the gallery, per-modality predictions and text search.


The API serves test products with every contestant's prediction, classifies any typed name with the text models, and searches test images by description with CLIP.


| Finding | Value | Detail |
|---|---|---|
| Serving port | `8,014` |  |


*Code:* `server/inference.py`, `server/main.py`


> **Phase gate: PASSED.** Test-set predictions and CLIP search served from the training run's artifacts.

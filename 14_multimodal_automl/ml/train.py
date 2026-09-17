"""
Project 14 -- what does each modality add, and is fusing them worth it?

    python ml/train.py
    (then ml/autogluon_run.py in .venv-automl, then python ml/train.py again to include it)

Products are split so that every product sharing a display name lands on the
same side. Each kind of information is modelled alone -- image, name, name with
the label words removed, catalogue fields -- then fused, and every contestant is
scored on the same held-out products with a paired bootstrap against the
strongest single modality.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage

from data import MIN_PER_CLASS, TABULAR, TARGET, label_words, load, mask_label_words
from embed import MODEL, cached, embed_texts, load_clip

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
WORK = ARTIFACTS / "work"
SEED = 42
BOOTSTRAPS = 2000


def scores(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(1)
    top3 = np.argsort(-proba, axis=1)[:, :3]
    return {"macro_f1": float(f1_score(y, pred, average="macro")), "accuracy": float(accuracy_score(y, pred)),
            "top3_accuracy": float((top3 == y[:, None]).any(1).mean())}


def logistic(Xtr, ytr, Xva, yva, grid=(0.1, 1.0, 10.0)) -> tuple[LogisticRegression, float]:
    """Choose the regularisation on the validation products, then keep that model."""
    best, best_c, best_f1 = None, None, -1.0
    for c in grid:
        m = LogisticRegression(C=c, max_iter=3000).fit(Xtr, ytr)
        f = f1_score(yva, m.predict(Xva), average="macro")
        if f > best_f1:
            best, best_c, best_f1 = m, c, f
    return best, best_c


def bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = BOOTSTRAPS) -> dict:
    """Paired bootstrap of the accuracy difference A - B over the same resampled test products."""
    rng = np.random.default_rng(SEED)
    ca, cb = (a.argmax(1) == y).astype(float), (b.argmax(1) == y).astype(float)
    diff = ca - cb
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "verdict": "better" if lo > 0 else "worse" if hi < 0 else "inconclusive"}


def main() -> None:
    t0 = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()
    report = CrispDmReport(
        project_slug="14_multimodal_automl", project_title="Multimodal AutoML",
        business_question=(
            "Can a catalogue classify products by type from their photo, name and attributes, "
            "and does combining them beat the best single source?"
        ),
    )
    report.phase(
        "business_understanding",
        objective="Measure what each modality contributes to classifying products, and whether fusion pays.",
        narrative=(
            "A retailer receives products from many suppliers, and each has to be filed under the "
            "right article type. There are three kinds of evidence: a photograph, a free-text "
            "name and a few catalogue fields. Multimodal AutoML promises to combine them "
            "automatically. This project measures each source alone, fuses them several ways, "
            "and asks whether fusion beats the strongest single source by more than chance -- "
            "and how much of the text's apparent strength is simply the label written in the name."
        ),
        code_refs=["ml/train.py:main"],
        gate_notes="Target, modalities and the leak to be measured fixed before modelling.",
    ).add("Target", "article type").add("Modalities", "image, name, catalogue fields")

    print("[1/9] loading products and CLIP embeddings ...")
    df, prov = load()
    prov.save(ARTIFACTS / "provenance.json")
    emb = cached(df["id"].to_numpy(), df["image_bytes"].tolist(), df["productDisplayName"].fillna("").tolist())
    counts = df[TARGET].value_counts()
    keep = df[TARGET].isin(counts[counts >= MIN_PER_CLASS].index).to_numpy()
    data = df[keep].reset_index(drop=True)
    img_all, txt_all = emb["image"][keep], emb["text"][keep]
    classes = sorted(data[TARGET].unique())
    y_all = data[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    vocab = set().union(*(label_words(c) for c in classes))
    data["masked_name"] = [mask_label_words(n, vocab) for n in data["productDisplayName"]]
    name_words = [set(re.findall(r"[a-z]+", (n or "").lower())) for n in data["productDisplayName"]]
    contains_own = float(np.mean([bool(label_words(lbl) & w) for lbl, w in zip(data[TARGET], name_words)]))
    dup_names = int(data["name_key"].duplicated().sum())

    report.phase(
        "data_understanding",
        objective="Find what in the catalogue would flatter a model.",
        narrative=(
            f"{len(df):,} products in {df[TARGET].nunique()} article types. The {len(classes)} types with at "
            f"least {MIN_PER_CLASS} products are kept: {len(data):,} products.\n\n"
            f"**The name often states the answer.** {contains_own * 100:.1f}% of names contain a word of their "
            f"own article type. **Siblings share names.** {dup_names:,} products repeat a display name "
            f"already used by another."
        ),
        code_refs=["ml/data.py:load", "ml/data.py:label_words"],
        gate_notes="Label-in-name rate and duplicate names measured.",
    ).add("Products kept", len(data)).add("Article types", len(classes)) \
     .add("Names containing their label", f"{contains_own * 100:.1f}%").add("Repeated names", dup_names)

    print("[2/9] grouped split ...")
    groups = data["name_key"].to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trval_idx, test_idx = next(outer.split(data, y_all, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=SEED)
    tr_rel, va_rel = next(inner.split(trval_idx, y_all[trval_idx], groups[trval_idx]))
    train_idx, val_idx = trval_idx[tr_rel], trval_idx[va_rel]
    crossing = len(set(groups[train_idx]) & set(groups[test_idx])) + len(set(groups[val_idx]) & set(groups[test_idx])) \
        + len(set(groups[train_idx]) & set(groups[val_idx]))
    leakage.require("assert_no_name_in_two_splits", crossing == 0,
                    f"{len(train_idx):,} training, {len(val_idx):,} validation and {len(test_idx):,} test products "
                    f"grouped by display name; {crossing} names appear in more than one split.",
                    train=len(train_idx), validation=len(val_idx), test=len(test_idx), crossing_names=crossing)
    missing = sorted(set(range(len(classes))) - set(y_all[test_idx]))
    leakage.require("assert_every_class_in_test", not missing,
                    f"All {len(classes)} article types have test products; missing: {missing}.", classes=len(classes))
    leakage.require("assert_hierarchy_fields_excluded", not {"masterCategory", "subCategory"} & set(TABULAR),
                    f"Catalogue features {TABULAR}; the coarser category levels, which determine most of the "
                    f"article type, are excluded.", features=len(TABULAR))
    ytr, yva, yte = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    run = RunRecord(project_slug="14_multimodal_automl", project_title="Multimodal AutoML",
                    task=f"{len(classes)}-class product classification from image, text and catalogue fields",
                    primary_metric="macro_f1", higher_is_better=True)
    run.set_provenance(prov)
    run.set_split(SplitInfo(strategy="grouped by display name: 70% train, 15% validation, 15% test",
                            n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx)))
    probs: dict[str, dict[str, np.ndarray]] = {}

    def enter(name, family, p_val, p_test, fit_s, notes="", **params):
        run.add_model(ModelScore(name=name, family=family, metrics=scores(yte, p_test), fit_seconds=round(fit_s, 1),
                                 notes=notes, params=params))
        probs[name] = {"val": p_val, "test": p_test}
        m = run.models[-1].metrics
        print(f"      {name:<40} macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}")

    print("[3/9] baselines and single modalities ...")
    prior = np.bincount(ytr, minlength=len(classes)) / len(ytr)
    enter("Most common article type", "baseline", np.tile(prior, (len(val_idx), 1)), np.tile(prior, (len(test_idx), 1)), 0,
          "Predicts the most frequent training class for everything.")

    model, processor = load_clip()
    prompts = [f"a photo of {c.lower()}" for c in classes]
    prompt_emb = embed_texts(model, processor, prompts)
    zs = lambda idx: torch.softmax(torch.tensor(img_all[idx] @ prompt_emb.T * 100.0), dim=1).numpy()
    enter("CLIP zero-shot, image vs class names", "baseline", zs(val_idx), zs(test_idx), 0,
          "No training on this catalogue: each image is matched to 'a photo of <article type>'.")

    t = time.perf_counter()
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    tab = data[TABULAR].astype(str)
    Ttr, Tva, Tte = ohe.fit_transform(tab.iloc[train_idx]), ohe.transform(tab.iloc[val_idx]), ohe.transform(tab.iloc[test_idx])
    gbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=8, verbose=-1)
    gbm.fit(Ttr.astype(np.float32), ytr, eval_set=[(Tva.astype(np.float32), yva)],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    enter("Catalogue fields only (LightGBM)", "tabular", gbm.predict_proba(Tva.astype(np.float32)),
          gbm.predict_proba(Tte.astype(np.float32)), time.perf_counter() - t, f"Fields: {', '.join(TABULAR)}.")

    fitted: dict[str, object] = {}
    for label, column in (("Name only", "productDisplayName"), ("Name only, label words removed", "masked_name")):
        t = time.perf_counter()
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        texts = data[column].fillna("")
        Xtr = vec.fit_transform(texts.iloc[train_idx])
        m, c = logistic(Xtr, ytr, vec.transform(texts.iloc[val_idx]), yva)
        enter(f"{label} (TF-IDF)", "text", m.predict_proba(vec.transform(texts.iloc[val_idx])),
              m.predict_proba(vec.transform(texts.iloc[test_idx])), time.perf_counter() - t,
              "Every word appearing in any article-type name is removed first." if "removed" in label else "",
              C=c)
        fitted[label] = (vec, m)

    t = time.perf_counter()
    img_model, img_c = logistic(img_all[train_idx], ytr, img_all[val_idx], yva)
    enter("Image only (CLIP + logistic)", "image", img_model.predict_proba(img_all[val_idx]),
          img_model.predict_proba(img_all[test_idx]), time.perf_counter() - t, f"{MODEL} image embeddings.", C=img_c)

    print("[4/9] late fusion ...")
    def late(parts: list[str], label: str, note: str):
        best_w, best_f = None, -1.0
        grid = np.linspace(0, 1, 11)
        for w in (np.array(v) for v in np.array(np.meshgrid(*[grid] * len(parts))).T.reshape(-1, len(parts))):
            if not np.isclose(w.sum(), 1):
                continue
            p = sum(wi * np.log(probs[n]["val"] + 1e-9) for wi, n in zip(w, parts))
            f = f1_score(yva, p.argmax(1), average="macro")
            if f > best_f:
                best_w, best_f = w, f
        combine = lambda split: np.exp(sum(wi * np.log(probs[n][split] + 1e-9) for wi, n in zip(best_w, parts)))
        pv, pt = combine("val"), combine("test")
        enter(label, "fusion", pv / pv.sum(1, keepdims=True), pt / pt.sum(1, keepdims=True), 0, note,
              **{f"weight: {n}": round(float(wi), 2) for n, wi in zip(parts, best_w)})

    late(["Image only (CLIP + logistic)", "Name only, label words removed (TF-IDF)", "Catalogue fields only (LightGBM)"],
         "Late fusion: image + masked name + fields", "Weighted geometric mean of probabilities; weights chosen on validation.")
    late(["Image only (CLIP + logistic)", "Name only (TF-IDF)", "Catalogue fields only (LightGBM)"],
         "Late fusion: image + name + fields", "Weighted geometric mean of probabilities; weights chosen on validation.")

    print("[5/9] early fusion and features for AutoGluon ...")
    t = time.perf_counter()
    vec_full, _ = fitted["Name only"]
    svd = TruncatedSVD(128, random_state=SEED).fit(vec_full.transform(data["productDisplayName"].fillna("").iloc[train_idx]))
    name_svd = svd.transform(vec_full.transform(data["productDisplayName"].fillna("")))
    fused = sparse.hstack([sparse.csr_matrix(img_all), sparse.csr_matrix(name_svd * 3), ohe.transform(tab)]).tocsr()
    early, early_c = logistic(fused[train_idx], ytr, fused[val_idx], yva)
    enter("Early fusion: image + name + fields", "fusion", early.predict_proba(fused[val_idx]), early.predict_proba(fused[test_idx]),
          time.perf_counter() - t, "Concatenated CLIP image embedding, 128-dimension name SVD and one-hot fields.", C=early_c)

    feats = pd.DataFrame(np.hstack([img_all, name_svd]), columns=[f"img{i}" for i in range(512)] + [f"name{i}" for i in range(128)])
    for c in TABULAR:
        feats[c] = data[c].astype(str)
    feats["label"] = data[TARGET]
    feats["split"] = np.where(np.isin(np.arange(len(data)), test_idx), "test", np.where(np.isin(np.arange(len(data)), val_idx), "val", "train"))
    feats["id"] = data["id"]
    feats.to_parquet(WORK / "fused_features.parquet")
    ag_path = ARTIFACTS / "autogluon.json"
    autogluon = None
    if ag_path.exists():
        ag = json.loads(ag_path.read_text(encoding="utf-8"))
        if ag["test_ids"] != data["id"].iloc[test_idx].astype(int).tolist() or ag["classes"] != classes:
            raise RuntimeError("autogluon.json was produced for different test products; rerun ml/autogluon_run.py")
        p = np.asarray(ag["test_proba"])
        enter("AutoGluon on fused features", "AutoML", p, p, ag["fit_seconds"],
              f"AutoGluon {ag['autogluon_version']}, {ag['presets']}, {ag['time_limit_seconds']}s; best {ag['best_model']}.")
        autogluon = {k: ag[k] for k in ("autogluon_version", "presets", "time_limit_seconds", "fit_seconds", "best_model", "models_trained")}
    else:
        print("      artifacts/autogluon.json not found; run ml/autogluon_run.py and train again to include AutoGluon.")

    print("[6/9] the name leak and the split leak ...")
    rand_tr, rand_te = train_test_split(np.arange(len(data)), test_size=0.15, stratify=y_all, random_state=SEED)
    vec_r = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    names = data["productDisplayName"].fillna("")
    lr_r = LogisticRegression(C=10.0, max_iter=3000).fit(vec_r.fit_transform(names.iloc[rand_tr]), y_all[rand_tr])
    random_split = scores(y_all[rand_te], lr_r.predict_proba(vec_r.transform(names.iloc[rand_te])))
    random_split["test_names_seen_in_train_pct"] = float(np.isin(groups[rand_te], groups[rand_tr]).mean() * 100)
    name_full = run.models[[m.name for m in run.models].index("Name only (TF-IDF)")].metrics
    name_masked = run.models[[m.name for m in run.models].index("Name only, label words removed (TF-IDF)")].metrics
    leaks = {"label_in_name_pct": contains_own * 100, "duplicate_names": dup_names,
             "name_full": name_full, "name_masked": name_masked,
             "random_split_name_model": random_split, "grouped_split_name_model": name_full}

    print(f"[7/9] significance ({BOOTSTRAPS} paired bootstraps), per-class results, search ...")
    single = ["Image only (CLIP + logistic)", "Name only, label words removed (TF-IDF)", "Catalogue fields only (LightGBM)"]
    best_single = max(single, key=lambda n: run.models[[m.name for m in run.models].index(n)].metrics["accuracy"])
    contenders = [m.name for m in run.models if m.family in ("fusion", "AutoML")] + ["Name only (TF-IDF)"]
    significance = [{"a": n, "b": best_single, **bootstrap(yte, probs[n]["test"], probs[best_single]["test"])} for n in contenders]
    for s in significance:
        print(f"      {s['a']} vs {s['b']}: {s['diff']:+.4f} [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] {s['verdict']}")

    fusion_name = "Late fusion: image + masked name + fields"
    f1_img = f1_score(yte, probs["Image only (CLIP + logistic)"]["test"].argmax(1), average=None, labels=range(len(classes)))
    f1_fus = f1_score(yte, probs[fusion_name]["test"].argmax(1), average=None, labels=range(len(classes)))
    support = np.bincount(yte, minlength=len(classes))
    per_class = sorted([{"article_type": classes[i], "test_products": int(support[i]), "image_f1": float(f1_img[i]),
                         "fusion_f1": float(f1_fus[i]), "gain": float(f1_fus[i] - f1_img[i])} for i in range(len(classes))],
                       key=lambda r: -r["gain"])
    cm = confusion_matrix(yte, probs["Image only (CLIP + logistic)"]["test"].argmax(1), labels=range(len(classes)))
    np.fill_diagonal(cm, 0)
    confusions = [{"actual": classes[i], "predicted": classes[j], "products": int(cm[i, j])}
                  for i, j in zip(*np.unravel_index(np.argsort(-cm, axis=None)[:12], cm.shape)) if cm[i, j] > 0]

    # Text-to-image search: can each test product's name (with its label words removed) find its own photo?
    masked_emb = embed_texts(model, processor, data["masked_name"].iloc[test_idx].replace("", "product").tolist())
    sims = masked_emb @ img_all[test_idx].T
    rank = (sims > sims[np.arange(len(test_idx)), np.arange(len(test_idx))][:, None]).sum(1)
    full_sims = txt_all[test_idx] @ img_all[test_idx].T
    full_rank = (full_sims > full_sims[np.arange(len(test_idx)), np.arange(len(test_idx))][:, None]).sum(1)
    search = {"gallery": int(len(test_idx)),
              "recall_at_10_full_name": float((full_rank < 10).mean()), "recall_at_10_masked_name": float((rank < 10).mean()),
              "median_rank_full_name": float(np.median(full_rank) + 1), "median_rank_masked_name": float(np.median(rank) + 1),
              "chance_at_10": 10 / len(test_idx)}

    print("[8/9] audit and report ...")
    own = leakage.audit_source(ROOT, temporal_modules=())
    leakage.save_report(own, ARTIFACTS / "audit.md", "Multimodal AutoML")
    crit = [f for f in own if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {"critical": len(crit), "warnings": sum(1 for f in own if f.severity == "WARNING"),
                 "total": len(own), "runtime_assertions": executed,
                 "assertions_passed": sum(1 for a in executed if a["passed"]), "assertions_run": len(executed)}

    M = {m.name: m.metrics for m in run.models}
    best = run.best
    fus_sig = next(s for s in significance if s["a"] == fusion_name)
    report.phase(
        "data_preparation",
        objective="Split so siblings never straddle train and test, and build each modality's features.",
        narrative=(
            f"Products are grouped by display name before splitting: {len(train_idx):,} train, {len(val_idx):,} "
            f"validation (used only to choose regularisation and fusion weights) and {len(test_idx):,} test. Images "
            f"and names are embedded once with {MODEL}. Names are also represented with TF-IDF, with a second "
            f"copy in which every word of any article-type name is removed. Catalogue features are "
            f"{', '.join(TABULAR)}; the coarser category levels are excluded because they nearly determine the answer."
        ),
        code_refs=["ml/train.py:main", "ml/embed.py:cached", "ml/data.py:mask_label_words"],
        gate_notes="Executed assertions: no shared names across splits, every class tested, hierarchy fields excluded.",
    ).add("Train / validation / test", f"{len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}")
    report.phase(
        "modeling",
        objective="Model each modality alone, then fuse them late, early and with AutoGluon.",
        narrative=(
            "Single-modality models: logistic regression on CLIP image embeddings, logistic regression on "
            "TF-IDF names (full and masked), and LightGBM on catalogue fields, beside a majority-class "
            "baseline and CLIP zero-shot matching against the class names. Late fusion combines "
            "probabilities with weights chosen on validation; early fusion concatenates features into one "
            "logistic regression" + ("; AutoGluon trains on the same fused features." if autogluon else ".")
        ),
        code_refs=["ml/train.py:logistic", "ml/train.py:main"],
        gate_notes=f"{len(run.models)} contestants scored on the same test products.",
    ).add("Contestants", len(run.models))
    report.phase(
        "evaluation",
        objective="Separate what fusion adds from what the name gives away.",
        narrative=(
            f"**{best.name}** has the highest macro-F1, {best.metrics['macro_f1']:.4f}. The image alone reaches "
            f"{M['Image only (CLIP + logistic)']['macro_f1']:.4f}; CLIP zero-shot, with no training, "
            f"{M['CLIP zero-shot, image vs class names']['macro_f1']:.4f}.\n\n"
            f"**The name leak.** The name model's macro-F1 falls from {name_full['macro_f1']:.4f} to "
            f"{name_masked['macro_f1']:.4f} when label words are removed. On a random split, where "
            f"{random_split['test_names_seen_in_train_pct']:.1f}% of test names also occur in training, it would have "
            f"reported {random_split['macro_f1']:.4f}.\n\n"
            f"**Does fusion pay?** Against the best honest single modality ({best_single}), late fusion of "
            f"image, masked name and fields changes accuracy by {fus_sig['diff']:+.4f} "
            f"(95% interval {fus_sig['ci_low']:+.4f} to {fus_sig['ci_high']:+.4f}): **{fus_sig['verdict']}**."
        ),
        code_refs=["ml/train.py:bootstrap", "ml/train.py:scores"],
        gate_passed=len(crit) == 0 and all(a["passed"] for a in executed),
        gate_notes="Paired bootstrap against the best single modality; leaks measured, not assumed.",
    ).add("Best macro-F1", f"{best.metrics['macro_f1']:.4f}").add("Fusion vs best single", fus_sig["verdict"])
    report.phase(
        "deployment",
        objective="Serve the gallery, per-modality predictions and text search.",
        narrative=(
            "The API serves test products with every contestant's prediction, classifies any typed name with "
            "the text models, and searches test images by description with CLIP."
        ),
        code_refs=["server/inference.py", "server/main.py"],
        gate_notes="Test-set predictions and CLIP search served from the training run's artifacts.",
    ).add("Serving port", 8014)

    test_ids = data["id"].iloc[test_idx].to_numpy()
    np.savez_compressed(WORK / "served.npz", ids=test_ids, labels=yte, image_emb=img_all[test_idx],
                        **{f"p{i}": probs[m.name]["test"].astype(np.float32) for i, m in enumerate(run.models)})
    joblib.dump({"classes": classes, "text_models": fitted, "model_names": [m.name for m in run.models], "vocab": vocab},
                ARTIFACTS / "model.joblib")
    run.extra = {
        "classes": classes, "class_counts": {c: int(n) for c, n in zip(classes, np.bincount(y_all, minlength=len(classes)))},
        "leaks": leaks, "significance": significance, "best_single": best_single, "per_class": per_class,
        "image_confusions": confusions, "search": search, "autogluon": autogluon,
        "clip": {"model": MODEL, "embedding_seconds": emb["seconds"], "products": int(len(df))},
        "fusion_reported": fusion_name,
    }
    run.duration_seconds = round(time.perf_counter() - t0, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")
    print("[9/9] done.")
    print(run.leaderboard_markdown(("accuracy", "top3_accuracy")))
    print(json.dumps({"search": search, "random_split": random_split}, indent=1))
    print(f"critical {len(crit)} | {run.duration_seconds}s")


if __name__ == "__main__":
    main()

"""
Project 02 -- train a small language model on TinyStories, and measure it honestly.

    python ml/train.py

1. Deduplicate: validation stories that also occur in the training file are
   removed before any score is computed.
2. Train a byte-level BPE tokenizer and encode the corpus.
3. Score n-gram baselines, so perplexity has something to be compared with.
4. Train the main model (RoPE, SwiGLU, RMSNorm).
5. Ablate each component on a shorter budget, with a second seed of the
   reference model to show how much of any difference is noise.
6. Check the causal mask and the KV cache by execution, and sample stories.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from dsfs import CrispDmReport, ModelScore, RunRecord, SplitInfo, leakage

from data import ARTIFACTS, SEPARATOR, VOCAB_SIZE, download, encode_to_file, stories, story_hash
from model import Config, NanoLM

ROOT = Path(__file__).resolve().parents[1]
WORK = ARTIFACTS / "work"
SEED = 1337
TRAIN_TOKENS_CAP = 320_000_000
TOKENIZER_STORIES = 200_000
BATCH, CONTEXT = 64, 256
MAIN_STEPS, ABLATION_STEPS = 12_000, 3_000
EVAL_EVERY, EVAL_BATCHES = 500, 100
PROMPTS = [
    "Once upon a time, there was a little girl named Lily. She loved",
    "Tom and his dog went to the park. Suddenly,",
    "The old tree in the garden was sad because",
    "One day, a small bird found a shiny key.",
    "Mia wanted to bake a cake for her mom, but",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def batches(tokens: np.ndarray, rng: np.random.Generator, batch: int = BATCH, context: int = CONTEXT):
    starts = rng.integers(0, len(tokens) - context - 1, batch)
    x = np.stack([tokens[s:s + context] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1:s + context + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(DEVICE, non_blocking=True), torch.from_numpy(y).to(DEVICE, non_blocking=True)


@torch.no_grad()
def evaluate(model: NanoLM, valid: np.ndarray) -> float:
    """Mean cross-entropy (nats per token) over fixed validation windows."""
    model.eval()
    rng = np.random.default_rng(0)  # the same windows every time, for every model
    total = 0.0
    for _ in range(EVAL_BATCHES):
        x, y = batches(valid, rng)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            _, loss, _ = model(x, y)
        total += float(loss)
    model.train()
    return total / EVAL_BATCHES


@torch.no_grad()
def evaluate_full(model: NanoLM, valid: np.ndarray) -> float:
    """
    Cross-entropy over every validation token, in consecutive windows. This is
    the score reported, so the model and the n-gram baselines are judged on
    exactly the same tokens.
    """
    model.eval()
    tokens = np.asarray(valid).astype(np.int64)
    total, count = 0.0, 0
    n_windows = (len(tokens) - 1) // CONTEXT
    for b in range(0, n_windows, BATCH):
        idx = np.arange(b, min(b + BATCH, n_windows))
        x = np.stack([tokens[i * CONTEXT:i * CONTEXT + CONTEXT] for i in idx])
        y = np.stack([tokens[i * CONTEXT + 1:i * CONTEXT + CONTEXT + 1] for i in idx])
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits, _, _ = model(torch.from_numpy(x).to(DEVICE))
        loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                                 torch.from_numpy(y).to(DEVICE).view(-1), reduction="sum")
        total += float(loss)
        count += y.size
    model.train()
    return total / count


def train_model(cfg: Config, steps: int, train: np.ndarray, valid: np.ndarray, seed: int, label: str,
                lr: float = 1e-3, warmup: int = 300) -> tuple[NanoLM, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NanoLM(cfg).to(DEVICE)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=DEVICE == "cuda")
    curve, t0, seen, last_train = [], time.perf_counter(), 0, float("nan")
    for step in range(1, steps + 1):
        scale = step / warmup if step <= warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr * max(scale, 0.05)
        x, y = batches(train, rng)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += x.numel()
        last_train = float(loss) if step % 50 == 0 else last_train
        if step % EVAL_EVERY == 0 or step == steps:
            val = evaluate(model, valid)
            elapsed = time.perf_counter() - t0
            curve.append({"step": step, "tokens_m": seen / 1e6, "train_loss": last_train, "val_loss": val,
                          "val_perplexity": math.exp(val), "minutes": elapsed / 60})
            print(f"      {label:<28} step {step:>6}  train {last_train:.3f}  val {val:.3f}  "
                  f"ppl {math.exp(val):6.2f}  {seen / elapsed / 1e3:,.0f}k tok/s")
    elapsed = time.perf_counter() - t0
    full = evaluate_full(model, valid)
    return model, {"curve": curve, "seconds": elapsed, "tokens": seen, "tokens_per_second": seen / elapsed,
                   "final_val_loss": full, "params": model.n_params()}


def ngram_cross_entropy(train: np.ndarray, valid: np.ndarray, vocab: int) -> dict:
    """
    Add-one unigram, and a bigram interpolated with it. The interpolation weight
    is chosen on the last two million training tokens, which are held out of
    the counts so the choice is not made on data the counts already contain.
    """
    tune = train[-2_000_001:]
    train = train[:-2_000_001]
    uni = np.bincount(train, minlength=vocab).astype(np.float64) + 1
    p_uni = uni / uni.sum()
    bi = np.zeros(vocab * vocab, dtype=np.float64)
    step = 20_000_000
    for i in range(0, len(train) - 1, step):
        chunk = train[i:i + step + 1].astype(np.int64)
        bi += np.bincount(chunk[:-1] * vocab + chunk[1:], minlength=vocab * vocab)
    bi = bi.reshape(vocab, vocab)
    row = bi.sum(1, keepdims=True)
    p_bi = np.divide(bi, row, out=np.zeros_like(bi), where=row > 0)

    def ce(tokens, lam):
        prev, nxt = tokens[:-1].astype(np.int64), tokens[1:].astype(np.int64)
        return float(-np.log(lam * p_bi[prev, nxt] + (1 - lam) * p_uni[nxt]).mean())

    lam = min((0.5, 0.7, 0.8, 0.9, 0.95, 0.99), key=lambda l: ce(tune, l))
    return {"unigram": float(-np.log(p_uni[valid[1:].astype(np.int64)]).mean()), "bigram": ce(valid, lam), "lambda": lam}


def main() -> None:
    t_start = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    leakage.reset_ledger()
    report = CrispDmReport(
        project_slug="02_nano_llm", project_title="Nano LLM Transformer",
        business_question=(
            f"How good a storyteller can a {NanoLM(Config()).n_params() / 1e6:.1f}-million-parameter transformer become on one "
            "laptop GPU, and which of its modern components actually earn their place?"
        ),
    )
    report.phase(
        "business_understanding",
        objective="Train a small language model from scratch and measure it against baselines and ablations.",
        narrative=(
            "Modern language models share a recipe: rotary position embeddings, gated "
            "feed-forward layers and RMSNorm. This project implements that recipe from "
            "scratch, trains it on children's stories, and asks two questions a demo usually "
            "skips. How much better is it than simple word statistics? And does each piece "
            "of the recipe help at this scale, or is it carried along by habit?"
        ),
        code_refs=["ml/model.py:NanoLM", "ml/train.py:main"],
        gate_notes="Validation perplexity on deduplicated stories fixed as the measure before training.",
    ).add("Measure", "validation perplexity").add("Hardware", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU")

    print("[1/8] downloading TinyStories ...")
    train_path, train_prov, valid_path, valid_prov = download()
    train_prov.save(ARTIFACTS / "provenance.json")

    print("[2/8] hashing stories to find validation stories that were also trained on ...")
    valid_stories = list(stories(valid_path))
    valid_hashes = {story_hash(s): i for i, s in enumerate(valid_stories)}
    seen_train: set[str] = set()
    n_train_stories = train_dupes = 0
    in_train: set[str] = set()
    for s in stories(train_path):
        h = story_hash(s)
        n_train_stories += 1
        if h in seen_train:
            train_dupes += 1
        else:
            seen_train.add(h)
        if h in valid_hashes:
            in_train.add(h)
    clean_valid = [s for s in valid_stories if story_hash(s) not in in_train]
    # Checked against every training story's hash, independently of the removal list.
    remaining = sum(1 for s in clean_valid if story_hash(s) in seen_train)
    del seen_train
    leakage.require(
        "assert_validation_stories_unseen", remaining == 0,
        f"{len(valid_stories):,} validation stories hashed against {n_train_stories:,} training stories; "
        f"{len(in_train):,} also appeared in training and were removed; {remaining} remain.",
        validation_stories=len(valid_stories), training_stories=n_train_stories,
        removed=len(in_train), remaining=remaining,
    )
    dedup = {"train_stories": n_train_stories, "train_duplicate_copies": train_dupes,
             "valid_stories": len(valid_stories), "valid_in_train": len(in_train), "valid_scored": len(clean_valid)}
    print(f"      {dedup}")
    report.phase(
        "data_understanding",
        objective="Measure whether the corpus repeats itself, and keep any repeats out of the score.",
        narrative=(
            f"{n_train_stories:,} training stories and {len(valid_stories):,} validation stories. "
            f"A generated corpus can repeat itself, so every story was hashed after "
            f"whitespace normalisation rather than assumed unique: {train_dupes:,} training "
            f"{'story is an exact copy' if train_dupes == 1 else 'stories are exact copies'} of an "
            f"earlier one, and **{len(in_train):,} validation stories appear in the training "
            f"file**. {'Those are removed, leaving' if in_train else 'All'} "
            f"{len(clean_valid):,} validation stories {'the model has never seen' if in_train else 'are scored'}."
        ),
        code_refs=["ml/data.py:story_hash", "ml/train.py:main"],
        gate_notes="Validation-training overlap measured and removed by an executed assertion.",
    ).add("Training stories", n_train_stories).add("Validation stories removed as seen", len(in_train))

    print("[3/8] tokenizer and encoding ...")
    tok_path = ARTIFACTS / "tokenizer.json"
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    def first_stories(n):
        for i, s in enumerate(stories(train_path)):
            if i >= n:
                return
            yield s

    tokenizer.train_from_iterator(first_stories(TOKENIZER_STORIES), trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, special_tokens=[SEPARATOR], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    tokenizer.save(str(tok_path))
    eot = tokenizer.token_to_id(SEPARATOR)

    def capped_train():
        budget_chars = TRAIN_TOKENS_CAP * 4.2
        used = 0
        for s in stories(train_path):
            if used > budget_chars:
                return
            used += len(s)
            yield s

    n_train_tokens = encode_to_file(tokenizer, capped_train(), WORK / "train.bin", eot)
    n_valid_tokens = encode_to_file(tokenizer, clean_valid, WORK / "valid.bin", eot)
    train_tok = np.memmap(WORK / "train.bin", dtype=np.uint16, mode="r")
    valid_tok = np.memmap(WORK / "valid.bin", dtype=np.uint16, mode="r")
    chars_per_token = sum(len(s) for s in clean_valid) / (n_valid_tokens - len(clean_valid))
    report.phase(
        "data_preparation",
        objective="Turn text into tokens with a vocabulary learned from the training stories only.",
        narrative=(
            f"A byte-level BPE tokenizer with {VOCAB_SIZE:,} tokens is trained on the first "
            f"{TOKENIZER_STORIES:,} training stories; byte-level means no character can ever be "
            f"out of vocabulary. The model trains on the first {n_train_tokens / 1e6:,.0f}M "
            f"tokens of the training file and is scored on {n_valid_tokens / 1e6:,.1f}M tokens "
            f"of unseen validation stories, averaging {chars_per_token:.2f} characters per token."
        ),
        code_refs=["ml/data.py:encode_to_file"],
        gate_notes="Tokenizer fitted on training stories only.",
    ).add("Vocabulary", VOCAB_SIZE).add("Training tokens", n_train_tokens).add("Validation tokens", n_valid_tokens)

    print("[4/8] n-gram baselines ...")
    # Scored on the same tokens evaluate_full uses: every whole window of the validation stream.
    scored = np.asarray(valid_tok[: ((len(valid_tok) - 1) // CONTEXT) * CONTEXT + 1])
    ngram = ngram_cross_entropy(np.asarray(train_tok[:200_000_000]), scored, VOCAB_SIZE)
    print(f"      unigram ppl {math.exp(ngram['unigram']):.1f}  bigram ppl {math.exp(ngram['bigram']):.1f}")

    run = RunRecord(project_slug="02_nano_llm", project_title="Nano LLM Transformer",
                    task="next-token prediction (language modelling)", primary_metric="val_perplexity",
                    higher_is_better=False)
    run.set_provenance(train_prov)
    run.set_split(SplitInfo(strategy="official TinyStories V2 split; validation stories seen in training removed",
                            n_train=int(n_train_tokens), n_test=int(n_valid_tokens),
                            notes="Counts are tokens."))
    for name, key, note in (("Unigram frequencies", "unigram", "Each token's training frequency, ignoring context."),
                            ("Bigram, interpolated", "bigram", f"Previous-token statistics blended with unigram, λ={ngram['lambda']}.")):
        run.add_model(ModelScore(name=name, family="baseline", notes=note,
                                 metrics={"val_perplexity": math.exp(ngram[key]), "val_loss": ngram[key],
                                          "bits_per_char": ngram[key] / math.log(2) / chars_per_token}))

    print(f"[5/8] main model, {MAIN_STEPS:,} steps ...")
    main_cfg = Config(vocab_size=VOCAB_SIZE, context=CONTEXT)
    model, main_log = train_model(main_cfg, MAIN_STEPS, train_tok, valid_tok, SEED, "main (RoPE·SwiGLU·RMS)")
    torch.save({"config": main_cfg.to_dict(), "state": model.state_dict()}, ARTIFACTS / "model.pt")
    run.add_model(ModelScore(
        name="NanoLM, RoPE · SwiGLU · RMSNorm", family="transformer", fit_seconds=round(main_log["seconds"], 1),
        metrics={"val_perplexity": math.exp(main_log["final_val_loss"]), "val_loss": main_log["final_val_loss"],
                 "bits_per_char": main_log["final_val_loss"] / math.log(2) / chars_per_token,
                 "tokens_seen_m": main_log["tokens"] / 1e6},
        params={**main_cfg.to_dict(), "parameters": main_log["params"], "steps": MAIN_STEPS, "batch": BATCH}))

    print(f"[6/8] ablations, {ABLATION_STEPS:,} steps each ...")
    variants = [
        ("Reference: RoPE · SwiGLU · RMSNorm", dict(), SEED),
        ("Reference, second seed", dict(), SEED + 1),
        ("Learned positions instead of RoPE", dict(position="learned"), SEED),
        ("GELU instead of SwiGLU", dict(mlp="gelu"), SEED),
        ("LayerNorm instead of RMSNorm", dict(norm="layer"), SEED),
        ("GPT-2 style: learned · GELU · LayerNorm", dict(position="learned", mlp="gelu", norm="layer"), SEED),
    ]
    ablations = []
    for label, change, seed in variants:
        cfg = Config(vocab_size=VOCAB_SIZE, context=CONTEXT, **change)
        m, log = train_model(cfg, ABLATION_STEPS, train_tok, valid_tok, seed, label[:28])
        ablations.append({"variant": label, "change": change, "seed": seed, "params": log["params"],
                          "val_loss": log["final_val_loss"], "val_perplexity": math.exp(log["final_val_loss"]),
                          "tokens_per_second": log["tokens_per_second"], "seconds": log["seconds"],
                          "curve": log["curve"]})
        del m
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
    ref = ablations[0]["val_loss"]
    seed_gap = abs(ablations[1]["val_loss"] - ref)
    for a in ablations:
        a["loss_vs_reference"] = a["val_loss"] - ref
        a["beyond_seed_noise"] = abs(a["loss_vs_reference"]) > seed_gap

    print("[7/8] executed checks: causal mask, KV cache, and samples ...")
    model.eval()
    probe = torch.from_numpy(np.asarray(valid_tok[:64]).astype(np.int64))[None].to(DEVICE)
    altered = probe.clone()
    altered[0, 40:] = (altered[0, 40:] + 17) % VOCAB_SIZE
    with torch.no_grad():
        a_logits, _, _ = model(probe)
        b_logits, _, _ = model(altered)
    max_change = float((a_logits[0, :40] - b_logits[0, :40]).abs().max())
    leakage.require(
        "assert_causal_mask", max_change < 1e-4,
        f"Tokens 40-63 of a 64-token sequence replaced; largest change in logits at positions 0-39 "
        f"was {max_change:.2e} (limit 1e-4). The model cannot see the future.",
        changed_from=40, max_logit_change=max_change,
    )

    encode = lambda text: tokenizer.encode(text).ids
    cache_checks, timings = [], []
    for p in PROMPTS[:3]:
        ids = encode(p)
        with_cache = model.generate(ids, 60, temperature=0, repetition_penalty=1.0, use_cache=True)
        without = model.generate(ids, 60, temperature=0, repetition_penalty=1.0, use_cache=False)
        cache_checks.append(with_cache == without)
    leakage.require(
        "assert_kv_cache_matches_recompute", all(cache_checks),
        f"Greedy 60-token continuations of {len(cache_checks)} prompts generated with and without the KV "
        f"cache; {sum(cache_checks)} of {len(cache_checks)} identical.",
        prompts=len(cache_checks), identical=sum(cache_checks),
    )
    ids = encode(PROMPTS[0])
    for use_cache in (True, False):
        model.generate(ids, 20, use_cache=use_cache, seed=0)  # warm-up
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = model.generate(ids, 200, temperature=0.8, repetition_penalty=1.0, use_cache=use_cache, seed=0)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        timings.append({"cache": use_cache, "tokens": len(out), "ms_per_token": (time.perf_counter() - t) / len(out) * 1000})
    samples = []
    for i, p in enumerate(PROMPTS):
        out = model.generate(encode(p), 160, temperature=0.8, top_p=0.9, repetition_penalty=1.1, stop_id=eot, seed=i)
        samples.append({"prompt": p, "continuation": tokenizer.decode([t for t in out if t != eot])})

    print("[8/8] audit, report and artifacts ...")
    own = leakage.audit_source(ROOT, temporal_modules=())
    leakage.save_report(own, ARTIFACTS / "audit.md", "Nano LLM Transformer")
    crit = [f for f in own if f.severity == "CRITICAL"]
    executed = leakage.ledger()
    run.audit = {"critical": len(crit), "warnings": sum(1 for f in own if f.severity == "WARNING"),
                 "total": len(own), "runtime_assertions": executed,
                 "assertions_passed": sum(1 for a in executed if a["passed"]), "assertions_run": len(executed)}

    main_ppl = math.exp(main_log["final_val_loss"])
    report.phase(
        "modeling",
        objective="Train the reference recipe, and each variant that removes one of its components.",
        narrative=(
            f"The main model has {main_log['params']:,} parameters: {main_cfg.n_layers} layers, "
            f"{main_cfg.n_heads} heads, width {main_cfg.d_model}, a {CONTEXT}-token context and tied "
            f"input and output embeddings. It trained for {MAIN_STEPS:,} steps "
            f"({main_log['tokens'] / 1e6:,.0f}M tokens) in {main_log['seconds'] / 60:.0f} minutes at "
            f"{main_log['tokens_per_second'] / 1e3:,.0f}k tokens per second.\n\n"
            f"Each ablation trains for {ABLATION_STEPS:,} steps with one component swapped for its "
            f"older equivalent. The reference was also trained with a second seed: the two differ by "
            f"{seed_gap:.4f} nats per token, which sets how large a difference must be before it "
            f"means anything."
        ),
        code_refs=["ml/model.py:NanoLM", "ml/train.py:train_model"],
        gate_notes=f"Main model and {len(ablations)} ablation runs completed.",
    ).add("Parameters", main_log["params"]).add("Training tokens seen", f"{main_log['tokens'] / 1e6:,.0f}M") \
     .add("Seed-to-seed gap", f"{seed_gap:.4f} nats/token")

    beyond = [a for a in ablations[2:] if a["beyond_seed_noise"]]
    report.phase(
        "evaluation",
        objective="Compare against baselines, separate real ablation effects from seed noise, and verify the mechanics.",
        narrative=(
            f"Validation perplexity is **{main_ppl:.2f}**, against "
            f"{math.exp(ngram['bigram']):.1f} for an interpolated bigram and "
            f"{math.exp(ngram['unigram']):.1f} for unigram frequencies.\n\n"
            f"Of the {len(ablations) - 2} ablations, {len(beyond)} moved validation loss by more than "
            f"the seed-to-seed gap: " + ("; ".join(f"{a['variant']} ({a['loss_vs_reference']:+.4f})" for a in beyond)
                                          or "none") + ". "
            f"Every one of these is a single run at {ABLATION_STEPS:,} steps, so the ranking is "
            f"indicative rather than settled.\n\n"
            f"Two mechanics are checked by execution: changing future tokens leaves earlier logits "
            f"unchanged, and generation with the KV cache produces exactly the tokens full "
            f"recomputation does, at {timings[0]['ms_per_token']:.1f} ms per token against "
            f"{timings[1]['ms_per_token']:.1f} ms."
        ),
        code_refs=["ml/train.py:ngram_cross_entropy", "ml/model.py:NanoLM.generate"],
        gate_passed=len(crit) == 0 and all(a["passed"] for a in executed),
        gate_notes="Baselines, seed-noise comparison and executed mechanics checks.",
    ).add("Validation perplexity", f"{main_ppl:.2f}").add("Bigram perplexity", f"{math.exp(ngram['bigram']):.1f}") \
     .add("KV cache speed-up", f"{timings[1]['ms_per_token'] / timings[0]['ms_per_token']:.1f}×")

    report.phase(
        "deployment",
        objective="Serve generation, tokenisation and attention maps from the trained checkpoint.",
        narrative=(
            "The API loads `model.pt` and `tokenizer.json` and serves `/api/generate` with "
            "temperature, top-p and repetition penalty, `/api/tokenize`, and `/api/attention`, "
            "which returns every head's attention weights for a short prompt."
        ),
        code_refs=["server/inference.py", "server/main.py"],
        gate_notes="Checkpoint and tokenizer persisted together.",
    ).add("Serving port", 8002)

    run.extra = {
        "dedup": dedup, "valid_provenance": valid_prov.to_dict(), "chars_per_token": chars_per_token,
        "ngram": {"unigram_perplexity": math.exp(ngram["unigram"]), "bigram_perplexity": math.exp(ngram["bigram"]),
                  "lambda": ngram["lambda"]},
        "main": {"config": main_cfg.to_dict(), "params": main_log["params"], "curve": main_log["curve"],
                 "seconds": main_log["seconds"], "tokens": main_log["tokens"],
                 "tokens_per_second": main_log["tokens_per_second"], "steps": MAIN_STEPS, "batch": BATCH},
        "ablations": ablations, "seed_gap": seed_gap, "ablation_steps": ABLATION_STEPS,
        "checks": {"causal_max_logit_change": max_change, "cache_identical": sum(cache_checks),
                   "cache_prompts": len(cache_checks), "timings": timings},
        "samples": samples, "prompts": PROMPTS,
        "hardware": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU",
        "torch": torch.__version__,
    }
    run.duration_seconds = round(time.perf_counter() - t_start, 1)
    run.save(ARTIFACTS / "metrics.json")
    report.save(ARTIFACTS / "crispdm.json")
    (ARTIFACTS / "crispdm.md").write_text(report.to_markdown(), encoding="utf-8")
    print(run.leaderboard_markdown(("val_loss", "bits_per_char")))
    for s in samples[:2]:
        print("---", s["prompt"], s["continuation"][:300])
    print(json.dumps({k: v for k, v in run.extra["checks"].items()}, indent=1))


if __name__ == "__main__":
    main()

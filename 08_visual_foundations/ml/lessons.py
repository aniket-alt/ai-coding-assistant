"""
Project 08 -- the algorithms the lessons teach, written out in full.

Each is small enough to read in one sitting and is checked during training
against a trusted reference: Naive Bayes against scikit-learn, gradient descent
against the closed-form least-squares solution, and backpropagation against
finite differences. The API imports this module, so the interactive views run
exactly the code those checks covered.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

TOKEN = re.compile(r"[a-z0-9£$€']+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


# ---------------------------------------------------------------------------
# 1. Naive Bayes
# ---------------------------------------------------------------------------

@dataclass
class NaiveBayes:
    """Multinomial Naive Bayes with add-alpha smoothing."""

    alpha: float = 1.0
    vocab: dict[str, int] = field(default_factory=dict)
    log_prior: np.ndarray = field(default_factory=lambda: np.zeros(2))
    log_likelihood: np.ndarray = field(default_factory=lambda: np.zeros((2, 0)))
    class_docs: list[int] = field(default_factory=lambda: [0, 0])

    def fit(self, texts: list[str], y: np.ndarray) -> "NaiveBayes":
        counts = [Counter(), Counter()]
        for text, label in zip(texts, y):
            counts[int(label)].update(tokenize(text))
        self.vocab = {w: i for i, w in enumerate(sorted(set(counts[0]) | set(counts[1])))}
        matrix = np.zeros((2, len(self.vocab)))
        for c in (0, 1):
            for w, k in counts[c].items():
                matrix[c, self.vocab[w]] = k
        smoothed = matrix + self.alpha
        self.log_likelihood = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        self.class_docs = [int((y == 0).sum()), int((y == 1).sum())]
        self.log_prior = np.log(np.asarray(self.class_docs, dtype=float) / len(y))
        return self

    def token_counts(self, text: str) -> Counter:
        return Counter(t for t in tokenize(text) if t in self.vocab)

    def log_odds(self, text: str) -> float:
        """log P(spam | text) - log P(ham | text). Words never seen in training are ignored."""
        odds = self.log_prior[1] - self.log_prior[0]
        for w, k in self.token_counts(text).items():
            i = self.vocab[w]
            odds += k * (self.log_likelihood[1, i] - self.log_likelihood[0, i])
        return float(odds)

    def spam_probability(self, text: str) -> float:
        z = self.log_odds(text)
        return float(1 / (1 + math.exp(-z))) if z > -700 else 0.0

    def explain(self, text: str) -> dict:
        words = []
        seen = set()
        for t in tokenize(text):
            if t in seen:
                continue
            seen.add(t)
            if t in self.vocab:
                i = self.vocab[t]
                k = self.token_counts(text)[t]
                words.append({"token": t, "count": k, "log_ratio": float(k * (self.log_likelihood[1, i] - self.log_likelihood[0, i])),
                              "p_given_spam": float(math.exp(self.log_likelihood[1, i])),
                              "p_given_ham": float(math.exp(self.log_likelihood[0, i]))})
            else:
                words.append({"token": t, "count": 1, "log_ratio": 0.0, "p_given_spam": None, "p_given_ham": None,
                              "unseen": True})
        prior = float(self.log_prior[1] - self.log_prior[0])
        odds = self.log_odds(text)
        return {"prior_log_odds": prior, "prior_spam": float(math.exp(self.log_prior[1])),
                "words": words, "log_odds": odds, "spam_probability": self.spam_probability(text)}

    def top_tokens(self, n: int, min_count_docs: int = 0) -> dict[str, list[dict]]:
        ratio = self.log_likelihood[1] - self.log_likelihood[0]
        words = sorted(self.vocab, key=self.vocab.get)
        order = np.argsort(ratio)
        return {"spam": [{"token": words[i], "log_ratio": float(ratio[i])} for i in order[::-1][:n]],
                "ham": [{"token": words[i], "log_ratio": float(ratio[i])} for i in order[:n]]}


# ---------------------------------------------------------------------------
# 2. Gradient descent for least squares
# ---------------------------------------------------------------------------

def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X])


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The exact least-squares solution, for gradient descent to be checked against."""
    return np.linalg.lstsq(add_intercept(X), y, rcond=None)[0]


def mse(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    r = add_intercept(X) @ w - y
    return float(r @ r / len(y))


def gradient_descent(X: np.ndarray, y: np.ndarray, lr: float, steps: int, record_every: int = 1) -> dict:
    """
    Batch gradient descent on mean squared error, from zero weights.

    Returns the loss trajectory and whether the run diverged (loss became
    non-finite or grew a thousandfold past its starting value).
    """
    A = add_intercept(X)
    w = np.zeros(A.shape[1])
    start = mse(w, X, y)
    trajectory = []
    diverged = False
    for step in range(steps + 1):
        loss = mse(w, X, y)
        if step % record_every == 0 or step == steps:
            trajectory.append({"step": step, "mse": loss if math.isfinite(loss) else None})
        if not math.isfinite(loss) or loss > 1000 * start:
            diverged = True
            break
        grad = 2 * A.T @ (A @ w - y) / len(y)
        w = w - lr * grad
    return {"weights": w.tolist(), "trajectory": trajectory, "diverged": diverged, "final_mse": mse(w, X, y) if not diverged else None}


def largest_stable_rate(X: np.ndarray) -> float:
    """Gradient descent on MSE converges only when lr < 2 / (largest eigenvalue of 2AᵀA/n)."""
    A = add_intercept(X)
    return float(2 / np.linalg.eigvalsh(2 * A.T @ A / len(A)).max())


# ---------------------------------------------------------------------------
# 3. A small neural network, with backpropagation by hand
# ---------------------------------------------------------------------------

@dataclass
class TinyMLP:
    """inputs -> tanh hidden layer -> linear output, trained on mean squared error."""

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    @classmethod
    def init(cls, n_in: int, n_hidden: int, seed: int) -> "TinyMLP":
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0, 1 / math.sqrt(n_in), (n_hidden, n_in)), np.zeros(n_hidden),
                   rng.normal(0, 1 / math.sqrt(n_hidden), (1, n_hidden)), np.zeros(1))

    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X: np.ndarray) -> dict[str, np.ndarray]:
        z1 = X @ self.W1.T + self.b1
        h = np.tanh(z1)
        out = h @ self.W2.T + self.b2
        return {"x": X, "z1": z1, "h": h, "out": out[:, 0]}

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        r = self.forward(X)["out"] - y
        return float(r @ r / len(y))

    def backward(self, X: np.ndarray, y: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Every gradient by the chain rule, returned beside the intermediate values it used."""
        f = self.forward(X)
        n = len(y)
        d_out = 2 * (f["out"] - y) / n                  # dL/d out
        dW2 = d_out[:, None].T @ f["h"]                  # d out / d W2 = h
        db2 = np.array([d_out.sum()])
        d_h = d_out[:, None] @ self.W2                   # d out / d h = W2
        d_z1 = d_h * (1 - f["h"] ** 2)                   # d tanh(z) / dz = 1 - tanh(z)^2
        dW1 = d_z1.T @ X
        db1 = d_z1.sum(axis=0)
        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return grads, {**f, "d_out": d_out, "d_h": d_h, "d_z1": d_z1}

    def step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        grads, _ = self.backward(X, y)
        for k, g in grads.items():
            setattr(self, k, getattr(self, k) - lr * g)
        return self.loss(X, y)


def gradient_check(model: TinyMLP, X: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> dict:
    """Compare every analytic gradient with a central finite difference."""
    grads, _ = model.backward(X, y)
    worst = 0.0
    checked = 0
    for name, param in model.params().items():
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = param[idx]
            param[idx] = old + eps
            up = model.loss(X, y)
            param[idx] = old - eps
            down = model.loss(X, y)
            param[idx] = old
            numeric = (up - down) / (2 * eps)
            analytic = grads[name][idx]
            rel = abs(numeric - analytic) / max(1e-12, abs(numeric) + abs(analytic))
            worst = max(worst, rel)
            checked += 1
    return {"parameters_checked": checked, "max_relative_error": worst}

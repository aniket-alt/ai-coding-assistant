"""
dsfs -- the shared kernel behind this 16-project data science portfolio.

Every project imports from here so that data sourcing, CRISP-DM structure,
metric recording, leakage auditing and document generation are implemented
once and behave identically across the portfolio.
"""

__version__ = "1.0.0"

from .provenance import Provenance, sha256_file
from .datasets import fetch, fetch_via, stamp_frame, cache_dir
from .crispdm import CrispDmReport, Phase, Finding, PHASE_ORDER
from .metrics import RunRecord, ModelScore, SplitInfo, capture_env
from . import leakage
from .leakage import LeakageError, reset_ledger, ledger

__all__ = [
    "Provenance", "sha256_file",
    "fetch", "fetch_via", "stamp_frame", "cache_dir",
    "CrispDmReport", "Phase", "Finding", "PHASE_ORDER",
    "RunRecord", "ModelScore", "SplitInfo", "capture_env",
    "leakage", "LeakageError", "reset_ledger", "ledger",
    "__version__",
]

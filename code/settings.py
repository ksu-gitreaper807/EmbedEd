"""All fixed constants in one place (IMPLEMENTATION_PLAN §3).

Nothing here is tunable per-condition. Changing a value = a new experiment:
bump VERSION so artifact hashes make staleness obvious.
"""
import os
from pathlib import Path

VERSION = "phase01-v1"

ROOT = Path(__file__).resolve().parent.parent
# env overrides let tests (and Colab-on-Drive) relocate artifacts/report
ARTIFACTS = Path(os.environ.get("EMBEDED_ARTIFACTS", ROOT / "code" / "artifacts"))
REPORT_MD = Path(os.environ.get("EMBEDED_REPORT", ROOT / "report" / "measurements.md"))

# --- identity / determinism (SCOPE P1-3, P2-12) ---------------------------
SEED = 13                # the one seed that samples positives, shuffles, splits
N_SEEDS = 3              # training seeds: SEED+0, +1, +2
CORPUS_SHA_KEY = "corpus"

# --- dataset (FINAL_SPEC §4) ------------------------------------------------
HF_DATASET = "google/code_x_glue_cc_clone_detection_big_clone_bench"
EXPECTED_FRAGMENTS = 9_134
EXPECTED_SPLIT_ROWS = {"train": 901_028, "valid": 415_416, "test": 415_416}

# --- model (FINAL_SPEC §5, Option A — decision frozen here) ----------------
MODEL_ID = "microsoft/graphcodebert-base"
POOLING = "mean"
OPTION_B_DATAFLOW = False          # full-DFG loading is out of scope for us (correction 3)
EMBED_DIM = 768

# --- mining -----------------------------------------------------------------
K_NEGATIVES = 20                   # same k for ALL conditions (SCOPE P2-6)
HARDNESS_MARGIN = 0.02             # "visible gap" for C1 < C2 (SCOPE P1-6)
HARDNESS_CHECK_ANCHORS = 1_000     # anchors sampled (seeded) for the hardness check

# --- training sizing: ILLUSTRATIVE until the throughput test replaces them --
TRAIN_PAIRS_CAP = 50_000           # [illustrative] triples per condition
EPOCHS = 1                          # [illustrative] 1-2
BATCH = 32
LR = 2e-5
MAX_LEN = 256                       # provisional; Phase 0 token-length check may move it to 512

# --- generalisation (FINAL_SPEC §9.3, Route A) -------------------------------
SPRIME_DOI = "10.5281/zenodo.17238379"
SPRIME_HELDOUT_K = 3

# --- pinned environment for the shared Colab notebook -------------------------
PINNED = [
    "datasets==2.20.0",
    "transformers==4.43.4",
    "sentence-transformers==3.0.1",
    "rank-bm25==0.2.2",
    "umap-learn==0.5.6",
    "scikit-learn==1.5.1",
    "gradio==4.42.0",
    "pytest==8.3.2",
]


def artifact(name: str) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS / name

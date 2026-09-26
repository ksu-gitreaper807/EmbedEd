"""All fixed constants in one place (IMPLEMENTATION_PLAN §3).

Nothing here is tunable per-condition. Changing a value = a new experiment:
bump VERSION so artifact hashes make staleness obvious.
"""
import os
from pathlib import Path

VERSION = "phase01-v3"   # v3: env refresh for Colab py3.13 (torch 2.6.0 / transformers 4.46.3 / scikit-learn 1.6.1 / gradio 5.49.1); v2: canonical data.jsonl fragment ids (8,063 unique texts); v1 derived ids from pairs

ROOT = Path(__file__).resolve().parent.parent
# env overrides let tests (and Colab-on-Drive) relocate artifacts/report
ARTIFACTS = Path(os.environ.get("EMBEDED_ARTIFACTS", ROOT / "embeded" / "artifacts"))
REPORT_MD = Path(os.environ.get("EMBEDED_REPORT", ROOT / "report" / "measurements.md"))

# --- identity / determinism (SCOPE P1-3, P2-12) ---------------------------
SEED = 13                # the one seed that samples positives, shuffles, splits
N_SEEDS = 3              # training seeds: SEED+0, +1, +2
CORPUS_SHA_KEY = "corpus"

# --- dataset (FINAL_SPEC §4, correction 9 — re-verified against the file) ----
HF_DATASET = "google/code_x_glue_cc_clone_detection_big_clone_bench"
# The CodeXGLUE PAPER says 9,134; the released data.jsonl has 9,126 lines, and
# 1,063 of them are byte-identical to another fragment → 8,063 unique texts.
EXPECTED_DATA_LINES = 9_126
EXPECTED_FRAGMENTS = 8_063
EXPECTED_SPLIT_ROWS = {"train": 901_028, "valid": 415_416, "test": 415_416}
# Canonical fragment list for --hf mode (pairs carry text, not idx). Pinned to
# a commit so reruns are deterministic. Cached under ARTIFACTS;
# EMBEDED_DATA_JSONL points at a local copy to skip the fetch. The API URL is
# a fallback for networks that block raw.githubusercontent.com.
CODEXGLUE_COMMIT = "ac74a62802a0dd159b3258c78a2df8ad36cdf2b9"  # microsoft/CodeXGLUE main, 2026-09-22
_CODEXGLUE_DATA_PATH = "Code-Code/Clone-detection-BigCloneBench/dataset/data.jsonl"
CODEXGLUE_DATA_JSONL_URL = (
    f"https://raw.githubusercontent.com/microsoft/CodeXGLUE/{CODEXGLUE_COMMIT}/{_CODEXGLUE_DATA_PATH}"
)
CODEXGLUE_DATA_JSONL_API_URL = (
    f"https://api.github.com/repos/microsoft/CodeXGLUE/contents/{_CODEXGLUE_DATA_PATH}"
    f"?ref={CODEXGLUE_COMMIT}"
)
CODEXGLUE_DATA_JSONL_SIZE = 15_174_797

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
# torch included here too (it used to live only in the notebook cell and drifted
# to a version without wheels for current Colab's Python). Colab is on Python
# 3.13 as of 2026-09: torch 2.6.0 is the oldest line whose Linux deps (triton
# 3.2.0) ship cp313 wheels; transformers had to move so its tokenizers pin
# (>=0.20,<0.21) can resolve to a cp313 build (4.43.4's tokenizers<0.20 predates
# 3.13 and would need a Rust build); scikit-learn needed 1.6 (first with cp313
# wheels); gradio needed 5.4+ (4.42's pydub imports the `audioop` stdlib module
# removed in 3.13; 5.49.1 = mature end of the py3.13-capable 5.x line).
PINNED = [
    "torch==2.6.0",
    "transformers==4.46.3",
    "datasets==2.20.0",
    "sentence-transformers==3.0.1",
    "rank-bm25==0.2.2",
    "umap-learn==0.5.6",
    "scikit-learn==1.6.1",
    "gradio==5.49.1",
    "pytest==8.3.2",
]


def artifact(name: str) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS / name

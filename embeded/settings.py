"""All fixed constants in one place (IMPLEMENTATION_PLAN §3).

Nothing here is tunable per-condition. Changing a value = a new experiment:
bump VERSION so artifact hashes make staleness obvious.
"""
import os
from pathlib import Path

VERSION = "phase01-v5"   # v5: MAX_LEN 256 -> 512 from the measured Phase 0.4 token lengths (p50=474, p99=5944); training sizing measured on the T4 (BATCH 8, TRAIN_PAIRS_CAP 32k, fp16 AMP — fp32 OOMs); v4: torch policy = Colab platform build (no torch pin); v3: env refresh for Colab py3.13 (transformers 4.46.3 / scikit-learn 1.6.1 / gradio 5.49.1); v2: canonical data.jsonl fragment ids (8,063 unique texts); v1 derived ids from pairs

ROOT = Path(__file__).resolve().parent.parent
# env overrides let tests and Colab/HF sync relocate artifacts/report
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

# --- training sizing: MEASURED, Phase 0.5 on the Colab T4, 2026-09-26 -------
# scripts/phase0_throughput.py, fp16 AMP, 512 tokens (report/measurements.md):
#   512:4  11.9 triples/s  peak  4.4 GB      512:16  14.1 triples/s  peak 11.4 GB
#   512:8  13.5 triples/s  peak  6.7 GB      512:32  OOM   (T4 = 14.5 GB)
# BATCH 8 is within 5% of the fastest fitting candidate and leaves ~8 GB of
# headroom for a 40-minute run; 16 would leave ~3 GB. Frozen (SCOPE P2-13).
# negatives.py mines TRAIN_PAIRS_CAP // K_NEGATIVES anchors, so the cap also
# sizes Phase 1 mining: 32,000 = 1,600 anchors x k=20 = 4,000 steps of 8.
TRAIN_PAIRS_CAP = 32_000           # triples per condition (measured 32,367 @ 13.5 triples/s x 40 min)
EPOCHS = 1                          # 40 min/run -> 9 main runs = 6 GPU-h; "try 2" = doubled budget
BATCH = 8                           # triples per step (the encoder sees 3x = 24 sequences)
LR = 2e-5
# Fixed by the Phase 0.4 token-length measurement (report/measurements.md, SCOPE P2-5):
# GraphCodeBERT tokens per fragment p50=474 p90=1535 p95=2233 p99=5944 max=36823.
# 512 is also the model's positional-embedding ceiling, so it is the max we can
# take; even so, roughly half the fragments exceed it and are truncated — state
# this as a limitation. (256 would have truncated the median fragment.)
MAX_LEN = 512

# --- training precision / memory recipe (fixed for the Colab T4) --------------
# scripts/phase0_throughput.py measures throughput with exactly this recipe, so
# train.py (Phase 2) MUST use the same one or the measured caps do not transfer.
# Plain fp32 does not fit: a 3-stream triplet step at 256 tokens x 32 triples
# already OOMs the 14.5 GB T4 (Phase 0.5, 2026-09). T4 = Turing: fp16 tensor
# cores, no bfloat16.
AMP_DTYPE = "float16"              # torch.autocast dtype; "float32" disables AMP
GRAD_CHECKPOINT = False            # recompute activations in backward: far less memory,
                                   # ~30% slower; turn on only if no useful batch fits at MAX_LEN

# --- generalisation (FINAL_SPEC §9.3, Route A) -------------------------------
SPRIME_DOI = "10.5281/zenodo.17238379"
SPRIME_HELDOUT_K = 3

# --- pinned environment for the shared Colab notebook -------------------------
# torch is intentionally NOT in this list: the notebook uses Colab's preinstalled
# torch, already matched to the image's GPU/CUDA and its torchvision/torchaudio.
# History: (a) torch was first pinned only in the notebook cell and drifted to a
# version with no wheels for Colab's py3.13 (2.3.1, broke 2026-09); (b) hard-
# pinning torch==2.6.0 installed, but forced a ~3 GB re-download every session
# and left Colab's matched torchvision broken. Policy: never add `torch==` to
# PINNED, requirements.txt, or the notebook cell — the cell prints torch's
# version instead, and test_env_pins.py rejects any torch== pin. Everything
# below must keep installing on current Colab (py3.13, 2026-09): transformers
# moved so its tokenizers pin (>=0.20,<0.21) resolves to cp313 wheels (4.43.4's
# tokenizers<0.20 predates 3.13 and would need a Rust build); scikit-learn
# needed 1.6 (first cp313 wheels); gradio needed 5.4+ (4.42's pydub imports the
# `audioop` stdlib module removed in 3.13; 5.49.1 = mature end of the
# py3.13-capable 5.x line). transformers 4.46.3 is verified against torch 2.11
# incl. .bin checkpoints (weights_only flip) — see PR #4.
PINNED = [
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

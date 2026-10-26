# Phase 6 - Seven-model webpage

The app compares **two editable Java pairs** using C0-C6, one preselected
seed-13 run per condition. C0 is untuned. The previous `demo.app` remains
unchanged; this app reuses its underlying `embeded.encoder`, checkpoint loader
and `embeded.evaluate.pair_cosines` rather than duplicating the inference math.

## Launch on Colab

1. Open `notebooks/Phase6_Colab.ipynb` in Colab; choose a T4 runtime.
2. Run the source cell and upload `output/phase6/EmbedEd_Phase6_source.zip`.
   No Git push or stale remote checkout is needed.
3. Run preparation, verification, and serving cells. The private webpage appears
   inside the notebook. It is not an always-on public URL.
4. When finished, stop the server and disconnect/delete the runtime. A recycled
   runtime needs the source ZIP and fixed assets rehydrated; do not invent results.

The source ZIP contains no credentials, model weights, raw datasets or caches.
Preparation downloads 39 publicly readable files, about 3.51 GB, to a separate
Colab directory and verifies every file. A different destination identity fails
closed. Checkpoints are trusted project artifacts, not arbitrary uploaded pickle
files; do not substitute untrusted checkpoints.

## Local UI preview / inference

With NumPy installed, `python -m demo.compare --port 7866` serves an explicitly
unavailable-runtime UI at `http://127.0.0.1:7866`. It never invents scores.
For actual local inference, prepare assets with
`python -m demo.prepare_compare --dest /absolute/inference_assets`, then launch
`python -m demo.compare --assets /absolute/inference_assets`. CPU inference is
supported but has not been promised interactive latency; Colab T4 is the tested path.

## Identity and precision

Dataset revision: `e3144bea7732c20ab2803782282282be4fede7b4`.
Base revision: `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`.
The old training config did not pin the base revision; the demo's explicit pin
is disclosed, not a retroactive proof of the old download. Checkpoint hashes,
fingerprints, full saved NPZ thresholds and source revisions are visible on the page.

The shared token-only encoder uses masked mean pooling, normalized float32
vectors, a 512-token cap, fp16 autocast on CUDA / fp32 on CPU, and the existing
dot-product/NumPy inclusive threshold predicate. No threshold rounding, mean
thresholds, ensembling or calibration. Near-one values and input truncation are
flagged. The two examples are illustrative and unlabeled; their outputs say
nothing about benchmark superiority or semantic equivalence.

One encoder resides on the GPU. Its untuned weights are kept on CPU for C0;
six trained state dictionaries are swapped sequentially. First-request latency
includes initialization; GPU peak memory measures the comparison after loading
the base model. Model load and request latency are exposed as measurements,
not promises of simultaneous seven-model residency.

## Safety and privacy

Java is never compiled or executed. The app doesn't log request bodies/snippets
or persist user comparisons. Four snippets, at most 20,000 characters each;
100 KB request cap; one comparison at a time; 15-second socket read timeout;
custom same-origin request header; no public CORS or arbitrary file serving.
The frontend clears stale results when input changes or inference fails. A
client timeout does not cancel ongoing inference: check Colab before retrying.

Private Colab access still depends on the owner's notebook/account access.
Do not paste confidential code. No public tunnel, deployment, user account
system, database or third-party inference API is included. Production hosting
or broader access is a separate decision.

## Evidence

`output/phase6/live_verification.json` records real T4 HTTP inference on the four
checked-in snippets, all 14 offline-path parity checks, safety HTTP checks,
truncation, cold latency and GPU memory. It contains no user-entered code.
`output/phase6/closure.json` records the actual completion gates. Phase 5 results,
audits and release ZIP remain historical and are not changed by this demo.

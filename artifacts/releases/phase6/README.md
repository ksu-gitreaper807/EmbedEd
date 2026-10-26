# Phase 6 deliverables - 25 October 2026

Open `notebooks/Phase6_Colab.ipynb` in Colab, select a T4 GPU, and upload
`EmbedEd_Phase6_source.zip` in its first code cell. Run preparation,
verification and serving cells. The webpage appears privately inside the
notebook. Stop the app and disconnect/delete the runtime after use.

`EmbedEd_Phase6_release.zip` contains the source/notebook, launch guide, plan,
and verification evidence. Model weights and credentials are excluded. Assets
are rehydrated from fixed public HF revisions, about 3.51 GB. No Git/HF push,
public tunnel or permanent website deployment has been performed.

## Observed verification

- 235 tests passed without skips; 31 new focused Phase 6 tests passed locally.
- 39 inference assets passed size/content-hash verification.
- All 14 scores and decisions matched the original inference path exactly
  on a Tesla T4. Cold comparison: 18.0865 seconds. Peak comparison GPU
  allocation: 673.5 MB. These are observations, not performance promises.
- Long-input truncation and invalid input, missing header and concurrent
  request handling passed. Browser preview, errors, reset and stale-result
  clearing were checked. Positive-result rendering used a clearly labelled
  replay of the real T4 response; it was not a second live inference request.
- The T4 session was stopped after evidence export. There is no active endpoint.
- The historical Phase 5 ZIP remains unchanged.

`ui-preview.png` shows the unavailable-weights local preview.
`ui-verified-results.png` shows the explicitly labelled browser rendering
check using recorded real T4 results. Neither image promises an active runtime.
The webpage is responsive in CSS; a mobile viewport was not visually tested.
The notebook's source cells parse and its iframe API matches the installed
Colab signature. Interactive source-upload and iframe embedding still require
the owner's normal Colab notebook UI; that interactive path was not automated.

## Interpretation

The two editable examples are illustrative/unlabeled, not a validation set.
Similarities are not probabilities or equivalence proofs. C0-C6 all use seed
13, fixed before inspecting results. Exact NPZ thresholds are kept; near-one
values and truncation are visible. One encoder swaps weights sequentially.

All historical study claims and AI-audit limitations remain unchanged. This
demo is separate from Phase 5, independent human review and new experiments.
See `demo/PHASE6_README.md` for privacy, inference precision and base-revision
provenance limitations. The receipt `closure.json` hashes the deliverables.

# Audit judgements — who labelled `audit_labels.csv`, and by what rule

**Judge:** the Arena coding agent (me), not a human and not an LLM-judge API.
**Sheet:** `artifacts/audit/audit_pairs.md`, 100 pairs, seed 13, version `phase01-v6`.
**Blind:** `audit_key.csv` was never present in this sandbox. Verified before labelling with
`git ls-tree -r --name-only origin/arena/01a0e257-embeded | grep -i audit` → only
`audit_pairs.md` and `audit_labels.csv`. The condition (C2/C3) of each pair is therefore
unknown to me, and so is which fragment came from the labelled corpus.

## Rubric actually applied

The sheet's own rubric, plus four tie-break rules I had to fix up front so that 100 calls
stay consistent:

1. **The functionality is what the method does for its caller**, not which APIs it touches.
   Two methods that both use `FileChannel.transferTo` are not clones if one compresses and the
   other copies (P009, P047).
2. **Different business operation, same boilerplate → `not_clone`.** Every "JDBC update inside
   a transaction" pair where the SQL does different work is `not_clone` (P021, P043, P065,
   P072, P093). Generic *helpers* that run arbitrary SQL are the exception and are `clone`
   (P090).
3. **Containment → `not_clone`.** Where one fragment is the payload of the other plus unrelated
   setup (P006: B is A's handler body verbatim inside a server `setUp`), the pair is not the
   same functionality.
4. **Hash/encode granularity → `clone`.** "Return a digest of this string" is one functionality
   whatever the algorithm or the encoding (SHA-1 hex vs MD5 base64, P074; FNV-1a vs MD5, P058,
   P084). "Hash a password to *verify* it" vs "hash it to *store* it" are two (P011), and
   "derive a cipher key" vs "return a digest" are two (P008).

## Result

| label | n |
|---|---|
| `clone` | 36 |
| `not_clone` | 63 |
| `unsure` | 1 (P036) |

`clone` ids: P001 P005 P010 P013 P017 P018 P027 P032 P033 P038 P039 P040 P044 P046 P049 P052
P055 P056 P057 P058 P060 P062 P063 P067 P068 P074 P075 P080 P084 P085 P086 P087 P090 P094
P098 P100.

Every row carries a one-line reason in the `note` column — spot-check any call by reading the
pair and its note together.

**P036 is the only `unsure`:** A registers a temp copy of a file (rename, copy as fallback) and
adds it to a managed list; B extracts an APK asset to disk. The shared part is one
`IOUtils.copy` call; whether that makes them "copy a file" is a granularity call I did not want
to force. It is counted in `fn_rate_upper_incl_unsure`.

Other calls a human might flip, listed so the disagreement is visible rather than buried:
P005 (batch copy into a project dir vs a one-file `.bak` backup), P008 (cipher construction vs
MD5 digest), P017 (media-serving proxy vs FOXML download proxy), P035 (whole-file checksum vs
per-contig FASTA hashes), P060 (cache materialisation vs single-file archive copy), P067
(`copyFile` vs path-resolution + duplicate check + copy), P089 (PNG→S3 pipeline vs upload +
thumbnail + DB record).

## What I did not read in full

The sheet's fragments run to 541 lines (p50 = 25, p90 = 71). I read each pair with fragments
capped at 80 lines, so **14 fragments were truncated** for the reading pass:
P009A P016A P019A P019B P026B P031B P037A P050B P071A P079A P081B P091B P095A P099A.

Every one of those 13 pairs is labelled `not_clone`. That makes the truncation a
**one-directional** risk: a clone relationship hiding below line 80 of a 500-line method would
have been missed, so 36 is a floor rather than a central estimate. Nobody can call a 461-line
servlet and a 12-line unit test the same functionality from the tail, but the honest statement
is that these 13 calls rest on the first 80 lines of each long side.

## Known bias

I have read this repository's `GROUND_TRUTH.md`, so I know the audit exists to test whether
mined negatives are contaminated, and that a high clone rate would hurt the project's claim.
That is an incentive to under-call `clone`; I cannot remove it, only disclose it. Two
structural counterweights: rule 4 above calls `clone` generously on the hashing family, and the
sheet's `score` reports `fn_rate_upper_incl_unsure`, so anything I could not settle counts
against the mining, not for it.

## Scoring

The key is still only on the Colab VM (`/content/embeded-artifacts/audit/audit_key.csv`), so
**the false-negative rate cannot be computed here.** To score:

```python
# Colab, cell 12 — put the labelled file where settings.py looks for it:
#   S.ARTIFACTS/audit/audit_labels.csv  ==  /content/embeded-artifacts/audit/audit_labels.csv
!python -m scripts.audit_sample score
```

`score` refuses to run unless the key still matches the triples on disk
(`check_key_matches_triples`), so a stale mining run cannot be scored by accident.

## Verification run here (system `python3`, `EMBEDED_ARTIFACTS` pointed at the audit dir)

- `filled_labels()` → **100** ids, `P001 … P100`, in sheet order; csv ids == the 100
  `## Pxxx` headers in `audit_pairs.md`.
- Vocabulary check that `score_rows` applies (`clone|not_clone|unsure`, stripped/lowercased) →
  **0 offending rows**, so cell 12's `score` cannot abort on my labels.
- `score_rows()` executed end-to-end on a **dummy** key (first 50 → C2, last 50 → C3, purely to
  exercise the code path) and returned the expected per-condition structure. The real split is
  unknown to me; those two numbers are not results.
- `python -m scripts.audit_sample score` → exits 1 with `missing …/audit_key.csv`: correct, and
  the only blocker.
- Clobber guard: `make(50, 13, force=False)` (data loader stubbed, guard real) →
  `SystemExit: … already has 100 label(s) (P001 ... P100) — refusing to overwrite`;
  md5 of both audit files identical before and after.

## Limits of this sheet

100 pairs at 36 clones gives a Wilson 95% CI of roughly [27%, 46%] on the pooled rate — wide
enough to separate "the mining is clean" from "about a third is contaminated", not wide enough
to compare C2 with C3 (that needs ~272–652 per condition, per the power table in
`PROGRESS.md`). Any per-condition conclusion needs a larger stratified sample with `rank` in
the key.

## Second-pass recheck (2026-10-01, same judge/sandbox)

All 100 pairs were re-read from `audit_pairs.md` and re-adjudicated against the same four-rule
rubric, blind to the key (still VM-only). **Result: 100/100 agreement, 0 flips, 0 rubric
inconsistencies found.** Mechanical checks re-verified: ids `P001..P100` in sheet order, one
label per id, vocabulary {clone, not_clone, unsure}, counts 36/63/1, unsure = P036, every row
carries a note.

- The seven disclosed-flippable calls were each re-examined and **maintained**, with the
  original granularity reasoning: P005 (batch-copy vs single backup = same copy functionality,
  consistent with P060/P067), P008 (cipher construction vs digest string), P017
  (proxy-download granularity), P035 (whole-file integrity vs per-contig digest map), P089
  (encode+upload vs ingest+thumbnail+record).
- The rubric's family structure held up under re-reading: GUID-generation pairs (P056, P080)
  are clones while a password-hash main vs GUID generation (P078) is not; string→digest-string
  pairs (P039/P040/P052/P057/P062/P074/P075/P086/P100) are uniformly clones; same-boilerplate-
  different-SQL (P021/P043/P065/P072/P093) uniformly not_clone; the generic-SQL-helper
  exception (P090) is clone. Consistency, not per-pair mood, drove the original labels.
- The 13 truncated-fragment (80-line cap) pairs were re-checked from the heads: all 13
  not_clone calls are head-decisive (different functionality visible without the tails). The
  one-directional floor caveat stands unchanged: 36 clones is a floor, so 42%/30% are FN
  *floors*.

**Caveat:** this was an *intra-judge* recheck (same judge, ~4 days apart). It catches
processing errors and rubric drift; it cannot cure the disclosed judge-orientation bias. An
independent second judge remains the real reliability check. Headline numbers unchanged:
C2 FN floor 42% [29, 56], C3 30% [19, 44], pooled 36% [27, 46].

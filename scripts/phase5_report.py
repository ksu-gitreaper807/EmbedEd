"""Generate reproducible Phase 5 Markdown/LaTeX tables from a pinned snapshot.

No training, scoring-threshold selection, labeling or remote writes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics as st

INPUT_REVISION = 'e3144bea7732c20ab2803782282282be4fede7b4'


def input_identity(root):
    """Require a pinned manifest; never stamp unrelated inputs with our revision."""
    for name, key in (('snapshot_manifest.json', 'sha256'), ('input_manifest.json', 'files')):
        path = root / name
        if path.is_file():
            identity = json.loads(path.read_text())
            if identity.get('revision') != INPUT_REVISION or identity.get('repo') != 'Kusshal/Embed':
                raise SystemExit('Wrong pinned input identity')
            return identity[key]
    raise SystemExit('A verified snapshot/input manifest is required')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(argv)
    root, out = args.snapshot.resolve(), args.output.resolve()
    pinned_hashes = input_identity(root)
    from embeded import settings as S
    import numpy as np
    S.ARTIFACTS = root / 'artifacts'
    os.environ['EMBEDED_AUDIT_PROTOCOL'] = str(root / 'report/phase4_ai_audit_protocol.json')
    os.environ['EMBEDED_AUDIT_FREEZE'] = str(root / 'report/phase4_ai_audit_freeze.json')
    from scripts.phase4_closure import audit_mismatches, verified_audits
    if audit_mismatches():
        raise SystemExit('Benchmark metrics do not replay; refusing final tables')
    log = root / 'report/phase4_audit_outcome_log.md'
    if not log.exists():
        log = root / 'sanitycheck.md'
    audits = verified_audits(log)
    inputs = {}

    def track(path):
        rel = str(path.relative_to(root))
        source_name = 'report/phase4_audit_outcome_log.md' if rel == 'sanitycheck.md' else rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if pinned_hashes.get(source_name) != digest:
            raise SystemExit(f'Pinned input changed or absent from manifest: {rel}')
        inputs[source_name] = digest
        return path

    def load(name):
        return json.loads(track(S.ARTIFACTS / 'runs' / name / 'eval_metrics.json').read_text())

    def aggregate(rows, key):
        vals = [float(r[key]) for r in rows]
        return {'mean': st.mean(vals), 'sd': st.stdev(vals) if len(vals) > 1 else 0.0}

    benchmark, extension, generalisation = {}, {}, {}
    reference_labels = None
    for c in range(7):
        cond = f'C{c}'
        seeds = [13] if c == 0 else list(range(13, 19))
        rows = []
        for seed in seeds:
            name = f'{cond}_{seed}'
            m = load(name)
            if m['condition'] != cond:
                raise SystemExit(f'Wrong condition in {name}')
            with np.load(track(S.ARTIFACTS / 'runs' / name / 'predictions.npz')) as p:
                labels, scores, thr = p['test_labels'], p['test_scores'], float(np.asarray(p['threshold']).ravel()[0])
                if reference_labels is None:
                    reference_labels = labels.copy()
                if not np.array_equal(labels, reference_labels):
                    raise SystemExit('Different test labels between benchmark runs')
                m = {**m, 'predicted_positive_fraction': float((scores >= thr).mean())}
            rows.append(m)
        def bench_summary(ms, used_seeds):
            return {'seeds': used_seeds, 'n_runs': len(ms), 'f1': aggregate(ms, 'test_f1'),
                    'map_at_r': aggregate(ms, 'test_map_at_r'),
                    'thresholds': [float(m['threshold']) for m in ms],
                    'pinned_ge_0999': sum(float(m['threshold']) >= .999 for m in ms),
                    'predicted_positive_fraction': [m['predicted_positive_fraction'] for m in ms]}
        benchmark[cond] = bench_summary(rows[:1 if c == 0 else 3], seeds[:1 if c == 0 else 3])
        if c:
            extension[cond] = bench_summary(rows, seeds)
            ms = [load(f'sprime_{cond}_{s}') for s in (13, 14, 15)]
            delta = [float(m['f1_seen']) - float(m['f1_unseen']) for m in ms]
            generalisation[cond] = {'seeds': [13, 14, 15], 'seen': aggregate(ms, 'f1_seen'),
                'unseen': aggregate(ms, 'f1_unseen'), 'delta': {'mean': st.mean(delta), 'sd': st.stdev(delta)},
                'pinned_ge_0999': sum(float(m['threshold']) >= .999 for m in ms)}
    c0 = load('sprime_transfer_C0_13')
    generalisation['C0'] = {'n_runs': 1, 'seen': {'mean': c0['f1_seen'], 'sd': None},
                           'unseen': {'mean': c0['f1_unseen'], 'sd': None}}
    for c in (4, 5, 6):
        for file in ('audit_key.csv', 'audit_pairs.md', 'audit_labels.csv', 'audit_result.json'):
            track(S.ARTIFACTS / f'audit_c{c}' / file)
        track(S.ARTIFACTS / f'triples_C{c}.jsonl')
    for file in ('phase4_ai_audit_protocol.json', 'phase4_ai_audit_freeze.json'):
        track(root / 'report' / file)
    track(log)
    prevalence = float(reference_labels.mean())
    summary = {'input_revision': INPUT_REVISION,
        'primary_benchmark': benchmark, 'six_seed_benchmark_extension': extension,
        'generalisation': generalisation, 'ai_audits_not_human_validated': audits,
        'baseline': {'n_test': len(reference_labels), 'n_positive': int(reference_labels.sum()),
                     'prevalence': prevalence, 'all_negative_f1': 0.0,
                     'all_positive_f1': 2*prevalence/(1+prevalence)},
        'pinned_definition': 'validation threshold >= 0.999; not equivalent to all-negative predictions',
        'input_sha256': dict(sorted(inputs.items()))}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    md = ['## Phase 5 — verified primary results', '',
          'Primary benchmark: seeds 13–15; C0 once. Mean ± sample SD. Pinned = threshold ≥0.999.', '',
          '| condition | F1 | MAP@R | thresholds | pinned | predicted-positive range |',
          '|---|---|---|---|---|---|']
    tex = [r'\begin{tabular}{lrrrrr}', r'\toprule',
           r'Condition & F1 $\pm$ SD & MAP@R & threshold range & pinned & predicted + \\', r'\midrule']
    aliases = ['CZero', 'COne', 'CTwo', 'CThree', 'CFour', 'CFive', 'CSix']
    macros = []
    for i, (c, m) in enumerate(benchmark.items()):
        f, sd, mp = m['f1']['mean'], m['f1']['sd'], m['map_at_r']['mean']
        lo, hi = min(m['thresholds']), max(m['thresholds'])
        pmin, pmax = min(m['predicted_positive_fraction']), max(m['predicted_positive_fraction'])
        f_text = f'{f:.4f} ± {sd:.4f}' if i else f'{f:.4f} (single reference)'
        md.append(f'| {c} | {f_text} | {mp:.4f} | ' + ', '.join(f'{v:.7g}' for v in m['thresholds']) +
                  f' | {m["pinned_ge_0999"]}/{m["n_runs"]} | {pmin:.1%}–{pmax:.1%} |')
        f_tex = f'{f:.4f} $\\pm$ {sd:.4f}' if i else f'{f:.4f}'
        tex.append(f'{c} & {f_tex} & {mp:.4f} & {lo:.4f}--{hi:.4f} & '
                   f'{m["pinned_ge_0999"]}/{m["n_runs"]} & {pmin*100:.1f}--{pmax*100:.1f}\\% \\\\')
        macros += [f'\\newcommand{{\\{aliases[i]}F}}{{{f:.4f}}}',
                   f'\\newcommand{{\\{aliases[i]}M}}{{{mp:.4f}}}']
    tex += [r'\bottomrule', r'\end{tabular}']
    (out / 'benchmark.tex').write_text('\n'.join(tex) + '\n')
    md += ['', '### Generalisation (different pair regime; do not compare absolute F1 across corpora)', '',
           '| condition | F1 seen | F1 unseen | Δ ± sample SD | pinned |', '|---|---|---|---|---|']
    tex = [r'\begin{tabular}{lrrrr}', r'\toprule',
           r'Condition & Seen F1 & Unseen F1 & $\Delta\pm$SD & pinned \\', r'\midrule']
    for i in (1, 2, 3, 4, 5, 6, 0):
        c, m = f'C{i}', generalisation[f'C{i}']
        seen, unseen = m['seen']['mean'], m['unseen']['mean']
        ds = f'{m["delta"]["mean"]:+.4f} ± {m["delta"]["sd"]:.4f}' if i else '—'
        pinned = f'{m["pinned_ge_0999"]}/3' if i else '—'
        md.append(f'| {c} | {seen:.4f} | {unseen:.4f} | {ds} | {pinned} |')
        dt = f'{m["delta"]["mean"]:+.4f} \\pm {m["delta"]["sd"]:.4f}' if i else r'\text{--}'
        tex.append(f'{c} & {seen:.4f} & {unseen:.4f} & ${dt}$ & {pinned.replace("—", "--")} \\\\')
        macros.append(f'\\newcommand{{\\{aliases[i]}U}}{{{unseen:.4f}}}')
    tex += [r'\bottomrule', r'\end{tabular}']
    (out / 'generalisation.tex').write_text('\n'.join(tex) + '\n')
    md += ['', '### Audit evidence (agent/AI-reviewed, not independently human-validated)', '',
           '| condition | clone / n | point rate [Wilson 95%] | including unsure | method |',
           '|---|---|---|---|---|',
           '| C1 | — | ≈0–2% | — | historical estimate, not a new sample |',
           '| C2 | 21/50 | 42% [29.4%, 55.8%] | 42% | historical agent review |',
           '| C3 | 15/50 | 30% [19.1%, 43.8%] | 32% | historical agent review |']
    tex = [r'\begin{tabular}{lrrrr}', r'\toprule',
           r'Condition & Clones/$n$ & Rate & Wilson 95\% & Incl. unsure \\', r'\midrule',
           r'C1 & -- & $\approx$0--2\% & -- & -- \\',
           r'C2 & 21/50 & 42\% & 29.4--55.8\% & 42\% \\',
           r'C3 & 15/50 & 30\% & 19.1--43.8\% & 32\% \\']
    for c, a in audits.items():
        low, high = a['fn_rate_ci95']
        md.append(f'| {c} | {a["clone"]}/{a["n"]} | {a["fn_rate"]:.0%} [{low:.1%}, {high:.1%}] | '
                  f'{a["fn_rate_upper_incl_unsure"]:.0%} | P4-AI-AUDIT-2026-10-23 |')
        tex.append(f'{c} & {a["clone"]}/{a["n"]} & {a["fn_rate"]*100:.0f}\\% & '
                   f'{low*100:.1f}--{high*100:.1f}\\% & {a["fn_rate_upper_incl_unsure"]*100:.0f}\\% \\\\')
    tex += [r'\bottomrule', r'\end{tabular}']
    (out / 'audits.tex').write_text('\n'.join(tex) + '\n')
    (out / 'numbers.tex').write_text('% Generated from pinned metric files; do not edit.\n' + '\n'.join(macros) + '\n')
    md += ['', '### Benchmark extension — six seeds 13–18 (not the primary table)', '',
           '| condition | F1 ± sample SD | MAP@R | pinned |', '|---|---|---|---|']
    tex = [r'\begin{tabular}{lrrr}', r'\toprule',
           r'Condition & F1 $\pm$ SD & MAP@R & pinned \\', r'\midrule']
    for c, m in extension.items():
        f, sd, mp = m['f1']['mean'], m['f1']['sd'], m['map_at_r']['mean']
        md.append(f'| {c} | {f:.4f} ± {sd:.4f} | {mp:.4f} | {m["pinned_ge_0999"]}/6 |')
        tex.append(f'{c} & {f:.4f} $\\pm$ {sd:.4f} & {mp:.4f} & {m["pinned_ge_0999"]}/6 \\\\')
    tex += [r'\bottomrule', r'\end{tabular}']
    (out / 'extension.tex').write_text('\n'.join(tex) + '\n')
    (out / 'tables.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'Generated tables and {len(inputs)} input hashes in {out}')


if __name__ == '__main__':
    main()

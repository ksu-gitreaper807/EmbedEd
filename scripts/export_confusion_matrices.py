"""Export verified benchmark confusion matrices, per run and per condition.

Reuses the existing [[TP,FN],[FP,TN]] arithmetic. No model loading, training,
threshold selection, averaging of thresholds, or ensemble prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scripts.phase5_report import INPUT_REVISION, input_identity
from scripts.plot_confusion_matrices import CONDITIONS, COND_COLOR, CELL_TAG, INK, confusion

NAMES = ['C0_13'] + [f'C{c}_{s}' for c in range(1, 7) for s in range(13, 19)]
DESCRIPTIONS = ['Untuned', 'Random', 'BM25', 'Semantic', 'BM25 + bouncer',
                'Random + bouncer', 'Semantic + bouncer']


def metrics(matrix):
    tp, fn, fp, tn = np.asarray(matrix, dtype=float).ravel()
    return {'precision': float(tp/(tp+fp)) if tp+fp else 0.0,
            'recall': float(tp/(tp+fn)) if tp+fn else 0.0,
            'f1': float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0}


def row_shares(matrix):
    m = np.asarray(matrix, dtype=float)
    return np.divide(m, m.sum(axis=1, keepdims=True), out=np.zeros_like(m),
                     where=m.sum(axis=1, keepdims=True) != 0)


def read_run(root, name, pinned_hashes, input_hashes):
    def verified(filename):
        rel = f'artifacts/runs/{name}/{filename}'
        path = root / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if pinned_hashes.get(rel) != digest:
            raise ValueError(f'Pinned input changed: {rel}')
        input_hashes[rel] = digest
        return path
    with np.load(verified('predictions.npz'), allow_pickle=False) as z:
        y, scores = z['test_labels'], z['test_scores']
        threshold = np.asarray(z['threshold'])
        if threshold.size != 1:
            raise ValueError(f'{name}: expected one frozen threshold')
        thr = float(threshold.ravel()[0])
        if (y.ndim != 1 or scores.shape != y.shape or not len(y)
                or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all()
                or not np.isfinite(thr)):
            raise ValueError(f'{name}: invalid predictions')
        matrix = confusion(scores, y, thr)
        labels = y.copy()
    stored = json.loads(verified('eval_metrics.json').read_text())
    condition, seed = name.split('_')
    if stored['condition'] != condition or int(stored['seed']) != int(seed) or stored.get('smoke'):
        raise ValueError(f'{name}: wrong run identity')
    scores = metrics(matrix)
    if abs(scores['f1']-float(stored['test_f1'])) > 5e-4:
        raise ValueError(f'{name}: f1 does not replay stored metrics')
    metric_differences = {key: {'stored': float(stored['test_'+key]), 'replayed': value}
                          for key,value in scores.items()
                          if abs(value-float(stored['test_'+key])) > 5e-4}
    count_differences = {}
    for key, value in zip(('tp', 'fn', 'fp', 'tn'), matrix.ravel()):
        if key in stored.get('test', {}) and stored['test'][key] != value:
            count_differences[key] = {'stored': stored['test'][key], 'replayed': int(value)}
    return {'run': name, 'condition': condition, 'seeds': [int(seed)], 'n_runs': 1,
            'thresholds': [thr], 'matrix': matrix.astype(int).tolist(),
            'row_shares': row_shares(matrix).tolist(), 'n_pairs_per_run': int(matrix.sum()),
            'mean_seed_f1': scores['f1'], 'sd_seed_f1': None,
            'metrics_from_counts': scores, 'aggregation': 'single run',
            'legacy_count_differences': count_differences,
            'legacy_metric_differences': metric_differences}, labels


def average_runs(rows):
    if not rows or len({r['condition'] for r in rows}) != 1:
        raise ValueError('Average must contain one condition')
    if len({r['run'] for r in rows}) != len(rows):
        raise ValueError('Duplicate run in average')
    matrices = np.array([r['matrix'] for r in rows], dtype=float)
    if not np.all(matrices.sum(axis=2) == matrices[0].sum(axis=1)):
        raise ValueError('Cannot average different class supports')
    mean = matrices.mean(axis=0)
    f1 = [r['mean_seed_f1'] for r in rows]
    return {'condition': rows[0]['condition'], 'runs': [r['run'] for r in rows],
            'seeds': [r['seeds'][0] for r in rows], 'n_runs': len(rows),
            'thresholds': [r['thresholds'][0] for r in rows],
            'matrix': mean.tolist(), 'row_shares': row_shares(mean).tolist(),
            'n_pairs_per_run': int(matrices[0].sum()), 'mean_seed_f1': float(np.mean(f1)),
            'sd_seed_f1': float(np.std(f1, ddof=1)) if len(f1)>1 else None,
            'metrics_from_counts': metrics(mean),
            'aggregation': 'elementwise mean of per-run counts; not an ensemble'}


def draw_matrix(ax, row, *, title, average, compact=False):
    m, shares = np.array(row['matrix']), np.array(row['row_shares'])
    colour = COND_COLOR[row['condition']]
    cmap = LinearSegmentedColormap.from_list(row['condition'], ['white', colour])
    im = ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks([0, 1], ['Clone', 'Not clone'], fontsize=12)
    ax.set_yticks([0, 1], ['Clone', 'Not clone'], fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=13)
    ax.set_ylabel('Actual label', fontsize=13)
    ax.set_title(title, fontsize=14 if compact else 16, pad=12, weight='bold', color=colour)
    for i in range(2):
        for j in range(2):
            count = f'{m[i,j]:,.1f}' if average and row['n_runs']>1 else f'{m[i,j]:,.0f}'
            if compact:
                count = f'{m[i,j]/1000:.1f}k'
            dark = shares[i,j] > .6
            # Reference hierarchy, with its original 15/8/8 pt cell fonts halved.
            ax.text(j, i-.13, CELL_TAG[i][j], ha='center', va='center', fontsize=4,
                    color='white' if dark else '#64748b')
            ax.text(j, i+.05, f'{100*shares[i,j]:.1f}%', ha='center', va='center',
                    fontsize=7.5, weight='bold', color='white' if dark else INK)
            ax.text(j, i+.24, count, ha='center', va='center', fontsize=4,
                    color='#e2e8f0' if dark else '#64748b')
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def render_single(row, path, *, average, dpi):
    fig, ax = plt.subplots(figsize=(5.8, 5.4), layout='constrained')
    c = row['condition']
    seed_text = str(row['seeds'][0]) if len(row['seeds'])==1 else f'{row["seeds"][0]}-{row["seeds"][-1]}'
    title = f'{c}: {DESCRIPTIONS[int(c[1])]}\n'
    title += f'Mean over {row["n_runs"]} runs; seeds {seed_text}' if average and row['n_runs']>1 else f'Single reference; seed {seed_text}' if c=='C0' else f'Seed {seed_text}'
    if not average:
        title += f'\nF1 = {row["mean_seed_f1"]:.4f}'
    if not average:
        title += f'; threshold = {row["thresholds"][0]:.9g}'
    im = draw_matrix(ax, row, title=title, average=average)
    bar = fig.colorbar(im, ax=ax, shrink=.75, ticks=[0, .5, 1])
    bar.ax.set_yticklabels(['0%', '50%', '100%'])
    bar.set_label('Share of actual-label row', fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)


def render_board(rows, path, *, title, dpi):
    fig, axes = plt.subplots(3, 3, figsize=(12.4, 10.8))
    fig.subplots_adjust(left=.08, right=.96, bottom=.07, top=.84, wspace=.65, hspace=.75)
    for ax, c in zip(axes.flat, CONDITIONS):
        row = rows[c]
        desc = f'{c}: {DESCRIPTIONS[int(c[1])].replace(" + bouncer", "")}\n'
        desc += f'+ bouncer; n={row["n_runs"]}' if c in ('C4','C5','C6') else 'single reference' if c=='C0' else f'mean of {row["n_runs"]} runs'
        im = draw_matrix(ax, row, title=desc, average=True, compact=True)
    note, scale = axes.flat[7], axes.flat[8]
    note.axis('off')
    note.text(0, .98, 'READING\nTHE PANELS', fontsize=14, weight='bold', va='top')
    note.text(0, .66, 'Mean counts in thousands (k).\nPercent = actual-row share.\n\nNot an ensemble.\nC0 = one reference.\nFrozen threshold per run.',
              fontsize=12, va='top', linespacing=1.35)
    scale.axis('off')
    scale.text(0, .98, 'CONDITION COLOURS\nSAME 0-100% SCALE', fontsize=14, weight='bold', va='top')
    bar_ax = scale.inset_axes([.14, .19, .80, .50])
    ramps = [LinearSegmentedColormap.from_list(c, ['white', COND_COLOR[c]])
             (np.linspace(0, 1, 256)) for c in CONDITIONS]
    bar_ax.imshow(np.array(ramps), aspect='auto', extent=(0, 100, 6.5, -.5), interpolation='nearest')
    bar_ax.set_yticks(range(7), CONDITIONS, fontsize=9)
    bar_ax.set_xticks([0, 50, 100], ['0%', '50%', '100%'], fontsize=10)
    bar_ax.tick_params(length=0)
    for spine in bar_ax.spines.values():
        spine.set_visible(False)
    scale.text(0, .04, 'Hue = condition; shade = row share.', fontsize=10, va='top')
    fig.suptitle(title, fontsize=20, weight='bold', y=.985)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)


def render_runs_board(rows, path, *, dpi):
    """Reference-style five-column overview; separate PNGs retain full resolution."""
    nrows = (len(rows)+4)//5
    fig, axes = plt.subplots(nrows, 5, figsize=(24, 4.6*nrows+.6), squeeze=False)
    for ax, row in zip(axes.flat, rows.values()):
        title = (f'{row["condition"]} - seed {row["seeds"][0]}\n'
                 f'F1 {row["mean_seed_f1"]:.4f} - threshold {row["thresholds"][0]:.9g}')
        draw_matrix(ax, row, title=title, average=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([0, 1], ['pred clone', 'pred not-clone'], fontsize=8)
        ax.set_yticks([0, 1], ['actual clone', 'actual not-clone'], fontsize=8)
        ax.title.set_fontsize(12)
    for ax in axes.flat[len(rows):]:
        ax.axis('off')
    fig.suptitle(f'Test-split confusion matrices - {len(rows)} verified runs',
                 fontsize=17, weight='bold', color=INK, y=.99)
    fig.text(.012, .005, '415,416 shared test pairs per panel; full pair counts; frozen validation threshold per run; '
             'colour = actual-row share; C0 = one untuned reference.', fontsize=9, color='#64748b')
    fig.tight_layout(rect=(0, .015, 1, .96), h_pad=3.2)
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('report/figures/confusion_matrices'))
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args(argv)
    root, out = args.snapshot.resolve(), args.out_dir.resolve()
    if root == out or root in out.parents:
        raise ValueError('Keep generated figures outside the immutable snapshot')
    if args.dpi < 150:
        raise ValueError('Use at least 150 DPI')
    pinned_hashes = input_identity(root)
    hashes, runs, reference = {}, {}, None
    for name in NAMES:
        row, labels = read_run(root, name, pinned_hashes, hashes)
        if reference is None:
            reference = labels
        if not np.array_equal(labels, reference) or len(labels)!=415416:
            raise ValueError(f'{name}: different benchmark test labels')
        runs[name] = row
    primary, all_seeds = {}, {}
    for c in CONDITIONS:
        rows = [r for r in runs.values() if r['condition']==c]
        primary[c] = average_runs([r for r in rows if r['seeds'][0] in (13,14,15)])
        all_seeds[c] = average_runs(rows)
    out.mkdir(parents=True, exist_ok=True)
    for name, row in runs.items():
        render_single(row, out/'individual'/f'{name}.png', average=False, dpi=args.dpi)
    render_runs_board(runs, out/'all_runs_board.png', dpi=150)
    for folder, rows in (('mean_primary',primary), ('mean_all_seeds',all_seeds)):
        for c,row in rows.items():
            render_single(row, out/folder/f'{c}_mean.png', average=True, dpi=args.dpi)
        label = 'Primary means: seeds 13-15 (C0 once)' if folder=='mean_primary' else 'All-seed means: seeds 13-18 (C0 once)'
        render_board(rows, out/f'{folder}_board.png', title=label, dpi=args.dpi)
    summary = {'input_revision': INPUT_REVISION, 'layout': [['TP','FN'],['FP','TN']],
               'positive_class':'clone', 'decision':'saved test score >= saved validation threshold',
               'average_definition':'elementwise mean of per-seed counts, not pooled totals, mean thresholds or an ensemble',
               'f1_note':'mean_seed_f1 is distinct from metrics_from_counts.f1 for averages',
               'legacy_discrepancy_note':'Figures replay frozen NPZ arrays and thresholds, not legacy JSON counts. All F1 values pass 5e-4 tolerance, but that does not certify precision/recall/count equality. Differences are recorded per run.',
               'runs':runs, 'mean_primary':primary, 'mean_all_seeds':all_seeds,
               'input_sha256':hashes,
               'output_sha256':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(out.rglob('*.png'))}}
    (out/'confusion_matrices.json').write_text(json.dumps(summary,indent=2)+'\n')
    readme = ['# Benchmark confusion matrices', '',
              '37 separate model/seed images, 7 primary means, 7 all-seed means and 3 comparison boards.',
              'Style follows Image_Graphs/phase2_confusion_matrices(1).png: condition colours, bold percentages, small tags/counts.',
              'Reference cell fonts are halved: percentages 7.5 pt, tags/counts 4 pt. Numerical data are unchanged.',
              'The 37-run overview is 150 DPI; separate images and mean boards use the requested DPI (default 300).',
              'Only CodeXGLUE benchmark runs are combined; s′ training/transfer/smoke runs are excluded.',
              'C0 is evaluated once, not repeated to manufacture seed replication.', '',
              'Rows = actual, columns = predicted; order is clone then not-clone: [[TP,FN],[FP,TN]].',
              'Colours/percentages use actual-row normalization. Every panel uses a common 0–100% scale.',
              'Counts in a mean panel are mean pairs per run (may be fractional), not summed pairs.',
              'Each run retains its saved threshold; no mean threshold is used. This is not an ensemble.',
              'Mean seed F1 and F1 recomputed from mean counts are both recorded separately in the JSON.', '',
              '![Primary means](mean_primary_board.png)', '', '![All-seed means](mean_all_seeds_board.png)', '',
              '[All 37 runs](all_runs_board.png)', '',
              '## Individual runs', '']
    readme += [f'- [{name}](individual/{name}.png)' for name in runs]
    readme += ['', '## Per-model averages', '']
    readme += [f'- {c}: [primary](mean_primary/{c}_mean.png), [all seeds](mean_all_seeds/{c}_mean.png)' for c in CONDITIONS]
    readme += ['', '## Historical count/metric discrepancies', '',
               'Plots replay the pinned NPZ scores and thresholds, not legacy JSON counts.',
               'F1 agreement within 5e-4 alone does not establish precision/recall or count equality.',
               'Float32 score/threshold persistence and historical threshold rounding can change ties.',
               'No source artifact or threshold was changed for this export. Exact differences follow:', '']
    readme += [f'- {name}: counts {r["legacy_count_differences"]}; metrics {r["legacy_metric_differences"]}'
               for name,r in runs.items() if r['legacy_count_differences'] or r['legacy_metric_differences']]
    (out/'README.md').write_text('\n'.join(readme)+'\n')
    print(f'Verified {len(runs)} runs; wrote {len(summary["output_sha256"])} PNGs and numerical provenance to {out}')
    return summary


if __name__ == '__main__':
    main()

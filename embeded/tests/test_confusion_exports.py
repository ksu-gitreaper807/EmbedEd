"""Correct threshold predicate, averages and frozen-input guard for figure exports."""
import hashlib
import json

import numpy as np
import pytest

from scripts.export_confusion_matrices import (NAMES, average_runs, draw_matrix, metrics,
                                               read_run, render_single, row_shares)
from scripts.plot_confusion_matrices import confusion
from scripts.plot_confusion_matrices import COND_COLOR


def row(name, matrix, threshold):
    return dict(run=name, condition=name.split('_')[0], seeds=[int(name.split('_')[1])],
                threshold=threshold, thresholds=[threshold], matrix=matrix,
                mean_seed_f1=metrics(matrix)['f1'])


def test_inventory_is_37_runs_and_not_smoke_or_sprime():
    assert len(NAMES)==37 and len(set(NAMES))==37
    assert [n for n in NAMES if n.startswith('C0')]==['C0_13']
    assert all(n.startswith('C') for n in NAMES)


def test_saved_threshold_is_inclusive_and_cells_are_positive_first():
    np.testing.assert_array_equal(confusion(np.array([.5,.49,.5,.49]),
                                           np.array([1,1,0,0]), .5), [[1,1],[1,1]])


def test_averages_counts_not_thresholds_or_seed_f1_ratios():
    a=row('C1_13',[[2,0],[0,4]],.7)
    b=row('C1_14',[[0,2],[3,1]],.2)
    avg=average_runs([a,b])
    np.testing.assert_array_equal(avg['matrix'], [[1,1],[1.5,2.5]])
    assert avg['thresholds']==[.7,.2] and 'threshold' not in avg
    assert avg['mean_seed_f1']==.5
    assert avg['metrics_from_counts']['f1']==pytest.approx(2/4.5)
    assert avg['n_pairs_per_run']==6
    assert avg['sd_seed_f1']==pytest.approx(np.sqrt(.5))


def test_c0_is_one_reference_not_replicated():
    r=row('C0_13',[[1,1],[2,2]],.99)
    avg=average_runs([r])
    assert avg['n_runs']==1 and avg['seeds']==[13] and avg['sd_seed_f1'] is None


@pytest.mark.parametrize('rows', [[], [row('C1_13',[[1,1],[1,1]],.5)]*2,
    [row('C1_13',[[1,1],[1,1]],.5),row('C2_13',[[1,1],[1,1]],.5)],
    [row('C1_13',[[1,1],[1,1]],.5),row('C1_14',[[1,0],[1,1]],.5)]])
def test_invalid_averages_are_rejected(rows):
    with pytest.raises(ValueError):
        average_runs(rows)


def test_row_shares_handle_zero_support():
    np.testing.assert_array_equal(row_shares([[0,0],[1,3]]),[[0,0],[.25,.75]])


def test_pinned_file_hash_and_metric_replay(tmp_path):
    d=tmp_path/'artifacts/runs/C1_13'
    d.mkdir(parents=True)
    np.savez(d/'predictions.npz', test_scores=np.array([.5,.1,.6,.2]),
             test_labels=np.array([1,1,0,0]), threshold=np.array(.5))
    meta=dict(condition='C1',seed=13,test_f1=.5,test_precision=.5,test_recall=.5)
    (d/'eval_metrics.json').write_text(json.dumps(meta))
    hashes={str(p.relative_to(tmp_path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}
    result,_=read_run(tmp_path,'C1_13',hashes,{})
    assert result['matrix']==[[1,1],[1,1]]
    # F1 equality must never silently imply precision/recall or count equality.
    meta['test_precision']=.6
    meta['test']={'tp':1,'fp':0,'fn':1}
    (d/'eval_metrics.json').write_text(json.dumps(meta))
    hashes['artifacts/runs/C1_13/eval_metrics.json']=hashlib.sha256((d/'eval_metrics.json').read_bytes()).hexdigest()
    result,_=read_run(tmp_path,'C1_13',hashes,{})
    assert result['legacy_count_differences']['fp']=={'stored':0,'replayed':1}
    assert result['legacy_metric_differences']['precision']=={'stored':.6,'replayed':.5}
    meta['test_f1']=.9
    (d/'eval_metrics.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='Pinned input changed'):
        read_run(tmp_path,'C1_13',hashes,{})
    hashes['artifacts/runs/C1_13/eval_metrics.json']=hashlib.sha256((d/'eval_metrics.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='does not replay'):
        read_run(tmp_path,'C1_13',hashes,{})


def test_float32_saved_threshold_can_include_a_decimal_threshold_tie():
    scores=np.array([.999999],dtype=np.float32)
    labels=np.array([1])
    saved=float(np.float32(.999999))
    assert confusion(scores,labels,saved)[0,0]==1
    assert confusion(scores.astype(np.float64),labels,.999999)[0,0]==0


def test_render_average_keeps_fractional_counts(tmp_path):
    avg=average_runs([row('C1_13',[[2,0],[0,4]],.7),row('C1_14',[[0,2],[3,1]],.2)])
    avg['row_shares']=row_shares(avg['matrix']).tolist()
    path=tmp_path/'mean.png'
    render_single(avg,path,average=True,dpi=150)
    assert path.stat().st_size>1000


@pytest.mark.parametrize('compact', [False, True])
def test_cell_font_is_halved_without_changing_titles_or_axes(compact):
    import matplotlib.pyplot as plt

    avg=average_runs([row('C1_13',[[2,0],[0,4]],.7)])
    fig, ax=plt.subplots()
    try:
        draw_matrix(ax,avg,title='Example',average=True,compact=compact)
        assert len(ax.texts)==12
        for tag, pct, count in zip(ax.texts[::3],ax.texts[1::3],ax.texts[2::3]):
            assert tag.get_fontsize()==4 and count.get_fontsize()==4
            assert pct.get_fontsize()==7.5 and pct.get_fontweight()=='bold'
        assert ax.title.get_fontsize()==(14 if compact else 16)
        assert ax.xaxis.label.get_fontsize()==13
        assert ax.yaxis.label.get_fontsize()==13
        assert all(t.get_fontsize()==12 for t in ax.get_xticklabels()+ax.get_yticklabels())
    finally:
        plt.close(fig)


@pytest.mark.parametrize('condition', list(COND_COLOR))
def test_reference_palette_preserves_shared_row_share_normalization(condition):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    avg=average_runs([row(f'{condition}_13',[[1,3],[2,2]],.7)])
    fig, ax=plt.subplots()
    try:
        im=draw_matrix(ax,avg,title='Example',average=True)
        assert im.get_clim()==(0,1)
        np.testing.assert_allclose(im.get_array(),[[.25,.75],[.5,.5]])
        np.testing.assert_allclose(im.cmap(1.0),to_rgba(COND_COLOR[condition]))
        assert ax.title.get_color()==COND_COLOR[condition]
    finally:
        plt.close(fig)

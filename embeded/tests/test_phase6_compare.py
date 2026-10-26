"""Offline Phase 6 contracts. Synthetic vectors never appear in the user UI."""
import importlib
import ast
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from demo import compare as C
from demo.prepare_compare import BASE, BASE_FILES, BASE_REVISION, CONDITIONS, digest
from scripts.phase5_report import INPUT_REVISION


@pytest.mark.parametrize('pairs',[None,[],C.EXAMPLES[:1],C.EXAMPLES*2,
    [{'a':'','b':'x'},C.EXAMPLES[1]],[{'a':1,'b':'x'},C.EXAMPLES[1]],
    [{'a':'x'*20001,'b':'x'},C.EXAMPLES[1]],[{'a':'x','b':'x','extra':1},C.EXAMPLES[1]]])
def test_bounded_two_pair_input(pairs):
    with pytest.raises(ValueError): C.validate_pairs(pairs)


def test_examples_are_unlabeled_text_only():
    assert len(C.validate_pairs(C.EXAMPLES))==4
    assert all(set(p)=={'a','b'} for p in C.EXAMPLES)
    text='Runtime.getRuntime().exec("never-run-this"); <script>alert(1)</script>'
    assert C.validate_pairs([{'a':text,'b':'x'},C.EXAMPLES[1]])[0]==text


def test_scores_reuse_original_evaluation_predicate_at_ties():
    vectors=np.array([[1,0],[1,0],[0,1],[1,0]],dtype=np.float32)
    scores,decisions=C.score_vectors(vectors,1.)
    assert scores.dtype==np.float64
    assert scores.tolist()==[1.,0.] and decisions.tolist()==[True,False]


@pytest.mark.parametrize('vectors',[np.zeros((4,2)),np.ones((3,2)),np.full((4,2),np.nan)])
def test_invalid_encoder_output_is_rejected(vectors):
    with pytest.raises(ValueError): C.score_vectors(vectors,.5)


def fake_catalog(root):
    for name in BASE_FILES:
        p=root/'base'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fake base asset')
    for c in CONDITIONS:
        d=root/'artifacts/runs'/f'{c}_13';d.mkdir(parents=True)
        cfg=dict(condition=c,seed=13,smoke=False,model_id=BASE,pooling='mean',max_len=512)
        (d/'eval_config.json').write_text(json.dumps(cfg))
        (d/'eval_metrics.json').write_text(json.dumps({**cfg,'threshold':.1}))
        np.savez(d/'predictions.npz',threshold=np.array(.9999989867210388))
        if c!='C0':
            from embeded.train import fingerprint
            cfg['amp_dtype']='float16';cfg['fingerprint']=fingerprint(cfg)
            (d/'run_config.json').write_text(json.dumps(cfg));(d/'checkpoint.pt').write_bytes(b'never loaded')
    hashes={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
    m=dict(repo='Kusshal/Embed',revision=INPUT_REVISION,base_model=BASE,base_revision=BASE_REVISION,seed=13,sha256=hashes)
    (root/'inference_manifest.json').write_text(json.dumps(m))
    return m


def test_catalog_uses_exact_npz_threshold_not_rounded_json(tmp_path):
    fake_catalog(tmp_path);rows,_=C.load_catalog(tmp_path)
    assert len(rows)==7 and rows[0]['checkpoint_sha256'] is None
    assert all(r['seed']==13 and r['threshold']==.9999989867210388 for r in rows)


@pytest.mark.parametrize('failure',['hash','missing_hash','wrong_seed','wrong_revision','path_escape'])
def test_catalog_fails_closed(tmp_path,failure):
    m=fake_catalog(tmp_path)
    if failure=='hash': (tmp_path/'base/config.json').write_text('tampered')
    elif failure=='missing_hash': m['sha256'].pop('base/pytorch_model.bin')
    elif failure=='wrong_seed': m['seed']=14
    elif failure=='wrong_revision': m['revision']='main'
    elif failure=='path_escape': m['sha256']['../outside']='0'*64
    (tmp_path/'inference_manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError): C.load_catalog(tmp_path)


@pytest.fixture()
def api():
    server=C.make_server(port=0)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown();server.server_close();worker.join()


def request(api,path='/api/compare',payload=None,header=True):
    headers={'Content-Type':'application/json'}
    if header: headers['X-EmbedEd-Request']='1'
    data=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(api+path,data=data,headers=headers)
    with urllib.request.urlopen(req) as response: return response.read()


def test_static_ui_and_status_work_without_weights(api):
    assert b'14 predictions' in request(api,'/')
    status=json.loads(request(api,'/api/status'))
    assert not status['ready'] and len(status['catalog'])==7


@pytest.mark.parametrize('path',['/.embeded-colab-secrets.json','/../paper/paper.tex','/api/status?x=1'])
def test_server_never_serves_arbitrary_paths(api,path):
    with pytest.raises(urllib.error.HTTPError) as e: request(api,path)
    assert e.value.code==404


@pytest.mark.parametrize('payload,header,status',[({'pairs':C.EXAMPLES},True,503),
    ({'pairs':C.EXAMPLES},False,403),({'pairs':[]},True,400),({'extra':1},True,400)])
def test_errors_contain_no_fabricated_predictions(api,payload,header,status):
    with pytest.raises(urllib.error.HTTPError) as e: request(api,payload=payload,header=header)
    assert e.value.code==status
    assert set(json.loads(e.value.read()))=={'error'}


def test_import_is_lazy_and_does_not_load_encoder(monkeypatch):
    import embeded.encoder as encoder
    monkeypatch.setattr(encoder,'load_encoder',lambda *a,**k:pytest.fail('Model loaded during import'))
    importlib.reload(C)


def test_browser_code_uses_text_not_user_html_and_invalidates_stale_results():
    source=(Path(C.__file__).parent/'web/app.js').read_text()
    assert 'innerHTML' not in source and 'textContent' in source
    assert "addEventListener('input'" in source
    assert 'AbortController' in source and '180000' in source


def test_request_body_limit_is_enforced(api):
    req=urllib.request.Request(api+'/api/compare',data=b'x'*100001,
        headers={'Content-Type':'application/json','X-EmbedEd-Request':'1'})
    with pytest.raises(urllib.error.HTTPError) as e: urllib.request.urlopen(req)
    assert e.value.code==413


def test_phase6_notebook_parses_and_defaults_to_private_serving():
    root=Path(__file__).resolve().parents[2]
    nb=json.loads((root/'notebooks/Phase6_Colab.ipynb').read_text())
    code={c['id']:''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'}
    for source in code.values(): ast.parse(source)
    assert 'serve_kernel_port_as_iframe' in code['p6-serve']
    assert 'cache_in_notebook=False' in code['p6-serve']
    assert 'share=True' not in ''.join(code.values())
    assert 'phase6_source_manifest.json' in code['p6-source']
    assert 'hashlib.sha256(data)' in code['p6-source']
    assert "'..' in rel.parts" in code['p6-source']

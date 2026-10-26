"""Real Phase 6 parity/safety check on the fixed illustrative examples only."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import numpy as np

from demo.compare import ComparisonEngine, EXAMPLES, make_server, score_vectors
from embeded import settings as S


def verify(assets,out,device=None):
    import torch
    from embeded.evaluate import default_embed_fn
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    engine=ComparisonEngine(assets,device)
    server=make_server(engine,port=0)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    url=f'http://127.0.0.1:{server.server_port}'
    def post(payload,header=True):
        h={'Content-Type':'application/json'}
        if header: h['X-EmbedEd-Request']='1'
        request=urllib.request.Request(url+'/api/compare',data=json.dumps(payload).encode(),headers=h)
        with urllib.request.urlopen(request,timeout=180) as r: return json.load(r)
    try:
        result=post({'pairs':EXAMPLES})
        assert len(result['results'])==7
        # Use the original default inference entry point, not a reimplementation.
        S.ARTIFACTS=Path(assets)/'artifacts'
        S.MODEL_ID=str(Path(assets)/'base')
        texts=[p[k] for p in EXAMPLES for k in ('a','b')]
        parity=[]
        for row in result['results']:
            fn,_,_=default_embed_fn(row['condition'],13,device=engine.device)
            scores,decisions=score_vectors(fn(texts),row['threshold'])
            np.testing.assert_allclose(scores,row['scores'],rtol=0,atol=1e-6)
            assert decisions.tolist()==row['decisions']
            parity.append(dict(condition=row['condition'],scores=scores.tolist(),
                               decisions=decisions.tolist(),max_abs_error=float(np.max(np.abs(scores-row['scores'])))))
            del fn
            gc.collect()
            if str(engine.device).startswith('cuda'): torch.cuda.empty_cache()
        safety={}
        for name,payload,header,expected in (
            ('csrf_header',{'pairs':EXAMPLES},False,403),
            ('wrong_pair_count',{'pairs':EXAMPLES[:1]},True,400),
            ('empty_snippet',{'pairs':[{'a':'','b':'x'},EXAMPLES[1]]},True,400),
            ('oversized_snippet',{'pairs':[{'a':'x'*20001,'b':'x'},EXAMPLES[1]]},True,400)):
            try: post(payload,header);raise AssertionError(name+' was accepted')
            except urllib.error.HTTPError as exc: assert exc.code==expected;safety[name]=exc.code
        engine.lock.acquire()
        try:
            try: post({'pairs':EXAMPLES});raise AssertionError('Concurrent request accepted')
            except urllib.error.HTTPError as exc: assert exc.code==429;safety['concurrent_request']=429
        finally: engine.lock.release()
        long_pairs=[{'a':'int identifier = 1; '*500,'b':EXAMPLES[0]['b']},EXAMPLES[1]]
        long_result=engine.compare(long_pairs)
        assert long_result['truncated'][0] and long_result['token_counts'][0]>512
        proof=dict(status='passed',device=engine.device,cuda_name=torch.cuda.get_device_name() if str(engine.device).startswith('cuda') else None,
                   source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__),Path(__file__).with_name('compare.py'),Path(__file__).with_name('prepare_compare.py')]},
                   live_result=result,offline_parity=parity,parity_scores_checked=14,
                   max_abs_error=max(p['max_abs_error'] for p in parity),safety_http_checks=safety,
                   long_input_truncation_confirmed=True,token_counts_long_input=long_result['token_counts'],
                   examples='demo.compare.EXAMPLES; illustrative/unlabeled, not benchmark evidence',
                   privacy='No user-entered code was used or stored',
                   data_revision=engine.manifest['revision'],base_revision=engine.manifest['base_revision'])
        (out/'live_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
        (out/'inference_manifest.json').write_text(json.dumps(engine.manifest,indent=2)+'\n')
        print(json.dumps({k:proof[k] for k in ('status','device','cuda_name','parity_scores_checked','max_abs_error','long_input_truncation_confirmed')}),flush=True)
        print(json.dumps({'latency_ms':result['latency_ms'],'peak_cuda_mb':result['peak_cuda_mb']}),flush=True)
        return proof
    finally:
        server.shutdown();server.server_close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--device',default=None)
    args=p.parse_args();verify(args.assets,args.out,args.device)

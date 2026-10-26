"""Phase 6: bounded seven-condition inference, private loopback webpage, no training."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import numpy as np

from demo.prepare_compare import BASE, BASE_REVISION, BASE_FILES, CONDITIONS, digest
from embeded import settings as S
from embeded.evaluate import pair_cosines
from scripts.phase5_report import INPUT_REVISION

NAMES=('Untuned reference','Random negatives','BM25 negatives','Semantic negatives',
       'BM25 + bouncer','Random + bouncer','Semantic + bouncer')
EXAMPLES=[
    {'a':'int sum(int[] values) {\n    int total = 0;\n    for (int value : values) total += value;\n    return total;\n}',
     'b':'int addAll(int[] numbers) {\n    int result = 0;\n    for (int number : numbers) result += number;\n    return result;\n}'},
    {'a':'int max(int[] values) {\n    int best = Integer.MIN_VALUE;\n    for (int value : values) best = Math.max(best, value);\n    return best;\n}',
     'b':'int countPositive(int[] values) {\n    int count = 0;\n    for (int value : values) if (value > 0) count++;\n    return count;\n}'}]
MAX_CHARS=20_000
MAX_BODY=100_000


class BusyError(Exception):
    pass


def validate_pairs(pairs):
    if not isinstance(pairs,list) or len(pairs)!=2:
        raise ValueError('Supply exactly two pairs.')
    texts=[]
    for pair in pairs:
        if not isinstance(pair,dict) or set(pair)!= {'a','b'}:
            raise ValueError('Each pair needs exactly snippets a and b.')
        for key in ('a','b'):
            value=pair[key]
            if not isinstance(value,str) or not value.strip() or len(value)>MAX_CHARS:
                raise ValueError(f'Each snippet must contain 1-{MAX_CHARS:,} characters.')
            texts.append(value)
    return texts


def score_vectors(vectors, threshold):
    vectors=np.asarray(vectors,dtype=np.float32)
    if vectors.ndim!=2 or vectors.shape[0]!=4 or not np.isfinite(vectors).all():
        raise ValueError('Invalid encoder output')
    if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
        raise ValueError('Encoder output is not normalized')
    # Keep the evaluation module's exact dot-product and NumPy dtype/predicate.
    scores=pair_cosines(dict(enumerate(vectors)),[(0,1,0),(2,3,0)])
    return scores,(scores>=threshold)


def load_catalog(root, *, verify=True):
    root=Path(root)
    manifest=json.loads((root/'inference_manifest.json').read_text())
    if (manifest['revision']!=INPUT_REVISION or manifest['base_revision']!=BASE_REVISION
            or manifest['repo']!='Kusshal/Embed' or manifest['base_model']!=BASE
            or manifest['seed']!=13):
        raise ValueError('Wrong pinned inference identity')
    required={f'base/{name}' for name in BASE_FILES}
    for c in CONDITIONS:
        folder=f'artifacts/runs/{c}_13/'
        required.update(folder+n for n in ('eval_config.json','eval_metrics.json','predictions.npz'))
        if c!='C0': required.update(folder+n for n in ('run_config.json','checkpoint.pt'))
    if set(manifest['sha256'])!=required:
        raise ValueError('Incomplete or unexpected inference asset inventory')
    for name,expected in manifest['sha256'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe asset path')
        if verify and digest(root/path)!=expected:
            raise ValueError('Inference asset hash mismatch: '+name)
    rows=[]
    for cond,name in zip(CONDITIONS,NAMES):
        folder=f'artifacts/runs/{cond}_13'
        cfg=json.loads((root/folder/'eval_config.json').read_text())
        if (cfg['condition']!=cond or cfg['seed']!=13 or cfg['smoke']
                or cfg['model_id']!=BASE or cfg['pooling']!='mean' or cfg['max_len']!=512):
            raise ValueError('Wrong evaluation configuration: '+cond)
        npz=folder+'/predictions.npz'
        with np.load(root/npz,allow_pickle=False) as z:
            threshold=np.asarray(z['threshold'])
            if threshold.size!=1 or not np.isfinite(threshold).all():
                raise ValueError('Invalid frozen threshold')
            threshold=float(threshold.ravel()[0])
        fingerprint=None
        checkpoint=None
        if cond!='C0':
            training=json.loads((root/folder/'run_config.json').read_text())
            from embeded.train import fingerprint as config_fingerprint
            if (training['condition']!=cond or training['seed']!=13 or training['smoke']
                    or training['model_id']!=BASE or training['max_len']!=512
                    or training['pooling']!='mean' or training['amp_dtype']!='float16'
                    or config_fingerprint(training)!=training['fingerprint']):
                raise ValueError('Wrong training identity: '+cond)
            fingerprint=training['fingerprint']
            checkpoint=manifest['sha256'][folder+'/checkpoint.pt']
        rows.append(dict(condition=cond,name=name,seed=13,threshold=threshold,
                         threshold_sha256=manifest['sha256'][npz],
                         checkpoint_sha256=checkpoint,fingerprint=fingerprint))
    return rows,manifest


class ComparisonEngine:
    """One encoder resident on the device; no model-per-condition cache."""
    def __init__(self,root,device=None):
        self.root=Path(root)
        self.catalog,self.manifest=load_catalog(root)
        self.device=device
        self.lock=threading.Lock()
        self.model=None

    def _load(self):
        if self.model is not None: return
        from embeded.encoder import load_encoder
        self.tokenizer,self.model,self.device=load_encoder(str(self.root/'base'),self.device)
        self.model.eval()
        self.base={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}

    def compare(self,pairs):
        texts=validate_pairs(pairs)
        if not self.lock.acquire(blocking=False):
            raise BusyError('Another comparison is running. Please try again shortly.')
        try:
            import torch
            from embeded.encoder import encode_texts
            from embeded.train import load_trained_checkpoint
            start=time.perf_counter()
            self._load()
            cuda=str(self.device).startswith('cuda')
            if cuda: torch.cuda.reset_peak_memory_stats()
            counts=[len(self.tokenizer(t,truncation=False)['input_ids']) for t in texts]
            rows=[]
            for entry in self.catalog:
                tick=time.perf_counter()
                cond=entry['condition']
                if cond=='C0':
                    self.model.load_state_dict(self.base)
                else:
                    ck=load_trained_checkpoint(self.root/f'artifacts/runs/{cond}_13/checkpoint.pt',self.model)
                    if ck['fingerprint']!=entry['fingerprint']:
                        raise ValueError('Checkpoint/config identity mismatch')
                    del ck
                self.model.eval()
                vectors=encode_texts(self.model,self.tokenizer,texts,max_len=512,
                                     device=self.device,batch_size=S.EVAL_BATCH,
                                     amp_dtype=S.AMP_DTYPE if cuda else 'float32')
                scores,decisions=score_vectors(vectors,entry['threshold'])
                rows.append({**entry,'scores':scores.tolist(),'decisions':decisions.tolist(),
                             'elapsed_ms':round(1000*(time.perf_counter()-tick),1),
                             'saturated':bool(entry['threshold']>=.999 or (scores>=.999).any())})
            return dict(results=rows,token_counts=counts,truncated=[n>512 for n in counts],
                        latency_ms=round(1000*(time.perf_counter()-start),1),device=self.device,
                        peak_cuda_mb=round(torch.cuda.max_memory_allocated()/2**20,1) if cuda else None,
                        revision=INPUT_REVISION,base_revision=BASE_REVISION,
                        precision='float16 autocast' if cuda else 'float32',max_tokens=512)
        finally:
            self.lock.release()


def make_server(engine=None, *, port=7860):
    web=Path(__file__).parent/'web'
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)
        def log_message(self,*args): pass  # Never log snippets, bodies or request URLs.

        def reply(self,status,value):
            data=json.dumps(value,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path=='/api/status':
                return self.reply(200,dict(ready=engine is not None,seed=13,examples=EXAMPLES,
                    catalog=engine.catalog if engine else [dict(condition=c,name=n,seed=13) for c,n in zip(CONDITIONS,NAMES)],
                    revision=INPUT_REVISION,base_revision=BASE_REVISION))
            if self.path not in ('/','/app.js','/style.css'):
                return self.reply(404,dict(error='Not found'))
            name='index.html' if self.path=='/' else self.path[1:]
            data=(web/name).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type',{'html':'text/html','js':'text/javascript','css':'text/css'}[name.split('.')[-1]]+'; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers(); self.wfile.write(data)

        def do_POST(self):
            if self.path!='/api/compare': return self.reply(404,dict(error='Not found'))
            if self.headers.get('X-EmbedEd-Request')!='1':
                return self.reply(403,dict(error='Same-origin request header required'))
            if self.headers.get_content_type()!='application/json':
                return self.reply(400,dict(error='JSON request required'))
            try: size=int(self.headers.get('Content-Length','0'))
            except ValueError: return self.reply(400,dict(error='Invalid body length'))
            if not 0<size<=MAX_BODY: return self.reply(413,dict(error='Request is too large or empty'))
            try:
                request=json.loads(self.rfile.read(size))
                if not isinstance(request,dict) or set(request)!={'pairs'}: raise ValueError('Supply pairs only.')
                validate_pairs(request['pairs'])
            except (ValueError,UnicodeDecodeError): return self.reply(400,dict(error='Supply two non-empty code pairs, at most 20,000 characters per snippet.'))
            if engine is None: return self.reply(503,dict(error='Weights are unavailable. Start the Phase 6 Colab notebook; no scores were generated.'))
            try: return self.reply(200,engine.compare(request['pairs']))
            except BusyError: return self.reply(429,dict(error='A comparison is already running. Try again shortly.'))
            except (Exception,SystemExit): return self.reply(503,dict(error='Inference failed. No partial or placeholder predictions are shown. Check the runtime and pinned assets.'))

    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    return server


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,help='Verified inference asset directory; omit for an unavailable-runtime UI preview')
    p.add_argument('--port',type=int,default=7860)
    args=p.parse_args(argv)
    engine=ComparisonEngine(args.assets) if args.assets else None
    server=make_server(engine,port=args.port)
    print(f'EmbedEd comparison: http://127.0.0.1:{server.server_port} (private loopback; no public tunnel)',flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__=='__main__': main()

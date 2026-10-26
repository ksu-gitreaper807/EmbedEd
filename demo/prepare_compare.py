"""Prepare only the fixed Phase 6 inference assets; no training or remote writes."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

from scripts.phase5_snapshot import get
from scripts.phase5_report import INPUT_REVISION

REPO = 'Kusshal/Embed'
BASE = 'microsoft/graphcodebert-base'
BASE_REVISION = '2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d'
CONDITIONS = tuple(f'C{i}' for i in range(7))
BASE_FILES = ('config.json','pytorch_model.bin','merges.txt','vocab.json',
              'special_tokens_map.json','tokenizer_config.json')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(dest):
    from huggingface_hub import hf_hub_download
    dest=Path(dest).resolve()
    dest.mkdir(parents=True,exist_ok=True)
    identity={'repo':REPO,'revision':INPUT_REVISION,'base_model':BASE,
              'base_revision':BASE_REVISION,'seed':13}
    marker=dest/'identity.json'
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError('Destination belongs to different inference assets')
    marker.write_text(json.dumps(identity,indent=2)+'\n')
    entries=[]
    for cond in CONDITIONS:
        folder=f'artifacts/runs/{cond}_13'
        with get(f'https://huggingface.co/api/datasets/{REPO}/tree/{INPUT_REVISION}/{folder}') as r:
            rows=json.load(r)
        names=['eval_config.json','eval_metrics.json','predictions.npz']
        if cond!='C0': names += ['run_config.json','checkpoint.pt']
        for filename in names:
            entry=next(x for x in rows if x['path']==folder+'/'+filename)
            entries.append((REPO,'dataset',INPUT_REVISION,entry['path'],entry,dest))
    with get(f'https://huggingface.co/api/models/{BASE}/revision/{BASE_REVISION}?blobs=true') as r:
        base_info=json.load(r)
    assert base_info['sha']==BASE_REVISION
    for filename in BASE_FILES:
        row=next(x for x in base_info['siblings'] if x['rfilename']==filename)
        entry={'size':row['size'],'oid':row['blobId']}
        if row.get('lfs'): entry['lfs']={'oid':row['lfs']['sha256']}
        entries.append((BASE,'model',BASE_REVISION,filename,entry,dest/'base'))
    total=sum(e[4]['size'] for e in entries)
    if total>4_000_000_000:
        raise ValueError('Inference assets exceed the fixed 4 GB download budget')
    print(f'Pinned inference assets: {len(entries)} files, {total/1e9:.2f} GB',flush=True)

    def fetch(args):
        repo,kind,rev,name,expected,folder=args
        path=Path(hf_hub_download(repo,name,repo_type=kind,revision=rev,
                                 local_dir=folder,token=False))
        if path.stat().st_size!=expected['size']:
            raise ValueError('Asset size mismatch: '+name)
        actual=digest(path)
        if expected.get('lfs'):
            if actual!=expected['lfs']['oid']: raise ValueError('Asset SHA mismatch: '+name)
        else:
            h=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode()+path.read_bytes()).hexdigest()
            if h!=expected['oid']: raise ValueError('Asset Git-blob mismatch: '+name)
        print('Verified '+name,flush=True)
        return str(path.relative_to(dest)),actual

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        hashes=dict(pool.map(fetch,entries))
    manifest={**identity,'total_bytes':total,'sha256':hashes}
    (dest/'inference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dest',required=True,type=Path)
    prepare(p.parse_args().dest)

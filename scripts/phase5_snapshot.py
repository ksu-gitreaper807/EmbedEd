"""Download an immutable, hash-verified HF analysis snapshot using the stdlib.

Unlike the legacy full-backup helper, fixes one revision for listing/downloads
and verifies Git-blob or LFS content hashes. Weights/raw caches are omitted.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
import urllib.error
import urllib.parse
import urllib.request


def selected(path):
    return not (path.endswith(('.pt', '.npy', '.zip', '.tar', '.gz'))
                or path.startswith(('artifacts/sprime/', 'artifacts/sprime_v6_archive/',
                                    'artifacts/v5-backup/')))


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    data = path.read_bytes()
    if entry.get('lfs'):
        return hashlib.sha256(data).hexdigest() == entry['lfs']['oid']
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == entry['oid']


def get(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(url, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise SystemExit('HF rate limit: stop retries and honour the recorded cooldown.') from exc
            if exc.code not in (408, 500, 502, 503, 504) or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', default='Kusshal/Embed')
    ap.add_argument('--revision', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--max-bytes', type=int, default=1_000_000_000)
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= 4:
        ap.error('workers must be between 1 and 4')
    base = 'https://huggingface.co'
    with get(f'{base}/api/datasets/{args.repo}/revision/{urllib.parse.quote(args.revision, safe="")}') as response:
        revision = json.load(response)['sha']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise SystemExit('Expected an immutable commit SHA')
    entries = []
    url = f'{base}/api/datasets/{args.repo}/tree/{revision}?recursive=true&limit=1000'
    while url:
        with get(url) as response:
            entries.extend(r for r in json.load(response) if r['type'] == 'file')
            match = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get('Link', ''))
            url = match.group(1) if match else None
    for entry in entries:
        name = PurePosixPath(entry['path'])
        if name.is_absolute() or '..' in name.parts:
            raise SystemExit('Unsafe repository path')
    wanted = [r for r in entries if selected(r['path'])]
    total = sum(r['size'] for r in wanted)
    if total > args.max_bytes:
        raise SystemExit(f'{total} bytes exceeds max-bytes; choose scope explicitly.')
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    identity = {'repo': args.repo, 'revision': revision, 'mode': 'analysis-only'}
    identity_path = dest / 'snapshot_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise SystemExit('Destination contains a different snapshot; choose a new directory.')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    print(f'Analysis snapshot: {len(wanted)}/{len(entries)} files; {total/1e6:.1f} MB; {revision}', flush=True)

    def download(entry):
        target = dest / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not matches(target, entry):
            if target.exists():
                raise RuntimeError(f'Preserving differing existing file: {target}')
            part = target.with_name(target.name + '.download-part')
            path = urllib.parse.quote(entry['path'], safe='/')
            with get(f'{base}/datasets/{args.repo}/resolve/{revision}/{path}') as response, part.open('wb') as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if not matches(part, entry):
                raise RuntimeError(f'Content hash mismatch; partial retained: {part}')
            part.replace(target)
        return entry['path'], hashlib.sha256(target.read_bytes()).hexdigest()

    hashes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(download, r) for r in wanted]):
            name, digest = future.result()
            hashes[name] = digest
            if len(hashes) % 10 == 0 or len(hashes) == len(wanted):
                print(f'Hash verified {len(hashes)}/{len(wanted)}', flush=True)
    manifest = {**identity, 'verified_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'n_remote_files': len(entries), 'n_verified_files': len(hashes), 'total_bytes': total,
                'sha256': dict(sorted(hashes.items())),
                'omitted': [r['path'] for r in entries if not selected(r['path'])]}
    (dest / 'snapshot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('VERIFIED SNAPSHOT:', dest, flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build a fresh code-only distribution from an explicit Visual allowlist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = 'decd/visual_expert/RELEASE_FILES.txt'
ZIP_NAME = 'visual-expert-code-only-RC1.zip'
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.csv',
    '.jsonl', '.png', '.jpg', '.jpeg', '.webp', '.gif', '.pdf', '.pkl', '.npz'}
PRIVATE_PATTERNS = (
    r'/root/(?:autodl-tmp|\.config|\.ssh)/',
    r'[A-Za-z]:[\\/](?:Users|work|autodl)[\\/]',
    r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
    r'\b(?:sk|hf)_[A-Za-z0-9]{24,}\b',
    r'\bsk-[A-Za-z0-9_-]{24,}\b',
    r'connect\.[A-Za-z0-9.-]+\.seetacloud\.com',
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def collect(source):
    source = Path(source).resolve()
    names = (source/ALLOWLIST).read_text(encoding='utf-8').splitlines()
    if not names or len(names) != len(set(names)):
        raise ValueError('empty or duplicate release allowlist')
    blobs = {}
    for name in names:
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or '..' in relative.parts or '\\' in name
                or ':' in name or relative.suffix.lower() in FORBIDDEN_SUFFIXES):
            raise ValueError(f'forbidden release member: {name}')
        if relative.suffix.lower() not in {'.py','.md','.yaml','.txt'} and name not in {'.gitignore','LICENSE'}:
            raise ValueError(f'non-code/document member in release allowlist: {name}')
        if any(part.lower() in {'outputs','runs','data','datasets','.git','.venv','__pycache__'}
               for part in relative.parts):
            raise ValueError(f'private output/cache directory in allowlist: {name}')
        if name.startswith(('decd/visual_expert/decd/', 'decd/visual_expert/scripts/',
                            'decd/visual_expert/tools/', 'decd/visual_expert/configs/')):
            raise ValueError('historical nested Visual projects are not release inputs')
        path = (source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f'missing or escaping release member: {name}')
        data = path.read_bytes()
        text = data.decode('utf-8')
        for pattern in PRIVATE_PATTERNS:
            if re.search(pattern, text, re.I):
                raise ValueError(f'possible private path/credential in {name}; review locally')
        if relative.suffix == '.py':
            compile(data, name, 'exec')
        blobs[name] = data
    return blobs


def build(source, output_dir):
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError('RC output directory exists; use a fresh directory, never overwrite')
    blobs = collect(source)
    output.mkdir(parents=True, exist_ok=False)
    distribution = output/'distribution'
    distribution.mkdir()
    for name, data in blobs.items():
        target = distribution/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    archive = output/ZIP_NAME
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, data in blobs.items():
            item = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            item.external_attr = 0o100644 << 16
            item.compress_type = zipfile.ZIP_DEFLATED
            zipped.writestr(item, data)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None or zipped.namelist() != list(blobs):
            raise ValueError('ZIP verification failed')
        if any(zipped.read(n) != b for n,b in blobs.items()):
            raise ValueError('ZIP source bytes differ')
    manifest = {'file': ZIP_NAME, 'bytes': archive.stat().st_size,
        'sha256': digest(archive.read_bytes()), 'files': len(blobs),
        'members_sha256': {n:digest(b) for n,b in blobs.items()},
        'private_scan': 'NONE FOUND in current whitelist; not a Git history audit'}
    with (output/'package_manifest.json').open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,indent=2); stream.write('\n')
    with (output/(ZIP_NAME+'.sha256')).open('x',encoding='ascii') as stream:
        stream.write(manifest['sha256']+'  '+ZIP_NAME+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root,args.output_dir),indent=2))


if __name__ == '__main__':
    main()

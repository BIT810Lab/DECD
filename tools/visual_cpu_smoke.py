#!/usr/bin/env python3
"""Synthetic data-only CLI smoke test. Never creates or trains a detector."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from decd.visual_expert.manifests import DOMAIN_TO_ID,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    out=a.output_dir.resolve() if a.output_dir else Path(tempfile.mkdtemp(prefix='visual-expert-cpu-smoke-'))
    if a.output_dir:
        out.mkdir(parents=True,exist_ok=False)
    for split,index in [('train',1),('dev',2)]:
        rows=[]
        for d,domain in enumerate(DOMAIN_TO_ID):
            for label in (0,1):
                path=out/f'{split}_{d}_{label}.png'
                Image.new('RGB',(17,11),(d*70,label*150,index*60)).save(path)
                rows.append(dict(image_path=path.name,domain=domain,label=label,split=split,sha256=file_sha256(path)))
        with (out/f'{split}.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    cmd=[sys.executable,'-B',str(ROOT/'scripts/train_reference.py'),'--config',str(ROOT/'configs/visual_expert_reported.yaml'),
         '--train-manifest',str(out/'train.csv'),'--dev-manifest',str(out/'dev.csv'),'--device','cpu','--dry-run']
    result=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=60)
    report={'purpose':'synthetic data-only smoke; not paper reproduction','model_loaded':False,'optimizer_updates':0,
            'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    with (out/'smoke_report.json').open('x') as f:json.dump(report,f,indent=2)
    print(result.stdout,end='');print(f'CPU smoke artifacts: {out}')
    if result.returncode:
        print(result.stderr,file=sys.stderr);raise SystemExit(result.returncode)


if __name__=='__main__':main()

import json
import runpy
from collections import Counter
from pathlib import Path
import pytest
from PIL import Image

from conftest import csv_file
from decd.visual_expert.manifests import DOMAIN_TO_ID, file_sha256

TOOL=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/build_mosaic_benchmark.py'))
select=TOOL['select_stratified']


def test_6000_exact_balanced_deterministic(virtual_pool):
    a,rest,_,_=select(virtual_pool,seed=2026)
    b,_,_,_=select(list(reversed(virtual_pool)),seed=2026)
    assert [r['sha256'] for r in a]==[r['sha256'] for r in b]
    assert len(a)==6000 and len(rest)==12
    assert len({r['original_sha256'] for r in a})==6000
    assert set(Counter((r['domain'],r['label']) for r in a).values())=={1000}
    assert not {r['sha256'] for r in a}&{r['sha256'] for r in rest}


def test_exclusions_and_dedup(virtual_pool):
    original=virtual_pool[0]['original_sha256']
    virtual_pool[1]['group_id']='excluded-group'
    a,rest,rejected,_=select(virtual_pool+[dict(virtual_pool[2])],exclusion_hashes={original},exclusion_groups={'excluded-group'})
    assert len(a)==6000 and len(rejected)==3
    assert not any(r['original_sha256']==original or r['group_id']=='excluded-group' for r in a)


def test_insufficient_no_replacement(virtual_pool):
    pool=[r for r in virtual_pool if not (r['domain']=='Cartoon' and r['label']==0 and int(r['sample_id'][-5:])>=999)]
    with pytest.raises(ValueError,match='fewer than 1000'): select(pool)


def test_exclusions_can_make_a_cell_insufficient(virtual_pool):
    hashes={r['sha256'] for r in virtual_pool[:3]}
    with pytest.raises(ValueError,match='fewer than 1000'):
        select(virtual_pool,exclusion_hashes=hashes)


def test_duplicate_conflicting_labels_rejected(virtual_pool):
    duplicate=dict(virtual_pool[0],label=1)
    with pytest.raises(ValueError,match='conflicting'): select(virtual_pool+[duplicate])


def test_freeze_verify_and_tamper_real_synthetic_files(tmp_path):
    # 6006 distinct 1x1 synthetic PNGs (~0.4MB payload), no real images/checkpoints.
    rows=[]; i=0
    for domain in DOMAIN_TO_ID:
        for label in (0,1):
            for _ in range(1001):
                image=tmp_path/f'{i}.png'
                Image.new('RGB',(1,1),(i%256,(i//256)%256,10)).save(image)
                rows.append(dict(file_name=image.name,domain=domain,label=f'{label}_'+('real' if label==0 else 'fake'),sha256=file_sha256(image)))
                i+=1
    metadata=csv_file(tmp_path/'metadata.csv',rows)
    out=tmp_path/'freeze'
    report=TOOL['freeze_benchmark'](metadata,tmp_path,out)
    assert report['selected_rows']==6000 and report['remaining_rows']==6
    assert report['isolation']['near_duplicate']=='NOT_VERIFIED'
    before={p.name:file_sha256(p) for p in out.iterdir()}
    assert TOOL['verify_frozen'](out,metadata=metadata,image_root=tmp_path)['seed']==2026
    assert before=={p.name:file_sha256(p) for p in out.iterdir()}
    with pytest.raises(FileExistsError): TOOL['freeze_benchmark'](metadata,tmp_path,out,seed=2027)
    assert before=={p.name:file_sha256(p) for p in out.iterdir()}
    with (out/'mosaic_benchmark_6000.csv').open('a') as f:f.write('tampered\n')
    with pytest.raises(ValueError,match='SHA mismatch'): TOOL['verify_frozen'](out)

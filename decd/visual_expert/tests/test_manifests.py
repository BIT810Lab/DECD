import csv
import runpy
from pathlib import Path
import pytest
import torch
from PIL import Image

from conftest import csv_file
from decd.visual_expert.manifests import (read_manifest, assert_disjoint, exclusion_identities, public_rows, file_sha256, write_csv_exclusive)
from decd.visual_expert.data import ManifestDataset, load_manifest

TOOLS = Path(__file__).resolve().parents[1]/'tools'


def test_hf_schema_and_aliases(tiny_pool):
    root, rows = tiny_pool
    rows = [dict(r, domain='Photo' if r['domain']=='Photo_and_poster' else r['domain']) for r in rows]
    loaded = read_manifest(csv_file(root/'hf.csv', rows), image_root=root, require_sha=True)
    assert len(loaded) == 12
    assert {r['label'] for r in loaded} == {0, 1}
    assert {r['domain'] for r in loaded} == {'Cartoon','Chart','Photo_and_poster'}
    assert all(r['sha256']==r['original_sha256'] for r in loaded)


@pytest.mark.parametrize('failure', ['missing', 'sha', 'duplicate', 'escape', 'label'])
def test_bad_manifest(tiny_pool, failure):
    root, rows = tiny_pool; row = dict(rows[0])
    if failure=='missing': row['file_name']='missing.png'
    if failure=='sha': row['sha256']='0'*64
    if failure=='escape': row['file_name']='../escape.png'
    if failure=='label': row['label']='2'
    data = [row, row] if failure=='duplicate' else [row]
    with pytest.raises((ValueError, FileNotFoundError)):
        read_manifest(csv_file(root/'bad.csv', data), image_root=root, require_sha=True)


@pytest.mark.parametrize('extension', ['png','jpg','webp'])
def test_decode_and_preprocess(tiny_pool, extension):
    root, rows = tiny_pool; image = root/f'new.{extension}'
    Image.new('RGB',(11,17)).save(image)
    r = dict(rows[0],file_name=image.name,sha256=file_sha256(image))
    loaded=read_manifest(csv_file(root/'fmt.csv',[r]),image_root=root)
    tensor=ManifestDataset(loaded)[0]['image']
    assert tensor.shape==(3,224,224) and torch.isfinite(tensor).all()


def test_broken_image_not_zero_tensor(tiny_pool):
    root, rows=tiny_pool; p=root/'empty.png'; p.write_bytes(b'')
    loaded=read_manifest(csv_file(root/'broken.csv',[dict(rows[0],file_name=p.name,sha256=file_sha256(p))]),image_root=root)
    with pytest.raises(RuntimeError): ManifestDataset(loaded)[0]


def test_disjoint_and_group_overlap(tiny_pool):
    root, rows=tiny_pool
    loaded=read_manifest(csv_file(root/'all.csv',rows),image_root=root)
    train=[r for r in loaded if r['split']=='train']; dev=[r for r in loaded if r['split']=='dev']
    assert_disjoint(train,dev)
    with pytest.raises(ValueError): assert_disjoint(train,train)
    dev[0]['group_id']=train[0]['group_id']
    with pytest.raises(ValueError): assert_disjoint(train,dev)


@pytest.mark.parametrize('role', ['train','dev'])
def test_benchmark_forbidden(tiny_pool,role):
    root,rows=tiny_pool; p=csv_file(root/'test.csv',[dict(rows[0],split='test',benchmark_id='mosaic_benchmark_6000_v1')])
    with pytest.raises(ValueError): load_manifest(p,role=role)


def test_legacy_original_and_converted_hashes(tiny_pool):
    root, rows=tiny_pool; source=root/rows[0]['file_name']; dest=root/'converted.jpg'
    Image.open(source).convert('RGB').save(dest)
    r=dict(rows[0],file_name=dest.name,standardized_sha256=file_sha256(dest))
    loaded=read_manifest(csv_file(root/'legacy.csv',[r]),image_root=root)[0]
    assert loaded['sha256']==file_sha256(dest)
    assert loaded['original_sha256']==file_sha256(source)
    assert loaded['sha256']!=loaded['original_sha256']


def test_exclusion_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError): exclusion_identities([csv_file(tmp_path/'bad.csv',[{'image_path':'missing.png'}])])
    with pytest.raises(ValueError): exclusion_identities([csv_file(tmp_path/'empty.csv',[{'unknown':'anything'}])])


def test_converter_requires_selection_and_no_overwrite(tiny_pool):
    root, rows=tiny_pool
    metadata=csv_file(root/'metadata.csv',rows)
    selection=csv_file(root/'selection.csv',[{'file_name':r['file_name']} for r in rows if r['split']=='train'])
    ns=runpy.run_path(str(TOOLS/'convert_mosaic_manifest.py'))
    out=root/'manifests'/'train.csv'; result=ns['convert'](metadata,root,selection,out,'train')
    assert result['rows']==6 and not result['automatic_split']
    assert len(load_manifest(out,role='train'))==6
    with pytest.raises(FileExistsError): ns['convert'](metadata,root,selection,out,'train')


def test_empty_selection_rejected(tiny_pool):
    root,rows=tiny_pool; p=root/'empty.csv';p.write_text('file_name\n')
    ns=runpy.run_path(str(TOOLS/'convert_mosaic_manifest.py'))
    with pytest.raises(ValueError): ns['convert'](csv_file(root/'metadata.csv',rows),root,p,root/'train.csv','train')

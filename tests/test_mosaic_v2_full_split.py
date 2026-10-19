"""Small real-file fixtures only; no real dataset, model weights or GPU."""
import copy
import csv
import hashlib
from pathlib import Path

from PIL import Image
import pytest

import prepare_mosaic_v2_full_split as split


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@pytest.fixture
def pool(tmp_path, monkeypatch):
    project = tmp_path / 'decd-visual-expert'
    images = tmp_path / 'datasets/mosaic_v2'
    project.mkdir(); images.mkdir(parents=True)
    rows = []
    for cell, (domain, label) in enumerate(split.CELLS):
        for index in range(6):
            p = images / f'{cell}_{index}.png'
            Image.new('RGB', (2, 2), (cell * 30, index * 30, 11)).save(p)
            rows.append(dict(file_name=p.name, domain=domain, label=f'{label}_'+('real' if label == 0 else 'fake'),
                             sha256=split.sha256(p), candidate_id=p.stem,
                             source_dataset_claim='', source_revision_claim='',
                             source_image_id_claim='', group_id_claim=''))
    write_csv(images/'metadata.csv', rows)
    legacy = project/'e2e/manifests/mosaic_test.csv'
    target = next(r for r in rows if r['domain']=='Chart' and r['label']=='1_fake')
    write_csv(legacy, [dict(image_path=str(images/target['file_name']), domain='Photo', label='1'),
                       dict(image_path=str(images/target['file_name']), domain='Chart', label='1')])
    (project/'configs').mkdir()
    source = Path(__file__).resolve().parents[1]/'configs/visual_expert_mosaic_v2_full.yaml'
    (project/'configs/visual_expert_mosaic_v2_full.yaml').write_bytes(source.read_bytes())
    monkeypatch.setattr(split, 'TARGETS', {'train':2,'dev':1,'test':1})
    monkeypatch.setattr(split, 'EXPECTED_POOL_SIZE', len(rows))
    return project, images, rows, legacy


def test_legacy_duplicate_and_conflicting_domain_use_metadata(pool):
    project, images, rows, legacy = pool
    before = legacy.read_bytes()
    result, report, unused = split.plan(project, 2026)
    protected = split.legacy_protection(legacy)[0]
    reserved = [r for r in result['test'] if r['sha256'] in protected]
    assert len(reserved)==1 and reserved[0]['domain']=='Chart' and reserved[0]['label']=='1'
    assert report['legacy_test_duplicate_records']==1
    assert report['legacy_test_domain_conflicting_sha256']==1
    assert report['legacy_test_label_conflicting_sha256']==0
    assert all(r['sha256'] not in protected for role in ('train','dev') for r in result[role])
    assert legacy.read_bytes()==before
    assert len(unused)==12


def test_legacy_conflicting_labels_are_audited_not_copied(pool):
    project, _, _, legacy = pool
    old = split.read_csv(legacy)
    old[0]['label']='0'
    write_csv(legacy, old)
    _, report, _ = split.plan(project, 2026)
    assert report['legacy_test_label_conflicting_sha256']==1
    assert report['legacy_annotations_used_for_v2_labels'] is False


def test_plan_deterministic_even_when_metadata_order_changes(pool):
    project, images, rows, _ = pool
    first, _, _ = split.plan(project, 2026)
    second, _, _ = split.plan(project, 2026)
    assert first==second
    write_csv(images/'metadata.csv', list(reversed(rows)))
    third, _, _ = split.plan(project, 2026)
    assert first==third


def test_available_source_groups_and_unknowns(pool):
    project, images, rows, _ = pool
    chosen = [r for r in rows if r['domain']=='Cartoon' and r['label']=='0_real'][:2]
    for row in chosen:
        row.update(source_dataset_claim='example/source', source_image_id_claim='same-reference')
    write_csv(images/'metadata.csv', rows)
    result, report, unused = split.plan(project, 2026)
    roles={r['sha256']:role for role, items in {**result,'unused':unused}.items() for r in items}
    assert len({roles[r['sha256']] for r in chosen})==1
    assert report['group_checks']['rows_with_group_or_source_links']==2
    assert report['group_checks']['rows_without_group_evidence']==34
    assert report['group_checks']['near_duplicate_isolation']=='NOT_VERIFIED'


def test_protected_group_members_do_not_enter_train_dev(pool):
    project, images, rows, legacy = pool
    chosen = [r for r in rows if r['domain']=='Chart' and r['label']=='1_fake'][:2]
    for row in chosen:
        row.update(source_dataset_claim='example', group_id_claim='shared-task')
    write_csv(images/'metadata.csv', rows)
    result, _, unused = split.plan(project, 2026)
    protected_group = {r['sha256'] for r in chosen}
    assert not any(r['sha256'] in protected_group for role in ('train','dev') for r in result[role])
    # The protected group has two rows but Test quota is one; hold it out whole.
    assert protected_group <= {r['sha256'] for r in unused}


def test_more_protected_than_test_quota_uses_unused(pool):
    project, images, rows, legacy = pool
    chosen=[r for r in rows if r['domain']=='Chart' and r['label']=='1_fake'][:2]
    write_csv(legacy,[dict(image_path=str(images/r['file_name']),domain='Photo',label='0') for r in chosen])
    result, report, unused=split.plan(project,2026)
    assert report['new_test_legacy_sha_overlap']==1
    assert len({r['sha256'] for r in unused}&{r['sha256'] for r in chosen})==1


@pytest.mark.parametrize('problem',['missing','sha','duplicate','escape','invalid_label','insufficient','protected_shortage'])
def test_invalid_pool_and_infeasible_quota_fail(pool, monkeypatch, problem):
    project, images, rows, legacy=pool
    if problem=='missing':
        rows[0]['file_name']='missing.png'
    elif problem=='sha':
        rows[0]['sha256']='0'*64
    elif problem=='duplicate':
        rows[1]['sha256']=rows[0]['sha256']
    elif problem=='escape':
        outside=images.parent/'outside.png'
        outside.write_bytes((images/rows[0]['file_name']).read_bytes())
        rows[0]['file_name']='../outside.png'
    elif problem=='invalid_label':
        rows[0]['label']='2'
    elif problem=='insufficient':
        rows=rows[3:]
        monkeypatch.setattr(split,'EXPECTED_POOL_SIZE',len(rows))
    else:
        write_csv(legacy,[dict(image_path=str(images/r['file_name']),domain='Cartoon',label='0') for r in rows[:4]])
    write_csv(images/'metadata.csv',rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        split.plan(project,2026)
    assert not (project/'e2e/manifests/mosaic_v2_frozen').exists()


def test_freeze_is_reused_without_overwrite_and_tamper_fails(pool):
    project, _, _, _=pool
    result, report, unused=split.plan(project,2026)
    directory, config=split.write(project,result,report,unused)
    all_files=[*directory.iterdir(),config]
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in all_files}
    split.write(project,result,report,unused,verify_only=True)
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in all_files}
    (directory/'mosaic_train.csv').write_text('tampered',encoding='utf-8')
    with pytest.raises(ValueError,match='refusing to overwrite'):
        split.write(project,result,report,unused)


def test_untrusted_initialization_file_rejected(pool):
    project, images, _, _=pool
    result, report, unused=split.plan(project,2026)
    with pytest.raises(ValueError,match='pinned official'):
        split.write(project,result,report,unused,pretrained_path=next(images.glob('*.png')))


def test_partial_existing_output_is_not_overwritten(pool):
    project, _, _, _=pool
    result, report, unused=split.plan(project,2026)
    path=project/'e2e/manifests/mosaic_v2_frozen'
    path.mkdir()
    sentinel=path/'keep.txt'; sentinel.write_text('keep')
    with pytest.raises(FileExistsError): split.write(project,result,report,unused)
    assert sentinel.read_text()=='keep'

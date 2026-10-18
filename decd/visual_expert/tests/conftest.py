import csv
import hashlib
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decd.visual_expert.manifests import DOMAIN_TO_ID, file_sha256


def csv_file(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return path


@pytest.fixture
def tiny_pool(tmp_path):
    rows = []
    for domain_index, domain in enumerate(DOMAIN_TO_ID):
        for label in (0, 1):
            for split_index, split in enumerate(('train', 'dev')):
                image = tmp_path / f'{domain_index}_{label}_{split}.png'
                Image.new('RGB', (7, 11), (domain_index * 60, label * 120, split_index * 70)).save(image)
                rows.append(dict(file_name=image.name, domain=domain,
                                 label=f'{label}_' + ('real' if label == 0 else 'fake'),
                                 sha256=file_sha256(image), split=split,
                                 group_id=f'group_{domain_index}_{label}_{split}'))
    return tmp_path, rows


@pytest.fixture
def virtual_pool():
    rows = []
    for d in DOMAIN_TO_ID:
        for y in (0, 1):
            for i in range(1002):
                sid = f'{d}_{y}_{i:05d}'; h = hashlib.sha256(sid.encode()).hexdigest()
                rows.append(dict(sample_id=sid, image_path=Path('unused') / (sid + '.png'),
                                 domain=d, label=y, label_name='Real' if y == 0 else 'Fake',
                                 sha256=h, original_sha256=h, group_id=''))
    return rows

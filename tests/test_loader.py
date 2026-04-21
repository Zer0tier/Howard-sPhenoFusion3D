import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file_io.loader import get_default_intrinsics, load_intrinsics


def test_default_intrinsics_shape():
    K, dist = get_default_intrinsics(640, 480)
    assert K.shape == (3, 3), 'K must be 3x3'
    assert len(dist) == 5, 'dist must have 5 elements'


def test_default_intrinsics_values():
    K, dist = get_default_intrinsics(640, 480, fov_deg=60)
    assert K[0, 2] == 320.0, 'cx should be width/2'
    assert K[1, 2] == 240.0, 'cy should be height/2'
    assert K[2, 2] == 1.0
    assert all(d == 0.0 for d in dist)


def test_load_intrinsics_missing_file():
    result = load_intrinsics('nonexistent_path.txt')
    assert result is None, 'Should return None for missing file'


def test_load_intrinsics_valid(tmp_path):
    import json
    data = {
        'K': [[481.2, 0, 319.5], [0, 480.0, 239.5], [0, 0, 1]],
        'dist': [0, 0, 0, 0, 0],
        'width': 640,
        'height': 480
    }
    f = tmp_path / 'kdc_intrinsics.txt'
    f.write_text(json.dumps(data))
    result = load_intrinsics(str(f))
    assert result is not None
    K, dist, w, h = result
    assert K.shape == (3, 3)
    assert w == 640
    assert h == 480
    assert K[0, 0] == pytest.approx(481.2)
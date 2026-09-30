"""Read compatibility for pgvector codec versions without dependency pinning."""

import numpy as np
import pytest
from pgvector import Vector

from engram.storage.postgres import _vector_to_list


@pytest.mark.parametrize("value", [None, [0.5, 1.0], np.array([0.5, 1.0]), Vector([0.5, 1.0])])
def test_vector_conversion(value):
    assert _vector_to_list(value) == (None if value is None else [0.5, 1.0])

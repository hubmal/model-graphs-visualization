"""Fixtures shared by several test modules."""

from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest


def save_model(model: ov.Model, directory: Path, name: str = "model") -> Path:
    path = directory / f"{name}.xml"
    ov.save_model(model, path, compress_to_fp16=False)
    return path


@pytest.fixture
def linear_model_path(tmp_path: Path) -> Path:
    """x[?, 4] -> MatMul(w[3, 4], transpose_b) -> Add(b[3]) -> Result, saved as OpenVINO IR."""
    x = ops.parameter(ov.PartialShape([-1, 4]), ov.Type.f32, name="x")
    w = ops.constant(np.ones((3, 4), np.float32), name="w")
    mm = ops.matmul(x, w, transpose_a=False, transpose_b=True, name="mm")
    b = ops.constant(np.zeros(3, np.float32), name="b")
    add = ops.add(mm, b, name="add")
    out = ops.result(add, name="out")
    return save_model(ov.Model([out], [x], "linear"), tmp_path)

"""Model loaders: convert a model file into the intermediate Graph."""

from pathlib import Path

from graph_viewer.loaders.base import ModelLoader, ModelLoadError
from graph_viewer.loaders.openvino_loader import OpenVINOLoader

# File extension -> loader class. Supporting a new format = one new loader + one entry here.
LOADERS: dict[str, type[ModelLoader]] = {
    ".xml": OpenVINOLoader,
}


def get_loader(path: str | Path) -> ModelLoader:
    suffix = Path(path).suffix.lower()
    if suffix not in LOADERS:
        supported = ", ".join(sorted(LOADERS))
        raise ModelLoadError(f"{path}: unsupported model format '{suffix}' (supported: {supported})")
    return LOADERS[suffix]()


__all__ = ["LOADERS", "ModelLoadError", "ModelLoader", "OpenVINOLoader", "get_loader"]

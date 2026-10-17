"""Abstract ModelLoader interface shared by all model formats."""

from abc import ABC, abstractmethod
from pathlib import Path

from graph_viewer.graph import Graph


class ModelLoadError(Exception):
    """Raised when a model file is missing, unsupported or invalid.

    Loaders translate library-specific exceptions into this one, so callers
    (e.g. the CLI) can show a readable message without knowing the model format.
    """


class ModelLoader(ABC):
    """Converts a model file into the framework-independent Graph."""

    @abstractmethod
    def load(self, path: str | Path) -> Graph:
        """Load the model at `path`.

        Raises:
            ModelLoadError: if the file is missing, unsupported or invalid.
        """

"""Registry mapping pass names (as typed on the command line) to pass classes."""

from graph_viewer.passes.base import PassConfigError, TransformationPass

_REGISTRY: dict[str, type[TransformationPass]] = {}


def register_pass(cls: type[TransformationPass]) -> type[TransformationPass]:
    """Class decorator that makes a pass available by its `name`."""
    if cls.name in _REGISTRY:
        raise ValueError(f"a pass named '{cls.name}' is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def available_passes() -> dict[str, type[TransformationPass]]:
    return dict(sorted(_REGISTRY.items()))


def create_pass(name: str, options: dict[str, str] | None = None) -> TransformationPass:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise PassConfigError(f"unknown pass '{name}' (available: {known})")
    return _REGISTRY[name].from_options(options or {})

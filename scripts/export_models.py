"""Export the example models to OpenVINO IR in ml_models/.

Requires the export extra: uv sync --extra export
Usage: uv run python scripts/export_models.py [tinyllama] [resnet18] [distilbert]
"""

import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "ml_models"


def export_with_optimum(model_id: str, output_dir: Path, *extra_args: str) -> None:
    """Hugging Face model -> OpenVINO IR, the way the assignment describes."""
    command = ["optimum-cli", "export", "openvino", "--model", model_id, *extra_args, str(output_dir)]
    subprocess.run(command, check=True)


def export_tinyllama(output_dir: Path) -> None:
    export_with_optimum("TinyLlama/TinyLlama-1.1B-Chat-v1.0", output_dir, "--weight-format", "fp16")


def export_distilbert(output_dir: Path) -> None:
    export_with_optimum("distilbert/distilbert-base-uncased", output_dir, "--task", "feature-extraction")


def export_resnet18(output_dir: Path) -> None:
    """torchvision model -> OpenVINO IR. Random weights: the graph structure is identical."""
    import openvino as ov
    import torch
    import torchvision

    model = torchvision.models.resnet18(weights=None).eval()
    ov_model = ov.convert_model(model, example_input=torch.zeros(1, 3, 224, 224))
    output_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, output_dir / "openvino_model.xml")


EXPORTERS = {
    "tinyllama": ("tinyllama_ov", export_tinyllama),
    "resnet18": ("resnet18_ov", export_resnet18),
    "distilbert": ("distilbert_ov", export_distilbert),
}


def main(names: list[str]) -> None:
    for name in names or EXPORTERS:
        if name not in EXPORTERS:
            sys.exit(f"unknown model '{name}', choose from: {', '.join(EXPORTERS)}")
        directory, export = EXPORTERS[name]
        output_dir = MODELS_DIR / directory
        if (output_dir / "openvino_model.xml").exists():
            print(f"{name}: already exported to {output_dir}")
            continue
        print(f"{name}: exporting to {output_dir}")
        export(output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])

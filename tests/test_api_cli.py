"""End-to-end tests: OpenVINO model -> passes -> SVG, through the Python API and the CLI."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from graph_viewer.api import load_model, visualize
from graph_viewer.cli import main, parse_pass_options
from graph_viewer.loaders import ModelLoadError
from graph_viewer.passes import HighlightPass, PassConfigError

needs_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz 'dot' not installed")


# --- Python API ------------------------------------------------------------


def test_load_model_picks_loader_by_extension(linear_model_path: Path) -> None:
    graph = load_model(linear_model_path)

    assert graph.num_nodes() == 6


def test_load_model_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="unsupported model format"):
        load_model(tmp_path / "model.onnx")


@needs_dot
def test_visualize_without_passes(linear_model_path: Path, tmp_path: Path) -> None:
    result = visualize(linear_model_path, tmp_path / "graph.svg")

    assert result.output_path.exists()
    assert (result.nodes_before, result.edges_before) == (6, 5)
    assert (result.nodes_after, result.edges_after) == (6, 5)
    svg = result.output_path.read_text()
    assert "Constant" in svg
    assert "out 0 → in 1" in svg


@needs_dot
def test_visualize_with_passes_end_to_end(linear_model_path: Path, tmp_path: Path) -> None:
    result = visualize(
        linear_model_path,
        tmp_path / "graph.svg",
        passes=["hide-constants", HighlightPass(["mm"])],
    )

    assert (result.nodes_after, result.edges_after) == (4, 3)
    assert [name for name, _ in result.pass_timings] == ["hide-constants", "highlight"]
    assert min(result.load_seconds, result.passes_seconds, result.render_seconds) >= 0

    root = ET.parse(result.output_path).getroot()
    text = "".join(root.itertext())
    assert "[1]: f32[3, 4]" in text   # hidden weight on the MatMul
    assert "[1]: f32[3]" in text      # hidden bias on the Add
    assert "Constant" not in text
    assert 'stroke="red"' in result.output_path.read_text()


def test_visualize_does_not_modify_model_files(linear_model_path: Path, tmp_path: Path) -> None:
    before = linear_model_path.read_bytes(), linear_model_path.with_suffix(".bin").read_bytes()

    if shutil.which("dot"):
        visualize(linear_model_path, tmp_path / "graph.svg", passes=["hide-constants"])
    else:
        load_model(linear_model_path)

    assert (linear_model_path.read_bytes(), linear_model_path.with_suffix(".bin").read_bytes()) == before


# --- CLI argument handling -------------------------------------------------


def test_parse_pass_options() -> None:
    options = parse_pass_options(["highlight.names=a,b", "highlight.color=blue"])

    assert options == {"highlight": {"names": "a,b", "color": "blue"}}


@pytest.mark.parametrize("bad", ["highlight", "highlight=red", ".names=a", "names=a"])
def test_parse_pass_options_rejects_bad_format(bad: str) -> None:
    with pytest.raises(PassConfigError, match="PASS.KEY=VALUE"):
        parse_pass_options([bad])


@needs_dot
def test_cli_renders_svg_and_reports_timings(
    linear_model_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out.svg"

    code = main([
        "--model", str(linear_model_path),
        "--output", str(output),
        "--passes", "hide-constants,highlight",
        "-O", "highlight.names=mm",
    ])

    assert code == 0
    assert output.exists()
    stdout = capsys.readouterr().out
    assert "6 nodes, 5 edges -> 4 nodes, 3 edges" in stdout
    assert "load:" in stdout and "render:" in stdout and "hide-constants" in stdout


def test_cli_list_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-passes"]) == 0

    stdout = capsys.readouterr().out
    assert "hide-constants" in stdout and "highlight" in stdout and "fold-converts" in stdout


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--passes", "no-such-pass"], "unknown pass"),
        (["--passes", "highlight"], "needs names"),
        (["--passes", "hide-constants", "-O", "highlight.names=a"], "not in --passes"),
        (["--passes", "highlight", "-O", "highlight"], "PASS.KEY=VALUE"),
    ],
)
def test_cli_reports_bad_pass_configuration(
    linear_model_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    message: str,
) -> None:
    code = main(["--model", str(linear_model_path), "--output", str(tmp_path / "o.svg"), *extra_args])

    assert code == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("model_name", "content", "message"),
    [
        ("missing.xml", None, "file not found"),
        ("broken.xml", "not a model", "not a valid OpenVINO IR model"),
        ("model.onnx", "x", "unsupported model format"),
    ],
)
def test_cli_reports_bad_model_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], model_name: str, content: str | None, message: str
) -> None:
    model = tmp_path / model_name
    if content is not None:
        model.write_text(content)

    code = main(["--model", str(model), "--output", str(tmp_path / "o.svg")])

    err = capsys.readouterr().err
    assert code == 1
    assert message in err
    assert "Traceback" not in err


def test_cli_requires_model_and_output(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--model", "m.xml"])

    assert exit_info.value.code == 2
    assert "--model and --output are required" in capsys.readouterr().err

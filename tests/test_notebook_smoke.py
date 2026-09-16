import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    Path("notebooks/main_DataProcessing_ClustAssess.ipynb"),
    Path("notebooks/main_Model_Selection_tuVI.ipynb"),
    Path("notebooks/main_Training_tuVI.ipynb"),
    Path("notebooks/main_Training_scVI.ipynb"),
    Path("notebooks/main_Model_Selection_TRVI.ipynb"),
    Path("notebooks/main_Training_TRVI.ipynb"),
    Path("notebooks/main_Inference_tuVI_scVI.ipynb"),
    Path("notebooks/main_Inference_TRVI.ipynb"),
]


def _load_notebook(path: Path) -> dict:
    return json.loads((REPO_ROOT / path).read_text())


def _code_source(notebook: dict) -> str:
    source_chunks = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source_chunks.append("".join(source) if isinstance(source, list) else source)
    return "\n".join(source_chunks)


def test_notebooks_parse_and_are_in_mkdocs_nav():
    mkdocs_yml = (REPO_ROOT / "mkdocs.yml").read_text()
    for notebook_path in NOTEBOOKS:
        notebook = _load_notebook(notebook_path)
        assert notebook.get("cells"), notebook_path
        assert notebook_path.as_posix() in mkdocs_yml


def test_notebooks_install_from_hollfelder_lab_repository():
    for notebook_path in NOTEBOOKS:
        raw = (REPO_ROOT / notebook_path).read_text()
        assert "github.com/fweberling/crecerelle.git" not in raw
        assert "github.com/Hollfelder-Lab/crecerelle.git" in raw


def test_notebooks_use_package_imports_without_copied_implementations():
    top_level_definition = re.compile(r"^\s*(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
    package_names = set()
    for source_path in (REPO_ROOT / "src" / "crecerelle").glob("*.py"):
        source = source_path.read_text()
        package_names.update(match.group(2) for match in top_level_definition.finditer(source))

    for notebook_path in NOTEBOOKS:
        source = _code_source(_load_notebook(notebook_path))
        assert "from crecerelle." in source
        notebook_names = {match.group(2) for match in top_level_definition.finditer(source)}
        assert not (package_names & notebook_names), notebook_path


def test_colab_urls_match_notebook_paths():
    for notebook_path in NOTEBOOKS:
        url = (
            "https://colab.research.google.com/github/"
            f"Hollfelder-Lab/crecerelle/blob/main/{notebook_path.as_posix()}"
        )
        assert url.endswith(notebook_path.as_posix())

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_NOTEBOOKS = REPO_ROOT / "docs" / "notebooks"
SOURCE_NOTEBOOKS = REPO_ROOT / "notebooks"


def on_config(config):
    if DOCS_NOTEBOOKS.exists() or DOCS_NOTEBOOKS.is_symlink():
        return config

    DOCS_NOTEBOOKS.symlink_to(SOURCE_NOTEBOOKS, target_is_directory=True)
    return config

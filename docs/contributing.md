# Contributing to Crecerelle

We welcome bug fixes, improvements to documentation and tutorials, and new features. For substantial changes, please open an issue first to discuss the approach.

## Development setup

Use Python 3.12, matching the documentation workflow, and Git. The commands below target macOS, Linux, or Windows through WSL. Python must include pip and venv support.

1. Fork https://github.com/Hollfelder-Lab/crecerelle on GitHub.
2. Clone your fork, replacing YOUR-USERNAME with your GitHub username:

```bash
git clone https://github.com/YOUR-USERNAME/crecerelle.git
cd crecerelle
git remote add upstream https://github.com/Hollfelder-Lab/crecerelle.git
git switch -c my-contribution
chmod +x contribute.sh
./contribute.sh
source .venv/bin/activate
```

The script creates or reuses `.venv`, installs Crecerelle in editable mode, and installs the existing documentation and test dependencies from `requirements-docs.txt`. Source edits take effect without reinstalling the package. The first installation may take several minutes because it includes scientific computing and machine learning dependencies.

To select a Python interpreter explicitly, run `PYTHON=/path/to/python3.12 bash contribute.sh` when creating the environment. The repository already ignores `.venv/`.

## Make and check changes

Edit Python code in `src/crecerelle/`, documentation in `docs/`, and source notebooks in `notebooks/`. The documentation build creates `docs/notebooks` as a link to the source notebooks; edit the originals.

Run the same checks as the current documentation workflow:

```bash
python -m pytest tests/test_notebook_smoke.py tests/test_public_api_smoke.py
python -m mkdocs build --strict
```

To preview the documentation locally:

```bash
python -m mkdocs serve
```

The smoke tests and documentation build do not reproduce all paper analyses or train the models. Validate any changed analysis or model behavior separately, using the necessary datasets, and describe what you tested. Add focused tests for changed Python behavior where appropriate. Keep notebook changes relevant and avoid committing datasets, credentials, virtual environments, or generated site files.

## Submit a pull request

Review `git diff`, stage the intended files, commit, and push your branch:

```bash
git add path/to/changed-file
git commit -m "Describe the change"
git push -u origin my-contribution
```

Open a pull request from your fork's branch to `Hollfelder-Lab/crecerelle:main`. Explain the problem, your changes, and the checks you ran. A maintainer will review the contribution before merging it.

For bug reports, include the Python and package versions, the error message, and a small reproducible example when possible.

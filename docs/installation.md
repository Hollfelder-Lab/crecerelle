# Installation

Crecerelle requires **Python 3.12 or newer** and Git. Python 3.12 is recommended because it is used by the automated checks.

## Local environment

For a local installation, first create and activate a virtual environment. On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Choose one of the installation methods below.

## Local Development

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/Hollfelder-Lab/crecerelle.git
cd crecerelle
python -m pip install -e .
```

If you already have a checkout, run the installation command from its root directory, where `setup.py` is located. Editable installation lets changes to the Python source take effect without reinstalling the package.

Package dependencies are declared in `setup.py` and installed automatically.

## Documentation

From the repository root, with the virtual environment activated, install both the package and documentation dependencies:

```bash
python -m pip install -e . -r requirements-docs.txt
python -m mkdocs build --strict
```

The documentation build renders notebooks without executing them. To preview the website locally, run `python -m mkdocs serve`.

## Google Colab

Use a runtime with Python 3.12 or newer. Check the version in a notebook cell:

```python
import sys
print(sys.version)
assert sys.version_info >= (3, 12), "Crecerelle requires Python 3.12 or newer."
```

Then install into the current notebook kernel:

```python
%pip install git+https://github.com/Hollfelder-Lab/crecerelle.git
```

If Colab requests a runtime restart after changing dependencies, restart before importing Crecerelle. A virtual environment is not needed for this Colab workflow.

## Python package

To install the package directly from GitHub into your activated local environment:

```bash
python -m pip install git+https://github.com/Hollfelder-Lab/crecerelle.git
```

This installs the Python package from GitHub; it does not create a local checkout of the tutorial notebooks. Use the Local Development instructions if you need the repository files.

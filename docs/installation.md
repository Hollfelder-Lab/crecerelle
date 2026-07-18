# Installation

Crecerelle is packaged with `setup.py` and requires Python 3.10 or newer.

## Local Development

From a checkout of this repository:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The package dependencies are declared in `setup.py`, including PyTorch, AnnData, Scanpy, NumPyro, pandas, NumPy, Matplotlib, seaborn, scikit-learn, UMAP, Leiden, igraph, scib, upsetplot, gprofiler-official, and matplotlib-venn.

## Documentation

Install the documentation dependencies from the repository root:

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

The documentation build renders notebooks without executing them.

## Google Colab

The notebooks use a Colab secret named `Crecerelle-GitHub-Access` and install the package from this private repository:

```python
import os
from google.colab import userdata

github_token = userdata.get("Crecerelle-GitHub-Access")
os.environ["Crecerelle-GitHub-Access"] = github_token
```

```bash
pip install git+https://${Crecerelle-GitHub-Access}@github.com/Hollfelder-Lab/crecerelle.git
```

Some notebooks pin `pandas==2.2.2` in the install cell because that pin is present in the runnable notebook source.

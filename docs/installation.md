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

Install the package from this repository in Colab

```python
!pip install git+https://github.com/Hollfelder-Lab/crecerelle.git
```

## Python package

Install the Python package via pip

```bash
pip install git+https://github.com/Hollfelder-Lab/crecerelle.git
```


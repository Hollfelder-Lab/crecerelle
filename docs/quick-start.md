# Quick Start

Most users should start from the notebooks because they define the data paths, Colab runtime setup, model checkpoints, and analysis outputs used in the paper workflows.

Create the expected results directory layout with the same helper used by the notebooks:

```python
from crecerelle.utils import setup_crecerelle

setup_crecerelle("/content/drive/My Drive", dataset_name="tabulaMuris")
```

For a local scratch run, choose a local root directory instead:

```python
from crecerelle.utils import setup_crecerelle

setup_crecerelle(".", dataset_name="tabulaMuris")
```

The notebooks expect data and outputs below `crecerelle_results`, with subdirectories for `data`, `figures`, and model checkpoints under `models/scVI`, `models/scTUVI`, and `models/scGETUVI`.

Recommended notebook order:

1. [Data Processing and Clustering Assessment](notebooks/main_DataProcessing_ClustAssess.ipynb)
2. [scTUVI Training](notebooks/main_Training_scTUVI.ipynb)
3. [scVI Training](notebooks/main_Training_scVI.ipynb)
4. [scGETUVI Training](notebooks/main_Training_scGETUVI.ipynb)
5. [scTUVI Model Selection](notebooks/main_Model_Selection_scTUVI.ipynb)
6. [scGETUVI Model Selection](notebooks/main_Model_Selection_scGETUVI.ipynb)
7. [scTUVI/scVI Inference](notebooks/main_Inference_scTUVI_scVI.ipynb)
8. [scGETUVI Inference](notebooks/main_Inference_scGETUVI.ipynb)

See the [tutorial index](tutorials/index.md) for the input files and expected outputs for each notebook.

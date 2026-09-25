# Crecerelle

This repo contains the processing scripts, code and evaluation methods for the paper XX.

![crecerelle_model_overview](figures/figure1_crecerelle_overview.png)

## 📦 Installation

Requires Git and Python 3.12 or newer; Python 3.12 is recommended to match the automated checks. Install into an activated virtual environment.

**Via GitHub checkout**:

```bash
git clone https://github.com/Hollfelder-Lab/crecerelle.git
cd crecerelle
python -m pip install -e .
```

**Directly with pip from GitHub**:

```bash
python -m pip install git+https://github.com/Hollfelder-Lab/crecerelle.git
```

## 🚀 Usage

For data-processing see
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main_DataProcessing_ClustAssess.ipynb)

For training tuVI see
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main_Training_scTUVI.ipynb)

For training scVI see
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main_Training_scVI.ipynb)

For training TRVI see
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main_Training_scGETUVI.ipynb)

For model selection of tuVI see
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main_Model_Selection_scTUVI.ipynb)

For model selection of TRVI see
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main_Model_Selection_scGETUVI.ipynb)

For inference with tuVI and scVI see
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main_Inference_scTUVI_scVI.ipynb)

For inference with TRVI see
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main_Inference_scGETUVI.ipynb)

## 🧪 Processed datasets and trained models 

Our data and trained models are available on Zenodo: XX

## 📜 License

This code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📃 Citing this work

Please cite our paper if you use this code in your own work:

```bibtex
@article {Weberling2026,
	author = {Weberling, Friedrich-Maximilian and Ampartzidis, Ioakeim and Mohorianu, Irina and Hollfelder, Florian},
	title = {Deep generative embeddings of gene expression and splicing reposition the interpretation of single-cell transcriptomic signatures},
	elocation-id = {2026.09.23.753495},
	year = {2026},
	doi = {10.64898/2026.09.23.753495},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/09/25/2026.09.23.753495},
	eprint = {https://www.biorxiv.org/content/early/2026/09/25/2026.09.23.753495.full.pdf},
	journal = {bioRxiv}
}
```

## 👥 Authors

- [Hollfelder Lab](https://hollfelder.bioc.cam.ac.uk/), Department of Biochemistry, University of Cambridge, UK

## 📧 Contact

For questions, please contact

- Friedrich-Maximilian Weberling (fmw37(at)cam.ac.uk)
- Florian Hollfelder (fh111(at)cam.ac.uk)

## 🤝 Contributing

We welcome contributions to Crecerelle, including bug fixes, documentation improvements, tutorials, and new features.

To set up the development environment with Python 3.12:

```bash
git clone https://github.com/Hollfelder-Lab/crecerelle.git
cd crecerelle
chmod +x contribute.sh
./contribute.sh
source .venv/bin/activate
```

If you do not have write access, fork the repository first and clone your fork instead. Submit your changes through a pull request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full setup, testing, and submission instructions.




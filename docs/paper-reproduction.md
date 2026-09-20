# Paper Reproduction

To reproduce the analyses and figures of the paper for the Tabula Muris data, run the Colab notebooks in the order described below. Ensure that you downloaded the AnnData objects from Zenodo and deposited in your file system (we provide a detailed explanation for integrating Google Drive alongside Colab).

## Workflow Order

1. [Data Processing and Clustering Assessment](notebooks/main_DataProcessing_ClustAssess.ipynb) prepares the gene expression and transcript usage AnnData objects, selects highly variable genes, creates train/validation/test splits, and writes the preprocessed data.
2. [tuVI Training](notebooks/main_Training_tuVI.ipynb) trains transcript usage tuVI model and writes tuVI checkpoints.
3. [scVI Training](notebooks/main_Training_tuVI.ipynb) trains the gene expression scVI model and writes the scVI model directory.
4. [TRVI Training](notebooks/main_Training_TRVI.ipynb) trains the joint gene expression and transcript usage TRVI model and writes TRVI checkpoints.
5. [tuVI Model Selection](notebooks/main_Model_Selection_tuVI.ipynb) evaluates tuVI checkpoints across configured random seeds and observation models.
6. [TRVI Model Selection](notebooks/main_Model_Selection_TRVI.ipynb) evaluates TRVI checkpoints across configured random seeds.
7. [tuVI/scVI Inference](notebooks/main_Inference_tuVI_scVI.ipynb) runs inference and downstream analyses for the unimodal models.
8. [TRVI Inference](notebooks/main_Inference_TRVI.ipynb) runs inference and downstream analyses for the joint model.

## Paper figures

To reproduce the paper figures 2 and 3, you need to run the tuVI/scVI Inference notebook. To reproduce the Figures 4,5, and 6, you need to run the TRVI Inference notebook. All notebooks use the default settings to reproduce the results from the paper.

## Data And Checkpoints

The notebooks use `dataset_name = "tabulaMuris"` by default and construct paths under `./data/<dataset_name>/`, `./figures/<dataset_name>/`, and `./models/`.

The preprocessing notebook expects raw gene expression and splicing AnnData files named `adata_exp.h5ad` and `adata_spl.h5ad`. 

Training notebooks create or copy checkpoint outputs under `./models/scVI`, `./models/tuVI`, and `./models/TRVI`. The model-selection and inference notebooks expect those trained model files to already exist. You can either use our trained models and download them from Zenodo or train them from scratch.

# Paper Reproduction

The notebooks are the paper-reproduction workflow. They should be run in Colab or an equivalent environment with the required data files and model-training resources.

The documentation workflow does not execute these notebooks. This keeps the GitHub Actions documentation build lightweight and avoids re-running expensive model training.

## Workflow Order

1. [Data Processing and Clustering Assessment](notebooks/main_DataProcessing_ClustAssess.ipynb) prepares the GE and TU AnnData objects, selects highly variable genes, creates train/validation/test splits, and writes preprocessed data.
2. [tuVI Training](notebooks/main_Training_tuVI.ipynb) trains transcript usage tuVI model and writes tuVI checkpoints.
3. [scVI Training](notebooks/main_Training_tuVI.ipynb) trains the gene expression scVI model and writes the scVI model directory.
4. [TRVI Training](notebooks/main_Training_TRVI.ipynb) trains the joint gene expression and transcript usage TRVI model and writes TRVI checkpoints.
5. [tuVI Model Selection](notebooks/main_Model_Selection_tuVI.ipynb) evaluates tuVI checkpoints across configured random seeds and observation models.
6. [TRVI Model Selection](notebooks/main_Model_Selection_TRVI.ipynb) evaluates TRVI checkpoints across configured random seeds.
7. [tuVI/scVI Inference](notebooks/main_Inference_tuVI_scVI.ipynb) runs inference and downstream analyses for the unimodal models.
8. [TRVI Inference](notebooks/main_Inference_TRVI.ipynb) runs inference and downstream analyses for the joint model.

## Data And Checkpoints

The notebooks use `dataset_name = "tabulaMuris"` by default and construct paths under `./data/<dataset_name>/`, `./figures/<dataset_name>/`, and `./models/`.

The preprocessing notebook expects raw gene-expression and splicing AnnData files named `adata_exp.h5ad` and `adata_spl.h5ad`. For the default Tabula Muris workflow it also references `adata_cellxgene.h5ad` and the gene annotation file `Mus_musculus.GRCm38.102.chr.gtf.gz`.

Training notebooks create or copy checkpoint outputs under `./models/scVI`, `./models/tuVI`, and `./models/TRVI`. The model-selection and inference notebooks expect those trained model files to already exist.

Runtime and exact memory requirements are not recorded in the notebooks. The notebook metadata requests Colab high-memory runtimes and GPU runtimes for model selection, training, and inference, so these should be measured and recorded during a verified reproduction run.

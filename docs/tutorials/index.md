# Tutorial Index

The tutorials are the repository notebooks rendered directly by `mkdocs-jupyter`. The documentation does not manually duplicate notebook code or outputs.

Each runnable notebook page includes an "Open in Colab" link.

## Data Processing and Clustering Assessment

[Open notebook](../notebooks/main_DataProcessing_ClustAssess.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_DataProcessing_ClustAssess.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: preprocess raw gene expression and transcript usage data, perform quality control, run ClustAssess-based highly variable gene assessment, and split GE/TU data.
- Input data: `adata_exp.h5ad`, `adata_spl.h5ad`, the default `dataset_name = "tabulaMuris"`, and for that default workflow `adata_cellxgene.h5ad` plus `Mus_musculus.GRCm38.102.chr.gtf.gz`.
- Expected runtime: not stated in the notebook; the workflow includes preprocessing and ClustAssess steps and should be timed during a verified reproduction run.
- Memory requirements: exact memory is not stated; Colab metadata requests a high-memory runtime.
- GPU recommended: not indicated by notebook metadata.
- Expected outputs: `clustassess_plots.pdf`, `full_hvg_annotation.npy`, `adata_GE_full_preprocessed.h5ad`, `adata_TU_full_preprocessed.h5ad`, and GE/TU train, validation, and test `.h5ad` files.

## scTUVI Model Selection

[Open notebook](../notebooks/main_Model_Selection_scTUVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Model_Selection_scTUVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: compare transcript-usage scTUVI model variants and configured random seeds.
- Input data: preprocessed TU data and train/validation/test splits from preprocessing, plus trained scTUVI checkpoints under `./models/scTUVI/`.
- Expected runtime: not stated in the notebook; it evaluates multiple seeds and checkpoints and should be treated as a GPU evaluation workflow.
- Memory requirements: exact memory is not stated; Colab metadata requests a high-memory runtime.
- GPU recommended: yes; notebook metadata requests a GPU runtime.
- Expected outputs: `model_selection_scTUVI_<observation_model>_eval_metric_summary.csv` and rendered comparison plots.

## scTUVI Training

[Open notebook](../notebooks/main_Training_scTUVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Training_scTUVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: train transcript-usage VAE models with DM, ZANIDM, or ZIDM observation-model settings.
- Input data: `adata_TU_full_preprocessed.h5ad` and TU train/validation/test `.h5ad` files from preprocessing.
- Expected runtime: not stated in the notebook; the configured training cell uses `num_epochs = 300`.
- Memory requirements: exact memory is not stated; Colab metadata requests a high-memory GPU runtime.
- GPU recommended: yes; notebook metadata requests a GPU runtime.
- Expected outputs: scTUVI checkpoint files matching `epochs_*_checkpoint.pth` under `./models/scTUVI/`.

## scVI Training

[Open notebook](../notebooks/main_Training_scVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Training_scVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: train the scVI gene-expression model using `scvi-tools`.
- Input data: `adata_GE_full_preprocessed.h5ad` and GE train/validation/test `.h5ad` files from preprocessing.
- Expected runtime: not stated in the notebook; the configured training cell uses `num_epochs = 300`.
- Memory requirements: exact memory is not stated; Colab metadata requests a high-memory GPU runtime.
- GPU recommended: yes; notebook metadata requests a GPU runtime.
- Expected outputs: a saved scVI model directory under `./models/scVI/`.

## scGETUVI Model Selection

[Open notebook](../notebooks/main_Model_Selection_scGETUVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Model_Selection_scGETUVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: compare joint GE/TU scGETUVI checkpoints across configured random seeds.
- Input data: preprocessed GE and TU data, GE/TU train/validation/test splits, and trained scGETUVI checkpoints under `./models/scGETUVI/`.
- Expected runtime: not stated in the notebook; it evaluates multiple seeds and checkpoints and should be treated as a GPU evaluation workflow.
- Memory requirements: exact memory is not stated; Colab metadata requests a high-memory GPU runtime.
- GPU recommended: yes; notebook metadata requests a GPU runtime.
- Expected outputs: `model_selection_scGETUVI_eval_metric_summary.csv` and rendered comparison plots.

## scGETUVI Training

[Open notebook](../notebooks/main_Training_scGETUVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Training_scGETUVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: train the joint gene-expression and transcript-usage scGETUVI model.
- Input data: preprocessed GE/TU data and matching GE/TU train/validation `.h5ad` files from preprocessing.
- Expected runtime: not stated in the notebook; the configured training cell uses `num_epochs = 1000` with a notebook comment noting `usually 200`.
- Memory requirements: exact memory is not stated; Colab metadata requests a high-memory GPU runtime.
- GPU recommended: yes; notebook metadata requests a GPU runtime.
- Expected outputs: scGETUVI checkpoint files matching `epochs_*_checkpoint.pth` under `./models/scGETUVI/`.

## scTUVI/scVI Inference

[Open notebook](../notebooks/main_Inference_scTUVI_scVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Inference_scTUVI_scVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: run inference and downstream analyses for the gene-expression scVI model and transcript-usage scTUVI models.
- Input data: preprocessed GE/TU data, GE/TU splits, the saved scVI model under `./models/scVI/`, and scTUVI checkpoints under `./models/scTUVI/`.
- Expected runtime: not stated in the notebook; it loads models, evaluates embeddings, and performs downstream analyses.
- Memory requirements: exact memory is not stated; Colab metadata requests a high-memory GPU runtime.
- GPU recommended: yes; notebook metadata requests a GPU runtime.
- Expected outputs: GE and TU cluster-evaluation CSV files, `scvi_sctuvi_evaluation_df.csv`, DEG and DSG CSV files, and `dsg_analysis_heart_sort_<sorting>_clustergroup_<group_id>_marker_introns.pdf`.

## scGETUVI Inference

[Open notebook](../notebooks/main_Inference_scGETUVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Inference_scGETUVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: run inference and downstream analyses for the joint GE/TU scGETUVI model.
- Input data: preprocessed GE/TU data, GE/TU splits, and scGETUVI checkpoints under `./models/scGETUVI/`.
- Expected runtime: not stated in the notebook; it loads model checkpoints, evaluates embeddings, and performs downstream analyses.
- Memory requirements: exact memory is not stated; Colab metadata requests a high-memory GPU runtime.
- GPU recommended: yes; notebook metadata requests a GPU runtime.
- Expected outputs: `evaluation_embeddings_scgetuvi.csv`, DEG and DSG CSV files, `fig6_final_draft.pdf`, `cell_atlas_scgetuvi.pdf`, and `fig6_draft.pdf`.

## Notes Requiring Reproduction Review

The notebooks do not record exact wall-clock runtimes or memory consumption. The runtime and memory entries above therefore report the notebook configuration and observed workflow type rather than measured benchmarks.

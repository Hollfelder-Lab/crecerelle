# Tutorial Index

The tutorials are the repository notebooks rendered directly by `mkdocs-jupyter`. They comprise our data processing, the training of tuVI, scVI, and TRVI, as well as the Crecerelle workflows for downstream tasks (e.g. UMAP visualisations, data-driven cell annotation, functional enrichment analysis). All tutorials use the default parameters for the Tabule Muris dataset (`dataset_name = "tabulaMuris"`).

Each runnable notebook page includes an "Open in Colab" link.

## Data Processing and Clustering Assessment

[Open notebook](../notebooks/main_DataProcessing_ClustAssess.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_DataProcessing_ClustAssess.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: Pre-processed matched single-cell gene expression (GE) and alternative splicing-induced transcript usage (TU) data are submitted to annotation of isoform groups (intron groups), quality control checks and data filtering, selection of a suitable number of highly variable genes (HVG) using ClustAssess, and finally split into a training, validation, and test dataset.
- Input data: AnnData file of single cell gene expression `adata_exp.h5ad`, AnnData file of single cell alternative splicing-induced transcript usage `adata_spl.h5ad`, a dataset name `dataset_name = "tabulaMuris"`, and a gene annotation file (Mus Musculus by default) `Mus_musculus.GRCm38.102.chr.gtf.gz`.
- GPU recommended: yes, it is recommended to use GPUs to process large datasets
- Expected outputs: plots of QC and ClustAssess, a numpy arrat of the highly variable genes used `full_hvg_annotation.npy`, an AnnData object containing the full processed gene expresssion data `adata_GE_full_preprocessed.h5ad`, an AnnData object containing the full processed alternative splicing-induced transcript usage data`adata_TU_full_preprocessed.h5ad`, further six `.h5ad` files containing the training, validation, and test dataset.

## tuVI Training

[Open notebook](../notebooks/main_Training_tuVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Training_tuVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: Training of the variational autoencoder transcript usage Variational Inference (tuVI) where different observation models can be specified. We accomodate the Dirichlet-Multinomial DM) distribution, the zero-and-N-inflated Dirichlet-Multinomial (ZANIDM) distribution, and the heurstic zero-inflated Dirichlet-Multinomial (ZIDM) surroagte with DM for observation model settings. This notebook requires to have executed the data processing and clustering assessment of Crecerelle (previous workflow).
- Input data: AnnData file of processed single cell alternative splicing-induced transcript usage data `adata_TU_full_preprocessed.h5ad`, and three  `.h5ad` files containing the training, validation, and test data.
- GPU recommended: yes, the training of the deep learning models requires a GPU.
- Expected outputs: the best tuVI checkpoint files (`.pth`), as evaluated on the validation partition, are stored in `./models/tuVI/`.

## scVI Training

[Open notebook](../notebooks/main_Training_scVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Training_scVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: Training of the variational autoencoder single-cell Variational Inference (scVI) using `scvi-tools` with its default training settings.
- Input data: AnnData file of processed single cell gene expression data `adata_GE_full_preprocessed.h5ad`, and three  `.h5ad` files containing the training, validation, and test data.
- GPU recommended: yes, the training of the deep learning models requires a GPU.
- Expected outputs: a best scVI checkpoint file (`.pth`) is saved in `./models/scVI/`.

## TRVI Training

[Open notebook](../notebooks/main_Training_TRVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Training_TRVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: Training of the relevance-weighted mixture-of-experts multimodal variational autoencoder Transcriptomic Regulation Variational Inference (TRVI) where different observation models can be specified. This notebook requires to have executed the data processing and clustering assessment of Crecerelle.
- Input data: AnnData files of processed single cell gene expression data `adata_GE_full_preprocessed.h5ad` and single-cell alternative splicing-induced transcript usage data `adata_TU_full_preprocessed.h5ad`, and six  `.h5ad` files containing the training, validation, and test data (three for each modality).
- GPU recommended: yes, the training of the deep learning models requires a GPU.
- Expected outputs: the best TRVI checkpoint files (`.pth`), as evaluated on the validation partition, are stored in `./models/TRVI/`.

## tuVI Model Selection

[Open notebook](../notebooks/main_Model_Selection_tuVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Model_Selection_tuVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: evaluate the tuVI models trained across different random seeds and select the best model for downstream tasks. This requires to have executed tuVI training using different random seeds and observation models as detailed by Crecerelle's tuVI training workflow.
- Input data: AnnData file of processed single cell alternative splicing-induced transcript usage data `adata_TU_full_preprocessed.h5ad`, three  `.h5ad` files containing the training, validation, and test data, and tuVI model checkpoints stored in `models/tuVI`.
- GPU recommended: yes, the training of the deep learning models requires a GPU.
- Expected outputs: figure panels visualising the performance metrics, Pandas dafaframes saved as `.csv` files containing the exact performance metrics values per model evaluated.

## TRVI Model Selection

[Open notebook](../notebooks/main_Model_Selection_TRVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Model_Selection_TRVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: evaluate the tuVI models trained across different random seeds and select the best model for downstream tasks. This requires to have executed TRVI training using different random seeds and observation models as detailed by Crecerelle's tuVI training workflow. In addition, the stability of the latent space geomtry of the joint gene expression and transcritp usage latent space is assessed across different random seedss
- Input data: AnnData files of processed single cell gene expression data `adata_GE_full_preprocessed.h5ad` and single-cell alternative splicing-induced transcript usage data `adata_TU_full_preprocessed.h5ad`, six  `.h5ad` files containing the training, validation, and test data (three for each modality), and TRVI model checkpoints stored in `models/TRVI`.
- GPU recommended: yes, the training of the deep learning models requires a GPU.
- Expected outputs: figure panels visualising the performance metrics, Pandas dafaframes saved as `.csv` files containing the exact performance metrics values per model evaluated. Figure panels visualising the latent space geometry assessment.

## tuVI/scVI Inference

[Open notebook](../notebooks/main_Inference_tuVI_scVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Inference_tuVI_scVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: The gene expression cell embeddings and transcript usage cell embeddings are inferred using the trained scVI and tuVI models respectively and stored. The notebook contains Crecerelle's downstream notebook for data-driven cell annotation (inlcuding UMAP visualisation, clustering, differential analysis), differential analysis comparisons, identification of isoform-specific subpopulations is carried out per transcriptomic facets. This workflow requires to have executed scVI training and having selected a suitable tuVI model.
- Input data: AnnData files of processed single cell gene expression data `adata_GE_full_preprocessed.h5ad` and single-cell alternative splicing-induced transcript usage data `adata_TU_full_preprocessed.h5ad`, six  `.h5ad` files containing the training, validation, and test data (three for each modality), the saved scVI model under `./models/scVI/`, and tuVI checkpoints under `./models/tuVI/`.
- GPU recommended: yes, for the inference of the cell embeddings; the other downstream task can be run with a CPU but we still recommend a GPU.
- Expected outputs: AnnData objects (`.h5ad` files) with inferred cell embeddings for gene expression and transcript usage for the full dataset and selected tissues, figure panels visualising data-driven cell annotation results, UMAPs of scVI and tuVI for the full datasets and selected tissues, Pandas dataframes stored as `.csv` files containing performance metrics (e.g. NMI, ARI, ASW, AvgBio), differentially expressed genes, and differentially spliced genes.
## TRVI Inference

[Open notebook](../notebooks/main_Inference_TRVI.ipynb){ .md-button }
[Open in Colab](https://colab.research.google.com/github/Hollfelder-Lab/crecerelle/blob/main/notebooks/main_Inference_TRVI.ipynb){ .md-button .md-button--primary target="_blank" rel="noopener" }

- Purpose: The cell embeddings of TRVI (P-GE, P-TU, S-GE, S-TU, GE-TU), and the single-cell modality-relevance weights are inferred using the trained TRVI model. The notebook contains Crecerelle's downstream notebook for joint data-driven cell annotation (inlcuding UMAP visualisation, clustering, differential analysis), UMAP visualisation of TRVI's latent spaces, differential analysis comparisons, identification of cell-cluster-specific isoform regulation, functional enrichment analysis, biological pathway comparisons. This workflow requires to have executed TRVI training and having selected a suitable TRVI model.
- Input data: AnnData files of processed single cell gene expression data `adata_GE_full_preprocessed.h5ad` and single-cell alternative splicing-induced transcript usage data `adata_TU_full_preprocessed.h5ad`, six  `.h5ad` files containing the training, validation, and test data (three for each modality), the saved scVI model under `./models/scVI/`, and TRVI checkpoints under `./models/TRVI/`.
- GPU recommended: yes, for the inference of the cell embeddings; the other downstream task can be run with a CPU but we still recommend a GPU.
- Expected outputs: AnnData objects (`.h5ad` files) with inferred cell embeddings for gene expression and transcript usage for the full dataset and selected tissues, figures visualising analysis of TRVI's latent spaces via UMAPs and distance matrices, full datasets and subset-specific case studies of integrating gene expression and transcript usage data, distributions of cell-specific, cell-type-specific, subset-specific, and dataset-specfic modality-relevance weights, GO-figure plots visualising identfied biological pathways. Pandas dataframes saved as `.csv` files containing differentially expressed and spliced genes, and functional enrichment terms, and further benchmarking metrics (NMI, ARI, ASW, AvgBio)


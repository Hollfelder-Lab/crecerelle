#!/usr/bin/env python3
import pdb

import numpy as np
from typing import Tuple, Any, List, Dict
import torch
import torch.nn as nn

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, iqr
import scipy.sparse

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import collections
import pandas as pd
from anndata import AnnData
import anndata as ad

import sklearn
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, accuracy_score
from collections import Counter
from umap import UMAP
import os
import itertools
import scanpy as sc

import networkx as nx

# Crecrelle modules
from .models import TRVI, LogisticRegressionClassifier
from .cell import TABULA_MURIS_CELL_TYPES, TABULA_MURIS_TISSUE_CELL_DICTIONARY
from .training_utils import train_classifier, train_vae_embedding_cell_type_classifier

def add_gene_annotation(
        adata: AnnData,
        gtf_path: str,
        filter_unique_gene: bool =True
) -> AnnData:
    r"""
    Legacy function of scQuint. Given an AnnData object containing the splicing data (exon junction reads mapped to
    introns and intron groups) and a gtf file containing the gene annotation, add the gene annotation to the AnnData
    object. If filter_unique_gene is True, only unique gene correspondences are considered. The AnnData object is
    returned.

    :param adata: AnnData object
    :param gtf_path: str
    :param filter_unique_gene: bool
    :return adata: AnnData object
    """
    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        header=None,
        comment="#",
        names=[
            "chromosome",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attribute",
        ],
    )
    gtf = gtf[gtf.feature == "exon"]
    gtf["gene_id"] = gtf.attribute.str.extract(r'gene_id "([^;]*)";')
    gtf["gene_name"] = gtf.attribute.str.extract(r'gene_name "([^;]*)";')
    gtf.chromosome = "chr" + gtf.chromosome.astype(str)

    gene_id_name = gtf[["gene_id", "gene_name"]].drop_duplicates()

    exon_starts = (
        gtf[["chromosome", "start", "gene_id"]].copy().rename(columns={"start": "pos"})
    )
    exon_starts.pos -= 1
    exon_ends = (
        gtf[["chromosome", "end", "gene_id"]].copy().rename(columns={"end": "pos"})
    )
    exon_ends.pos += 1
    exon_boundaries = pd.concat(
        [exon_starts, exon_ends], ignore_index=True
    ).drop_duplicates()

    genes_by_exon_boundary = exon_boundaries.groupby(
        ["chromosome", "pos"]
    ).gene_id.unique()

    adata.var = (
        adata.var.merge(
            genes_by_exon_boundary,
            how="left",
            left_on=["chromosome", "start"],
            right_on=["chromosome", "pos"],
        )
        .rename(columns={"gene_id": "gene_id_start"})
        .set_index(adata.var.index)
    )
    adata.var = (
        adata.var.merge(
            genes_by_exon_boundary,
            how="left",
            left_on=["chromosome", "end"],
            right_on=["chromosome", "pos"],
        )
        .rename(columns={"gene_id": "gene_id_end"})
        .set_index(adata.var.index)
    )

    def fill_na_with_empty_array(val):
        return val if isinstance(val, np.ndarray) else np.array([])

    adata.var.gene_id_start = adata.var.gene_id_start.apply(fill_na_with_empty_array)
    adata.var.gene_id_end = adata.var.gene_id_end.apply(fill_na_with_empty_array)

    adata.var["gene_id_list"] = adata.var.apply(
        lambda row: np.unique(np.concatenate([row.gene_id_start, row.gene_id_end])),
        axis=1,
    )
    adata.var["n_genes"] = adata.var.gene_id_list.apply(len)
    adata.var.gene_id_list = adata.var.gene_id_list.apply(
        lambda x: ",".join(x.tolist())
    )
    adata.var.gene_id_start = adata.var.gene_id_start.apply(
        lambda x: ",".join(x.tolist())
    )
    adata.var.gene_id_end = adata.var.gene_id_end.apply(
        lambda x: ",".join(x.tolist())
    )

    if filter_unique_gene:
        print("Filtering to introns associated to 1 and only 1 gene.")
        adata = adata[:, adata.var.n_genes == 1]
        adata.var["gene_id"] = adata.var.gene_id_list
        adata.var.drop(columns=["gene_id_list",], inplace=True)
        adata.var = adata.var.merge(gene_id_name, how="left", on="gene_id").set_index(
            adata.var.index
        )
        adata.var.index = adata.var.gene_name.astype(str) + "_" + adata.var.index.astype(str)

    return adata


def group_introns(
        adata : AnnData,
        by: str = "three_prime",
        filter_unique_gene_per_group: bool = True
) -> AnnData:
    r"""
    Legacy function of scQuint. Group the introns and their exon junction reads into intron groups. The options are to
    group introns by "three_prime", "five_prime", or "gene". The default is by "three_prime. If
    filter_unique_gene_per_group is true, only intron groups that are associated with 1 and only 1 gene are kept. The
    function returns the AnnData object with the introns grouped into intron groups.

    :param adata: AnnData
    :param by: str
    :param filter_unique_gene_per_group: bool
    :return: AnnData
    """
    if by == "three_prime":
        adata.var["intron_group"] = adata.var.apply(
            lambda intron: intron.chromosome
            + "_"
            + (str(intron.end) if intron.strand == "+" else str(intron.start))
            + "_"
            + intron.strand,
            axis=1,
        )
    elif by == "five_prime":
        adata.var["intron_group"] = adata.var.apply(
            lambda intron: intron.chromosome
            + "_"
            + (str(intron.end) if intron.strand == "-" else str(intron.start))
            + "_"
            + intron.strand,
            axis=1,
        )
    elif by == "gene":
        adata.var["intron_group"] = adata.var.gene_id
    else:
        raise Exception(f"Grouping by {by} not yet supported.")

    intron_group_sizes = (
        adata.var.intron_group.value_counts()
        .rename("intron_group_size")
        .to_frame()
    )
    adata.var = adata.var.merge(
        intron_group_sizes, how="left", left_on="intron_group", right_index=True
    ).set_index(adata.var.index)
    print("Filtering singletons.")
    adata = adata[:, adata.var.intron_group_size > 1]

    if filter_unique_gene_per_group:
        print("Filtering intron groups associated with more than 1 gene.")
        n_genes_per_intron_group = adata.var.groupby("intron_group").gene_id.nunique().to_frame().rename(columns={"gene_id": "n_genes_per_intron_group"})
        adata.var = adata.var.merge(n_genes_per_intron_group, how="left", left_on="intron_group", right_index=True)
        adata = adata[:, adata.var.n_genes_per_intron_group==1]
        adata.var.intron_group = adata.var.gene_name.astype(str) + "_" + adata.var.intron_group.astype(str)

    return adata


def relabel(labels):
    r"""
    """
    all_old_labels = pd.unique(labels).tolist()
    mapping = {c: i for i, c in enumerate(all_old_labels)}
    new_labels = np.array([mapping[l] for l in labels])

    return new_labels

def filter_min_cells_per_feature(adata, min_cells_per_feature, idx_cells_to_count=slice(None)):
    # from scquint_data.py
    print("filter_min_cells_per_feature")
    idx_features = np.where((adata.X[idx_cells_to_count] > 0).sum(axis=0).A1 >= min_cells_per_feature)[0]
    adata = adata[:, idx_features]
    adata = filter_singletons(adata)
    return adata

def filter_min_cells_per_intron_group(adata, min_cells_per_intron_group, idx_cells_to_count=slice(None)):
    # from scquint_data.py
    print("filter_min_cells_per_intron_group")
    intron_groups = relabel(adata.var.intron_group.values)
    intron_group_summation = make_intron_group_summation_cpu(intron_groups)
    n_cells_per_intron_group = ((((adata.X[idx_cells_to_count]) @ intron_group_summation) > 0).sum(axis=0)).A1
    idx_intron_groups = np.where(n_cells_per_intron_group >= min_cells_per_intron_group)[0]
    idx_features = np.where(np.isin(intron_groups, idx_intron_groups))[0]
    adata = adata[:, idx_features]
    adata = filter_singletons(adata)
    return adata

def filter_singletons(adata):
    # from scquint_data.py
    print("filter_singletons")
    intron_group_counter = Counter(adata.var.intron_group.values)
    intron_group_counts = np.array([intron_group_counter[c] for c in adata.var.intron_group.values])
    idx_features = np.where(intron_group_counts > 1)[0]
    adata = adata[:, idx_features]
    return adata

def relabel(labels):
    r"""
    """
    all_old_labels = pd.unique(labels).tolist()
    mapping = {c: i for i, c in enumerate(all_old_labels)}
    new_labels = np.array([mapping[l] for l in labels])

    return new_labels


def make_intron_group_summation_cpu(intron_groups: np.ndarray) -> scipy.sparse.csr_matrix:
    r"""
    Given an array of intron groups, create a sparse matrix that sums the intron counts for each group.
    The function first counts the number of unique intron groups and then creates a sparse matrix
    where each row corresponds to an intron and each column corresponds to an intron group. The
    values in the matrix are the counts of introns in each group.
    :param intron_groups: Array of intron groups
    :return: Sparse matrix of intron group summation
    :rtype: scipy.sparse.csr_matrix
    Example:
    >>> intron_groups = np.array([0, 1, 0, 2, 1])
    >>> intron_group_summation = make_intron_group_summation_cpu(intron_groups)
    >>> print(intron_group_summation)
    >>> # Output:
    >>> #   (0, 0)	1
    >>> #   (1, 1)	1
    >>> #   (2, 0)	1
    >>> #   (3, 2)	1
    """
    n_introns = len(intron_groups)
    n_intron_groups = len(np.unique(intron_groups))
    rows, cols = zip(*list(enumerate(intron_groups)))
    vals = np.ones(n_introns, dtype=int)
    intron_group_summation = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), (n_introns, n_intron_groups)
    ).tocsr()

    return intron_group_summation

def compute_psi(adata: AnnData) -> np.ndarray:
    r"""
    Given an AnnData object, compute the PSI values for each cell and intron group. The PSI (percent-spliced-in) score
    is calculated via

    .. math::
        \psi_g^{c} = \frac{1}{\sum_i (\mathbf{x}_g^c)_i} \mathbf{x}_g^c

    where :math:`\mathbf{x}_g^c` are the intron counts of intron group :math:`g` for a specifc cell :math:`c`.

    The function first relabels the intron groups and then computes the PSI values using the formula given above.

    :param adata: AnnData object containing the data
    :return: PSI values
    :rtype: numpy.ndarray

    Example:
    >>> adata = AnnData(X, var=var)
    >>> psi_values = compute_psi(adata)
    >>> print(psi_values)
    >>> # Output:
    >>> # [[0.5, 0.5, 0.0],
    >>> #  [0.0, 1.0, 0.0],
    >>> #  [0.0, 0.0, 1.0]]
    The PSI values are calculated for each cell and intron group, and the resulting matrix is returned.
    """
    groups = adata.var.intron_group.values
    intron_groups = relabel(groups)

    # Calculate in each cell for each intron group the PSI scores
    intron_count_matrix = adata.X.toarray()
    psi_matrix = np.zeros(intron_count_matrix.shape)

    for unique_intron_group in np.unique(intron_groups):
        # Determine intron indices of group and intron group sum
        intron_indices = np.argwhere(intron_groups == unique_intron_group).squeeze()

        # Ensure intron_indices is treated as a 1D array of indices even if there's only one
        if intron_indices.ndim == 0:
            raise ValueError("Zero count intron columns have been removed in the pre-processing step.")
        elif intron_indices.ndim > 1:
            intron_indices = intron_indices.flatten()

        intron_counts_in_group = intron_count_matrix[:, intron_indices]
        intron_group_sums = np.sum(intron_counts_in_group, axis=1)

        # Calculate PSI score
        psi_scores = intron_counts_in_group / intron_group_sums.reshape(-1, 1)

        # Set NaN values to zero (elements where intron counts where not present)
        psi_scores = np.nan_to_num(psi_scores, nan=0.0)

        psi_matrix[:, intron_indices] = psi_scores

    return psi_matrix


def smoothed_psi(adata: AnnData) -> np.ndarray:
    r"""
    Given an AnnData object, compute the smoothed PSI values for each cell and intron group.
    The function first relabels the intron groups and then computes the PSI values using the
    `make_intron_group_summation_cpu` function. The smoothed PSI values are then returned.

    :param adata: AnnData object containing the data
    :return: smoothed PSI values
    :rtype: numpy.ndarray

    Example:
    >>> adata = AnnData(X, var=var)
    >>> smoothed_psi_values = smoothed_psi(adata)
    >>> print(smoothed_psi_values)

    """

    groups = adata.var.intron_group.values

    intron_groups = relabel(groups)
    intron_group_summation = make_intron_group_summation_cpu(intron_groups)
    x = adata.X.toarray()
    intron_sum = x.sum(axis=0)
    intron_group_sum = intron_sum @ intron_group_summation
    psi_global = intron_sum / intron_group_sum[intron_groups]

    x_smoothed = x + psi_global
    intron_group_sums = x_smoothed @ intron_group_summation
    psi_smoothed = x_smoothed / intron_group_sums[:, intron_groups]

    return psi_smoothed

def remove_cell_genes(adata1: AnnData, adata2: AnnData) -> AnnData:
    r"""
    Given two Anndata objects where the first one contains the pre-processed gene expression data and the second one
    contains the raw transcript usage data, remove all intron count columns from the second Anndata object genes are
    not considered as highly variable in the first Anndata object. Then, all cells in the second Anndata object that
    are not present in the first Anndata object are removed. The function returns the filtered version of the second
    Anndata object.

    :param adata1: AnnData object containing the pre-processed gene expression data
    :param adata2: AnnData object containing the raw transcript usage data
    :return: AnnData object with filtered transcript usage data
    :rtype: AnnData

    """

    # Remove genes that are not highly variable
    hvg_gene_ids = adata1.var["gene_id"][adata1.var["highly_variable"] == True].to_numpy()
    intron_gene_ids = np.unique(adata2.var["gene_id"].to_numpy())
    num_removed_genes = 0
    num_tu_hvgs = 0

    for intron_gene_id in intron_gene_ids:
        if intron_gene_id not in hvg_gene_ids:
            #print(f"{intron_gene_id} is not a highly variable gene, the according intron count columns will be removed")
            adata2 = adata2[:, adata2.var["gene_id"] != intron_gene_id]
            num_removed_genes += 1
        else:
            #print(f"{intron_gene_id} is a highly variable gene")
            num_tu_hvgs += 1

    print(f"Number of removed genes: {num_removed_genes}")
    print(f"Number of highly variable genes in transcript usage data: {num_tu_hvgs}")

    # Remove cells that are not present in the first AnnData object
    cell_ids_1 = adata1.obs_names
    cell_ids_2 = adata2.obs_names

    # Get the intersection of cell IDs
    common_cell_ids = np.intersect1d(cell_ids_1, cell_ids_2)
    adata2 = adata2[adata2.obs_names.isin(common_cell_ids), :]

    assert np.all(adata1.obs_names == adata2.obs_names), "Cell IDs do not match between the two AnnData objects."

    # Remove columns in second Anndata object that are all zeros
    #if np.all(np.array(adata2.X.sum(axis=0)).flatten() != 0):
    #    print("All intron count columns are non-zero.")
    #else:
    #    print("Some intron count columns are zero, they will be removed.")
    #    # Remove columns that are all zeros
    #    adata2 = adata2[:, np.array(adata2.X.sum(axis=0)).flatten() != 0]

    return adata2



def pca_on_transcript_usage_data(adata: AnnData) -> AnnData:
    r"""
    Given an AnnData object, perform PCA on the transcript usage data. The function first
    computes the smoothed PSI values and then applies PCA to reduce the dimensionality of
    the data. The PCA results are stored in the `obsm` attribute of the AnnData object.

    :param adata: AnnData object containing the data
    :return: AnnData object with PCA results stored in `obsm`
    :rtype: AnnData

    Example:
    >>> adata = AnnData(X, var=var)
    >>> adata = pca_on_transcript_usage_data(adata)
    >>> print(adata.obsm["X_pca"])
    """
    psi_matrix = adata.layers["Psi"]
    intron_groups = adata.var.intron_group.values
    all_intron_groups = np.unique(intron_groups)
    first_indices_dict = {}
    for i, c in enumerate(intron_groups):
        if c not in first_indices_dict:
            first_indices_dict[c] = i
    first_indices = np.array([first_indices_dict[c] for c in all_intron_groups])
    psi_matrix = np.delete(psi_matrix, first_indices, axis=1)

    pc_latent = sklearn.decomposition.PCA(n_components=50).fit_transform(psi_matrix)

    adata.obsm["X_pca"] = pc_latent

    return adata


def setup_crecerelle(root_directory: str, **kwargs) -> None:
    r"""
    Given a root_directory, set-up the directory structure for Crecerelle. The function creates the following directory
    structure if it does not already exist:
    - root_directory/crecerelle_results
    - root_directory/crecerelle_results/models
    - root_directory/crecerelle_results/data
    - root_directory/crecerelle_results/figures
    - root_directory/crecerelle_results/models/scVI
    - root_directory/crecerelle_results/models/tuVI
    - root_directory/crecerelle_results/models/TRVI

    If a key word argument for the dataset name (dataset_name) is provided, following directories are created in
    addition:
    - root_directory/crecerelle_results/data/dataset_name
    - root_directory/crecerelle_results/figures/dataset_name

    :param root_directory: Root directory for crecerelle results
    """

    # Define the base directory for results
    results_dir = os.path.join(root_directory, "crecerelle_results")

    # List of baseline directories to create
    directories = [
        results_dir,
        os.path.join(results_dir, "models"),
        os.path.join(results_dir, "data"),
        os.path.join(results_dir, "figures"),
        os.path.join(results_dir, "models", "scVI"),
        os.path.join(results_dir, "models", "tuVI"),
        os.path.join(results_dir, "models", "TRVI"),
    ]

    # Check if dataset_name was provided in kwargs
    dataset_name = kwargs.get("dataset_name")
    if dataset_name:
        directories.extend([
            os.path.join(results_dir, "data", dataset_name),
            os.path.join(results_dir, "figures", dataset_name)
        ])

    # Create each directory only if does not already exist
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created: {directory}")
        else:
            print(f"Skipped (already exists): {directory}")

def rank_intron_groups_groups(adata: AnnData, diff_spl_intron_groups: pd.DataFrame, groupby: str) -> None:
    r"""
    Given an AnnData object and a DataFrame containing differential splicing results for intron groups, store the logfold
    changes, p-values, adjusted p-values, and names of the intron groups in the `uns` attribute of the AnnData object as
    dictionary "rank_intron_groups_groups".
    
    :param adata: AnnData object containing the data
    :param diff_spl_intron_groups: DataFrame containing differential splicing results for intron groups
    :param groupby: str specifying the groupby variable used in the differential splicing analysis (e.g. leiden_res_0.3)
    """
    # Extract results as arrays

    test_groups, counts = np.unique(diff_spl_intron_groups["test_group"].to_numpy(), return_counts=True)
    max_counts = np.max(counts)

    # create numpy arrays
    pvals = np.zeros((max_counts, len(test_groups)))
    pvals_adj = np.zeros((max_counts, len(test_groups)))
    logfoldchanges = np.zeros((max_counts, len(test_groups)))
    names = np.zeros((max_counts, len(test_groups)), dtype=object)
    gene_names = np.zeros((max_counts, len(test_groups)), dtype=object)

    for i, test_group in enumerate(test_groups):
        sig_diff_spl_test_group_df = diff_spl_intron_groups[diff_spl_intron_groups["test_group"] == test_group]
        logfoldchanges_group = sig_diff_spl_test_group_df["max_abs_lfc_psi"].to_numpy()
        pvals_group = sig_diff_spl_test_group_df["p_value"].to_numpy()
        pvals_adj_group = sig_diff_spl_test_group_df["p_value_adj"].to_numpy()
        names_group = sig_diff_spl_test_group_df["name"].to_numpy()
        gene_names_group = sig_diff_spl_test_group_df["gene_name"].to_numpy()

        logfoldchanges[:len(logfoldchanges_group), i] = logfoldchanges_group
        pvals[:len(pvals_group), i] = pvals_group
        pvals_adj[:len(pvals_adj_group), i] = pvals_adj_group
        names[:len(names_group), i] = names_group
        gene_names[:len(gene_names_group), i] = gene_names_group

    # Convert gene_names and names to string dtype to avoid issues with rec arrays
    names = names.astype(str)
    gene_names = gene_names.astype(str)

    logfoldchanges_array_list = []
    logfoldchanges_list_dtype = []
    pvals_array_list = []
    pvals_list_dtype = []
    pvals_adj_array_list = []
    pvals_adj_list_dtype = []
    names_array_list = []
    names_list_dtype = []
    gene_names_array_list = []
    gene_names_list_dtype = []

    """
    for i in range(len(test_groups)):
        logfoldchanges_array_list.append(logfoldchanges[:, i])
        logfoldchanges_list_dtype.append((str(i), np.float64))
        pvals_array_list.append(pvals[:, i])
        pvals_list_dtype.append((str(i), np.float64))
        pvals_adj_array_list.append(pvals_adj[:, i])
        pvals_adj_list_dtype.append((str(i), np.float64))
        names_array_list.append(names[:, i])
        names_list_dtype.append((str(i), object))
        gene_names_array_list.append(gene_names[:, i])
        gene_names_list_dtype.append((str(i), object))
    """

    for i, test_group in enumerate(test_groups):
        logfoldchanges_array_list.append(logfoldchanges[:, i])
        logfoldchanges_list_dtype.append((test_group, np.float64))
        pvals_array_list.append(pvals[:, i])
        pvals_list_dtype.append((test_group, np.float64))
        pvals_adj_array_list.append(pvals_adj[:, i])
        pvals_adj_list_dtype.append((test_group, np.float64))
        names_array_list.append(names[:, i])
        names_list_dtype.append((test_group, object))
        gene_names_array_list.append(gene_names[:, i])
        gene_names_list_dtype.append((test_group, object))


    # Create record arrays from pvals, pvals_adj, names
    logfoldchanges_rec = np.rec.fromarrays(
        arrayList=logfoldchanges_array_list,
        dtype=logfoldchanges_list_dtype
    )
    pvals_rec = np.rec.fromarrays(
        arrayList=pvals_array_list,
        dtype=pvals_list_dtype
    )
    pvals_adj_rec = np.rec.fromarrays(
        arrayList=pvals_adj_array_list,
        dtype=pvals_adj_list_dtype
    )
    names_rec = np.rec.fromarrays(
        arrayList=names_array_list,
        dtype=names_list_dtype
    )
    gene_names_rec = np.rec.fromarrays(
        arrayList=gene_names_array_list,
        dtype=gene_names_list_dtype
    )

    # Create dictionary
    rank_introns_groups_dict = {
        "params": {"groupby": groupby, "method": "dm_glm_likelihood_ratio"},
        "names": names_rec,
        "gene_names": gene_names_rec,
        "logfoldchanges": logfoldchanges_rec,
        "pvals": pvals_rec,
        "pvals_adj": pvals_adj_rec,
    }

    adata.uns["rank_intron_groups_groups"] = rank_introns_groups_dict

def rank_intron_groups_groups_df(adata: AnnData, group: str) -> pd.DataFrame:
    r"""
    Given an AnnData object and a group name, return a DataFrame containing the ranked intron groups for the specified group.

    :param adata: AnnData object containing the data
    :param group: Group name for which to retrieve the ranked intron groups
    :return: DataFrame containing the ranked intron groups for the specified group
    """

    # Assert if adata contains "rank_introns_groups" in .uns
    assert "rank_intron_groups_groups" in adata.uns, "adata does not contain 'rank_intron_groups_groups' in .uns"

    names_group = adata.uns["rank_intron_groups_groups"]["names"][group]
    gene_names_group = adata.uns["rank_intron_groups_groups"]["gene_names"][group]
    logfoldchanges_group = adata.uns["rank_intron_groups_groups"]["logfoldchanges"][group]
    pvals_group = adata.uns["rank_intron_groups_groups"]["pvals"][group]
    pvals_adj_group = adata.uns["rank_intron_groups_groups"]["pvals_adj"][group]

    return pd.DataFrame(
        {
            "names": names_group,
            "gene_names": gene_names_group,
            "logfoldchanges": logfoldchanges_group,
            "pvals": pvals_group,
            "pvals_adj": pvals_adj_group
        }
    )

def sort_dataframe_by_rank(df_to_sort: pd.DataFrame, sort_column: str, ordering_series: pd.Series) -> pd.DataFrame:
    """
    Sorts a DataFrame based on the explicit order of values in a separate Series.
    """
    # Create the Categorical type based on the external ordering
    df_to_sort[sort_column] = pd.Categorical(
        df_to_sort[sort_column],
        categories=ordering_series.unique(), # Use unique values from the ordering series
        ordered=True
    )

    # Sort the DataFrame
    return df_to_sort.sort_values(by=sort_column, ignore_index=True)

def rank_introns_groups(adata: AnnData, diff_spl_introns: pd.DataFrame, groupby: str, sortby: str, groups_test: List[str]) -> None:
    r"""
    Given an AnnData object and a DataFrame containing differential splicing results for introns, store the logfold
    changes, delta PSI, and names of the introns in the `uns` attribute of the AnnData object as dictionary
    "rank_introns_groups".

    :param adata: AnnData object containing the data
    :param diff_spl_introns: DataFrame containing differential splicing results for introns
    :param groupby: str specifying the groupby variable used in the differential splicing analysis (e.g. leiden_res_0.3)
    :param sortby: str specifying the column name by which to sort the introns in the resulting DataFrame (e.g. "intron_group", "logfoldchange", "delta_psi")
    :param groups_test: List of group names for which to rank the introns ["0", "1", "2", ...]
    """
    # Select those introns that are included (delta_psi > 0.05) as marker introns
    potential_marker_introns = diff_spl_introns[diff_spl_introns["delta_psi"] >= 0.05]

    _, counts = np.unique(potential_marker_introns["test_group"].to_numpy(), return_counts=True)

    max_num_introns = np.max(counts)

    logfoldchanges_array_list = []
    logfoldchanges_list_dtype = []
    delta_psi_array_list = []
    delta_psi_list_dtype = []
    names_array_list = []
    names_list_dtype = []
    gene_names_array_list = []
    gene_names_list_dtype = []

    for i, group_id in enumerate(groups_test):

        # Select the differentially spliced groups for test group (group_id)
        rank_sig_intron_groups_groups_df = rank_intron_groups_groups_df(adata, group_id)
        sig_intron_groups_gene_names = rank_sig_intron_groups_groups_df["gene_names"].to_numpy()

        potential_marker_introns_group_id = potential_marker_introns[potential_marker_introns["test_group"] == group_id]
        potential_marker_introns_genes_group_id = potential_marker_introns_group_id["gene_name"].to_numpy()

        # Keep only those introns in potential_marker_introns_group_id that belong to genes in sig_intron_groups_gene_names
        potential_marker_introns_group_id = potential_marker_introns_group_id[np.isin(potential_marker_introns_genes_group_id, sig_intron_groups_gene_names)]

        marker_introns_ig_df = sort_dataframe_by_rank(potential_marker_introns_group_id, "intron_group",
                                                      rank_sig_intron_groups_groups_df["names"])

        if sortby == "intron_group":
            pass
        elif sortby == "logfoldchange":
            marker_introns_ig_df = marker_introns_ig_df.sort_values(by="abs_lfc_psi", ascending=False, ignore_index=True)
        elif sortby == "delta_psi":
            marker_introns_ig_df = marker_introns_ig_df.sort_values(by="abs_delta_psi", ascending=False, ignore_index=True)
        else:
            raise ValueError(f"Invalid sortby value: {sortby}. Must be one of 'intron_group', 'logfoldchange', or 'delta_psi'.")

        # Extract values for rec array
        logfoldchanges_group = marker_introns_ig_df["lfc_psi"].to_numpy()
        delta_psi_group = marker_introns_ig_df["delta_psi"].to_numpy()
        names_group = marker_introns_ig_df["name"].to_numpy()
        gene_names_group = marker_introns_ig_df["gene_name"].to_numpy()

        # Pad array with zeros to a length of max_num_introns
        if len(logfoldchanges_group) < max_num_introns:
            diff = max_num_introns - len(logfoldchanges_group)
            logfoldchanges_group = np.pad(logfoldchanges_group, (0, diff), mode="constant")

        if len(delta_psi_group) < max_num_introns:
            diff = max_num_introns - len(delta_psi_group)
            delta_psi_group = np.pad(delta_psi_group, (0, diff), mode="constant")

        if len(names_group) < max_num_introns:
            diff = max_num_introns - len(names_group)
            names_group = np.pad(names_group, (0, diff), mode="constant")

        if len(gene_names_group) < max_num_introns:
            diff = max_num_introns - len(gene_names_group)
            gene_names_group = np.pad(gene_names_group, (0, diff), mode="constant")

        names_group = names_group.astype(str)
        gene_names_group = gene_names_group.astype(str)

        # Append to lists of arrays or dtyps
        logfoldchanges_array_list.append(logfoldchanges_group)
        logfoldchanges_list_dtype.append((group_id, np.float64))
        delta_psi_array_list.append(delta_psi_group)
        delta_psi_list_dtype.append((group_id, np.float64))
        names_array_list.append(names_group)
        names_list_dtype.append((group_id, object))
        gene_names_array_list.append(gene_names_group)
        gene_names_list_dtype.append((group_id, object))

    # Create rec arrays
    logfoldchanges_rec = np.rec.fromarrays(
        arrayList=logfoldchanges_array_list,
        dtype=logfoldchanges_list_dtype
    )
    delta_psi_rec = np.rec.fromarrays(
        arrayList=delta_psi_array_list,
        dtype=delta_psi_list_dtype
    )
    names_rec = np.rec.fromarrays(
        arrayList=names_array_list,
        dtype=names_list_dtype
    )
    gene_names_rec = np.rec.fromarrays(
        arrayList=gene_names_array_list,
        dtype=gene_names_list_dtype
    )

    # Create dictionary
    rank_introns_groups_dict = {
        "params": {"groupby": groupby, "method": "dm_glm_likelihood_ratio"},
        "names": names_rec,
        "gene_names": gene_names_rec,
        "logfoldchanges": logfoldchanges_rec,
        "delta_psi": delta_psi_rec
    }

    adata.uns["rank_introns_groups"] = rank_introns_groups_dict


def rank_introns_groups_df(adata: AnnData, group: str) -> pd.DataFrame:
    r"""
    Given an AnnData object and a group name, return a DataFrame containing the ranked introns for the specified group.

    :param adata: AnnData object containing the data
    :param group: Group name for which to retrieve the ranked intron groups
    :return: DataFrame containing the ranked intron groups for the specified group
    """

    # Assert if adata contains "rank_introns_groups" in .uns
    assert "rank_introns_groups" in adata.uns, "adata does not contain 'rank_intron_groups_groups' in .uns"

    names_group = adata.uns["rank_introns_groups"]["names"][group]
    gene_names_group = adata.uns["rank_introns_groups"]["gene_names"][group]
    logfoldchanges_group = adata.uns["rank_introns_groups"]["logfoldchanges"][group]
    delta_psi_group = adata.uns["rank_introns_groups"]["delta_psi"][group]


    return pd.DataFrame(
        {
            "names": names_group,
            "gene_names": gene_names_group,
            "logfoldchanges": logfoldchanges_group,
            "delta_psi": delta_psi_group
        }
    )

def determine_zanidm_cases(
        adata: AnnData,
        **kwargs
) -> Dict[str, int]:
    r"""
    Given an AnnData object containing pre-processed transcript usage data, determine the number of each case of the
    ZANIDM observation model present. For the observed intron counts `:math: \mathbf{x} \in \mathbb{N}^d` of splicing
    event with :math: d` components, the scenarios are defined as follows:

    Case 1: counts of each intron in intron group are greater than zero i.e. `:math: \forall j x_j > 0, N > 0`
    Case 2: all counts of each intron in intron group are zero i.e. `:math: \forall j x_j = 0, N = 0`
    Case 3: d - 1 introns in intron group are zero and number of trials N > 0
    Case 4: d - 2 introns are zero and number of trials N > 0

    The number of occurences of each case is returned.

    :param adata: AnnData object containing transcript usage data
    :return: Dictionary containing the number of occurences of each case
    :rtype: Dict[str, int]
    """

    # Determine that layer "counts" is present
    assert "counts" in adata.layers, "Layer 'counts' is not present in the AnnData object. Please ensure that the pre-processed transcript usage data contains a layer named 'counts'."

    cell_type_key = kwargs.get("cell_type_key")
    cell_type_groups_key = kwargs.get("cell_type_groups_key")

    # Intron groups definitions
    intron_groups_by_name = adata.var["intron_group"].values
    intron_groups_array = intron_names_2_integers(intron_groups_by_name)
    unique_intron_groups, intron_group_idx_start_array = np.unique(intron_groups_array, return_index=True)
    num_intron_groups = len(unique_intron_groups)

    # Create a dataset and data loader
    cell_intron_counts = torch.from_numpy(adata.layers["counts"].toarray())
    cell_intron_levels = torch.from_numpy(adata.X.toarray())
    cell_types, cell_counts = np.unique(adata.obs[cell_type_key].to_numpy(), return_counts=True)

    transform_intron_counts = None  # Previously "log" but incorrect
    transform_intron_levels = None
    transform_ontology = "one-hot"

    dataset = TranscriptUsageDataset(
        cell_intron_counts=cell_intron_counts,
        cell_intron_levels=cell_intron_levels,
        ontology=adata.obs[cell_type_key].to_numpy(),
        tissue=adata.obs[cell_type_groups_key].to_numpy(),
        cell_types=cell_types,
        transform_intron_counts=transform_intron_counts,
        transform_intron_levels=transform_intron_levels,
        transform_ontology=transform_ontology
    )

    print(f"The dataset is set up for evaluation.")

    # Tensors for handling of intron groups
    intron_group_idx_nostart = torch.tensor(np.delete(np.arange(dataset.num_introns), intron_group_idx_start_array),
                                                 dtype=torch.long)
    intron_group_idx_start = torch.tensor(intron_group_idx_start_array, dtype=torch.long)
    intron_groups = torch.tensor(intron_groups_array, dtype=torch.long)

    intron_group_indices = torch.stack((intron_groups, torch.arange(0, dataset.num_introns)), dim=0)

    intron_group_summation = torch.sparse_coo_tensor(
        intron_group_indices,
        torch.ones(dataset.num_introns, dtype=int),
        torch.Size([num_intron_groups, dataset.num_introns])
    ).to(torch.float)

    # Create an unshuffled dataloader from the data provided in dataset
    dataloader = DataLoader(dataset, 256, shuffle=False)

    num_cases_dict = {
        "case_1": 0,
        "case_2": 0,
        "case_3": 0,
        "case_4": 0
    }
    # Loop through the dataset and determine the number of occurences of each case
    for i, batch in enumerate(dataloader):
        input_batch_counts, _, _, _ = batch

        num_trials_batch = torch.sparse.mm(intron_group_summation, input_batch_counts.T).T

        # Case 1: all counts in intron group are greater than zero i.e. for all j: x_j > 0 and N > 0
        mask_non_zero_trials = (num_trials_batch > 0)
        group_indices = intron_group_indices[0].unsqueeze(0).expand(input_batch_counts.shape[0], -1)

        input_batch_counts_is_zero = (input_batch_counts == 0).float()
        num_zeros_group = torch.zeros(input_batch_counts.shape[0], num_trials_batch.shape[-1], dtype=input_batch_counts.dtype)
        num_zeros_group.scatter_add_(dim=1, index=group_indices.long(), src=input_batch_counts_is_zero)
        mask_min_count_positive = (num_zeros_group == 0)
        mask_case_1 = mask_non_zero_trials & mask_min_count_positive

        num_case_non_zeros = mask_case_1.sum()

        # Case 2: all counts in intron group are zero i.e. x = 0 and N = 0
        mask_all_zero = (num_trials_batch == 0)  # shape (batch size, num intron groups)

        num_case_zeros = mask_all_zero.sum()

        # Case 3: d - 1 categories are zero and N > 0
        mask_d_1 = torch.full((input_batch_counts.shape[0], num_trials_batch.shape[-1]), fill_value=False, dtype=torch.bool)

        for intron_group in range(0, num_trials_batch.shape[-1]):
            ig_mask = (intron_group_indices[0] == intron_group)
            intron_indices_mask = intron_group_indices[1, ig_mask]
            ig_counts = input_batch_counts[:, intron_indices_mask]
            num_trials_ig = num_trials_batch[:, intron_group]
            mask_d_1[:, intron_group] = (num_trials_ig > 0) & (
                        (ig_counts == 0).sum(dim=-1) == (ig_counts.shape[-1] - 1))

        num_case_d_1 = mask_d_1.sum()

        # Case 4: d - 2 categories are zero and N > 0
        mask_d_2 = torch.full((input_batch_counts.shape[0], num_trials_batch.shape[-1]), fill_value=False, dtype=torch.bool)

        for intron_group in range(0, num_trials_batch.shape[-1]):
            ig_mask = (intron_group_indices[0] == intron_group)
            intron_indices_mask = intron_group_indices[1, ig_mask]
            ig_counts = input_batch_counts[:, intron_indices_mask]
            num_trials_ig = num_trials_batch[:, intron_group]
            cond_1 = (num_trials_ig > 0)
            cond_2 = 1 <= (ig_counts == 0).sum(dim=-1)
            cond_3 = (ig_counts == 0).sum(dim=-1) <= (ig_counts.shape[-1] - 2)

            mask_d_2[:, intron_group] = cond_1 & cond_2 & cond_3

        num_case_d_2 = mask_d_2.sum()

        num_cases_all = num_case_non_zeros + num_case_zeros + num_case_d_1 + num_case_d_2

        assert num_cases_all == len(
            num_trials_batch.flatten()), "Number of cases do not sum up to total number of data points times intron groups"

        num_cases_dict["case_1"] += num_case_non_zeros.item()
        num_cases_dict["case_2"] += num_case_zeros.item()
        num_cases_dict["case_3"] += num_case_d_1.item()
        num_cases_dict["case_4"] += num_case_d_2.item()

    print(
        f"Case all nonzero: The number of case x_j > 0 and N > 0 is {num_cases_dict['case_1']}. \n"
        f"Case all zero: The number of case x = 0 and N = 0 is {num_cases_dict['case_2']}. \n"
        f"Case one nonzero: The number of case d - 1 categories are zero and N > 0 is {num_cases_dict['case_3']}. \n"
        f"Case subsets are zero: The number of case d - 2 categories are zero and N > 0 is {num_cases_dict['case_4']}."
    )

    return num_cases_dict

def distance_matrix(
        adata: AnnData,
        embedding_key: str,
        cluster_key: str,
        average_embeddings: bool = False,
        distance_metric: str = "euclidean",
        **kwargs
) -> Tuple[np.ndarray,np.ndarray]:
    r"""
    Given an AnnData object where cell embeddings have been inferred and added to .obsm[embedding_key], compute the
    distance matrix between the cell embeddings. The cells shall be ordered according to the cluster_key (e.g. cell
    type, Leiden cluster). If the number of cells is too large, it is recommended to average the embeddings and
    calculate the distance matrix per cluster. The distance metric can be chosen but at the moment only supports
    Euclidean distance. An np.ndarray containing the distance matrix and an np.ndarray containing the order of the cells
    according to cluster_key are returned.

    :param adata: AnnData object
    :param embedding_key: str
    :param cluster_key: str
    :param average_embeddings: bool
    :param distance_metric: str
    :param kwargs: dict
    :return:
    """
    # Check if embedding_key is present in .obsm
    assert embedding_key in adata.obsm, f"Embedding key '{embedding_key}' not found in adata.obsm. Please ensure that the specified embedding key is present in the AnnData object and contains the cell embeddings."
    # Check if cluster_key is present in .obs
    assert cluster_key in adata.obs, f"Cluster key '{cluster_key}' not found in adata.obs. Please ensure that the specified cluster key is present in the AnnData object and contains the cluster annotations for the cells."
    # Check if distance_metric is supported
    assert distance_metric in ["euclidean"], f"Distance metric '{distance_metric}' is not supported. Currently, only 'euclidean' distance is supported. Please choose a supported distance metric."

    # Get the cell embeddings and cluster annotations
    if average_embeddings:
        print("Averaging embeddings")
        cluster_adata = sc.get.aggregate(adata, by=cluster_key, func='mean', obsm=embedding_key)
        cell_embeddings = cluster_adata.layers["mean"]
        cluster_annotations = cluster_adata.obs[cluster_key].to_numpy()
    else:
        """ OLD
        unordered_cell_embeddings = adata.obsm[embedding_key]
        unordered_cluster_annotations = adata.obs[cluster_key].to_numpy()

        unique_cluster_annotations = np.sort(np.unique(unordered_cluster_annotations))
        num_cluster_annotations = len(unique_cluster_annotations)

        cell_embeddings = []
        cluster_annotations = []

        for cluster_annotation in unique_cluster_annotations:
            indices = np.where(unordered_cluster_annotations == cluster_annotation)
            cell_embeddings.append(unordered_cell_embeddings[indices])
            cluster_annotations.extend([unordered_cluster_annotations] * len(indices))

        cell_embeddings = np.concatenate(cell_embeddings, axis=0)
        cluster_annotations = np.array(cluster_annotations)
        """

        unordered_cell_embeddings = adata.obsm[embedding_key]
        unordered_cluster_annotations = adata.obs[cluster_key].to_numpy()

        sort_order = np.argsort(unordered_cluster_annotations)

        cell_embeddings = unordered_cell_embeddings[sort_order]
        cluster_annotations = unordered_cluster_annotations[sort_order]

    # Calculate distance matrix using pdist
    condensed_distance_matrix = pdist(cell_embeddings, metric=distance_metric)
    emb_distance_matrix = squareform(condensed_distance_matrix)

    return emb_distance_matrix, cluster_annotations

def calculate_distance_matrices_across_seeds(
    adata: AnnData,
    seeds: List[int],
    latent_mean_key: str,
    likelihood_keys: List[str],
    cluster_key: str,
    average_embeddings: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray]:
    r"""
    Given an AnnData object where cell embeddings have been inferred and added to .obsm[embedding_key], compute the
    distance matrices between the cell embeddings for each seed. The cells shall be ordered according to the cluster_key
    (e.g. cell type, Leiden cluster). If the number of cells is too large, it is recommended to average the embeddings
    and calculate the distance matrix per cluster. The distance metric can be chosen but at the moment only supports
    Euclidean distance. A list of np.ndarray containing the distance matrices and an np.ndarray containing the order of
    the cells according to cluster_key are returned.

    :param adata: AnnData object
    :param seeds: List of seeds for which to calculate the distance matrices
    :param latent_mean_key: str specifying the key for the latent mean embeddings in .obsm (e.g. "latent_mean", "private_latent_mean", "shared_latent_mean")
    :param likelihood_keys: List of str specifying the keys for the likelihood embeddings (e.g. ["ZINB"], ["ZIDM"], ["ZINB", "ZIDM"])
    :param cluster_key: str specifying the key for the cluster annotations in .obs (e.g. "cell_ontology_class")
    :param average_embeddings: bool specifying whether to average embeddings before calculating distance matrix
    :return: Tuple containing a list of np.ndarray distance matrices and an np.ndarray of cluster annotations
    """
    distance_matrices = []

    for seed_selected in seeds:

        if len(likelihood_keys) == 1:
            cell_embedding_key = likelihood_keys[0] + "_" + str(seed_selected) + "_" + latent_mean_key
        elif len(likelihood_keys) == 2:
            cell_embedding_key = likelihood_keys[0] + "_" + str(seed_selected) + "_" + likelihood_keys[1] + "_"  + str(seed_selected) + "_" + latent_mean_key
        else:
            raise ValueError

        emb_distance_matrix, cluster_annotations = distance_matrix(
              adata,
              cell_embedding_key,
              cluster_key,
              average_embeddings
        )

        upper_triangular_distance_matrix = np.triu(emb_distance_matrix)

        distance_matrices.append(upper_triangular_distance_matrix)

    print(f"The distance matrices of the embeddings each of shape {emb_distance_matrix.shape} have been computed.")

    return distance_matrices, cluster_annotations

def calculate_latent_space_geometry(
    distance_matrices: List[np.ndarray],
    cluster_annotation: np.ndarray,
    lineage_mapping: Dict[str, str],
) -> Dict[str, Any]:
    r"""
    Calculate the geometry of the latent space based on the provided distance matrices and cluster annotations.

    The seed similarity matrix measures using the Spearman correlation how well the latent space geometry is preserved across
    different random seeds. It is calculated for every pair of seeds :math:`(s,t)` as the Spearman correlation

    ..math::
        R_{s,t} = \rho_{Spearman}(\mathbf{x}^{(s)}, \mathbf{x}^{(t)})

    of their upper triangular distance matrices stacked into 1D arrays :math:`\mathbf{x}^{(s)}` and
    :math:`\mathbf{x}^{(t)}`.

    The consensus distance matrix :math:`\mathbf{D}_{\text{cons}} \in \mathbb{R}^{n \times n}` assesses the distance agreement across seeds and is calculated as the median of the
    normalised distance matrices across seeds :math:`s`

    ..math::
        D_{ij}^{cons} = median_s \widetilde{D}_{ij}^{(s)}

    where :math:`\widetilde{D}_{ij}^{(s)}` is the normalised distance matrix for seed :math:`s` and cell types :math:`i`
    and :math:`j`. It is normalised by the median of the off-diagonal distances for each seed :math:`s` to account for
    differences in scale across seeds. The consensus distance matrix should be evaluated in conjunction with the
    interquartile range matrix :math:`\mathbf{U} \in \mathbb{R}^{n \times n}` which is calculated for each pair
    :math:`(i,j)` as the interquartile range of the normalised distances across seeds :math:`s` as

    ..math::
        U_{ij} = Q_{0.75}(\widetilde{D}_{ij}^{(0)}, \dots, \widetilde{D}_{ij}^{(S)}) -  Q_{0.25}(\widetilde{D}_{ij}^{(0)}, \dots, \widetilde{D}_{ij}^{(S)})

    Read jointly for a pair of cell types :math:`(i,j)`, the consensus distance matrix and the interquartile range
    matrix indicate: large distance, small iqr means reproducibly separated; small distance, small iqr means
    reproducibly close; large distance, large iqr means inconsistent separation dependent on the seed.

    :param distance_matrices: List of np.ndarray distance matrices for each seed
    :param cluster_annotation: np.ndarray of cluster annotations for the cells
    :param lineage_mapping: Dict mapping cluster annotations to their respective lineages
    :return: Dict containing the latent space geometry metrics

    """
    # Calculate raw fingerprint distances
    num_seeds = len(distance_matrices)

    num_cell_types = distance_matrices[0].shape[0]
    num_elements_upper_triangle = int(num_cell_types * (num_cell_types - 1) / 2)

    raw_fingerprint_distances = np.zeros((num_seeds, num_elements_upper_triangle))

    # Get upper triangle index coordinates once (k=1 excludes diagonal)
    triu_indices = np.triu_indices(num_cell_types, k=1)

    for i, distance_matrix in enumerate(distance_matrices):
        raw_fingerprint_distances[i] = distance_matrix[triu_indices]

    # Calculate median across pairs for each seed (axis=1)
    median_off_diagonal_distances = np.median(raw_fingerprint_distances, axis=1)

    # Normalised distance matrices (by median off diagonal distances per seed)
    normalised_distance_matrices = []
    for i, distance_matrix in enumerate(distance_matrices):
        normalised_distance_matrix = distance_matrix / median_off_diagonal_distances[i]
        normalised_distance_matrices.append(normalised_distance_matrix)

    # Normalised fingerprint distances
    normalised_fingerprint_distances = np.zeros((num_seeds, num_elements_upper_triangle))

    for i, normalised_distance_matrix in enumerate(normalised_distance_matrices):
        normalised_fingerprint_distances[i] = normalised_distance_matrix[triu_indices]

    # Seed similarity matrix
    seed_similarity_matrix = np.ones((num_seeds, num_seeds)) # Pre-fill with 1s for the diagonal

    # 1. Optimize the loop: calculate only the upper triangle to halve computation time
    for i in range(num_seeds):
        for j in range(i + 1, num_seeds):
            corr = spearmanr(distance_matrices[i].flatten(), distance_matrices[j].flatten())[0]
            seed_similarity_matrix[i, j] = corr
            seed_similarity_matrix[j, i] = corr # Mirror to lower half for the heatmap

    # 2. Extract ONLY the off-diagonal upper triangular values into a 1D array
    # k=1 ensures we skip the diagonal (self-correlations of 1.0)
    pairwise_correlations = seed_similarity_matrix[np.triu_indices_from(seed_similarity_matrix, k=1)]

    # 3. Calculate statistics on just the valid pairwise correlations
    median_pairwise_seed_similarity = np.median(pairwise_correlations)
    lower_pairwise_seed_similarity = np.min(pairwise_correlations)
    upper_pairwise_seed_similarity = np.max(pairwise_correlations)

    # Stacked distance matrices (full not upper triangular)
    distance_matrices = [U + U.T - np.diag(np.diag(U)) for U in distance_matrices]
    distance_matrices_tensor = np.stack(distance_matrices, axis=0) # (num seeds x num cell types x num cell types)

    # Stacked normalised distance matrices (full not upper triangular)
    normalised_distance_matrices_full = [U + U.T - np.diag(np.diag(U)) for U in normalised_distance_matrices]
    normalised_distance_matrices_tensor = np.stack(normalised_distance_matrices_full, axis=0) # (num seeds x num cell types x num cell types)

    # Reorder stacked normalised distance matrices according to cell lineage
    # By default the order of the lineage mapping is preserved
    annotation_order = list(lineage_mapping.keys())
    cluster_map = {name: i for i, name in enumerate(cluster_annotation)}
    reorder_idx = [cluster_map[cell_type] for cell_type in annotation_order]

    reordered_normalised_distance_matrices = normalised_distance_matrices_tensor[:, reorder_idx, :][:, :, reorder_idx]
    reordered_distance_matrices = distance_matrices_tensor[:, reorder_idx, :][:, :, reorder_idx]

    # Reorder cluster annotations according to cell lineage
    cluster_annotations_reordered = np.array(annotation_order)

    # Median across seed dimension

    consensus_distance_matrix = np.median(
        reordered_normalised_distance_matrices,
        axis=0,
    )

    # Interquartile range matrix (0.25 -- 0.75)
    interquartile_range_matrix = iqr(reordered_normalised_distance_matrices, axis=0, rng=(25, 75))

    latent_space_geometry = {
        "raw_fingerprint_distances": raw_fingerprint_distances,
        "median_off_diagonal_distances": median_off_diagonal_distances,
        "normalised_distance_matrices": normalised_distance_matrices,
        "normalised_fingerprint_distances": normalised_fingerprint_distances,
        "seed_similarity_matrix": seed_similarity_matrix,
        "pairwise_correlations": pairwise_correlations,
        "median_pairwise_seed_similarity": median_pairwise_seed_similarity,
        "lower_pairwise_seed_similarity": lower_pairwise_seed_similarity,
        "upper_pairwise_seed_similarity": upper_pairwise_seed_similarity,
        "distance_matrices_tensor": distance_matrices_tensor,
        "normalised_distance_matrices_tensor": normalised_distance_matrices_tensor,
        "reordered_normalised_distance_matrices": reordered_normalised_distance_matrices,
        "reordered_distance_matrices": reordered_distance_matrices,
        "consensus_distance_matrix": consensus_distance_matrix,
        "interquartile_range_matrix": interquartile_range_matrix,
        "cluster_annotations_reordered": cluster_annotations_reordered,
        "lineage_order": annotation_order,
    }

    return latent_space_geometry

def construct_undirected_knn_graph(
    distance_matrix_for_graph: np.ndarray,
    k: int=5,
    mode: str="mutual",
    cell_type_labels: np.ndarray=None,
    weighted: bool=False
) -> Tuple[np.ndarray, nx.Graph]:
    r"""
    Constructs an undirected k-NN adjacency matrix and NetworkX Graph from a distance matrix.

    :param distance_matrix_for_graph: Square symmetric distance matrix (NumPy array)
    :param k: Number of nearest neighbors
    :param mode: 'union' (OR rule: edge exists if either is in top-k) or 'mutual' (AND rule: edge exists only if both are in each other's top-k)
    :param cell_type_labels: Optional list of node labels
    :param weighted: If True, edge weights store the actual distances
    :return: Tuple containing the undirected adjacency matrix and the NetworkX Graph
    """

    D = (
        distance_matrix_for_graph.values
        if hasattr(distance_matrix_for_graph, "values")
        else np.asarray(distance_matrix_for_graph)
    )
    N = D.shape[0]

    # 1. Identify top k neighbors for each cell (skip index 0, which is self-distance)
    nn_indices = np.argsort(D, axis=1)[:, 1 : k + 1]

    # 2. Build directed adjacency matrix
    row_indices = np.arange(N)[:, None]
    adj_directed = np.zeros((N, N), dtype=float)

    if weighted:
        adj_directed[row_indices, nn_indices] = D[row_indices, nn_indices]
    else:
        adj_directed[row_indices, nn_indices] = 1.0

    # 3. Symmetrize matrix to make the graph undirected
    if mode == "union":
        adj_undirected = np.maximum(adj_directed, adj_directed.T)
    elif mode == "mutual":
        adj_undirected = np.minimum(adj_directed, adj_directed.T)
    else:
        raise ValueError("mode must be either 'union' or 'mutual'")

    # 4. Create NetworkX Undirected Graph (nx.Graph)
    G = nx.from_numpy_array(adj_undirected, create_using=nx.Graph)

    if cell_type_labels is not None:
        label_mapping = {i: label for i, label in enumerate(cell_type_labels)}
        G = nx.relabel_nodes(G, label_mapping)

    return adj_undirected, G


# Example Usage:
# adj_mat, G = construct_undirected_knn_graph(
#     consensus_distance_matrix,
#     k=5,
#     mode='union', # or 'mutual'
#     cell_type_labels=cell_lineage_order
# )

def compute_neighbour_retention(
    distance_matrices_stacked: np.ndarray,
    k: int=5,
    cell_labels: np.ndarray=None
) -> pd.DataFrame:
    r"""
    Calculates the cell-type specific reference-free neighbor retention score :math:`R_i^(k)`. The score asses if a
    cell type  math:`i` retains its math:`k` nearest neighbors across different random seeds. For cell type  math:`i` of
    all  math:`N`cell types, compare it k-neighbor sets across all

    ..math::
        Rn_{sets} =
            \left(
                \begin{matrix}
                    S \\
                    2
                \end{matrix}
            \right)

    pairs of seeds. For seeds  math:`(s,t), compute the intersection of the k-nearest neighbor sets`

    ..math::
        O_i^{(s, t, k)} = \frac{|N_i^{(s, k)} \cap N_i^{(t, k)}|}{k}.

    The retention score is then the average of the intersection over all seed pairs given by

    ..math::
            R_i^{(k)} = \frac{1}{n_{sets}} \sum_{s < t} O_i^{(s, t, k)}

    For ten seeds, if the retention score is 0.8, it means that on average 80% of the k-nearest neighbors of cell type i
    are retained across all seed pairs. As a baseline, the expected intersection of two random k-nearest neighbor sets
    is given by

    ..math::
        \mathbb{E}[O_i] = \frac{k}{N-1}

    indicating chance level retention. The retention score is expected to be above the chance level if the local
    neighbors are preserved across seeds.

    :param distance_matrices_stacked: np.ndarray of shape (S, N, N) across S seeds.
    :param k: Number of nearest neighbors.
    :param cell_labels: Optional list of N cell type names.
    :return: pandas DataFrame with cell type scores and chance baseline.
    """
    S, N, _ = distance_matrices_stacked.shape

    if S < 2:
        raise ValueError(
            "At least S >= 2 seeds are required to compare seed pairs."
        )

    # 1. Identify top k neighbors per seed and cell type (excluding self-distance at index 0)
    knn_indices = np.argsort(distance_matrices_stacked, axis=2)[
        :, :, 1 : k + 1
    ]

    # 2. Construct boolean indicator tensor of shape (S, N, N) using put_along_axis
    is_neighbor = np.zeros((S, N, N), dtype=bool)
    np.put_along_axis(is_neighbor, knn_indices, True, axis=2)

    # 3. Vectorized intersection count via batch matrix multiplication: (N, S, N) @ (N, N, S) -> (N, S, S)
    is_neighbor_by_cell = np.transpose(is_neighbor, (1, 0, 2))
    overlap_matrices = np.matmul(
        is_neighbor_by_cell, np.transpose(is_neighbor_by_cell, (0, 2, 1))
    )

    # 4. Extract pair intersections s < t
    triu_s, triu_t = np.triu_indices(S, k=1)
    pairwise_overlaps = overlap_matrices[
        :, triu_s, triu_t
    ]  # Shape: (N, n_sets)

    # 5. Compute retention score R_i^(k) by averaging over seed pairs and normalizing by k
    R = np.mean(pairwise_overlaps, axis=1) / k

    # 6. Chance expectation baseline E[O_i]
    expected_baseline = k / (N - 1)

    # 7. Return formatted DataFrame
    if cell_labels is None:
        cell_labels = [f"Cell_Type_{i}" for i in range(N)]

    df = pd.DataFrame(
        {
            "cell_type": cell_labels,
            f"retention_score_k{k}": R,
            "expected_baseline": expected_baseline,
            "above_baseline": R > expected_baseline,
        }
    )

    #return df.sort_values(by=f"retention_score_k{k}", ascending=False).reset_index(drop=True)
    return df

def calculate_adjacency_and_edge_support_of_knn_graph(
    reordered_distance_matrices: np.ndarray,
    k: int = 5,
    mode: str = "mutual",
    cell_type_labels: np.ndarray = None,
    weighted: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Computes k-NN binary adjacency matrices for each seed and consensus edge support matrix across seeds.

    :param reordered_distance_matrices: np.ndarray of shape (S, N, N) across S seeds.
    :param k: int, number of nearest neighbors (excludes diagonal self-distance).
    :param mode: str, 'mutual' (AND condition), 'union' (OR condition), or 'directed'.
    :param cell_type_labels: list or array of N cell type names (optional).
    :param weighted: bool, whether to weight individual adjacency edges by inverse distance.
    :return: Tuple containing the adjacency matrices and edge support matrix.
    """

    S, N, _ = reordered_distance_matrices.shape

    # 1. Identify top k nearest neighbors per seed (indices 1 to k+1 exclude self at index 0)
    knn_indices = np.argsort(reordered_distance_matrices, axis=2)[
        :, :, 1 : k + 1
    ]

    # 2. Build 3D directed indicator tensor (S x N x N)
    adj_matrices = np.zeros((S, N, N), dtype=float)
    np.put_along_axis(adj_matrices, knn_indices, 1.0, axis=2)

    # 3. Enforce graph symmetry mode
    if mode == "mutual":
        adj_matrices = adj_matrices * np.transpose(adj_matrices, (0, 2, 1))
    elif mode in ["union", "undirected"]:
        adj_matrices = np.maximum(
            adj_matrices, np.transpose(adj_matrices, (0, 2, 1))
        )
    elif mode == "directed":
        pass
    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Choose 'mutual', 'union', or 'directed'."
        )

    # 4. Optional distance weighting
    if weighted:
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dist = np.where(
                reordered_distance_matrices > 0,
                1.0 / reordered_distance_matrices,
                0.0,
            )
        adj_matrices = adj_matrices * inv_dist

    # 5. Compute mean edge persistence across the seed dimension (S)
    binary_presence = (adj_matrices > 0).astype(float) if weighted else adj_matrices
    edge_support = np.mean(binary_presence, axis=0)

    # Ensure zero diagonal
    np.fill_diagonal(edge_support, 0.0)

    return adj_matrices, edge_support


def split_adata_dataset(
        adata: Tuple[AnnData, AnnData],
        dataset_ratio: np.ndarray = np.array([0.8, 0.1, 0.1])) -> Tuple[Tuple[AnnData, AnnData], Tuple[AnnData, AnnData], Tuple[AnnData, AnnData]]:
    r"""
    Given a tuple of two AnnData objects where the first contains gene expression data and the second transcript usage
    data, randomly split the data into training, validation, and test datasets. The ratio of the split is defined by
    dataset_ratio. The function returns a tuple of three tuples, each containing the training, validation, and test
    datasets for both gene expression and transcript usage data.

    :param adata: Tuple of two AnnData objects (gene expression data, transcript usage data)
    :param dataset_ratio: Ratio for splitting the data into training, validation, and test datasets
    :return: Tuple of three tuples, each containing the training, validation, and test datasets for both gene expression
             and transcript usage data
    :rtype: Tuple[Tuple[AnnData, AnnData], Tuple[AnnData, AnnData], Tuple[AnnData, AnnData]]
    Example:
    >>> adata1 = AnnData(X1, var=var1)
    >>> adata2 = AnnData(X2, var=var2)
    >>> adata = (adata1, adata2)
    >>> dataset_ratio = np.array([0.8, 0.1, 0.1])
    >>> train_data, val_data, test_data = split_adata_dataset(adata, dataset_ratio)
    >>> print(train_data)
    >>> # Output:
    >>> # (AnnData object with n_obs × n_vars = 800 × 1000, AnnData object with n_obs × n_vars = 800 × 1000)
    >>> print(val_data)
    >>> # Output:
    >>> # (AnnData object with n_obs × n_vars = 100 × 1000, AnnData object with n_obs × n_vars = 100 × 1000)
    >>> print(test_data)
    >>> # Output:
    >>> # (AnnData object with n_obs × n_vars = 100 × 1000, AnnData object with n_obs × n_vars = 100 × 1000)
    """
    # Check if the dataset ratio is valid
    if np.sum(dataset_ratio) != 1:
        raise ValueError("Dataset ratio must sum to 1.")
    if len(dataset_ratio) != 3:
        raise ValueError("Dataset ratio must be a 3-element array.")
    if np.any(dataset_ratio < 0):
        raise ValueError("Dataset ratio must be non-negative.")
    if np.any(dataset_ratio > 1):
        raise ValueError("Dataset ratio must be less than or equal to 1.")
    if len(adata) != 2:
        raise ValueError("Input must be a tuple of two AnnData objects.")
    if not isinstance(adata[0], AnnData) or not isinstance(adata[1], AnnData):
        raise ValueError("Input must be a tuple of two AnnData objects.")
    if adata[0].shape[0] != adata[1].shape[0]:
        raise ValueError("Number of cells in gene expression and transcript usage data must be the same.")

    # Get the number of cells in the dataset
    num_cells = adata[0].shape[0]

    # Get the number of cells for each dataset
    num_train_cells = int(num_cells * dataset_ratio[0])
    num_val_cells = int(num_cells * dataset_ratio[1])
    num_test_cells = num_cells - num_train_cells - num_val_cells

    # Get the indices for each dataset
    indices = np.arange(num_cells)
    np.random.shuffle(indices)
    train_indices = indices[:num_train_cells]
    val_indices = indices[num_train_cells:num_train_cells + num_val_cells]
    test_indices = indices[num_train_cells + num_val_cells:]

    # Split the data into training, validation, and test datasets
    train_dataset = (adata[0][train_indices], adata[1][train_indices])
    val_dataset = (adata[0][val_indices], adata[1][val_indices])
    test_dataset = (adata[0][test_indices], adata[1][test_indices])

    return train_dataset, val_dataset, test_dataset

class GeneExpressionTranscriptUsageDataset(Dataset):
    r"""
    Dataset class for gene expression and transcript usage data. Each data point is a tuple where the first entry is
    a torch.Tensor representing the gene expression counts, the second entry is a torch.Tensor representing the log(1+x)
    transformed normalised gene expression counts (called gene_levels), the third entry a torch.Tensor for the intron
    counts, the fourth entry a torch.Tensor for the PSI scores, the fifth entry a numpy.ndarray for the cell ontology
    annotation, and the sixth entry a numpy.ndarray for the tissue type annotation.  The cells for each entry of the
    four input tensors must be identical.

    :param cell_gene_counts: torch.Tensor of shape (num_cells, num_genes) representing the gene expression counts
    :param cell_gene_levels: torch.Tensor of shape (num_cells, num_genes) representing the transformed gene counts
    :param cell_intron_counts: torch.Tensor of shape (num_cells, num_introns) representing the intron counts
    :param cell_intron_levels: torch.Tensor of shape (num_cells, num_intron_groups) representing the PSI scores
    :param ontology: numpy.ndarray of shape (num_cells,) representing the cell ontology annotation
    :param tissue: numpy.ndarray of shape (num_cells,) representing the tissue type annotation
    :param cell_types: Tuple[str] of cell types to be used for one-hot encoding of the ontology
    :param transform_gene_counts: str specifying the transformation to be applied to the gene expression counts
    :param transform_intron_counts: str specifying the transformation to be applied to the intron counts
    :param transform_intron_levels: str specifying the transformation to be applied to the PSI scores
    :param transform_ontology: str specifying the transformation to be applied to the ontology

    """

    def __init__(
            self,
            cell_gene_counts: torch.Tensor,
            cell_gene_levels: torch.Tensor,
            cell_intron_counts: torch.Tensor,
            cell_intron_levels: torch.Tensor,
            ontology: np.ndarray,
            tissue: np.ndarray,
            cell_types: Tuple[str] = TABULA_MURIS_CELL_TYPES,
            transform_gene_counts: str = None,
            transform_intron_counts: str = None,
            transform_intron_levels: str = None,
            transform_ontology: str = "one-hot"
    ) -> None:
        super().__init__()
        assert ((cell_gene_counts.shape[0] == cell_intron_counts.shape[0])
                and (cell_gene_counts.shape[0] == cell_intron_levels.shape[0])
                and (cell_intron_levels.shape[0] == cell_gene_levels.shape[0])), \
            "Number of cells must be the same for gene expression and transcript usage data."
        assert cell_gene_levels.shape[-1] == cell_gene_counts.shape[-1], \
            "Number of genes in gene counts and expression levels must be identical"

        self.num_cells = cell_gene_counts.shape[0]
        self.num_genes = cell_gene_counts.shape[-1]
        self.num_introns = cell_intron_counts.shape[-1]
        self.cell_gene_counts = cell_gene_counts
        self.cell_gene_levels = cell_gene_levels
        self.cell_intron_counts = cell_intron_counts
        self.cell_intron_levels = cell_intron_levels
        self.ontology = ontology
        self.tissue = tissue
        self.cell_types = cell_types
        self.transform_gene_counts = transform_gene_counts
        self.transform_intron_counts = transform_intron_counts
        self.transform_intron_levels = transform_intron_levels
        self.transform_ontology = transform_ontology

        self.class_to_int = {class_label: i for i, class_label in enumerate(cell_types)}

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(
            self,
            idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | str, str]:
        gene_counts = self.cell_gene_counts[idx]
        gene_levels = self.cell_gene_levels[idx]
        intron_counts = self.cell_intron_counts[idx]
        intron_levels = self.cell_intron_levels[idx]

        if self.transform_gene_counts == "log":
            gene_counts = torch.log(1 + gene_counts)

        if self.transform_intron_counts == "log":
            intron_counts = torch.log(1 + intron_counts)

        if self.transform_intron_levels is not None:
            raise NotImplementedError("Transform for intron levels not implemented yet.")

        if self.transform_ontology == "one-hot":
            token = self.class_to_int[self.ontology[idx]]
            tokenized_ontology = torch.as_tensor(np.array([token])).squeeze()
            cell_ontology = nn.functional.one_hot(tokenized_ontology, num_classes=len(self.cell_types))
        else:
            cell_ontology = self.ontology[idx]

        cell_tissue = self.tissue[idx]

        return gene_counts, gene_levels, intron_counts, intron_levels, cell_ontology, cell_tissue

class GeneExpressionDataset(Dataset):
    r"""
    Dataset class for gene expression data. Each data point is a tuple where the first entry is a torch.Tensor
    representing the gene expression counts, the second entry is a torch.Tensor representing the log(1+x) transformed
    normalised gene expression counts (called gene_levels), the third entry a numpy.ndarray for the cell ontology
    annotation, and the fourth entry a numpy.ndarray for the tissue type annotation.
    """

    def __init__(
            self,
            cell_gene_counts: torch.Tensor,
            cell_gene_levels: torch.Tensor,
            ontology: np.ndarray,
            tissue: np.ndarray,
            cell_types: Tuple[str] = TABULA_MURIS_CELL_TYPES,
            transform_gene_counts: str = None,
            transform_ontology: str = "one-hot"
    ) -> None:
        super().__init__()

        self.num_cells = cell_gene_counts.shape[0]
        self.num_genes = cell_gene_counts.shape[-1]
        self.cell_gene_counts = cell_gene_counts
        self.cell_gene_levels = cell_gene_levels
        self.ontology = ontology
        self.tissue = tissue
        self.cell_types = cell_types
        self.transform_gene_counts = transform_gene_counts
        self.transform_ontology = transform_ontology

        self.class_to_int = {class_label: i for i, class_label in enumerate(cell_types)}

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[Tensor, Tensor, Tensor | np.ndarray[Any, np.dtype[Any] | Any], np.ndarray[Any, np.dtype[Any] | Any]]:
        gene_counts = self.cell_gene_counts[idx]
        gene_levels = self.cell_gene_levels[idx]

        if self.transform_gene_counts == "log":
            gene_counts = torch.log(1 + gene_counts)

        if self.transform_ontology == "one-hot":
            token = self.class_to_int[self.ontology[idx]]
            tokenized_ontology = torch.as_tensor(np.array([token])).squeeze()
            cell_ontology = nn.functional.one_hot(tokenized_ontology, num_classes=len(self.cell_types))
        else:
            cell_ontology = self.ontology[idx]

        cell_tissue = self.tissue[idx]

        return gene_counts, gene_levels, cell_ontology, cell_tissue

class TranscriptUsageDataset(Dataset):
    r"""
    Dataset class for transcript usage data. Each data point is a tuple where the first entry is a torch.Tensor for the
    intron counts, the second entry a torch.Tensor intron levels (this can be log(1+x)-transformed and normalised intron
    counts, or the PSI scores), the third entry a numpy.ndarray for the cell ontology annotation, and the fourth entry
    a numpy.ndarray for the tissue type annotation.
    """

    def __init__(
            self,
            cell_intron_counts: torch.Tensor,
            cell_intron_levels: torch.Tensor, # psi scores or intron levels
            ontology: np.ndarray,
            tissue: np.ndarray,
            cell_types: Tuple[str] = TABULA_MURIS_CELL_TYPES,
            transform_intron_counts: str = "log(1 + x)",
            transform_intron_levels: str = None,
            transform_ontology: str = "one-hot"
    ) -> None:
        super().__init__()

        self.num_cells = cell_intron_counts.shape[0]
        self.num_introns = cell_intron_counts.shape[-1]
        self.cell_intron_counts = cell_intron_counts
        self.cell_intron_levels = cell_intron_levels
        self.ontology = ontology
        self.tissue = tissue
        self.cell_types = cell_types
        self.transform_intron_counts = transform_intron_counts
        self.transform_intron_levels = transform_intron_levels
        self.transform_ontology = transform_ontology

        self.class_to_int = {class_label: i for i, class_label in enumerate(cell_types)}

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(
            self,
            idx
    ) -> Tuple[Tensor, Tensor, Tensor | np.ndarray[Any, np.dtype[Any] | Any], np.ndarray[Any, np.dtype[Any] | Any]]:
        intron_counts = self.cell_intron_counts[idx]
        intron_levels = self.cell_intron_levels[idx]

        if self.transform_intron_counts == "log":
            intron_counts = torch.log(1 + intron_counts)

        if self.transform_intron_levels is not None:
            raise NotImplementedError("Transform for intron levels not implemented yet.")

        if self.transform_ontology == "one-hot":
            token = self.class_to_int[self.ontology[idx]]
            tokenized_ontology = torch.as_tensor(np.array([token])).squeeze()
            cell_ontology = nn.functional.one_hot(tokenized_ontology, num_classes=len(self.cell_types))
        else:
            cell_ontology = self.ontology[idx]

        cell_tissue = self.tissue[idx]

        return intron_counts, intron_levels, cell_ontology, cell_tissue

class MarkerGeneCellTypeDataset(Dataset):
    r"""
    Dataset class for binary classification of cell types based on marker genes.
    """
    def __init__(
            self,
            marker_genes: torch.Tensor,  # shape (num_cells, num_marker_genes)
            ontology: np.ndarray,  # shape (num_cells,)
            cell_type: str,
    ) -> None:
        super().__init__()

        self.marker_genes = marker_genes
        self.ontology = ontology
        self.num_cells = marker_genes.shape[0]
        self.num_marker_genes = marker_genes.shape[1]
        self.cell_type = cell_type

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        marker_gene_expression = self.marker_genes[idx]
        cell_ontology = self.ontology[idx]

        if self.cell_type == cell_ontology:
            cell_label = torch.tensor(1, dtype=torch.float)
        else:
            cell_label = torch.tensor(0, dtype=torch.float)

        return marker_gene_expression, cell_label

class VAEEmbeddingsCellTypeDataset(Dataset):
    def __init__(
            self,
            vae_embeddings: torch.Tensor,
            ontology: np.ndarray,
            tissue: np.ndarray,
            cell_types: Tuple[str] = TABULA_MURIS_CELL_TYPES,
            transform_inputs=None,
            transform_ontology=None,
    ) -> None:
        super().__init__()

        self.vae_embeddings = vae_embeddings
        self.ontology = ontology
        self.tissue = tissue
        self.cell_types = cell_types
        self.transform_inputs = transform_inputs
        self.transform_ontology = transform_ontology

        self.num_cells = vae_embeddings.shape[0]
        self.embedding_dim = vae_embeddings.shape[-1]

        self.class_to_int = {class_label: i for i, class_label in enumerate(self.cell_types)}

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(
            self,
            idx
    ) -> Tuple[torch.Tensor, torch.Tensor | np.ndarray[Any, np.dtype[Any] | Any], np.ndarray[Any, np.dtype[Any] | Any]]:

        vae_embedding = self.vae_embeddings[idx]
        cell_tissue = self.tissue[idx]

        if self.transform_inputs:
            raise ValueError("No input transform supported")

        token = self.class_to_int[self.ontology[idx]]
        cell_ontology = torch.as_tensor(np.array([token])).squeeze().long()

        if self.transform_ontology == "one-hot":
            cell_ontology = nn.functional.one_hot(cell_ontology, num_classes=len(self.cell_types))

        return vae_embedding, cell_ontology, cell_tissue

class EmbeddingCellTypeDataset(Dataset):
    def __init__(
            self,
            private_embeddings: Tuple[torch.Tensor, torch.Tensor], # private GE, private TU
            shared_unimodal_embeddings: Tuple[torch.Tensor, torch.Tensor], # shared uni-modal GE, shared uni-modal TU
            shared_embeddings: torch.Tensor,
            ontology: np.ndarray,
            tissue: np.ndarray,
            cell_types: Tuple[str] = TABULA_MURIS_CELL_TYPES,
            transform_inputs=None,
            transform_ontology=None,
    ) -> None:
        super().__init__()

        self.private_embeddings_1 = private_embeddings[0]
        self.private_embeddings_2 = private_embeddings[-1]
        self.shared_unimodal_embeddings_1 = shared_unimodal_embeddings[0]
        self.shared_unimodal_embeddings_2 = shared_unimodal_embeddings[1]
        self.shared_embeddings = shared_embeddings
        self.ontology = ontology
        self.tissue = tissue
        self.cell_types = cell_types
        self.transform_inputs = transform_inputs
        self.transform_ontology = transform_ontology

        self.num_cells = private_embeddings[0].shape[0]
        self.private_1_dim = private_embeddings[0].shape[-1]
        self.private_2_dim = private_embeddings[1].shape[-1]
        self.shared_unimodal_1_dim = shared_unimodal_embeddings[0].shape[-1]
        self.shared_unimodal_2_dim = shared_unimodal_embeddings[1].shape[-1]
        self.shared_dim = shared_embeddings.shape[-1]

        self.class_to_int = {class_label: i for i, class_label in enumerate(self.cell_types)}

    def __len__(self) -> int:
        return self.num_cells

    def __getitem__(
            self,
            idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | np.ndarray[Any, np.dtype[Any] | Any], np.ndarray[Any, np.dtype[Any] | Any]]:

        private_embedding_1 = self.private_embeddings_1[idx]
        private_embedding_2 = self.private_embeddings_2[idx]
        shared_unimodal_embedding_1 = self.shared_unimodal_embeddings_1[idx]
        shared_unimodal_embedding_2 = self.shared_unimodal_embeddings_2[idx]
        shared_embedding = self.shared_embeddings[idx]
        cell_tissue = self.tissue[idx]

        if self.transform_inputs:
            raise ValueError("No input transform supported")

        token = self.class_to_int[self.ontology[idx]]
        cell_ontology = torch.as_tensor(np.array([token])).squeeze().long()

        if self.transform_ontology == "one-hot":
            cell_ontology = nn.functional.one_hot(cell_ontology, num_classes=len(self.cell_types))

        return private_embedding_1, private_embedding_2, shared_unimodal_embedding_1, shared_unimodal_embedding_2, shared_embedding, cell_ontology, cell_tissue


def intron_names_2_integers(intron_groups_by_name: np.ndarray) -> np.ndarray:
    r"""
    Given np.array of intron group names, an ordered dictionary is created mapping intron group names to integers
    which is then used to return a numpy array containing the mapped intron groups as integers.

    :param intron_groups_by_name:
    :return:
    """

    unique_intron_group_names = collections.OrderedDict.fromkeys(intron_groups_by_name)
    name_to_int = {name: i for i, name in enumerate(unique_intron_group_names)}
    intron_groups = np.array([name_to_int[name] for name in intron_groups_by_name])

    return intron_groups

def evaluate_random_seeds_reconstructions(
        seeds: List[int],
        adata: AnnData,
        likelihood_type: str,
        data_partition: str = "test",
        evaluation_metric: str = "nll",
) -> pd.DataFrame:
    r"""
    Given a list of random seeds, an AnnData object containing evaluation results, a likelihood type,
    a data partition (train, val, test, or full), and an evaluation metric (nll, rmse, or mae), return a
    DataFrame containing the evaluation results for each random seed.

    :param seeds: (List[int]) List of random seeds
    :param adata: (AnnData) AnnData object containing evaluation results
    :param likelihood_type: (str) The likelihood function used in the model, e.g. "NB, "ZINB, "Gaussian", "ZIDM", "DM", "ZANIDM"
    :param data_partition: (str) The data partition to evaluate on, either "train", "val", "test", or "full"
    :param evaluation_metric: (str) The evaluation metric to use, either "nll", "rmse", or "mae"
    :return: (pd.DataFrame) DataFrame containing evaluation results for each random seed

    """

    # Filter AnnData object based on data partition
    if data_partition == "train":
        adata_partition = adata[adata.obs["data_partition"] == "train"].copy()
    elif data_partition == "val":
        adata_partition = adata[adata.obs["data_partition"] == "val"].copy()
    elif data_partition == "test":
        adata_partition = adata[adata.obs["data_partition"] == "test"].copy()
    elif data_partition == "full":
        adata_partition = adata.copy()
    else:
        raise ValueError("data_partition must be either 'train', 'val', or 'test'.")

    eval_obs_name_list = []
    for seed in range(len(seeds)):
        random_seed = seeds[seed]
        eval_obs_name = likelihood_type + "_" + str(random_seed) + "_" + evaluation_metric
        eval_obs_name_list.append(eval_obs_name)

    eval_metric_values = adata_partition.obs[eval_obs_name_list].to_numpy()

    eval_metric_df = pd.DataFrame(
        data=eval_metric_values,
        columns=eval_obs_name_list
    )

    return eval_metric_df

def inference_vae(
        adata: AnnData,
        dataset: GeneExpressionDataset | TranscriptUsageDataset,
        model, # BetaVAE
        likelihood_type: str,
        data_modality: str = "Gene expression",
        count_data_included: bool = True,
        batch_size: int = 256,
) -> AnnData:
    r"""
    Given the data both as AnnData object and as Dataset, a trained model (instance of BetaVAE), infer the latent mean embeddings, the two UMAP
    dimensions, and the data reconstructions. The latent mean embeddings and the UMAP embeddings are both added as .obsm
    to the given AnnData object. The data reconstructions are added as an additional layer. The edited AnnData object
    is returned.

    :param adata: (AnnData)
    :param dataset:
    :param model: (BetaVAE)
    :param likelihood_type: (str) The likelihood function used in the model, e.g. "poisson", "gaussian", etc.
    :param data_modality: (str) The type of data modality, either "Gene expression" or "Transcript usage".
    :param count_data_included: (bool)
    :param batch_size: (int)
    :return: adata
    """

    model.eval()

    # Create an unshuffled dataloader from the data provided in dataset
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # Initialise latent mean embeddings, reconstructed data, and negative log-likelihood as zero tensor
    latent_mean_embeddings = torch.zeros(len(dataset), model.latent_dim)

    if data_modality == "Gene expression":
        mean_reconstructions = torch.zeros(len(dataset), dataset.num_genes)
    elif data_modality == "Transcript usage":
        mean_reconstructions = torch.zeros(len(dataset), dataset.num_introns)
    else:
        raise ValueError("data_modality must be either 'Gene expression' or 'Transcript_usage'.")

    nll = torch.zeros(dataset.num_cells)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if count_data_included:
                input_batch_counts, input_batch_levels, _, _ = batch

                input_batch_counts = input_batch_counts.to(model.device)
                input_batch_levels = input_batch_levels.to(model.device)

                input_batch = (input_batch_counts, input_batch_levels)
            else:
                _, input_batch, _, _ = batch

                input_batch = (input_batch.to(model.device))

            # Infer latent mean embeddings
            variational_posterior_dict = model.variational_posterior(input_batch[-1], None)

            # Predict mean of data reconstructions
            if data_modality == "Gene expression":
                if model.scaling_factor:
                    generative_model_dict = model.generative_model(
                        variational_posterior_dict["Latent mean"],
                        variational_posterior_dict["Scale factor"]
                    )
                else:
                    generative_model_dict = model.generative_model(variational_posterior_dict["Latent mean"], None)
            elif data_modality == "Transcript usage":
                generative_model_dict = model.generative_model(variational_posterior_dict["Latent mean"],
                                                               input_batch[0])

            # Evaluate the negative log-likelihood (NLL)
            batch_nll = model.negative_log_likelihood(input_batch[0], generative_model_dict)

            # Add latent mean and data reconstruction to tensor
            latent_mean_embeddings[i * batch_size:(i + 1) * batch_size, :] = variational_posterior_dict[
                "Latent mean"]
            if model.scaling_factor:
                mean_reconstructions[i * batch_size:(i + 1) * batch_size, :] = generative_model_dict["Scaled mean"]
            else:
                mean_reconstructions[i * batch_size:(i + 1) * batch_size, :] = generative_model_dict["Mean"]
            nll[i * batch_size:(i + 1) * batch_size] = batch_nll

        # Move tensors from device to cpu
        latent_mean_embeddings = latent_mean_embeddings.cpu().numpy()
        mean_reconstructions = mean_reconstructions.cpu().numpy()
        nll = nll.cpu().numpy()

    # Add latent mean embeddings, data reconstructions, and negative log-likelihood to AnnData object
    adata.obsm[likelihood_type + "_latent_mean"] = latent_mean_embeddings
    adata.layers[likelihood_type + "_reconstructions"] = mean_reconstructions
    adata.obs[likelihood_type + "_nll"] = nll

    # NOT NECESSARY Calculate neighbourhood graph using latent mean embeddings

    # Calculate UMAP dimensions using latent mean embeddings and add it to AnnData object
    adata.obsm[likelihood_type + "_X_umap"] = UMAP(n_components=2).fit_transform(adata.obsm[likelihood_type + "_latent_mean"])

    return adata

def run_random_seed_evaluation_VAE(
        model,
        observation_model: str,
        beta_str: str,
        list_of_random_seeds: List[int],
        list_of_epoch_checkpoints: List[int],
        adata: AnnData,
        dataset: GeneExpressionDataset | TranscriptUsageDataset,
        dataset_name: str,
        num_hvg: int,
        path2models: str,
        data_modality: str,
        device: str = "cuda",
        count_data_included: bool = True,
        batch_size: int = 256,
) -> AnnData:
    r"""
    Given a trained model (instance of BetaVAE), a list of random seeds, a list of epoch checkpoints, an AnnData object,
    a Dataset, dataset name, number of highly variable genes, path to stored models, data modality, device, whether count data is included,
    and batch size, run inference for each random seed and epoch checkpoint, and return the edited AnnData object.

    :param model: (BetaVAE)
    :param observation_model: (str) The likelihood function used in the model, e.g. "NB, "ZINB", "Gaussian", "ZIDM", "DM", "ZANIDM"
    :param beta_str: (str) The beta value used in the model as string
    :param list_of_random_seeds: (List[int]) List of random seeds
    :param list_of_epoch_checkpoints: (List[int]) List of epoch checkpoints
    :param adata: (AnnData)
    :param dataset: (GeneExpressionDataset | TranscriptUsageDataset)
    :param dataset_name: (str)
    :param num_hvg: (int)
    :param path2models: (str)
    :param data_modality: (str) The type of data modality, either "Gene expression" or "Transcript usage".
    :param device: (str)
    :param count_data_included: (bool)
    :param batch_size: (int)
    :return: adata
    """

    for i, seed in enumerate(list_of_random_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Name of the likelihood + seed for storing in AnnData object
        likelihood_seed = observation_model + "_" + str(seed)

        # Nomenclature geneExpression + likelihood + VAE + seed + beta
        if data_modality == "Gene expression":
            model_name = "geneExpression" + observation_model + "VAE" + "_" + str(seed) + "_Beta_" + beta_str
        elif data_modality == "Transcript usage":
            model_name = "tuVI_" + observation_model + "_" + str(seed) + "_Beta_" + beta_str
        else:
            raise ValueError("data_modality must be either 'Gene expression' or 'Transcript usage'.")

        # Epoch checkpoint
        epoch_checkpoint = list_of_epoch_checkpoints[i]

        model_checkpoint_name = model_name + "_epochs_" + str(epoch_checkpoint) + "_checkpoint.pth"
        checkpoint = torch.load(path2models + dataset_name + "_" + str(num_hvg) + "_" + model_checkpoint_name)

        model.load_state_dict(checkpoint["model_state_dict"])

        # Push model to GPU if available
        if torch.cuda.is_available():
            model = model.to(device)

        model.eval()

        """
        #### DEBUG START
        # Force actual feature precision to 1 because exp(0) = 1
        model.feature_precision = torch.zeros(
            model.feature_precision.shape,
            device=device,
            dtype=torch.float32,
        )
        #### DEBUG END
        """

        adata = inference_vae(
            adata,
            dataset,
            model,
            likelihood_type=likelihood_seed,
            data_modality=data_modality,
            count_data_included=count_data_included,
            batch_size=batch_size,
        )

    return adata

def run_random_seed_evaluation_TRVI(
        model: TRVI,
        likelihoods: List[str],
        beta_str: str,
        list_of_random_seeds: List[int],
        list_of_epoch_checkpoints: List[int],
        adata: Tuple[AnnData, AnnData],
        dataset: GeneExpressionTranscriptUsageDataset,
        dataset_name: str,
        num_hvg: int,
        path2models: str,
        device: str = "cuda",
        count_data_included: List[bool] = [True, True],
        batch_size: int = 256,
) -> Tuple[AnnData, AnnData]:
    r"""
    Given a trained model (TRVI), a list of random seeds, a list of epoch checkpoints, a Tuple of AnnData objects,
    infer for each random seed and epoch checkpoint the latent mean embeddings, the two UMAP dimensions, and the data
    reconstructions for both data modalities. The latent mean embeddings and the UMAP embeddings are both added as
    .obsm to the given AnnData object. The data reconstructions are added as an additional layer. The edited AnnData
    object is returned.

    :param model: (TRVI)
    :param likelihoods: (List[str])
    :param beta_str: (str)
    :param list_of_random_seeds: (List[int]) List of random seeds
    :param list_of_epoch_checkpoints: (List[int]) List of epoch checkpoints
    :param adata: (Tuple[AnnData, AnnData])
    :param dataset: (GeneExpressionTranscriptUsageDataset)
    :param dataset_name: (str)
    :param num_hvg: (int)
    :param path2models: (str)
    :param device: (str)
    :param count_data_included: (bool)
    :param batch_size: (int)
    :return adata: Tuple[AnnData, AnnData]

    """

    for i, seed in enumerate(list_of_random_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Name of the likelihood + seed for storing in AnnData object
        likelihood_1_seed = likelihoods[0] + "_" + str(seed)
        likelihood_2_seed = likelihoods[1] + "_" + str(seed)

        # Nomenclature TRVI + seed + likelihood 1 + likelihood 2 + beta
        model_name = "TRVI_" + str(seed) + "_" + likelihoods[0] + "_" + likelihoods[1] + "_" + beta_str

        #+ "_epochs_" + str(list_of_epoch_checkpoints[i]) + "_checkpoint.pth"

        # Epoch checkpoint
        epoch_checkpoint = list_of_epoch_checkpoints[i]
        model_checkpoint_name = model_name + "_epochs_" + str(epoch_checkpoint) + "_checkpoint.pth"
        checkpoint = torch.load(path2models + dataset_name + "_" + str(num_hvg) + "_" + model_checkpoint_name)

        model.load_state_dict(checkpoint["model_state_dict"])

        # Push model to GPU if available
        if torch.cuda.is_available():
            model = model.to(device)

        adata = inference_trvi(
            adata=adata,
            dataset=dataset,
            model=model,
            likelihood_types=[likelihood_1_seed, likelihood_2_seed],
            count_data_included=count_data_included,
            batch_size=batch_size
        )

    return adata

def create_differential_analysis_distriubtion_df(
        adata: AnnData,
        cluster_group: str,
        top_genes: List[str],
        transcriptomic_facet: str,
        **kwargs
) -> pd.DataFrame:
    r"""
    Calculate for either a set of genes or a set of isoforms of a cluster group, the distribution of expression levels
    or PSI scores, respectively, and return a DataFrame in long format.
    :param adata: (AnnData)
    :param cluster_group: (str)
    :param top_genes: (List[str])
    :param transcriptomic_facet: (str)
    """

    if transcriptomic_facet == "Gene expression":
        gene_in_group = "DEG group " + cluster_group + " vs other"
        var_name = 'Gene'
        value_name = 'Expression'
        value_matrix = adata[:, top_genes].X.todense() if 'sparse' in str(type(adata.X)) else adata[:, top_genes].X
    elif transcriptomic_facet == "Transcript usage":
        gene_in_group = "DSG group " + cluster_group + " vs other"
        var_name = 'Isoform'
        value_name = 'PSI-score'
        value_matrix = np.array(adata[:, top_genes].layers["PSI_raw"])
    else:
        raise ValueError("transcriptomic_facet must be either 'Gene expression' or 'Transcript usage'.")
    # DSG expression levels for violin plots
    groups_indices = adata.obs["leiden"].to_numpy()
    not_differential_group = "other"
    differential_group_indices = np.argwhere(groups_indices == cluster_group)
    not_differential_group_indices = np.argwhere(groups_indices != cluster_group)

    gene_in_group_values = np.zeros(adata.n_obs, dtype=object)
    gene_in_group_values[differential_group_indices] = "target"  # group_id_GE
    gene_in_group_values[not_differential_group_indices] = not_differential_group
    adata.obs[gene_in_group] = gene_in_group_values

    distribution_df = pd.DataFrame(value_matrix, columns=top_genes)
    distribution_df['Group'] = adata.obs[gene_in_group].to_numpy()
    distribution_df_long = pd.melt(distribution_df, id_vars=['Group'], value_vars=top_genes, var_name=var_name,
                              value_name=value_name)

    return distribution_df_long

def create_cell_type_groups_relevance_weight_df(
        adata_objects: Tuple[AnnData, AnnData],
        weighting_keys: List[str],
        cell_type_groups_key: str,
        cell_org_hierarchy_dictionary: dict | None = None,
        save_cell_type_groups_relevance_weight_df: bool = True,
        **kwargs
) -> pd.DataFrame:
    r"""
    Given gene expression and AS-induced transcript usage data as tuple of AnnData objects, a list of weighting keys for
    each data modality, a cell type groups key, and an optional cell ontology hierarchy dictionary, calculate the mean
    relevance weights for each cell type group and return a DataFrame in long format. Optionally save the DataFrame as
    a CSV file.

    :param adata_objects: (AnnData) Tuple of AnnData objects for gene expression and transcript usage data
    :param weighting_keys: (List[str]) List of weighting keys for each data modality
    :param cell_type_groups_key: (str) Key for cell type groups in AnnData
    :param cell_org_hierarchy_dictionary: (dict | None) Optional dictionary for cell ontology hierarchy
    :param save_cell_type_groups_relevance_weight_df: (bool) Whether to save the DataFrame as a CSV file
    :return: (pd.DataFrame) DataFrame containing mean modality-relevance weights for each cell type group in long format
    """

    adata_1 = adata_objects[0]
    adata_2 = adata_objects[1]

    weighting_key_1 = weighting_keys[0]
    weighting_key_2 = weighting_keys[1]

    cell_type_group_1_weighting_df = adata_1.obs.groupby(cell_type_groups_key)[weighting_key_1].mean().to_frame()
    cell_type_group_2_weighting_df = adata_2.obs.groupby(cell_type_groups_key)[weighting_key_2].mean().to_frame()

    cell_type_group_1_weighting_df.rename(columns={weighting_key_1: "GE weight"}, inplace=True)
    cell_type_group_2_weighting_df.rename(columns={weighting_key_2: "TU weight"}, inplace=True)

    cell_type_group_weighting_df = pd.concat([cell_type_group_1_weighting_df, cell_type_group_2_weighting_df], axis=1)

    if cell_org_hierarchy_dictionary is not None:
        all_cell_type_groups = list(adata_1.obs[cell_type_groups_key].unique())
        sorted_cell_type_groups = sorted(all_cell_type_groups, key=lambda x: (cell_org_hierarchy_dictionary.get(x, "Unknown"), x))
        cell_type_group_weighting_df = cell_type_group_weighting_df.reindex(sorted_cell_type_groups)

    cell_type_groups_weights_df = cell_type_group_weighting_df.reset_index().rename(columns={'index': cell_type_groups_key})
    cell_type_groups_weights_df = cell_type_groups_weights_df.melt(id_vars=cell_type_groups_key, var_name='Weight Type', value_name='Weight')

    if save_cell_type_groups_relevance_weight_df:
        dataset_name = kwargs.get("dataset_name", "default")
        cell_type_groups_weights_df.to_csv("./data/" + dataset_name + "/trvi_cell_type_groups_weights.csv", index=False)

    return cell_type_groups_weights_df


def run_functional_enrichment_analysis(
    gene_types: str,
    tissue = str | None,
    num_hvg: int = 3000,
    path2data: str = "./data/tabulaMuris/",
    organism: str = "mmusculus"
) -> pd.DataFrame:
    r"""
    Given a type of differentially analysed genes(either "DEG" or "DSG" ), an optional tissue, the number of highly
    variable genes, and a path to the stored Anndata files, load the Anndata objects, extract the top DEGs or DSGs for
    each tissue, run functional enrichment analysis using gprofile,and return the enriched terms as a DataFrame.

    :param gene_types: str specifying the type of differentially analysed genes, either "DEG" for differentially expressed genes or "DSG" for differentially spliced genes
    :param tissue: (str | None) Tissue to run functional enrichment analysis on, if None, run for all tissues
    :param num_hvg: (int) Number of highly variable genes used in the model
    :param path2data: (str) Path to the stored Anndata file containing the DEGs or DSGs
    :param organism: (str) The organism to use for functional enrichment analysis, e.g. "mmusculus" or "hsapiens"
    """

    if gene_types == "DEG":
        modality = "GE"
    elif gene_types == "DSG":
        modality = "TU"
    else:
        raise ValueError("gene_types must be either 'DEG' or 'DSG'.")

    if tissue is not None:
        path2adata = path2data + "adata_" + modality + "_" + str(num_hvg) + "_" + tissue + "_trvi_inference.h5ad"
    else:
        path2adata = path2data + "adata_" + modality + "_" + str(num_hvg) + "_trvi_inference.h5ad"

    # Assert if adata exists and load it
    assert os.path.exists(path2adata), "The specified Anndata file does not exist. Please check the path and file name."
    adata = ad.read_h5ad(path2adata)

    # Assert if adata.uns contains "rank_genes_groups" if "DEG" or "rank_introns_groups" if "DSG"
    if gene_types == "DEG":
        assert "rank_genes_groups" in adata.uns, "The Anndata file does not contain 'rank_genes_groups' in .uns. Please check if the file contains the results of differential expression analysis."
        rank_key = "rank_genes_groups"
    elif gene_types == "DSG":
        assert "rank_introns_groups" in adata.uns, "The Anndata file does not contain 'rank_introns_groups' in .uns. Please check if the file contains the results of differential splicing analysis."
        rank_key = "rank_introns_groups"
    else:
        raise ValueError("gene_types must be either 'DEG' or 'DSG'.")

    # For filter out all entries with the pval_adj < 0.05, for DSG this has been carried out before
    diff_analysis_results = adata.uns[rank_key].copy()
    groups = diff_analysis_results["logfoldchanges"].dtype.names
    sig_diff_genes = []

    if gene_types == "DEG":
        # For DEG only, filter for pval_adj < 0.05
        for group in groups:
            # ADD: Reduce to 50 or 25
            rank_df = sc.get.rank_genes_groups_df(adata, group=group).head(100)
            # Logfoldchange threshold > 0.5
            sig_diff_genes_group = rank_df[(rank_df["pvals_adj"] < 0.05) & (rank_df["logfoldchanges"] > 0.5)]["names"].tolist()
            sig_diff_genes.extend(sig_diff_genes_group)

        unique_sig_diff_genes = list(set(sig_diff_genes))

    elif gene_types == "DSG":
        # For DSG, the filtering for adjusted p-value < 0.05 has already been carried out before, so no need to filter again
        for group in groups:
            rank_df = rank_introns_groups_df(adata, group=group)
            # Logfoldchange threshold > 0.5
            #sig_diff_genes_group = rank_df[rank_df["logfoldchanges"] > 0.5]["gene_names"].tolist()
            sig_diff_genes_group = rank_df["gene_names"].tolist()
            sig_diff_genes.extend(sig_diff_genes_group)

        unique_sig_diff_genes = list(set(sig_diff_genes))
        # Remove "0" from unique_sig_diff_genes if it exists, as "0" is used to indicate non-significant introns in the DSG analysis
        if "0" in unique_sig_diff_genes:
            unique_sig_diff_genes.remove("0")
    else:
        raise ValueError("gene_types must be either 'DEG' or 'DSG'.")

    # Run functional enrichment analysis using gprofile
    #print(f"Using the following list of {len(unique_sig_diff_genes)} {gene_types}s for functional enrichment analysis: {unique_sig_diff_genes}")

    #enrichment_results_df = sc.queries.enrich(unique_sig_diff_genes, org=organism)

    # Run functional enrichment analysis using g:Profiler
    print(
        f"Using the following list of {len(unique_sig_diff_genes)} "
        f"{gene_types}s for functional enrichment analysis: "
        f"{unique_sig_diff_genes}"
    )

    enrichment_results_df = sc.queries.enrich(
        unique_sig_diff_genes,
        org=organism,
        gprofiler_kwargs={"no_evidences": False}
    )

    enrichment_results_df = enrichment_results_df.rename(
        columns={"intersections": "matched_genes"}
    )

    enrichment_results_df["matched_genes"] = (
        enrichment_results_df["matched_genes"]
        .apply(lambda genes: "; ".join(genes))
    )

    return enrichment_results_df

def determine_shared_unique_deg_dsg_terms(
    deg_enrichment_results_df: pd.DataFrame,
    dsg_enrichment_results_df: pd.DataFrame,
    tissue: str | None = None,
    save_df: bool = True,
) -> Tuple[pd.DataFrame]:

    r"""
    Given two pandas DataFrames containing the functional enrichment results for DEGs and DSGs respectively, an optional
    tissue, and whether to save the resulting three dataframes as csv files, determine the shared and unique
    significantly enriched terms for DEGs and DSGs, and return three DataFrames containing the shared and unique
    enrichment terms.

    :param deg_enrichment_results_df: (pd.DataFrame) DataFrame containing the functional enrichment results for DEGs
    :param dsg_enrichment_results_df: (pd.DataFrame) DataFrame containing the functional enrichment results for DSGs
    :param tissue: (str | None) The tissue to determine the shared and unique enrichment terms for, if None, determine for all tissues
    :param save_df: (bool) Whether to save the resulting three DataFrames as csv files
    :return: Tuple[pd.DataFrame] containing the shared and unique enrichment terms for DEGs and DSGs
    """

    deg_term_identifiers = deg_enrichment_results_df["native"].to_numpy()
    dsg_term_identifiers = dsg_enrichment_results_df["native"].to_numpy()

    # Intersection and unique pathway dataframes
    overlapping_terms = np.intersect1d(dsg_term_identifiers, deg_term_identifiers)
    dsg_unique_terms = np.setdiff1d(dsg_term_identifiers, deg_term_identifiers)
    deg_unique_terms = np.setdiff1d(deg_term_identifiers, dsg_term_identifiers)

    # Dataframes of overlapping and unique pathways
    dsg_unique_terms_df = dsg_enrichment_results_df[dsg_enrichment_results_df["native"].isin(dsg_unique_terms)]
    deg_unique_terms_df = deg_enrichment_results_df[deg_enrichment_results_df["native"].isin(deg_unique_terms)]
    overlapping_terms_df = deg_enrichment_results_df[deg_enrichment_results_df["native"].isin(overlapping_terms)]

    if save_df:
        if tissue is None:
            dsg_unique_terms_df.to_csv("./data/tabulaMuris/dsg_unique_terms_df.csv", index=False)
            deg_unique_terms_df.to_csv("./data/tabulaMuris/deg_unique_terms_df.csv", index=False)
            overlapping_terms_df.to_csv("./data/tabulaMuris/overlapping_terms_df.csv", index=False)
        else:
            dsg_unique_terms_df.to_csv(f"./data/tabulaMuris/dsg_unique_terms_df_{tissue}.csv", index=False)
            deg_unique_terms_df.to_csv(f"./data/tabulaMuris/deg_unique_terms_df_{tissue}.csv", index=False)
            overlapping_terms_df.to_csv(f"./data/tabulaMuris/overlapping_terms_df_{tissue}.csv", index=False)

    return dsg_unique_terms_df, deg_unique_terms_df, overlapping_terms_df

def load_merge_and_save_enrichment_terms(
        tax_level_list: List[str],
        file_prefix: str,
        output_filename: str,
        path2data: str,
        **kwargs
) -> pd.DataFrame:
    r"""
    This is a helper function to load and merge the dataframes of enrichment analysis results saved as csv files.
    Duplicate terms are retained only once. It returns one merged dataframe.

    :param tax_level_list: List of taxonomy levels to load the dataframes containing the enrichment analysis results
    :param file_prefix: The prefix for the CSV files to be loaded (i.e. "dsg_unique_terms_df", "dsg_unique_terms_df", "overlapping_terms_df")
    :param output_filename: The filename for the output file (i.e. "dsg_unique_terms_df_all.csv", "deg_unique_terms_df_all.csv", "overlapping_terms_df_all.csv")
    :param path2data: The path to the directory containing the CSV files

    """
    dataframes = []
    dataset_name = kwargs.get("dataset_name", "")

    for tax_level_member in tax_level_list:
        file_path = path2data + f"{file_prefix}_{tax_level_member}.csv"

        enrichment_df = pd.read_csv(file_path)
        enrichment_df["tissue"] = tax_level_member
        enrichment_df["dataset"] = dataset_name

        dataframes.append(enrichment_df)

    # Combine all taxonomic levels
    enrichment_df_all = pd.concat(dataframes, ignore_index=True)

    # Retain each term only once across all taxonomy levels
    enrichment_df_all = enrichment_df_all.drop_duplicates(
        subset="native",
        keep="first",
    )

    # Save the combined dataframe
    enrichment_df_all.to_csv(
        path2data + output_filename,
        index=False,
    )

    return enrichment_df_all

def inference_trvi(
        adata: Tuple[AnnData, AnnData], # adata[0] for gene expression, adata[1] for transcript usage
        dataset: GeneExpressionTranscriptUsageDataset,
        model: TRVI,
        likelihood_types: List[str],
        count_data_included: List[bool], # In general [True, True]
        batch_size: int = 256,
) -> Tuple[AnnData, AnnData]:
    r"""
    Given the data both as Tuple of two AnnData objects (gene expression and transcript usage) and as Dataset, a trained
    model (instance of TRVI), infer the latent mean embeddings (private gene expression, private transcript usage,
    shared), the individual two UMAP dimensions, the data reconstructions, (IN FUTURE: and the cell type predictions).
    The latent mean embeddings and the UMAP embeddings are added as .obsm to the given AnnData objects depending on the
    modality (the shared is added to both). The data reconstructions are added as an additional layer, and the cell type
    predictions as .obs. The edited AnnData objects are returned as Tuple

    :param adata: Tuple of two AnnData objects with adata[0] containing gene expression and adata[1] containing transcript usage data
    :param dataset: A customised PyTorch dataset of type GeneExpressionTranscriptUsageDataset
    :param model: An instance of TRVI
    :param likelihood_types: A list of strings of likelihoods where likelihoods[0] is for gene expression and likelihoods[1] for transcript usage e.g. ["ZINB", "ZIDM"]
    :param count_data_included: A list of bool, in general [True, True] for both modalities unless Gaussian likelihoods are used
    :param batch_size: (int)
    :return: adata Tuple of processed AnnData objects (gene expression, transcript usage)
    """

    # Create an unshuffled dataloader from the data provided in dataset
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # Initialize latent mean embeddings, reconstructed data, and negative log-likelihood as zero tensor
    latent_mean_embeddings_private_1 = torch.zeros(len(dataset), model.latent_dim[0]) # gene expression private
    latent_mean_embeddings_shared_1 = torch.zeros(len(dataset), model.latent_dim[2])  # gene expression uni-modal shared embedding
    latent_mean_embeddings_private_2 = torch.zeros(len(dataset), model.latent_dim[1]) # transcript usage private
    latent_mean_embeddings_shared_2 = torch.zeros(len(dataset), model.latent_dim[2]) # transcript usage uni-modal shared embedding
    latent_mean_embeddings_shared = torch.zeros(len(dataset), model.latent_dim[2]) # shared (mean of bimodal shared embedding)
    weights_1 = torch.zeros(dataset.num_cells)
    weights_2 = torch.zeros(dataset.num_cells)

    mean_reconstructions_1 = torch.zeros(len(dataset), dataset.num_genes) # gene expression
    mean_reconstructions_2 = torch.zeros(len(dataset), dataset.num_introns) # transcript usage

    nll_1 = torch.zeros(dataset.num_cells)
    nll_2 = torch.zeros(dataset.num_cells)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # 1 is for gene expression data and 2 for transcript usage data
            input_batch_1_counts, input_batch_1_levels, input_batch_2_counts, input_batch_2_levels, _, _ = batch
            batch_size = input_batch_1_counts.shape[0]

            # Push data to model device
            # Push data to device ("cuda" or "cpu")
            input_batch_1_counts = input_batch_1_counts.to(model.vae_1.device)
            input_batch_1_levels = input_batch_1_levels.to(model.vae_1.device)
            input_batch_2_counts = input_batch_2_counts.to(model.vae_2.device)
            input_batch_2_levels = input_batch_2_levels.to(model.vae_2.device)

            input_batch = (input_batch_1_counts, input_batch_1_levels, input_batch_2_counts, input_batch_2_levels)

            # Infer latent mean embeddings (unimodal)
            variational_posterior_dict = model.variational_posterior(
                (input_batch_1_levels, input_batch_2_levels),
                [None, None, None, None]
            )

            # Predict mean of data reconstructions
            generative_model_dict = model.generative_model(
                (variational_posterior_dict["Modality 1"]["Shared latent mean"], variational_posterior_dict["Modality 2"]["Shared latent mean"]),
                (variational_posterior_dict["Modality 1"]["Private latent mean"], variational_posterior_dict["Modality 2"]["Private latent mean"]),
                (None, None),
                (None, None),
                input_batch_2_counts
            )

            # Evaluate the negative log-likelihood (NLL) (unimodal)
            nll_1_unimodal = model.vae_1.negative_log_likelihood(input_batch_1_counts, generative_model_dict[
                "Modality 1 unimodal"])  # Needs to be more flexible for Gaussian likelihood
            nll_2_unimodal = model.vae_2.negative_log_likelihood(input_batch_2_counts, generative_model_dict[
                "Modality 2 unimodal"])  # Needs to be more flexible for Gaussian likelihood

            nll_1[i * batch_size:(i + 1) * batch_size] = nll_1_unimodal
            nll_2[i * batch_size:(i + 1) * batch_size] = nll_2_unimodal

            # Add latent mean and data reconstruction to tensor
            latent_mean_embeddings_private_1[i * batch_size:(i + 1) * batch_size, :] = variational_posterior_dict["Modality 1"]["Private latent mean"]
            latent_mean_embeddings_shared_1[i * batch_size:(i + 1) * batch_size, :] = variational_posterior_dict["Modality 1"]["Shared latent mean"]
            latent_mean_embeddings_private_2[i * batch_size:(i + 1) * batch_size, :] = variational_posterior_dict["Modality 2"]["Private latent mean"]
            latent_mean_embeddings_shared_2[i * batch_size:(i + 1) * batch_size, :] = variational_posterior_dict["Modality 2"]["Shared latent mean"]
            latent_mean_embeddings_shared[i * batch_size:(i + 1) * batch_size, :] = variational_posterior_dict["Mixture of experts"]["Latent mean"]
            weights_1 [i * batch_size:(i + 1) * batch_size] = variational_posterior_dict["Mixture of experts"]["Weighting modality 1"]
            weights_2[i * batch_size:(i + 1) * batch_size] = variational_posterior_dict["Mixture of experts"]["Weighting modality 2"]
            mean_reconstructions_1[i * batch_size:(i + 1) * batch_size, :] = generative_model_dict["Modality 1 unimodal"]["Mean"]
            mean_reconstructions_2[i * batch_size:(i + 1) * batch_size, :] = generative_model_dict["Modality 2 unimodal"]["Mean"]

        # Move tensors from device to cpu
        latent_mean_embeddings_private_1 = latent_mean_embeddings_private_1.cpu().numpy()
        latent_mean_embeddings_shared_1 = latent_mean_embeddings_shared_1.cpu().numpy()
        latent_mean_embeddings_private_2 = latent_mean_embeddings_private_2.cpu().numpy()
        latent_mean_embeddings_shared_2 = latent_mean_embeddings_shared_2.cpu().numpy()
        latent_mean_embeddings_shared = latent_mean_embeddings_shared.cpu().numpy()
        weights_1 = weights_1.cpu().numpy()
        weights_2 = weights_2.cpu().numpy()
        mean_reconstructions_1 =  mean_reconstructions_1.cpu().numpy()
        mean_reconstructions_2 = mean_reconstructions_2.cpu().numpy()
        nll_1 = nll_1.cpu().numpy()
        nll_2 = nll_2.cpu().numpy()

    # Add latent mean embeddings, data reconstructions, and negative log-likelihood to AnnData objects
    adata[0].obsm[likelihood_types[0] + "_private_latent_mean"] = latent_mean_embeddings_private_1
    adata[0].obsm[likelihood_types[0] + "_shared_latent_mean"] = latent_mean_embeddings_shared_1
    adata[0].obsm[likelihood_types[0] + "_" + likelihood_types[1] + "_shared_latent_mean"] = latent_mean_embeddings_shared
    adata[0].layers[likelihood_types[0] + "_reconstructions"] = mean_reconstructions_1
    adata[0].obs[likelihood_types[0] + "_weighting"] = weights_1
    adata[0].obs[likelihood_types[0] + "_nll"] = nll_1
    adata[0].obs[likelihood_types[0] + "_nll"] = nll_1

    adata[1].obsm[likelihood_types[1] + "_private_latent_mean"] = latent_mean_embeddings_private_2
    adata[1].obsm[likelihood_types[1] + "_shared_latent_mean"] = latent_mean_embeddings_shared_2
    adata[1].obsm[likelihood_types[0] + "_" + likelihood_types[1] + "_shared_latent_mean"] = latent_mean_embeddings_shared
    adata[1].layers[likelihood_types[1] + "_reconstructions"] = mean_reconstructions_2
    adata[1].obs[likelihood_types[1] + "_weighting"] = weights_2
    adata[1].obs[likelihood_types[1] + "_nll"] = nll_2

    # Calculate UMAP dimension using latent mean embeddings add it to AnnData object
    adata[0].obsm[likelihood_types[0] + "_private_X_umap"] = UMAP(n_components=2).fit_transform(adata[0].obsm[likelihood_types[0] + "_private_latent_mean"])
    adata[0].obsm[likelihood_types[0] + "_shared_X_umap"] = UMAP(n_components=2).fit_transform(adata[0].obsm[likelihood_types[0] + "_shared_latent_mean"])
    adata[1].obsm[likelihood_types[1] + "_private_X_umap"] = UMAP(n_components=2).fit_transform(adata[1].obsm[likelihood_types[1] + "_private_latent_mean"])
    adata[1].obsm[likelihood_types[1] + "_shared_X_umap"] = UMAP(n_components=2).fit_transform(adata[1].obsm[likelihood_types[1] + "_shared_latent_mean"])

    shared_umap = UMAP(n_components=2).fit_transform(adata[0].obsm[likelihood_types[0] + "_" + likelihood_types[1] + "_shared_latent_mean"])
    adata[0].obsm[likelihood_types[0] + "_" + likelihood_types[1] + "_shared_X_umap"] = shared_umap
    adata[1].obsm[likelihood_types[0] + "_" + likelihood_types[1] + "_shared_X_umap"] = shared_umap

    return adata

def analysis_trvi_relevance_weights_across_cell_types(
        adata: AnnData,
        list_of_random_seeds: List[int],
        transcriptomic_facet: str = "Gene expression",
        likelihood_key: str = "ZINB",
        cell_type_key: str = "cell_ontology_class",
        sort_cell_type_key: str | None = None,
        **kwargs
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    r"""
    Given an AnnData object (either of gene expression data or transcript usage data) containing the results of
    inference carried out with TRVI (across random seeds), a list of random seeds, a transcriptomic facet (i.e. Gene
    expression or Transcript usage), a likelihood key (e.g. ZINB for gene expression, or ZIDM for transcript usage), and
    a cell type key (e.g. cell_ontology_class), and an optional sort_cell_type_key (e.g. sort cell types according to
    organ systems), calculate the mean modality-relevance weight per cell type and the total mean modality-relevance
    weight across all cell types for each random seed, determine the total mean and standard deviation of the modality-
    relevance weights for the specified transcriptomic facet across random seeds, and return a DataFrame containing the
    mean modality-relevance weights per cell type for each random seed, and a dictionary containing the total mean and
    standard deviation of the modality-relevance weights for the specified transcriptomic facet.

    :param adata: AnnData object (either of gene expression data or transcript usage data)
    :param list_of_random_seeds: list of random seeds e.g. [0, 1, 2, 3]
    :param transcriptomic_facet: Gene expression or Transcript usage data
    :param likelihood_key: key of cell type (e.g. ZINB)
    :param cell_type_key: key of cell type (e.g. cell_ontology_class)
    :param sort_cell_type_key: optional key to sort cell types according to, e.g. organ system key
    :param kwargs: additional keyword arguments to pass to TRVI

    """
    # For each seed, calculate the mean relevance weight per cell type and the total mean relevance weight across all cell types
    list_of_means = []
    list_of_total_means = []
    mean_weights_dict = {}

    for random_seed in list_of_random_seeds:
        weighting_key = likelihood_key + "_" + str(random_seed) + "_weighting"

        # Assert if weighting_key is present in adata.obs otherwise the inference has not been done with the according random seed
        assert weighting_key in adata.obs, f"The weighting key {weighting_key} is not present in adata.obs. Please check if the inference has been done with random seed {random_seed} and likelihood {likelihood_key}."

        # Mean relevance weight per cell type
        means = adata.obs.groupby(cell_type_key)[weighting_key].mean()
        list_of_means.append(means)

        mean_weights_dict[likelihood_key + "_" + str(random_seed)] = means

        # Total mean across cell types
        total_mean = means.mean()
        list_of_total_means.append(total_mean)

    # Total mean and standard deviation for specified transcriptomic facet
    total_means = np.array(list_of_total_means)
    total_mean = total_means.mean()
    total_std = total_means.std()

    total_means_complement = 1 - total_means  # for other transcriptomic facet
    total_mean_complement = total_means_complement.mean()
    total_std_complement = total_means_complement.std()

    if transcriptomic_facet == "Gene expression":
        total_mean_GE = total_mean
        total_std_GE = total_std

        total_mean_TU = total_mean_complement
        total_std_TU = total_std_complement

    elif transcriptomic_facet == "Transcript usage":
        total_mean_GE = total_mean_complement
        total_std_GE = total_std_complement

        total_mean_TU = total_mean
        total_std_TU = total_std

    else:
        raise ValueError("Invalid transcriptomic facet")

    atlas_weight_dict = {
        "total_mean_GE": total_mean_GE,
        "total_std_GE": total_std_GE,
        "total_mean_TU": total_mean_TU,
        "total_std_TU": total_std_TU,
    }

    # Convert mean weights dictionary into dataframe
    mean_weights_df = pd.DataFrame(mean_weights_dict)

    # Check if sort cell_type_key exists
    if sort_cell_type_key is not None:

        cell_system_counts_df = adata.obs.value_counts([cell_type_key, sort_cell_type_key])

        mean_weights_df_cell_names = mean_weights_df.index.to_numpy()
        organ_system_list = []
        cell_list = []

        for i in range(len(mean_weights_df_cell_names)):
            organ_systems_cell_present = cell_system_counts_df[mean_weights_df_cell_names[i]]

            organ_system_list.append(organ_systems_cell_present.index[0])

            if len(organ_systems_cell_present) > 1:
                cell_list.append(mean_weights_df_cell_names[i] + "*")
            else:
                cell_list.append(mean_weights_df_cell_names[i])

        mean_weights_df.index = cell_list

        # Add column sort_cell_type_key after settings["cell_type_key"] to mean_weights_df
        mean_weights_df.insert(0, sort_cell_type_key, organ_system_list)

        # Sort mean_weights_df alphabetically according to sort_cell_type_key column
        mean_weights_df = mean_weights_df.sort_values(by=[sort_cell_type_key])

    return mean_weights_df, atlas_weight_dict




def training_vae_embeddings_cell_type_classification(
        adata: AnnData,
        tissue: str,
        modality: str,
        likelihood: str,
        cell_types: Tuple[str],
        classification: str = "LogisticRegression",
        device = "cuda",
        num_hvg: int = 3000,
        dataset_name: str = "tabulaMuris",
        num_epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64

) -> None:
    if tissue == "All":
        adata_tissue = adata.copy()
    else:
        adata_tissue = adata[adata.obs["tissue"] == tissue].copy()

    adata_tissue_train = adata_tissue[adata_tissue.obs["data_partition"] == "train"].copy()
    adata_tissue_val = adata_tissue[adata_tissue.obs["data_partition"] == "val"].copy()

    # Embeddings for training and validation dataset
    vae_embeddings_train = adata_tissue_train.obsm[likelihood + "_latent_mean"]
    vae_embeddings_val = adata_tissue_val.obsm[likelihood + "_latent_mean"]

    # Ontology for training and validation dataset
    ontology_train = adata_tissue_train.obs["cell_ontology_class"].to_numpy()
    ontology_val = adata_tissue_val.obs["cell_ontology_class"].to_numpy()

    # Tissue for training and validation dataset
    tissue_train = adata_tissue_train.obs["tissue"].to_numpy()
    tissue_val = adata_tissue_val.obs["tissue"].to_numpy()

    # Create training and validation dataset for cell type classification
    # The `cell_types` list passed here will define the mapping for one-hot encoding.
    train_dataset = VAEEmbeddingsCellTypeDataset(
        vae_embeddings=vae_embeddings_train,
        ontology=ontology_train,
        tissue=tissue_train,
        cell_types=cell_types
    )
    val_dataset = VAEEmbeddingsCellTypeDataset(
        vae_embeddings=vae_embeddings_val,
        ontology=ontology_val,
        tissue=tissue_val,
        cell_types=cell_types
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Define type of loss function
    if classification == "LogisticRegression":
        loss_type = "Cross Entropy"
    else:
        raise ValueError("Unsupported classification type. Currently only 'LogisticRegression' is supported.")

    classifier_name = tissue + "_" + "LogisticRegressionClassifier_sc" + modality + "VI_" + likelihood
    classifier = LogisticRegressionClassifier(
        input_dim=vae_embeddings_train.shape[-1],
        num_classes=len(cell_types),
        device=device
    )

    # Push the classifier to the device
    classifier = classifier.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Training classifier {classifier_name} on {dataset_name} dataset with sc{modality}VI embeddings")

    train_vae_embedding_cell_type_classifier(
        classifier,
        optimizer,
        (train_dataloader, val_dataloader),
        num_epochs,
        classifier_name,
        dataset_name + "_" + str(num_hvg),
        loss_type,
    )

def training_cell_type_classifiers_trvi(
        adata: List[AnnData],
        tissue: str,
        likelihoods: List[str],
        emb_types: List[str],
        dataset_name: str = "tabulaMuris",
        classification: str="LogisticRegression",
        device: str = "cuda",
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 64
) -> None:
    r"""
    Given a tuple of two AnnData objects (gene expression and transcript usage) that have the latent embeddings already
    inferred and stored in .obsm, a chosen tissue type, a list of likelihhoods, and a list of embedding types, as well
    as arguments for training a classifier, train classifiers for each embedding type. The best performing classifiers
    are saved.

    :param adata: List of two AnnData objects with adata[0] containing gene expression and adata[1] containing transcript usage data
    :param tissue: The tissue type to filter the data by
    :param likelihoods: List of strings of likelihoods where likelihoods[0] is for gene expression and likelihoods[1] for transcript usage e.g. ["ZINB", "ZIDM"]
    :param emb_types: List of strings of embedding types to use for classification, e.g. ["private_1", "private_2", "shared_uni_1", "shared_uni_2", "shared"]
    :param dataset_name: The name of the dataset, e.g. "tabulaMuris"
    :param classification: The type of classifier to use, e.g. "LogisticRegression", "RandomForest", etc.
    :param device: The device to use for training, e.g. "cuda" or "cpu"
    :param num_epochs: The number of epochs to train the classifier for
    :param learning_rate: The learning rate for the optimizer
    :param weight_decay: The weight decay for the optimizer
    :param batch_size: The batch size for the DataLoader
    """

    """
    # Ground truth cell types for dataset
    if tissue == "All":
        cell_types = TABULA_MURIS_CELL_TYPES
    else:
        
        cell_types = tuple(TABULA_MURIS_TISSUE_CELL_DICTIONARY[tissue])
    """

    # Filter the AnnData objects by tissue type
    if tissue != "All":
        adata[0] = adata[0][adata[0].obs["tissue"] == tissue].copy()
        adata[1] = adata[1][adata[1].obs["tissue"] == tissue].copy()

        cell_types = tuple(np.unique(adata[0].obs["cell_ontology_class"].to_numpy()))
    else:
        cell_types = tuple(np.unique(adata[0].obs["cell_ontology_class"].to_numpy()))

    # Extract the training and validation dataset partitions from the AnnData objects
    adata_train = (
        adata[0][adata[0].obs["data_partition"] == "train"].copy(),
        adata[1][adata[1].obs["data_partition"] == "train"].copy()
    )
    adata_val = (
        adata[0][adata[0].obs["data_partition"] == "val"].copy(),
        adata[1][adata[1].obs["data_partition"] == "val"].copy()
    )

    # Create the dataset for cell type classification
    train_dataset = create_cell_type_classification_dataset(
        adata_train,
        likelihoods,
        cell_types
    )
    val_dataset = create_cell_type_classification_dataset(
        adata_val,
        likelihoods,
        cell_types
    )

    # Create the DataLoader for training and validation datasets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define type of loss function
    if classification == "LogisticRegression":
        loss_type = "Cross Entropy"
    else:
        raise ValueError("Unsupported classification type. Currently only 'LogisticRegression' is supported.")

    # Loop through the embedding types and train classifiers
    checkpoint_indices = []

    for emb_type in emb_types:
        # Choose embedding indices based on the embedding type
        if emb_type == "private_1":
            embedding_indices = [0]  # private embedding 1
        elif emb_type == "private_2":
            embedding_indices = [1]  # private embedding 2
        elif emb_type == "shared_uni_1":
            embedding_indices = [2]  # shared uni-modal embedding 1
        elif emb_type == "shared_uni_2":
            embedding_indices = [3]  # shared uni-modal embedding 2
        elif emb_type == "shared":
            embedding_indices = [4]  # shared embedding
        else:
            raise ValueError("Invalid embedding type")

        classifier_name = tissue + "_" + "LogisticRegressionClassifier_" + emb_type
        classifier = LogisticRegressionClassifier(
            input_dim=10,
            num_classes=len(cell_types),
            device=device
        )

        # Push the classifier to the device
        classifier = classifier.to(device)

        # Define the optimizer
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

        print(f"Training classifier {classifier_name} on {dataset_name} dataset with {emb_type} embeddings")

        train_classifier(
            classifier,
            optimizer,
            (train_dataloader, val_dataloader),
            embedding_indices,
            num_epochs,
            classifier_name,
            dataset_name,
            loss_type,
            patience=10
        )

def evaluate_vae_emb_cell_type_classifiers(
        adata: Tuple[AnnData, AnnData],
        tissue: str,
        checkpoint_indices_GE: List[int],
        checkpoint_indices_TU: List[int],
        likelihoods_GE: List[str],
        likelihoods_TU: List[str],
        dataset_name: str = "tabulaMuris",
        device: str = "cuda"
) -> dict[Any, Any]:
    r"""
    Given a tuple of two AnnData objects (gene expression and transcript usage) that have the latent embeddings already
    inferred and stored in .obsm, a chosen tissue type, a list of likelihoods, and a list of embedding types, as well
    as a list of checkpoint indices, evaluate the classifiers for each embedding type. The evaluation metrics are stored
    for each classifier in a dictionary which is returned. The predicted cell types are added to the .obs of the AnnData
    objects which are also returned.

    :param adata: Tuple of two AnnData objects with adata[0] containing gene expression and adata[1] containing transcript usage data
    :param tissue: The tissue type to filter the data by
    :param checkpoint_indices_GE: List of gene expression checkpoint indices to evaluate
    :param checkpoint_indices_TU: List of transcript usage checkpoint indices to evaluate
    :param likelihoods_GE: List of gene expression likelihood strings
    :param likelihoods_TU: List of transcript usage likelihood strings
    :param dataset_name: The name of the dataset, e.g. "tabulaMuris"
    :param device: The device to use for evaluation, e.g. "cuda" or "cpu"
    :return: Tuple of two AnnData objects with predicted cell types in .obs and a dictionary with evaluation metrics
    """

    # Create an empty list that will store the classifiers
    classifiers = []
    path2scgevi_classifiers = "./models/scGEVI/classifiers/"
    path2tuvi_classifiers = ("./models/tu"
                               "VI/classifiers/")

    # Extract the ground truth cell types for the dataset
    if tissue == "All":
        cell_types = TABULA_MURIS_CELL_TYPES
    else:
        cell_types = tuple(TABULA_MURIS_TISSUE_CELL_DICTIONARY[tissue])

    # Load scGEVI classifiers
    for i, checkpoint_idx in enumerate(checkpoint_indices_GE):
        classifier_name = dataset_name + "_" + tissue + "_" + "LogisticRegressionClassifier_scGEVI_" + likelihoods_GE[i] + "_epochs_" + str(checkpoint_idx) + "_checkpoint.pth"
        classifier = LogisticRegressionClassifier(
            input_dim=10,
            num_classes=len(cell_types),
            device=device
        )
        classifier.load_state_dict(torch.load(path2scgevi_classifiers + classifier_name, map_location=device)["model_state_dict"])
        classifier.to(device)
        classifiers.append(classifier)

    # Load tuVI classifiers
    for i, checkpoint_idx in enumerate(checkpoint_indices_TU):
        classifier_name = dataset_name + "_" + tissue + "_" + "LogisticRegressionClassifier_tuVI_" + likelihoods_TU[i] + "_epochs_" + str(checkpoint_idx) + "_checkpoint.pth"
        classifier = LogisticRegressionClassifier(
            input_dim=10,
            num_classes=len(cell_types),
            device=device
        )
        classifier.load_state_dict(torch.load(path2tuvi_classifiers + classifier_name, map_location=device)["model_state_dict"])
        classifier.to(device)
        classifiers.append(classifier)

    # Create a dataset from the AnnData objects for evaluation
    if tissue != "All":
        adata_eval = (
            adata[0][adata[0].obs["tissue"] == tissue].copy(),
            adata[1][adata[1].obs["tissue"] == tissue].copy()
        )
    else:
        adata_eval = (adata[0].copy(), adata[1].copy())

    # Create evaluation dataset for gene expression data and transcript usage data
    eval_datasets_list = []

    for i, likelihood in enumerate(likelihoods_GE):
        scGEVI_embeddings = adata_eval[0].obsm[likelihood + "_latent_mean"]
        scGEVI_ontology = adata_eval[0].obs["cell_ontology_class"].to_numpy()
        scGEVI_tissue = adata_eval[0].obs["tissue"].to_numpy()

        eval_datasets_list.append(
            VAEEmbeddingsCellTypeDataset(
                vae_embeddings=scGEVI_embeddings,
                ontology=scGEVI_ontology,
                tissue=scGEVI_tissue,
                cell_types=cell_types
            )
        )

    for i, likelihood in enumerate(likelihoods_TU):
        tuVI_embeddings = adata_eval[1].obsm[likelihood + "_latent_mean"]
        tuVI_ontology = adata_eval[1].obs["cell_ontology_class"].to_numpy()
        tuVI_tissue = adata_eval[1].obs["tissue"].to_numpy()

        eval_datasets_list.append(
            VAEEmbeddingsCellTypeDataset(
                vae_embeddings=tuVI_embeddings,
                ontology=tuVI_ontology,
                tissue=tuVI_tissue,
                cell_types=cell_types
            )
        )

    # Create an empty dictionary to store evaluation metrics
    likelihoods = likelihoods_GE + likelihoods_TU
    classifiers_eval_dict = {}
    batch_size = 64

    # Loop through the likelihoods to evaluate classifiers
    for i, likelihood in enumerate(likelihoods):
        # Extract the evaluation dataset
        eval_dataset = eval_datasets_list[i]

        # Create a DataLoader for the evaluation dataset
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        # Select classifier and set it to evaluation mode
        classifier = classifiers[i]
        classifier.to(device)
        classifier.eval()

        # Arrays to store probability of class one, probability of class rest, and true labels
        class_probs_one = np.zeros((len(eval_dataset), len(cell_types)))
        binary_one_vs_rest_classes = np.zeros((len(eval_dataset), len(cell_types)))
        target_true = np.zeros(len(eval_dataset))

        class_probs = np.zeros((len(eval_dataset), len(cell_types)))
        pred_classes = np.zeros(len(eval_dataset))
        true_classes = np.zeros(len(eval_dataset))

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):

                input_batch_emb, target_batch, _ = batch

                # Push input_batch and target_batch to device
                input_batch = input_batch_emb.to(classifier.device)
                target_batch = target_batch.to(classifier.device)

                logits = classifier.forward(input_batch)

                # Map logits through softmax to obtain class probabilities
                class_probs_batch = torch.nn.functional.softmax(logits, dim=1)
                pred_classes_batch = torch.argmax(class_probs_batch, dim=1)

                class_probs[i * batch_size:(i + 1) * batch_size] = class_probs_batch.cpu().numpy()
                # Arguments for accuracy over all classes
                pred_classes[i * batch_size:(i + 1) * batch_size] = pred_classes_batch.cpu().numpy()
                true_classes[i * batch_size:(i + 1) * batch_size] = target_batch.cpu().numpy()

                # One vs rest
                for j, class_index in enumerate(range(len(cell_types))):
                    class_probs_one_batch = class_probs_batch[:, class_index].cpu().numpy()
                    class_probs_one[i * batch_size:(i + 1) * batch_size, j] = class_probs_one_batch
                    binary_one_vs_rest_classes[i * batch_size:(i + 1) * batch_size, j] = (
                                target_batch.cpu().numpy() == class_index).astype(int)

        # Accuracy
        accuracy = (true_classes == pred_classes).sum() / len(true_classes)
        # False positive and true positive rate
        roc_auc_dict = {}

        for class_index, cell_type in enumerate(cell_types):
            # One vs rest
            fpr, tpr, _ = roc_curve(binary_one_vs_rest_classes[:, class_index], class_probs_one[:, class_index])
            accuracy_one_vs_rest = (binary_one_vs_rest_classes[:, class_index] == (class_probs_one[:, class_index] > 0.5).astype(int)).sum() / len(binary_one_vs_rest_classes[:, class_index])
            auroc = auc(fpr, tpr)
            f1 = f1_score(binary_one_vs_rest_classes[:, class_index],
                          (class_probs_one[:, class_index] > 0.5).astype(int))
            roc_auc_dict[cell_type] = {
                "fpr": fpr,
                "tpr": tpr,
                "auroc": auroc,
                "f1": f1,
                "accuracy": accuracy_one_vs_rest,
                "num_samples": binary_one_vs_rest_classes[:, class_index].sum()
            }

        # Calculate macro and weighted classification metrics
        auroc_values = np.array([metrics["auroc"] for metrics in roc_auc_dict.values() if "auroc" in metrics])
        f1_values = np.array([metrics["f1"] for metrics in roc_auc_dict.values() if "f1" in metrics])
        num_samples_per_class = np.array([metrics["num_samples"] for metrics in roc_auc_dict.values()])
        macro_f1 = f1_values.mean()
        weighted_f1 = (f1_values * num_samples_per_class / num_samples_per_class.sum()).sum()
        auroc_values = auroc_values[~np.isnan(auroc_values)]
        macro_auroc = auroc_values.mean()
        weighted_auroc = (auroc_values * num_samples_per_class / num_samples_per_class.sum()).sum()

        classifiers_eval_dict[likelihood] = {
            "accuracy": accuracy,
            "macro_auroc": macro_auroc,
            "weighted_auroc": weighted_auroc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_roc_auc": roc_auc_dict
        }

    return classifiers_eval_dict


def evaluate_cell_type_classifiers_trvi(
        adata: Tuple[AnnData, AnnData],
        tissue: str,
        checkpoint_indices: List[int],
        likelihoods: List[str],
        emb_types: List[str],
        dataset_name: str = "tabulaMuris",
        device: str = "cuda"
) -> tuple[tuple[AnnData, AnnData], dict[Any, Any]] | None:
    r"""
    Given a tuple of two AnnData objects (gene expression and transcript usage) that have the latent embeddings already
    inferred and stored in .obsm, a chosen tissue type, a list of likelihoods, and a list of embedding types, as well
    as a list of checkpoint indices, evaluate the classifiers for each embedding type. The evaluation should be done on
    the test set.The evaluation metrics are stored for each classifier in a dictionary which is returned. The predicted
    cell types are added to the .obs of the AnnData objects which are also returned.

    :param adata: Tuple of two AnnData objects with adata[0] containing gene expression and adata[1] containing transcript usage data
    :param tissue: The tissue type to filter the data by
    :param checkpoint_indices: List of indices of the checkpoints to evaluate
    :param likelihoods: List of strings of likelihoods where likelihoods[0] is for gene expression and likelihoods[1] for transcript usage e.g. ["ZINB", "ZIDM"]
    :param emb_types: List of strings of embedding types to use for classification, e.g. ["private_1", "private_2", "shared_uni_1", "shared_uni_2", "shared"]
    :param dataset_name: The name of the dataset, e.g. "tabulaMuris"
    :param device: The device to use for evaluation, e.g. "cuda" or "cpu"
    :return: Tuple of two AnnData objects with predicted cell types in .obs and a dictionary with evaluation metrics
    """

    # Create an empty list that will store the classifiers
    classifiers = []
    path2models = "./models/TRVI/classifiers/"

    """
    # Extract the groud truth cell types for the dataset
    if tissue == "All":
        cell_types = TABULA_MURIS_CELL_TYPES
    else:
        cell_types = tuple(TABULA_MURIS_TISSUE_CELL_DICTIONARY[tissue])
    """

    # Create a dataset from the AnnData objects for evaluation
    if tissue != "All":
        adata_eval = (
            adata[0][adata[0].obs["tissue"] == tissue].copy(),
            adata[1][adata[1].obs["tissue"] == tissue].copy()
        )

        cell_types = tuple(np.unique(adata_eval[0].obs["cell_ontology_class"].to_numpy()))
    else:
        adata_eval = (adata[0].copy(), adata[1].copy())

        cell_types = tuple(np.unique(adata_eval[0].obs["cell_ontology_class"].to_numpy()))

    eval_dataset = create_cell_type_classification_dataset(
        adata_eval,
        likelihoods,
        cell_types
    )

    # Load classifiers from the checkpoints and append them to the list
    for i, checkpoint_idx in enumerate(checkpoint_indices):
        classifier_name = dataset_name + "_" + tissue + "_" + "LogisticRegressionClassifier_" + emb_types[
            i] + "_epochs_" + str(checkpoint_idx) + "_checkpoint.pth"
        classifier = LogisticRegressionClassifier(
            input_dim=10,
            num_classes=len(cell_types),
            device=device
        )
        classifier.load_state_dict(torch.load(path2models + classifier_name, map_location=device)["model_state_dict"])
        classifier.to(device)
        classifiers.append(classifier)

    # Create a DataLoader for the evaluation dataset
    batch_size = 64
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Create an empty dictionary to store evaluation metrics
    classifiers_eval_dict = {}

    # Loop through the embedding types and evaluate classifiers
    for i, emb_type in enumerate(emb_types):
        # Choose embedding indices
        if emb_type == "private_1":
            embedding_indices = [0]  # private embedding 1
        elif emb_type == "private_2":
            embedding_indices = [1]  # private embedding 2
        elif emb_type == "shared_uni_1":
            embedding_indices = [2]  # shared uni-modal embedding 1
        elif emb_type == "shared_uni_2":
            embedding_indices = [3]  # shared uni-modal embedding 2
        elif emb_type == "shared":
            embedding_indices = [4]  # shared embedding
        else:
            raise ValueError("Invalid embedding type")

        classifiers_eval_dict[emb_type] = {}
        classifier = classifiers[i]

        # Set the classifier to evaluation mode
        classifier.to(device)
        classifier.eval()

        # Arrays to store probabilitiy of class one, probablitiy of class rest, and true labels
        class_probs_one = np.zeros((len(eval_dataset), len(cell_types)))
        binary_one_vs_rest_classes = np.zeros((len(eval_dataset), len(cell_types)))
        target_true = np.zeros(len(eval_dataset))

        class_probs = np.zeros((len(eval_dataset), len(cell_types)))
        pred_classes = np.zeros(len(eval_dataset))
        true_classes = np.zeros(len(eval_dataset))

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):

                input_private_emb_1, input_private_emb_2, shared_unimodal_emb_1, shared_unimodal_emb_2, shared_emb, target_batch, _ = batch

                input_batch_embs = (input_private_emb_1, input_private_emb_2, shared_unimodal_emb_1,
                                    shared_unimodal_emb_2, shared_emb)

                if len(embedding_indices) == 1:
                    input_batch = [input_batch_embs[i].to(classifier.device) for i in embedding_indices][0]
                else:
                    input_batch = torch.cat([input_batch_embs[i].to(classifier.device) for i in embedding_indices])

                # Push input_batch and target_batch to device
                input_batch = input_batch.to(classifier.device)
                target_batch = target_batch.to(classifier.device)

                logits = classifier.forward(input_batch)

                # Map logits through softmax to obtain class probabilities
                class_probs_batch = torch.nn.functional.softmax(logits, dim=1)
                pred_classes_batch = torch.argmax(class_probs_batch, dim=1)

                class_probs[i * batch_size:(i + 1) * batch_size] = class_probs_batch.cpu().numpy()

                # Arguments for accuracy over all classes
                pred_classes[i * batch_size:(i + 1) * batch_size] = pred_classes_batch.cpu().numpy()
                true_classes[i * batch_size:(i + 1) * batch_size] = target_batch.cpu().numpy()

                # One vs rest
                for j, class_index in enumerate(range(len(cell_types))):
                    class_probs_one_batch = class_probs_batch[:, class_index].cpu().numpy()

                    class_probs_one[i * batch_size:(i + 1) * batch_size, j] = class_probs_one_batch
                    binary_one_vs_rest_classes[i * batch_size:(i + 1) * batch_size, j] = (
                                target_batch.cpu().numpy() == class_index).astype(int)

        # Accuracy
        accuracy = (true_classes == pred_classes).sum() / len(true_classes)

        # False positive and true positive rate
        roc_auc_dict = {}

        for class_index, cell_type in enumerate(cell_types):
            # One vs rest
            fpr, tpr, _ = roc_curve(binary_one_vs_rest_classes[:, class_index], class_probs_one[:, class_index])
            accuracy_one_vs_rest = (binary_one_vs_rest_classes[:, class_index] == (class_probs_one[:, class_index] > 0.5).astype(int)).sum() / len(binary_one_vs_rest_classes[:, class_index])
            auroc = auc(fpr, tpr)
            f1 = f1_score(binary_one_vs_rest_classes[:, class_index],
                          (class_probs_one[:, class_index] > 0.5).astype(int))
            roc_auc_dict[cell_type] = {
                "fpr": fpr,
                "tpr": tpr,
                "auroc": auroc,
                "f1": f1,
                "accuracy": accuracy_one_vs_rest,
                "num_samples": binary_one_vs_rest_classes[:, class_index].sum()
            }

        # Calculate macro and weighted classification metrics
        auroc_values = np.array([metrics["auroc"] for metrics in roc_auc_dict.values() if "auroc" in metrics])
        f1_values = np.array([metrics["f1"] for metrics in roc_auc_dict.values() if "f1" in metrics])
        num_samples_per_class = np.array([metrics["num_samples"] for metrics in roc_auc_dict.values()])

        macro_f1 = f1_values.mean()
        weighted_f1 = (f1_values * num_samples_per_class / num_samples_per_class.sum()).sum()

        auroc_values = auroc_values[~np.isnan(auroc_values)]
        macro_auroc = auroc_values.mean()
        num_samples_per_class_zero_excluded = num_samples_per_class[num_samples_per_class > 0]
        weighted_auroc = (
                auroc_values * num_samples_per_class_zero_excluded / num_samples_per_class_zero_excluded.sum()
        ).sum()

        # Store the evaluation metrics in the dictionary
        classifiers_eval_dict[emb_type]["accuracy"] = accuracy
        classifiers_eval_dict[emb_type]["macro_auroc"] = macro_auroc
        classifiers_eval_dict[emb_type]["weighted_auroc"] = weighted_auroc
        classifiers_eval_dict[emb_type]["macro_f1"] = macro_f1
        classifiers_eval_dict[emb_type]["weighted_f1"] = weighted_f1
        classifiers_eval_dict[emb_type]["roc_auc"] = roc_auc_dict

        # Add predicted cell types to the AnnData objects
        cell_type_list = list(cell_types)
        pred_named_classes = [cell_type_list[i] for i in pred_classes.astype(int)]
        adata_eval[0].obs[emb_type + "_pred_cell_ontology_class"] = pred_named_classes
        adata_eval[1].obs[emb_type + "_pred_cell_ontology_class"] = pred_named_classes

    return adata_eval, classifiers_eval_dict

def cell_type_classification_trvi_dataframe(
        classification_dict: dict[str, dict[str, dict[str, float]]],
        embedding_types: List[str]
) -> pd.DataFrame:
    r"""
    Given a dictionary outputted from the evaluate_cell_type_classifiers_trvi function and a list of embedding types,
    create a pandas DataFrame for reporting and plotting purposes.
    """
    accuracy_list = []
    weighted_auroc_list = []
    weighted_f1_list = []

    for emb_type in embedding_types:
        accuracy_list.append(classification_dict[emb_type]["accuracy"])
        weighted_auroc_list.append(classification_dict[emb_type]["weighted_auroc"])
        weighted_f1_list.append(classification_dict[emb_type]["weighted_f1"])

    classification_metrics_values = np.concatenate(
        (np.array(accuracy_list), np.array(weighted_auroc_list), np.array(weighted_f1_list)))
    classification_metrics_keys = np.sort(
        np.array([["accuracy", "weighted_auroc", "weighted_f1"] * len(embedding_types)])).squeeze()
    embedding_type_list = embedding_types * 3

    # Create dataframe
    classification_eval_df = pd.DataFrame({
        "Embedding": embedding_type_list,
        "Classification Metric": classification_metrics_keys,
        "Value": classification_metrics_values
    })

    return classification_eval_df

def cell_type_classification_dataframe(
        classification_dict,
        likelihoods: List[str]
) -> pd.DataFrame:
    r"""
    Given a dictionary from scGEVI or tuVI
    """
    accuracy_list = []
    weighted_auroc_list = []
    weighted_f1_list = []

    for likelihood in likelihoods:
        accuracy_list.append(classification_dict[likelihood]["accuracy"])
        weighted_auroc_list.append(classification_dict[likelihood]["weighted_auroc"])
        weighted_f1_list.append(classification_dict[likelihood]["weighted_f1"])

    classification_metrics_values = np.concatenate(
        (np.array(accuracy_list), np.array(weighted_auroc_list), np.array(weighted_f1_list)))
    classification_metrics_keys = np.sort(np.array([["ACC", "AUROC", "F1"] * len(likelihoods)])).squeeze()
    likelihoods_list = likelihoods * 3

    # Create dataframe
    classification_eval_df = pd.DataFrame({
        "Likelihood": likelihoods_list,
        "Classification Metric": classification_metrics_keys,
        "Value": classification_metrics_values
    })

    return classification_eval_df

def evaluate_cell_type_classifier_marker_gene(
        adata: AnnData,
        cell_type: str,
        tissue: str,
        checkpoint_index: int,
        marker_genes: List[str],
        dataset_name: str = "tabulaMuris",
        device: str = "cuda"
) -> Tuple[dict[str, dict[str, float]], np.ndarray]:
    r"""
    Based on an AnnData object where the vars have already been filtered to contain only the marker genes for the cell
    type and tissue given, a dataset of type MarkerGeneDataset is created and passed to a dataloader. Then, a trained
    binary cell type classifier is loaded from the checkpoint, and the dataset is evaluated. The evaluation metrics are
    returned as a list of dictionaries, and the predicted cell types are returned as a numpy array.

    :param adata: An AnnData object with the marker genes for the cell type and tissue given
    :param cell_type: The cell type to evaluate the classifier for
    :param tissue: The tissue type to filter the data by
    :param checkpoint_index: The index of the checkpoint to evaluate
    :param marker_genes: A list of marker genes for the cell type, if None, the marker genes are taken from the adata object
    :param dataset_name: The name of the dataset, e.g. "tabulaMuris"
    :param device: The device to use for evaluation, e.g. "cuda" or "cpu"

    :return: A tuple containing a list of dictionaries with evaluation metrics, false positive rate, true positive rate,
             and predicted cell types as a numpy array.
    """

    # Check if the marker genes given match the ones in the adata object
    assert np.all(adata.var_names.to_numpy() == np.array(marker_genes)), "The marker genes in the adata object do not match the ones given."

    # Create a dataset from the AnnData object for evaluation
    gene_expression_matrix = torch.from_numpy(adata.X.toarray())
    cell_ontology = adata.obs["cell_ontology_class"].to_numpy()

    dataset = MarkerGeneCellTypeDataset(
        gene_expression_matrix,
        cell_ontology,
        cell_type,
    )
    # Create a DataLoader for the evaluation dataset
    batch_size = 64
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load the classifier from the checkpoint
    model_name = dataset_name + "_" + tissue + "_" + cell_type.replace(" ", "_") + "_" + "MarkerGeneLogisticRegressionClassifier" + "_epochs_" + str(checkpoint_index) +"_checkpoint.pth"
    model = LogisticRegressionClassifier(
        input_dim=len(marker_genes),
        num_classes=1,
        device=device
    )
    model.load_state_dict(torch.load("./models/" + model_name, map_location=device)["model_state_dict"])
    model.to(device)

    # Create zeros arrays to store the predicted probabilities and true labels
    pred_classes = torch.zeros(len(dataset))
    true_classes = torch.zeros(len(dataset))
    class_probs = torch.zeros(len(dataset))

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation and reduce memory usage
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            input_batch, target_batch = batch

            # Push input_batch and target_batch to device
            input_batch = input_batch.to(model.device)
            target_batch = target_batch.to(model.device).reshape(-1, 1)  # Reshape target to match the model output

            # Forward pass through the model
            logits = model.forward(input_batch)

            # Map logits through sigmoid to obtain class probabilities
            class_probs_batch = torch.sigmoid(logits).squeeze()

            # Get predicted classes (0 or 1)
            pred_classes_batch = (class_probs_batch > 0.5).float()

            # Store the predicted classes and true labels
            pred_classes[i * batch_size:(i + 1) * batch_size] = pred_classes_batch.cpu()
            true_classes[i * batch_size:(i + 1) * batch_size] = target_batch.squeeze().cpu()
            class_probs[i * batch_size:(i + 1) * batch_size] = class_probs_batch

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_classes, pred_classes)
    f1 = f1_score(true_classes, pred_classes)
    fpr, tpr, _ = roc_curve(true_classes, class_probs)
    auroc_score = roc_auc_score(true_classes, pred_classes)

    pred_cell_types = np.where(pred_classes.cpu().numpy() == 1, cell_type, "other")

    classification_dict = {
        cell_type: {
            "fpr": fpr,
            "tpr": tpr,
            "auroc": auroc_score,
            "f1": f1,
            "accuracy": accuracy,
        }
    }
    """
    gene_marker_eval_data = [{
        "Embedding": "gene_markers",
        "Classification Metric": "accuracy",
        "Value": accuracy,
    }, {
        "Embedding": "gene_markers",
        "Classification Metric": "f1",
        "Value": f1,
    }, {
        "Embedding": "gene_markers",
        "Classification Metric": "auroc",
        "Value": auroc_score
    }
    ]

    return gene_marker_eval_data, fpr, tpr, pred_cell_types
    """
    return classification_dict, pred_cell_types

def evaluate_marker_gene_cell_type_classification(
        dataset_name: str,
        tissue: str,
        cell_type: str,
        checkpoint_index: int,
        data_partition: str = "test",
        device: str = "cuda"
) -> Tuple[dict[str, dict[str, float]], np.ndarray]:

    if dataset_name == "tabulaMuris":
        path2data = "./data/tabulaMuris/"
    else:
        raise ValueError("Unsupported dataset name. Currently only 'tabulaMuris' is supported.")

    # Load gene expression training data
    adata_train = ad.read_h5ad(path2data + "adata_GE_" + "train" + "_preprocessed.h5ad")
    adata_train.obs["data_partition"] = "train"

    # Load gene expression validation data
    adata_val = ad.read_h5ad(path2data + "adata_GE_" + "val" + "_preprocessed.h5ad")
    adata_val.obs["data_partition"] = "val"

    # Load gene expression test data
    adata_test = ad.read_h5ad(path2data + "adata_GE_" + "test" + "_preprocessed.h5ad")
    adata_test.obs["data_partition"] = "test"

    adata = ad.concat([adata_train, adata_val, adata_test], axis=0, join="outer", merge="same")

    if cell_type == "monocyte":
        marker_genes = ["Csf1r", "Ctsg", "Mpo", "Cd14", "Itgam", "Fcgr1", "Cd33"]
        print("Following marker genes of monocytes are employed: " + str(marker_genes))
    elif cell_type == "endothelial cell of coronary artery":
        marker_genes = ["Kdr", "Vwf", "Pecam1", "Cdh5"]
        print("Following marker genes of endothelial cells of coronary artery are employed: " + str(marker_genes))
    elif cell_type == "endocardial cell":
        marker_genes = ["Kdr", "Vwf", "Pecam1", "Cdh5", "Tie1", "Nfatc1", "Npr3", "Gata4", "Gata5", "Tbx5", "Tbx18",
                        "Tbx20", "Nkx2-5", "Bmp10"]
        print("Following marker genes of endocardial cells are employed: " + str(marker_genes))
    elif cell_type == "fibroblast of cardiac tissue":
        marker_genes = ["Vim", "Fn1", "Col1a1", "Col1a2", "Col3a1"]
        print("Following marker genes of fibroblast of cardiac tissue are employed: " + str(marker_genes))
    elif cell_type == "smooth muscle cell":
        marker_genes = ["Acta2", "Myh11", "Cnn1", "Tagln", "Cald1"]
        print("Following marker genes of smooth muscle cells are employed: " + str(marker_genes))
    else:
        raise ValueError("Invalid cell type")

    # Reduce gene expression features to marker genes
    adata = adata[:, marker_genes].copy()

    # Filter to test data
    adata_gene_marker_eval = adata[
        (adata.obs["data_partition"] == data_partition)
        & (adata.obs["tissue"] == tissue)
        ]

    classification_dict_mg, pred_cell_types = evaluate_cell_type_classifier_marker_gene(
        adata=adata_gene_marker_eval,
        cell_type=cell_type,
        tissue=tissue,
        checkpoint_index=checkpoint_index,
        marker_genes=marker_genes,
        dataset_name=dataset_name,
        device=device
    )

    return classification_dict_mg, pred_cell_types

def create_cell_type_classification_dataset(
        adata: Tuple[AnnData, AnnData],
        likelihoods: List[str],
        cell_types: Tuple[str]=TABULA_MURIS_CELL_TYPES
) -> EmbeddingCellTypeDataset:

    # Extract ontology and tissue from the first AnnData object
    ontology = adata[0].obs["cell_ontology_class"].to_numpy()
    tissue = adata[0].obs["tissue"].to_numpy()

    # Extract embeddings, ontology and tissue
    private_embeddings_1 = torch.from_numpy(adata[0].obsm[likelihoods[0] + "_private_latent_mean"])
    shared_unimodal_embeddings_1 = torch.from_numpy(adata[0].obsm[likelihoods[0] + "_shared_latent_mean"])

    private_embeddings_2 = torch.from_numpy(adata[1].obsm[likelihoods[1] + "_private_latent_mean"])
    shared_unimodal_embeddings_2 = torch.from_numpy(adata[1].obsm[likelihoods[1] + "_shared_latent_mean"])

    shared_embeddings = torch.from_numpy(adata[0].obsm[likelihoods[0] + "_" + likelihoods[1] + "_shared_latent_mean"])

    dataset = EmbeddingCellTypeDataset(
        (private_embeddings_1, private_embeddings_2),
        (shared_unimodal_embeddings_1, shared_unimodal_embeddings_2),
        shared_embeddings,
        ontology,
        tissue,
        cell_types,
        None,
        None  # No input transform since target must be integer encoded for classification
    )

    return dataset

def calculate_classifier_roc_auc(
        classifier,
        dataset: EmbeddingCellTypeDataset,
        class_index: int,
        embedding_type: str,
        device: str = "cpu",
        batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Given a classifier, a dataset of type EmbeddingCellTypeDataset, and the index of the class considered as positive in
    a one vs all scenario, create a Dataloader, and then iterate over the dataset to compute, the logits, the true
    labels, and the predicted class probabilities in a one vs all fashion. The true labels and the class probabilities
    are then used to calculate the false positive rate (FPR), true positive rate (TPR), and the area under the ROC curve
    (AUC) for each class. The function returns the FPR, TPR, and AUC as numpy arrays.

    :param classifier: A trained classifier
    :param dataset: An instance of EmbeddingCellTypeDataset
    :param class_index: The index of the class considered as positive in a one vs all scenario
    :param embedding_type: The type of embedding to use for classification, e.g. "private_1", "private_2", "shared_uni_1", "shared_uni_2", "shared", etc.
    :param device: The device to use for computation, e.g. "cpu" or "cuda"
    :param batch_size: The batch size for the DataLoader
    :return: A tuple of numpy arrays (FPR, TPR, AUC)
    """

    # Create a DataLoader from the given dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Arrays to store probabilitiy of class one, probablitiy of class rest, and true labels
    class_probs_one = np.zeros(len(dataset))
    class_probs_rest = np.zeros(len(dataset))
    target_true = np.zeros(len(dataset))

    # Set the classifier to evaluation mode
    classifier.to(device)
    classifier.eval()

    # Select the embedding type
    # Choose embedding indices
    if embedding_type == "private_1":
        embedding_indices = [0]  # private embedding 1
    elif embedding_type == "private_2":
        embedding_indices = [1]  # private embedding 2
    elif embedding_type == "shared_uni_1":
        embedding_indices = [2]  # shared uni-modal embedding 1
    elif embedding_type == "shared_uni_2":
        embedding_indices = [3]  # shared uni-modal embedding 2
    elif embedding_type == "shared":
        embedding_indices = [4]  # shared embedding
    else:
        raise ValueError("Invalid embedding type")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            input_private_emb_1, input_private_emb_2, shared_unimodal_emb_1, shared_unimodal_emb_2, shared_emb, target_batch, _ = batch

            input_batch_embs = (input_private_emb_1, input_private_emb_2, shared_unimodal_emb_1,
                                shared_unimodal_emb_2, shared_emb)

            if len(embedding_indices) == 1:
                input_batch = [input_batch_embs[i] for i in embedding_indices][0]
            else:
                input_batch = torch.cat([input_batch_embs[i] for i in embedding_indices])

            # Push input_batch and target_batch to device
            input_batch = input_batch.to(classifier.device)
            target_batch = target_batch.to(classifier.device)

            logits = classifier.forward(input_batch)

            # Map logits through softmax to obtain class probabilities
            class_probs = torch.nn.functional.softmax(logits, dim=1)

            # One vs rest
            class_probs_one_batch = class_probs[:, class_index].cpu().numpy()
            class_probs_rest_batch = 1 - class_probs_one_batch

            class_probs_one[i * batch_size:(i + 1) * batch_size] = class_probs_one_batch
            class_probs_rest[i * batch_size:(i + 1) * batch_size] = class_probs_rest_batch
            target_true[i * batch_size:(i + 1) * batch_size] = target_batch[:, class_index].cpu().numpy()

    # Calculate false positive rate, true positive rate, and area under the ROC curve
    fpr, tpr, _ = roc_curve(target_true, class_probs_one)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def evaluate_cell_type_classification(checkpoint_list: List[int], dataset: EmbeddingCellTypeDataset, cell_type: str, dataset_name: str, device="cpu", cell_ground_truth=TABULA_MURIS_CELL_TYPES) -> Dict[str, Dict[str, np.ndarray]]:
    r"""
    Given a list of checkpoint indices of the classifiers based on private_1, private_2, shared_uni_1, shared_uni_2,
    and shared embeddings, a dataset of type EmbeddingCellTypeDataset, and a cell type, calculate the ROC AUC for each
    checkpoint and return a dictionary containing the FPR, TPR, and AUC for each embedding type.

    :param checkpoint_list: List of checkpoint indices
    :param dataset: An instance of EmbeddingCellTypeDataset
    :param cell_type: The cell type to evaluate
    :param dataset_name: The name of the dataset, used to load the correct model checkpoints
    :param device: The device to use for computation, default is "cpu"
    :param cell_ground_truth: The ground truth cell types, default is TABULA_MURIS_CELL_TYPES
    """
    class_to_int = {class_label: i for i, class_label in enumerate(cell_ground_truth)}
    class_index = class_to_int[cell_type]

    embedding_types = ["private_1", "private_2", "shared_uni_1", "shared_uni_2", "shared"]
    path2models = "./models/"

    # Private 1
    classifier_checkpoint = "LogisticRegressionClassifier_" + embedding_types[0] + "_epochs_" + str(
        checkpoint_list[0]) + "_checkpoint.pth"
    classifier_private_1 = LogisticRegressionClassifier(
        input_dim=10,
        num_classes=len(TABULA_MURIS_CELL_TYPES),
        device=device
    )
    checkpoint = torch.load(path2models + dataset_name + "_" + classifier_checkpoint)
    classifier_private_1.load_state_dict(checkpoint["model_state_dict"])

    # Private 2
    classifier_checkpoint = "LogisticRegressionClassifier_" + embedding_types[1] + "_epochs_" + str(
        checkpoint_list[1]) + "_checkpoint.pth"
    classifier_private_2 = LogisticRegressionClassifier(
        input_dim=10,
        num_classes=len(TABULA_MURIS_CELL_TYPES),
        device=device
    )
    checkpoint = torch.load(path2models + dataset_name + "_" + classifier_checkpoint)
    classifier_private_2.load_state_dict(checkpoint["model_state_dict"])

    # Shared uni-modal 1
    classifier_checkpoint = "LogisticRegressionClassifier_" + embedding_types[2] + "_epochs_" + str(
        checkpoint_list[2]) + "_checkpoint.pth"
    classifier_shared_uni_1 = LogisticRegressionClassifier(
        input_dim=10,
        num_classes=len(TABULA_MURIS_CELL_TYPES),
        device=device
    )
    checkpoint = torch.load(path2models + dataset_name + "_" + classifier_checkpoint)
    classifier_shared_uni_1.load_state_dict(checkpoint["model_state_dict"])

    # Shared uni-modal 2
    classifier_checkpoint = "LogisticRegressionClassifier_" + embedding_types[3] + "_epochs_" + str(
        checkpoint_list[3]) + "_checkpoint.pth"
    classifier_shared_uni_2 = LogisticRegressionClassifier(
        input_dim=10,
        num_classes=len(TABULA_MURIS_CELL_TYPES),
        device=device
    )
    checkpoint = torch.load(path2models + dataset_name + "_" + classifier_checkpoint)
    classifier_shared_uni_2.load_state_dict(checkpoint["model_state_dict"])

    # Shared
    classifier_checkpoint = "LogisticRegressionClassifier_" + embedding_types[4] + "_epochs_" + str(
        checkpoint_list[4]) + "_checkpoint.pth"
    classifier_shared = LogisticRegressionClassifier(
        input_dim=10,
        num_classes=len(TABULA_MURIS_CELL_TYPES),
        device=device
    )
    checkpoint = torch.load(path2models + dataset_name + "_" + classifier_checkpoint)
    classifier_shared.load_state_dict(checkpoint["model_state_dict"])

    classifiers = [classifier_private_1, classifier_private_2, classifier_shared_uni_1, classifier_shared_uni_2,
                   classifier_shared]

    classifier_eval_dict = dict.fromkeys(embedding_types)

    for i, classifier in enumerate(classifiers):
        fpr, tpr, roc_auc = calculate_classifier_roc_auc(
            classifier=classifier,
            dataset=dataset,
            class_index=class_index,
            embedding_type=embedding_types[i],
            device=device,
        )
        classifier_eval_dict[embedding_types[i]] = {"FPR": fpr, "TPR": tpr, "AUC": roc_auc}

    return classifier_eval_dict


def infer_latent_embeddings_VAE(model, dataset, batch_size: int = 128) -> torch.Tensor:
    r"""
    Given a latent space model and a dataset, infer the latent embeddings of the data

    :param model:
    :param dataset:
    :param batch_size:
    :return:

    Example:

    """
    # Create unshuffled datatloader from training dataset
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # Initialize latent embeddings as zero tensor
    latent_embeddings = torch.zeros(len(dataset), model.dim_latent)
    cell_ontology_list = np.zeros(len(dataset), dtype=object)
    tissue_list = np.zeros(len(dataset), dtype=object)

    for i, batch in enumerate(dataloader):
        pdb.set_trace()
        input_batch_GE, _, input_batch_TU_psi , cell_ontology, tissue = batch
        cell_ontology_list[i * batch_size:(i + 1) * batch_size] = cell_ontology
        tissue_list[i * batch_size:(i + 1) * batch_size] = tissue
        input_batch = (input_batch_GE, input_batch_TU_psi)

        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        latent_mean, _ = model.infer_variational_posterior(input_batch)

        latent_embeddings[i * batch_size:(i + 1) * batch_size, :] = latent_mean.detach()

    latent_embeddings = latent_embeddings.cpu()

    return latent_embeddings


def infer_latent_embeddings_MMVAEplus(
        model,
        dataset: GeneExpressionTranscriptUsageDataset,
        batch_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    r"""
    Given a GeneExpressionTranscriptUsageMMVAEplus, a GeneExpressionTranscriptUsageDataset, infer a tuple consisting
    of the shared latent embedding of gene expression and transcript usage, the private latent embedding of gene
    expression, and the private latent embedding of transcript usage.

    :param model:
    :param dataset:
    :param batch_size:
    :return:

    Example:

    """
    # Create unshuffled dataloader from training dataset
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # Initialize latent embeddings as zero tensor
    num_classes = len(dataset.cell_types)
    shared_latent_embeddings = torch.zeros(len(dataset), model.dim_latent_shared)
    private_latent_embeddings_1 = torch.zeros(len(dataset), model.dim_latent_1)
    private_latent_embeddings_2 = torch.zeros(len(dataset), model.dim_latent_2)
    cell_ontology_onehot = torch.zeros((len(dataset), num_classes))
    tissue_list = np.zeros(len(dataset), dtype=object)

    for i, batch in enumerate(dataloader):
        _, input_batch_1_levels, _, input_batch_2_psi, cell_ontology, tissue = batch
        # input_batch_1, input_batch_2, _, _ = batch

        cell_ontology_onehot[i * batch_size:(i + 1) * batch_size] = cell_ontology
        tissue_list[i * batch_size:(i + 1) * batch_size] = tissue

        if torch.cuda.is_available():
            input_batch_1 = input_batch_1_levels.cuda()
            input_batch_2_psi = input_batch_2_psi.cuda()
            input_batch = (input_batch_1, input_batch_2_psi)
        else:
            input_batch = (input_batch_1_levels, input_batch_2_psi)

        latent_embeddings_dict = model.infer_variational_posterior(input_batch)

        shared_latent_embeddings[i * batch_size:(i + 1) * batch_size, :] = latent_embeddings_dict[
            "Shared Mean"].detach()
        private_latent_embeddings_1[i * batch_size:(i + 1) * batch_size, :] = latent_embeddings_dict[
            "Private Mean GE"].detach()
        private_latent_embeddings_2[i * batch_size:(i + 1) * batch_size, :] = latent_embeddings_dict[
            "Private Mean TU"].detach()

    shared_latent_embeddings = shared_latent_embeddings.cpu()
    private_latent_embeddings_1 = private_latent_embeddings_1.cpu()
    private_latent_embeddings_2 = private_latent_embeddings_2.cpu()

    return shared_latent_embeddings, private_latent_embeddings_1, private_latent_embeddings_2, cell_ontology_onehot, tissue_list


def impute_data_MMVAEplus(
        model,
        adata: Tuple[AnnData, AnnData],
        dataset: GeneExpressionTranscriptUsageDataset,
        batch_size: int = 256
) -> Tuple[AnnData, AnnData]:
    r"""
    Given a GeneExpressionTranscriptUsageMMVAEplus, a dataset given in the format of a Tuple of two AnnData objects
    (gene expression and transcript usage data) and as GeneExpressionTranscriptUsageDataset, impute the data using the
    model. The function adds imputed data to the AnnData objects given and returns these edited objects as a Tuple.

    TO DO: Add support for cross-modal imputation, add perturbation to the imputed data, and try out with TRVI

    :param model: GeneExpressionTranscriptUsageMMVAEplus
    :param adata: Tuple of two AnnData objects (gene expression and transcript usage data)
    :param dataset: GeneExpressionTranscriptUsageDataset
    :param batch_size: Batch size for the dataloader
    :return: Tuple of two AnnData objects (imputed gene expression and transcript usage data)
    :rtype: Tuple[AnnData, AnnData]

    Example:
    >>> model = GeneExpressionTranscriptUsageMMVAEplus()
    >>> adata1 = AnnData(X1, var=var1)
    >>> adata2 = AnnData(X2, var=var2)
    >>> adata = (adata1, adata2)
    >>> dataset = GeneExpressionTranscriptUsageDataset(adata1,adata2,,,,
    >>> imputed_data = impute_data_MMVAEplus(model, adata, dataset)
    >>> print(imputed_data)
    >>> # Output:
    >>> # (AnnData object with n_obs × n_vars = 800 × 1000, AnnData object with n_obs × n_vars = 800 × 1000)
    """

    # Create unshuffled dataloader from training dataset
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # Initialize imputed data as zero tensor
    imputed_gene_expressions = torch.zeros(adata[0].shape)
    imputed_gene_expressions_cross_modal = torch.zeros(adata[0].shape)
    imputed_psi = torch.zeros(adata[1].shape)
    imputed_psi_cross_modal = torch.zeros(adata[1].shape)

    # Set model to evaluation mode
    model.eval()

    # Iterate over the dataloader
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x_1, _, x_2_psi, _, _ = batch

            if torch.cuda.is_available():
                x_1 = x_1.cuda()
                x_2_psi = x_2_psi.cuda()

            # Infer the variational posterior for gene expression and transcript usage data
            variational_posterior_dict_1 = model.variational_posterior(x_1, None, "Gene expression")
            variational_posterior_dict_2 = model.variational_posterior(x_2_psi, None, "Transcript usage")

            # Create auxiliary latent variable for cross-modal imputation
            variational_posterior_dict_1["Auxiliary latent variable"] = torch.zeros(
                (x_2_psi.shape[0], variational_posterior_dict_1["Auxiliary latent variance"].shape[-1])
            )
            variational_posterior_dict_2["Auxiliary latent variable"] = torch.zeros(
                (x_1.shape[0], variational_posterior_dict_2["Auxiliary latent variance"].shape[-1])
            )
            if torch.cuda.is_available():
                variational_posterior_dict_1["Auxiliary latent variable"] = variational_posterior_dict_1[
                    "Auxiliary latent variable"].cuda()
                variational_posterior_dict_2["Auxiliary latent variable"] = variational_posterior_dict_2[
                    "Auxiliary latent variable"].cuda()

            # Concatenate the latent means of the shared and private latent variables
            variational_posterior_dict_1["Concatenated latent variable"] = torch.cat(
                (variational_posterior_dict_1["Shared latent mean"],
                 variational_posterior_dict_1["Private latent mean"]),
                dim=-1
            )
            variational_posterior_dict_2["Concatenated latent variable"] = torch.cat(
                (variational_posterior_dict_2["Shared latent mean"],
                 variational_posterior_dict_2["Private latent mean"]),
                dim=-1
            )
            variational_posterior_dict_1["Cross-modal latent variable"] = torch.cat(
                (variational_posterior_dict_1["Shared latent mean"],
                 variational_posterior_dict_1["Auxiliary latent variable"]),
                dim=-1
            )

            variational_posterior_dict_2["Cross-modal latent variable"] = torch.cat(
                (variational_posterior_dict_2["Shared latent mean"],
                 variational_posterior_dict_2["Auxiliary latent variable"]),
                dim=-1
            )

            # Impute data using the generative model given the latent embeddings
            generative_model_dict_1 = model.generative_model(
                None,
                None,
                variational_posterior_dict_1["Concatenated latent variable"],
                variational_posterior_dict_1["Cross-modal latent variable"],
                "Gene expression"
            )

            generative_model_dict_2 = model.generative_model(
                None,
                None,
                variational_posterior_dict_2["Concatenated latent variable"],
                variational_posterior_dict_2["Cross-modal latent variable"],
                "Transcript usage"
            )

            # Store the imputed data
            imputed_gene_expressions[i * batch_size:(i + 1) * batch_size, :] = generative_model_dict_1[
                "Mean reconstruction GE"].detach()
            imputed_psi[i * batch_size:(i + 1) * batch_size, :] = generative_model_dict_2[
                "Mean reconstruction TU"].detach()

            # Store the imputed data for cross-modal imputation
            imputed_gene_expressions_cross_modal[i * batch_size:(i + 1) * batch_size, :] = generative_model_dict_2[
                "Mean reconstruction GE"].detach()
            imputed_psi_cross_modal[i * batch_size:(i + 1) * batch_size, :] = generative_model_dict_1[
                "Mean reconstruction TU"].detach()

    # Convert the imputed data to numpy arrays
    imputed_gene_expressions = imputed_gene_expressions.cpu().numpy()
    imputed_psi = imputed_psi.cpu().numpy()
    imputed_gene_expressions_cross_modal = imputed_gene_expressions_cross_modal.cpu().numpy()
    imputed_psi_cross_modal = imputed_psi_cross_modal.cpu().numpy()

    # Add the imputed data to the AnnData objects
    adata[0].layers["imputed"] = imputed_gene_expressions
    adata[1].layers["imputed"] = imputed_psi
    adata[0].layers["imputed_cross_modal"] = imputed_gene_expressions_cross_modal
    adata[1].layers["imputed_cross_modal"] = imputed_psi_cross_modal

    return adata


def initialize_weights(module) -> None:
    r"""

    :param module:
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')


def auxiliary_noise(latent_shape: Tuple[int, int], num_samples: int = 1) -> torch.Tensor:
    r"""
    Sample noise from zero mean, unit variance Gaussian :math:`\epsilon \sim \mathcal{N}(0,1)` distribution.
    Shape of the sample is determined by the batch size and the dimensionality of the latent variable

    :param latent_shape:
    :return:
    """

    if num_samples > 1:
        noise_samples = torch.distributions.Normal(0, 1).sample((num_samples, *latent_shape))
    else:
        noise_samples = torch.distributions.Normal(0, 1).sample(latent_shape)

    return noise_samples


def log_sum_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""
    To avoid numerical instabilities caused by underflow or overflow, the log sum exp trick saves the maximum value according to

    ..math::
        \begin{split}
            \mathrm{log} \, \sum_{k=1}^K \mathrm{exp}(b_k) &= \mathrm{log} \sum_{k=1}^K (\mathrm{exp}(b_{k-1} -B) \, \mathrm{exp}(B))\\
            &=B + \mathrm{log} \sum_{k=1}^K \mathrm{exp}(b_k -B)
        \end{split}

    where the largest :math:`b_k` is saved as :math:`B = \mathrm{max}_{k=1}^K b_k`. Here, the implementation here is only for two summands.

    :param a: First tensor
    :param b: Second tensor
    :return: Log sum exp of the two tensors
    :rtype: torch.Tensor

    Example:

    """

    max_val = torch.max(a, b)
    return max_val + torch.log(torch.exp(a - max_val) + torch.exp(b - max_val))


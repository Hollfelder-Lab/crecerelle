#!/usr/bin/env python3
import pdb

import matplotlib
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from importlib import resources
import matplotlib.gridspec as gridspec
from matplotlib_venn import venn2 # Not installed in PyCharm atm
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from types import ModuleType
import warnings

from collections.abc import Mapping, Sequence

import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.colors import to_rgba
import matplotlib.colorbar as mcolorbar
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


import os
import itertools

import anndata as ad
import scanpy as sc
import upsetplot
from anndata import AnnData
from umap import UMAP
import scib
from upsetplot import plot, from_contents

import re

import torch

from scipy.spatial.distance import pdist, squareform

import seaborn as sns
import matplotlib.colors as mcolors

from .utils import filter_min_cells_per_feature, filter_min_cells_per_intron_group, cell_type_classification_trvi_dataframe

def plot_customized_UMAP_coordinates(
        axes: matplotlib.axes.Axes,
        skip_axes: List[int] | None = None,
        x_bottom: float = 0.0,
        y_bottom: float = 0.0,
        length: float = 0.1,
        font_size: int = 16
) -> None:
    r"""
    Customises the UMAP coordinates of the given axes by removing ticks and adding arrows and labels

    :param axes: matplotlib.axes.Axes, the axes to be customised
    :param skip_axes: List of int, the indices of the axes to be skipped
    :param x_bottom: float, the x coordinate of the bottom left corner of the arrow
    :param y_bottom: float, the y coordinate of the bottom left corner of the arrow
    :param length: float, the length of the arrow
    :param font_size: int, the font size
    :return: None
    """

    if isinstance(axes, np.ndarray) or isinstance(axes, list):

        if skip_axes is not None:
            axes = axes.flatten()
            axes = np.delete(axes, skip_axes)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

            # X-axis arrow
            ax.arrow(x_bottom, y_bottom, length, 0,
                     head_width=0.01, head_length=0.01, fc='k', ec='k',
                     transform=ax.transAxes, clip_on=False)  # clip_on=False ensures it's not cut off by axes limits

            # Y-axis arrow
            ax.arrow(x_bottom, y_bottom, 0, length,
                     head_width=0.01, head_length=0.01, fc='k', ec='k',
                     transform=ax.transAxes, clip_on=False)

            # Labels for UMAP1 and UMAP2
            #ax.text(x_bottom + length +0.02, y_bottom - 0.02, 'UMAP1'
            ax.text(x_bottom - 0.02, y_bottom + length, 'UMAP1', # + 0.02
                    horizontalalignment='center', verticalalignment='top',
                    transform=ax.transAxes, fontsize=font_size)

            ax.text(x_bottom + length, y_bottom - 0.02, 'UMAP2',
                    horizontalalignment='right', verticalalignment='center', rotation=90,
                    transform=ax.transAxes, fontsize=font_size)
    else:
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_xlabel('')
        axes.set_ylabel('')

        # X-axis arrow
        axes.arrow(x_bottom, y_bottom, length, 0,
                 head_width=0.01, head_length=0.01, fc='k', ec='k',
                 transform=axes.transAxes, clip_on=False)  # clip_on=False ensures it's not cut off by axes limits

        # Y-axis arrow
        axes.arrow(x_bottom, y_bottom, 0, length,
                 head_width=0.01, head_length=0.01, fc='k', ec='k',
                 transform=axes.transAxes, clip_on=False)

        # Labels for UMAP1 and UMAP2
        axes.text(x_bottom + length / 2, y_bottom - 0.02, 'UMAP1',
                horizontalalignment='center', verticalalignment='top',
                transform=axes.transAxes, fontsize=16)

        axes.text(x_bottom - 0.02, y_bottom + length / 2, 'UMAP2',
                horizontalalignment='right', verticalalignment='center', rotation=90,
                transform=axes.transAxes, fontsize=16)

def plot_customized_legend(
        ax: matplotlib.axes.Axes,
        tissues: List[str],
        colormap: str = 'tab10'
) -> None:

    cmap_colors = plt.cm.get_cmap(colormap, 10)

    # Iterate through tissues and plot a "bullet" (marker) for each
    for i, tissue in enumerate(tissues):
        reversed_index = len(tissues) - 1 - i

        color = cmap_colors(reversed_index) # Get the color for the current index

        # Plot a dummy point to create a legend entry
        # We use a large marker size to make it look like a bullet
        ax.plot(0, i, marker='o', markersize=10, linestyle='', color=color,
                label=tissue)

    # Set up the plot to look like a legend
    ax.set_xlim(-0.5, 0.5) # Keep the x-axis narrow
    ax.set_ylim(-1, len(tissues)) # Adjust y-limits to fit all labels

    ax.set_axis_off() # Hide the axes (spines, ticks, labels)

    # Create the legend
    legend = ax.legend(loc='center left', frameon=False, fontsize=14,
                      bbox_to_anchor=(0.2, 0.5))

    for line in ax.lines:
        line.remove()

    # Now, hide the axes completely
    ax.set_axis_off()

    # Adjust spacing between legend entries (optional, but needed for visual alignment)
    for text in legend.get_texts():
        text.set_x(text.get_position()[0] + 0.5)


def plot_random_seed_comparison_VAE(
        adata: AnnData,
        observation_model: str,
        list_of_random_seeds: List[int],
        eval_metric_df: pd.DataFrame,
        eval_metric: str,
        display_box_plot: bool =False,
        save_fig: bool = True,
        **kwargs
) -> None:
    r"""
    Given an AnnData object containing the results of training a VAE model with different random seeds,
    plot comparison of reconstruction evaluation metric (e.g. "nll", "rmse") across different random seeds

    :param adata: AnnData object containing the results of training a VAE model with different random seeds
    :param observation_model: str, the observation model used in the VAE model (e.g. "ZINB", "NB", "Gaussian", "DM", "ZIDM", "ZANIDM")
    :param list_of_random_seeds: List of int, the list of random seeds used in training the VAE model
    :param eval_metric_df: pd.DataFrame, the dataframe containing the reconstruction evaluation metric for each seed
    :param eval_metric: str, the reconstruction evaluation metric used (e.g. "nll", "rmse")
    :param display_box_plot: bool, whether to display a box plot of the reconstruction evaluation metric
    :param save_fig: bool, whether to save the figure
    """
    # Max values (outliers) for each seed
    max_values = eval_metric_df.max(axis=0).to_numpy()

    # Mean and standard deviation of reconstruction evaluation metric of different seeds
    eval_metric_mean_std_df = pd.concat([eval_metric_df.mean(), eval_metric_df.std()], axis=1, join="inner")
    mean_col_name = "Mean of " + eval_metric.upper()
    std_col_name = "Std of " + eval_metric.upper()
    eval_metric_mean_std_df.columns = [mean_col_name, std_col_name]

    # Add columns for seeds
    eval_metric_mean_std_df["Seeds"] = list_of_random_seeds

    # Mean of means and std of stds
    mean_across_seeds = eval_metric_mean_std_df[mean_col_name].mean()
    std_across_seeds = eval_metric_mean_std_df[std_col_name].mean()

    means_seeds = eval_metric_mean_std_df[mean_col_name].to_numpy()
    stds_seeds = eval_metric_mean_std_df[std_col_name].to_numpy()

    # Best and worst seeds according to mean of evaluation metric
    best_seed = means_seeds.argmin()
    worst_seed = means_seeds.argmax()

    # Select seeds below mean of means and below std of stds
    seeds_below_mean = np.argwhere(means_seeds <= mean_across_seeds)
    seeds_below_std = np.argwhere(stds_seeds <= std_across_seeds)

    # Choose seeds present in both sets
    seeds_selected_mean_std = np.intersect1d(seeds_below_mean, seeds_below_std)

    # Check for outliers
    mean_max_values_selected = max_values[seeds_selected_mean_std].mean()
    seeds_selected_max_values = np.argwhere(max_values <= mean_max_values_selected)

    seeds_selected = np.intersect1d(seeds_selected_mean_std, seeds_selected_max_values)

    # Plot
    fig = plt.figure(figsize=(10, 12), dpi=300)
    gs = fig.add_gridspec(3, 2)

    # Mean of evaluation metric for each seed
    ax00 = fig.add_subplot(gs[0, 0])
    sns.barplot(data=eval_metric_mean_std_df, x="Seeds", y=mean_col_name, ax=ax00)
    ax00.axhline(y=mean_across_seeds, color='r', linestyle='--')
    # ax00.set_title("NLL mean")

    # Stdv of evaluation metric for each seed
    ax01 = fig.add_subplot(gs[0, 1])
    sns.barplot(data=eval_metric_mean_std_df, x="Seeds", y=std_col_name, ax=ax01)
    ax01.axhline(y=std_across_seeds, color='r', linestyle='--')
    # ax01.set_title("NLL standard deviation")

    # Plot boxplots of selected seeds spanning the second row
    if display_box_plot:
        ax_boxplot = fig.add_subplot(gs[1, :])
        sns.boxplot(
            data=eval_metric_df.iloc[:, seeds_selected],
            ax=ax_boxplot,
            flierprops={
                'marker': '.',  # Use a point/dot instead of a circle
                'markersize': 4,  # Control the size
                'markerfacecolor': 'gray',
                'markeredgecolor': 'none'  # Remove the outline for a cleaner 'dot' look
            }
        )
        ax_boxplot.set_xticklabels(seeds_selected)
        ax_boxplot.set_xlabel("Seeds")
        ax_boxplot.set_ylabel("NLL")

        ax_best = fig.add_subplot(gs[2, 0])
        ax_worst = fig.add_subplot(gs[2, 1])
    else:
        ax_best = fig.add_subplot(gs[1, 0])
        ax_worst = fig.add_subplot(gs[1, 1])

    # Plot UMAP of best seed
    torch.manual_seed(best_seed)
    np.random.seed(best_seed)

    #adata.obsm["X_umap"] = adata.obsm[observation_model + "_" + str(best_seed) + "_X_umap"].copy()
    embedding_best = observation_model + "_" + str(best_seed) + "_latent_mean"

    adata.obsm["X_umap"] = UMAP(n_components=2, random_state=best_seed).fit_transform(adata.obsm[embedding_best])
    sc.pl.umap(adata, color="tissue", ax=ax_best, legend_loc=None, show=False, frameon=False)
    # Title
    ax_best.set_title(" ")  # Best seed
    del adata.obsm["X_umap"]

    # Plot UMAP of worst seed
    torch.manual_seed(worst_seed)
    np.random.seed(worst_seed)

    #adata.obsm["X_umap"] = adata.obsm[observation_model + "_" + str(worst_seed) + "_X_umap"].copy()
    embedding_worst = observation_model + "_" + str(worst_seed) + "_latent_mean"
    adata.obsm["X_umap"] = UMAP(n_components=2, random_state=worst_seed).fit_transform(adata.obsm[embedding_worst])
    sc.pl.umap(adata, color="tissue", ax=ax_worst, legend_loc=None, show=False, frameon=False)
    ax_worst.set_title(" ")  # Worst Seed
    del adata.obsm["X_umap"]

    plot_customized_UMAP_coordinates(ax_best, length=1.0)
    plot_customized_UMAP_coordinates(ax_worst, length=1.0)

    alphabet = ["a", "b", "c", "d", "e"]
    # Titles
    if display_box_plot:
        ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')
        ax01.set_title('b', loc='left', fontsize=20, fontweight='bold')
        ax_boxplot.set_title('c', loc='left', fontsize=20, fontweight='bold')
        ax_best.set_title('d', loc='left', fontsize=20, fontweight='bold')
        ax_worst.set_title('e', loc='left', fontsize=20, fontweight='bold')
    else:
        ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')
        ax01.set_title('b', loc='left', fontsize=20, fontweight='bold')
        ax_best.set_title('c', loc='left', fontsize=20, fontweight='bold')
        ax_worst.set_title('d', loc='left', fontsize=20, fontweight='bold')

    print(f"The best seed is {best_seed} and the worst seed is {worst_seed}")

    plt.tight_layout()

    if save_fig:
        dataset_name = kwargs.get("dataset_name", "default")
        fig.savefig("./figures/" + dataset_name + "/random_seed_comparison_tuVI_" + observation_model + ".pdf", dpi=300,
                    bbox_inches='tight')
    plt.show()

def plot_random_seed_comparison_trvi(
        adata_tuple: Tuple[AnnData, AnnData],
        observation_models: List[str],
        list_of_random_seeds: List[int],
        eval_metric_dfs: Tuple[pd.DataFrame],
        eval_metric: str,
        display_weights: bool =False,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given a tuple of two Anndata objects (gene expression and transcript usage data), a list of random seeds, and a Tuple of two dataframes (gene expression and transcript usage) containing the evaluation metric for each seed, plot comparison of reconstruction evaluation metric (e.g. "nll", "rmse") across different random seeds for both modalities.
    """
    adata_1 = adata_tuple[0]
    adata_2 = adata_tuple[1]

    eval_metric_df_1 = eval_metric_dfs[0]
    eval_metric_df_2 = eval_metric_dfs[1]

    # Gene expression evaluation
    eval_metric_mean_std_df_1 = pd.concat([eval_metric_df_1.mean(), eval_metric_df_1.std()], axis=1, join="inner")
    mean_col_name_1 = "GE: Mean of " + eval_metric.upper()
    std_col_name_1 = "GE: Std of " + eval_metric.upper()
    eval_metric_mean_std_df_1.columns = [mean_col_name_1, std_col_name_1]

    # Add columns for seeds
    eval_metric_mean_std_df_1["Seeds"] = list_of_random_seeds

    # Mean of means and std of stds
    mean_across_seeds_1 = eval_metric_mean_std_df_1[mean_col_name_1].mean()
    std_across_seeds_1 = eval_metric_mean_std_df_1[std_col_name_1].mean()

    # Transcript usage evaluation
    eval_metric_mean_std_df_2 = pd.concat([eval_metric_df_2.mean(), eval_metric_df_2.std()], axis=1, join="inner")
    mean_col_name_2 = "TU: Mean of " + eval_metric.upper()
    std_col_name_2 = "TU: Std of " + eval_metric.upper()
    eval_metric_mean_std_df_2.columns = [mean_col_name_2, std_col_name_2]

    # Add columns for seeds
    eval_metric_mean_std_df_2["Seeds"] = list_of_random_seeds

    # Mean of means and std of stds
    mean_across_seeds_2 = eval_metric_mean_std_df_2[mean_col_name_2].mean()
    std_across_seeds_2 = eval_metric_mean_std_df_2[std_col_name_2].mean()


    # Best seed and worst seed selection
    means_seeds_1 = eval_metric_mean_std_df_1[mean_col_name_1].to_numpy()
    stds_seeds_1 = eval_metric_mean_std_df_1[std_col_name_1].to_numpy()

    means_seeds_2 = eval_metric_mean_std_df_2[mean_col_name_2].to_numpy()
    stds_seeds_2 = eval_metric_mean_std_df_2[std_col_name_2].to_numpy()

    seeds_below_mean_1 = np.argwhere(means_seeds_1 <= mean_across_seeds_1)
    seeds_below_std_1 = np.argwhere(stds_seeds_1 <= std_across_seeds_1)

    seeds_below_mean_2 = np.argwhere(means_seeds_2 <= mean_across_seeds_2)
    seeds_below_mean_2 = np.argwhere(stds_seeds_2 <= std_across_seeds_2)

    seeds_selected_1 = np.intersect1d(seeds_below_mean_1, seeds_below_std_1)
    seeds_selected_2 = np.intersect1d(seeds_below_mean_2, seeds_below_mean_2)

    seeds_selected = np.intersect1d(seeds_selected_1, seeds_selected_2)
    seeds_selected_mean_value_1 = means_seeds_1[seeds_selected]
    seeds_selected_mean_value_2 = means_seeds_2[seeds_selected]

    # Plot
    fig = plt.figure(figsize=(10, 12), dpi=300)
    gs = fig.add_gridspec(3, 2)

    # Gene expression: Mean of evaluation metric for each seed
    ax00 = fig.add_subplot(gs[0, 0])
    sns.barplot(data=eval_metric_mean_std_df_1, x="Seeds", y=mean_col_name_1, ax=ax00)
    ax00.axhline(y=mean_across_seeds_1, color='r', linestyle='--')
    # ax00.set_title("NLL mean")

    # Gene expression: Stdv of evaluation metric for each seed
    ax01 = fig.add_subplot(gs[0, 1])
    sns.barplot(data=eval_metric_mean_std_df_1, x="Seeds", y=std_col_name_1, ax=ax01)
    ax01.axhline(y=std_across_seeds_1, color='r', linestyle='--')
    # ax01.set_title("NLL standard deviation")

    # Transcript usage: Mean of evaluation metric for each seed
    ax10 = fig.add_subplot(gs[1, 0])
    sns.barplot(data=eval_metric_mean_std_df_2, x="Seeds", y=mean_col_name_2, ax=ax10)
    ax10.axhline(y=mean_across_seeds_2, color='r', linestyle='--')

    ax11 = fig.add_subplot(gs[1, 1])
    sns.barplot(data=eval_metric_mean_std_df_2, x="Seeds", y=std_col_name_2, ax=ax11)
    ax11.axhline(y=std_across_seeds_2, color='r', linestyle='--')

    ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')
    ax01.set_title('b', loc='left', fontsize=20, fontweight='bold')
    ax10.set_title('c', loc='left', fontsize=20, fontweight='bold')
    ax11.set_title('d', loc='left', fontsize=20, fontweight='bold')

    plt.tight_layout()

    print(f"The best seeds are: {seeds_selected} \n"
          f"The corresponding mean evaluation metric values for gene expression are: {seeds_selected_mean_value_1} \n"
          f"The corresponding mean evaluation metric values for transcript usage are: {seeds_selected_mean_value_2}"
          )

    if save_fig:
        dataset_name = kwargs.get("dataset_name", "default")
        fig.savefig("./figures/" + dataset_name + "/random_seed_comparison_trvi_" + observation_models[0] + "_" + observation_models[1] + ".pdf",
                    dpi=300,
                    bbox_inches='tight')
    plt.show()






def plot_cell_embeddings_umaps_trvi(
        adata_objects: Tuple[ad.AnnData, ad.AnnData],
        tissue_list: List[str] | None,
        cell_list: List[str] | None,
        likelihood_keys: List[str] = ["ZINB", "ZIDM"],
        dataset_name: str = "tabulaMuris",
        seed: int = 0,
        min_cells: int = 50, # minimum number of cells for cell type to be included otherwise filtered out
        save_fig: bool = True,
):
    r"""
    Given two AnnData objects containing the results of training a TRVI on gene expression and transcript usage data,
    plot UMAPs of the shared latent space of each modality and the joint latent space, as well as a bar plot of the
    modality-relevance weights. The UMAPs are colored by cell type and the bar plot shows the mean modality-relevance
    weight for each cell type. The figure panel is saved if save_fig is True.

    :param adata_objects:
    :param tissue_list:
    :param cell_list:
    :param likelihood_keys:
    :param dataset_name:
    :param seed:
    :param min_cells:
    :param save_fig:
    """
    import matplotlib.lines as mlines  # Needed for custom legend

    adata_1, adata_2 = adata_objects
    shared_embedding_1 = likelihood_keys[0] + "_shared_latent_mean"
    shared_embedding_2 = likelihood_keys[1] + "_shared_latent_mean"
    joint_embedding = likelihood_keys[0] + "_" + likelihood_keys[1] + "_shared_latent_mean"

    if tissue_list is not None:
        num_tissues = len(tissue_list)
    else:
        num_tissues = 1

    fig, axs = plt.subplots(num_tissues, 4, figsize=(20, 4 * num_tissues))

    # Ensure axs is always 2D for consistent indexing
    if num_tissues == 1:
        axs = axs.reshape(1, -1)

    # Subfigures numbering
    subfigures_alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]

    for i in range(num_tissues):
        if min_cells > 0:
            if tissue_list is not None:
                tissue = tissue_list[i]
                if cell_list is not None:
                    # Filter AnnData to tissues AND cells listed
                    adata_1_tissue = adata_1[adata_1.obs["tissue"] == tissue].copy()
                    adata_1_tissue = adata_1_tissue[adata_1_tissue.obs["cell_ontology_class"].isin(cell_list)].copy()

                    adata_2_tissue = adata_2[adata_2.obs["tissue"] == tissue].copy()
                    adata_2_tissue = adata_2_tissue[adata_2_tissue.obs["cell_ontology_class"].isin(cell_list)].copy()

                else:
                    # Filter AnnData to tissues listed
                    adata_1_tissue = adata_1[adata_1.obs["tissue"] == tissue].copy()
                    cell_types, cell_counts = np.unique(adata_1_tissue.obs["cell_ontology_class"].to_numpy(),
                                                        return_counts=True)
                    filtered_cell_types = cell_types[np.argwhere(cell_counts >= min_cells)].squeeze()
                    filtered_cell_types = np.atleast_1d(filtered_cell_types)
                    adata_1_tissue = adata_1_tissue[
                        adata_1_tissue.obs["cell_ontology_class"].isin(filtered_cell_types)].copy()

                    adata_2_tissue = adata_2[adata_2.obs["tissue"] == tissue].copy()
                    cell_types, cell_counts = np.unique(adata_2_tissue.obs["cell_ontology_class"].to_numpy(),
                                                        return_counts=True)
                    filtered_cell_types = cell_types[np.argwhere(cell_counts >= min_cells)].squeeze()
                    filtered_cell_types = np.atleast_1d(filtered_cell_types)
                    adata_2_tissue = adata_2_tissue[
                        adata_2_tissue.obs["cell_ontology_class"].isin(filtered_cell_types)].copy()
            else:
                if cell_list is not None:
                    # Filter AnnData to cells listed

                    adata_1_tissue = adata_1[adata_1.obs["cell_ontology_class"].isin(cell_list)].copy()
                    cell_types, cell_counts = np.unique(adata_1_tissue.obs["cell_ontology_class"].to_numpy(),
                                                        return_counts=True)
                    filtered_cell_types = cell_types[np.argwhere(cell_counts >= min_cells)].squeeze()
                    filtered_cell_types = np.atleast_1d(filtered_cell_types)
                    adata_1_tissue = adata_1_tissue[
                        adata_1_tissue.obs["cell_ontology_class"].isin(filtered_cell_types)].copy()

                    adata_2_tissue = adata_2[adata_2.obs["cell_ontology_class"].isin(cell_list)].copy()
                    cell_types, cell_counts = np.unique(adata_2_tissue.obs["cell_ontology_class"].to_numpy(),
                                                        return_counts=True)
                    filtered_cell_types = cell_types[np.argwhere(cell_counts >= min_cells)].squeeze()
                    filtered_cell_types = np.atleast_1d(filtered_cell_types)
                    adata_2_tissue = adata_2_tissue[
                        adata_2_tissue.obs["cell_ontology_class"].isin(filtered_cell_types)].copy()
                else:
                    # Display full AnnData with all tissues and cells, but filter out cell types with less than min_cells
                    adata_1_tissue = adata_1.copy()
                    cell_types, cell_counts = np.unique(adata_1_tissue.obs["cell_ontology_class"].to_numpy(),
                                                        return_counts=True)
                    filtered_cell_types = cell_types[np.argwhere(cell_counts >= min_cells)].squeeze()
                    filtered_cell_types = np.atleast_1d(filtered_cell_types)
                    adata_1_tissue = adata_1_tissue[
                        adata_1_tissue.obs["cell_ontology_class"].isin(filtered_cell_types)].copy()

                    adata_2_tissue = adata_2.copy()
                    cell_types, cell_counts = np.unique(adata_2_tissue.obs["cell_ontology_class"].to_numpy(),
                                                        return_counts=True)
                    filtered_cell_types = cell_types[np.argwhere(cell_counts >= min_cells)].squeeze()
                    filtered_cell_types = np.atleast_1d(filtered_cell_types)
                    adata_2_tissue = adata_2_tissue[
                        adata_2_tissue.obs["cell_ontology_class"].isin(filtered_cell_types)].copy()

        else:
            # Display AnnData without filtering out cell types less than min cells
            if tissue_list is not None:
                # Filter AnnData to tissues
                tissue = tissue_list[i]
                if cell_list is not None:
                    # Filter AnnData to tissues AND cells
                    adata_1_tissue = adata_1[adata_1.obs["tissue"] == tissue].copy()
                    adata_1_tissue = adata_1_tissue[adata_1_tissue.obs["cell_ontology_class"].isin(cell_list)].copy()
                    adata_2_tissue = adata_2[adata_2.obs["tissue"] == tissue].copy()
                    adata_2_tissue = adata_2_tissue[adata_2_tissue.obs["cell_ontology_class"].isin(cell_list)].copy()
                else:
                    # Filter AnnData to tissues
                    adata_1_tissue = adata_1[adata_1.obs["tissue"] == tissue].copy()
                    adata_2_tissue = adata_2[adata_2.obs["tissue"] == tissue].copy()
            else:
                # Display full AnnData
                if cell_list is not None:
                    # Filter AnnData to cells
                    adata_1_tissue = adata_1[adata_1.obs["cell_ontology_class"].isin(cell_list)].copy()
                    adata_2_tissue = adata_2[adata_2.obs["cell_ontology_class"].isin(cell_list)].copy()
                else:
                    # Display full AnnData
                    adata_1_tissue = adata_1.copy()
                    adata_2_tissue = adata_2.copy()

        if tissue_list is None and cell_list is None:
            # Shared gene expression UMAP
            sc.pp.neighbors(adata_1_tissue, use_rep=shared_embedding_1, random_state=seed)
            adata_1_tissue.obsm["X_umap"] = UMAP(n_components=2, random_state=seed).fit_transform(
                adata_1_tissue.obsm[shared_embedding_1])

            sc.pl.umap(adata_1_tissue, color="cell_ontology_class", ax=axs[0], legend_loc=None, show=False, frameon=False,
                          title="S-GE")
            del adata_1_tissue.obsm["X_umap"]
            plot_customized_UMAP_coordinates(axs[0], length=1.0)
            axs[0].set_title(subfigures_alphabet[0], loc='left', fontsize=20, fontweight='bold')

            # Shared transcript usage UMAP
            sc.pp.neighbors(adata_2_tissue, use_rep=shared_embedding_2, random_state=seed)
            adata_2_tissue.obsm["X_umap"] = UMAP(n_components=2, random_state=seed).fit_transform(
                adata_2_tissue.obsm[shared_embedding_2])

            sc.pl.umap(adata_2_tissue, color="cell_ontology_class", ax=axs[1], legend_loc=None, show=False, frameon=False,
                            title="S-TU")
            del adata_2_tissue.obsm["X_umap"]
            plot_customized_UMAP_coordinates(axs[1], length=1.0)

            # Joint UMAP
            sc.pp.neighbors(adata_1_tissue, use_rep=joint_embedding, random_state=seed)
            adata_1_tissue.obsm["X_umap"] = UMAP(n_components=2, random_state=seed).fit_transform(
                adata_1_tissue.obsm[joint_embedding])

            sc.pl.umap(adata_1_tissue, color="cell_ontology_class", ax=axs[2], legend_loc=None, show=False, frameon=False,
                            title="Joint")
            del adata_1_tissue.obsm["X_umap"]
            plot_customized_UMAP_coordinates(axs[2], length=1.0)

        else:
            # UMAPs need to be recomputed for each tissue/cell type subset

            # Shared gene expression UMAP
            sc.pp.neighbors(adata_1_tissue, use_rep=shared_embedding_1, random_state=seed)
            adata_1_tissue.obsm["X_umap"] = UMAP(n_components=2, random_state=seed).fit_transform(adata_1_tissue.obsm[shared_embedding_1])
            sc.pl.umap(adata_1_tissue, color="cell_ontology_class", ax=axs[i, 0], legend_loc=None, show=False, frameon=False,
                       title="S-GE")
            del adata_1_tissue.obsm["X_umap"]
            plot_customized_UMAP_coordinates(axs[i, 0], length=1.0)
            axs[i, 0].set_title(subfigures_alphabet[i], loc='left', fontsize=20, fontweight='bold')

            # Shared transcript usage UMAP
            sc.pp.neighbors(adata_2_tissue, use_rep=shared_embedding_2, random_state=seed)
            adata_2_tissue.obsm["X_umap"] = UMAP(n_components=2, random_state=seed).fit_transform(adata_2_tissue.obsm[shared_embedding_2])
            sc.pl.umap(adata_2_tissue, color="cell_ontology_class", ax=axs[i, 1], legend_loc=None, show=False, frameon=False,
                         title="S-TU")
            del adata_2_tissue.obsm["X_umap"]
            plot_customized_UMAP_coordinates(axs[i, 1], length=1.0)

            # Joint UMAP
            sc.pp.neighbors(adata_1_tissue, use_rep=joint_embedding, random_state=seed)
            adata_1_tissue.obsm["X_umap"] = UMAP(n_components=2, random_state=seed).fit_transform(adata_1_tissue.obsm[joint_embedding])
            sc.pl.umap(adata_1_tissue, color="cell_ontology_class", ax=axs[i, 2], legend_loc=None, show=False, frameon=False,
                         title="Joint")
            del adata_1_tissue.obsm["X_umap"]
            plot_customized_UMAP_coordinates(axs[i, 2], length=1.0)

        # 4. Modality-relevance weights bar plot
        key1, key2 = likelihood_keys[0], likelihood_keys[1]
        cell_weights_1_mean = adata_1_tissue.obs.groupby("cell_ontology_class")[
            f"{key1}_weighting"].mean().reset_index()
        cell_weights_2_mean = adata_2_tissue.obs.groupby("cell_ontology_class")[
            f"{key2}_weighting"].mean().reset_index()

        cell_weights_1_mean.columns = ["cell_ontology_class", "Weighting"]
        cell_weights_1_mean["Modality"] = "GE"
        cell_weights_2_mean.columns = ["cell_ontology_class", "Weighting"]
        cell_weights_2_mean["Modality"] = "TU"

        weights_df = pd.concat([cell_weights_1_mean, cell_weights_2_mean], axis=0)

        sns.barplot(data=weights_df, x="cell_ontology_class", y="Weighting", hue="Modality", ax=axs[i, 3])
        axs[i, 3].axhline(y=0.5, color='k', linestyle='--')

        # REMOVE X-AXIS LABELS
        axs[i, 3].set_xticklabels([])
        axs[i, 3].set_xlabel("")
        axs[i, 3].set_ylabel("Relevance")

        # --- CREATE CUSTOM CELL TYPE LEGEND ON THE RIGHT ---
        # Extract the color palette used in the AnnData
        categories = adata_1_tissue.obs["cell_ontology_class"].cat.categories
        colors = adata_1_tissue.uns["cell_ontology_class_colors"]

        # The x-positions for the bars in a categorical Seaborn plot are 0, 1, 2, ...
        x_positions = range(len(categories))

        # Plot the dots slightly below the x-axis (transform_offset or just setting y)
        # We use clip_on=False so the dots aren't cut off by the plot borders
        axs[i, 3].scatter(x_positions, [-0.03] * len(categories),
                          c=colors, s=100, marker='o',
                          zorder=3, clip_on=False,
                          transform=axs[i, 3].get_xaxis_transform())

        # Create "bullet point" markers for the legend
        legend_handles = [
            mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8, label=cat)
            for cat, color in zip(categories, colors)
        ]

        # Add legend to the right of the barplot
        # bbox_to_anchor moves it outside the plot area
        axs[i, 3].legend(handles=legend_handles, title="Cell Types",
                         loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.tight_layout()
    if save_fig:
        # Sanitize strings to avoid path errors
        t_name = "_".join(tissue_list).replace("/", "-") if tissue_list else "all_tissues"
        c_name = "_".join([c.replace("/", "-") for c in cell_list]) if cell_list else "all_cell_types"

        filename = f"./figures/{dataset_name}/{t_name}_{c_name}_trvi_umaps.pdf"

        # Explicitly save the 'fig' object
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {filename}")

    plt.show()

def plot_zanidm_cases(
        num_cases_dict: Dict[str, int],
        log_scale: bool = True,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given a dictionary containing the occurrences of each ZANIDM case (case 1: all nonzero; case 2: all zero; case 3:
    one nonzero; case 4: subsets are zero), generate two plots in one panel. The first plot shows the absoulte results
    as bar chart with the option of a log-scale y-acis. The second plot shows the relative results as a stacked bar chart.
    The figure panel is saved if save_fig is True.

    :param num_cases_dict:
    :param log_scale:
    :param save_fig:
    """
    # Create a dataframe from the dictionary
    df = pd.DataFrame(list(num_cases_dict.items()), columns=['Category', 'Value'])

    # Rename categories for better readability
    mapping = {
        'case_1': 'All\nnonzero',
        'case_2': 'All\nzero',
        'case_3': 'One\nnonzero',
        'case_4': 'Subsets\nare zero'
    }
    df['Category'] = df['Category'].replace(mapping)

    # Setup the figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- FIRST PLOT: Matplotlib Bar Chart ---
    # Generate colors from the viridis colormap to match the second plot
    colors = cm.viridis(np.linspace(0, 0.9, len(df)))

    # Create the bar chart using matplotlib directly
    bars = axes[0].bar(df['Category'], df['Value'], color=colors, edgecolor='none')

    if log_scale:
        axes[0].set_yscale("log")

    #axes[0].set_title("# occurrences per ZANIDM cases")
    axes[0].set_ylabel("# occurrences (log scale)" if log_scale else "# occurrences")
    axes[0].set_xlabel("Cases")

    # Add data labels for the absolute plot
    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:  # Ensure we don't annotate zero/negative if log scale
            axes[0].annotate(f'{int(height):,}',
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom',
                             xytext=(0, 5),
                             textcoords='offset points',
                             fontsize=10, weight='bold')

    # Determine the y-axis limit dynamically
    top_limit = df['Value'].max()
    if log_scale:
        axes[0].set_ylim(bottom=1, top=top_limit * 5)  # bottom=1 is safer for log scales
    else:
        axes[0].set_ylim(top=top_limit * 1.15)

    axes[0].set_title('a', loc='left', fontsize=20, fontweight='bold')

    # --- SECOND PLOT: Relative stacked bar ---
    df['Relative Value'] = df['Value'] / df['Value'].sum()
    df_stacked = df.set_index('Category')[['Relative Value']].T

    # Using pandas helper (which uses matplotlib under the hood)
    df_stacked.plot(kind='bar', stacked=True, ax=axes[1], colormap='viridis', width=0.5, legend=False)

    #axes[1].set_title("Relative Distribution")
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("All cases")
    axes[1].set_xticks([])

    #axes[1].legend(title="Cases", loc='center left', bbox_to_anchor=(1, 0.5))

    # Add percentage labels
    cumulative_height = 0
    for category in df['Category']:
        val = df.loc[df['Category'] == category, 'Relative Value'].values[0]
        if val > 0.03:
            axes[1].text(0, cumulative_height + val / 2, f'{val:.1%}',
                         ha='center', va='center', color='white', weight='bold')
        cumulative_height += val

    axes[1].set_title('b', loc='left', fontsize=20, fontweight='bold')

    plt.tight_layout()

    if save_fig:
        import os
        dataset_name = kwargs.get("dataset_name", "default")
        save_path = f"./figures/{dataset_name}/"
        os.makedirs(save_path, exist_ok=True)
        fig_name = os.path.join(save_path, "occurrences_zanidm_cases.pdf")
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')

    plt.show()

def plot_comparison_tuvi_zanidm_zidm(
        distance_matrices = Tuple[np.ndarray, np.ndarray],
        cluster_annotations = Tuple[np.ndarray, np.ndarray],
        computation_times = Dict[str, Dict[str, float]],
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given a tuple of two distance matrices where the first contains the distances of embeddings computed with
    tuVI-ZANIDM, and the second one distances of embeddings computed with tuVI-ZIDM, and a dictionary of computation
    times for both models, plot a 2 x 2 panel. The first row contains two horizontal bar charts where the first one
    compares the average epoch time for both models and the second one the total training time. The second row displays
    the distance matrices of both models annotated with the corresponding annotations given.

    :param distance_matrices:
    :param cluster_annotations:
    :param computation_times:
    :param save_fig:
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot average epoch training time
    avg_epoch_times = np.array([computation_times["ZANIDM"]["Average time per epoch"], computation_times["ZIDM"]["Average time per epoch"]])
    axes[0,0].barh(np.array(["ZANIDM", "ZIDM"]), avg_epoch_times)
    axes[0,0].set_xlabel("Average time per epoch (s)")
    axes[0,0].set_ylabel("Observation model of tuVI")
    axes[0,0].tick_params(axis="both")
    axes[0,0].set_title('a', loc='left', fontsize=20, fontweight='bold')

    # Plot total training time
    total_training_times = np.array(
        [computation_times["ZANIDM"]["Total training time"], computation_times["ZIDM"]["Total training time"]])
    axes[0, 1].barh(np.array(["ZANIDM", "ZIDM"]), total_training_times)
    axes[0, 1].set_xlabel("Total training time (s)")
    axes[0, 1].set_ylabel("Observation model of tuVI")
    axes[0, 1].tick_params(axis="both")
    axes[0, 1].set_title('b', loc='left', fontsize=20, fontweight='bold')

    # Heatmaps
    vmin = min(np.min(distance_matrices[0]), np.min(distance_matrices[1]))
    vmax = max(np.max(distance_matrices[0]), np.max(distance_matrices[1]))

    if kwargs.get("average_embeddings", True):
        heatmap_label = "Mean cell embeddings"
    else:
        heatmap_label = "Cell embeddings"

    sns.heatmap(
        distance_matrices[0],
        ax=axes[1, 0],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Distance'}
    )
    axes[1, 0].set_title('c', loc='left', fontsize=20, fontweight='bold')
    axes[1, 0].set_xlabel(heatmap_label)
    axes[1, 0].set_ylabel(heatmap_label)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    #axes[1, 0].set_xticks(np.arange(len(cluster_annotations[0])))
    #axes[1, 0].set_xticklabels(cluster_annotations[0], rotation=90)
    #axes[1, 0].set_yticks(np.arange(len(cluster_annotations[0])))
    #axes[1, 0].set_yticklabels(cluster_annotations[0], rotation=0)

    sns.heatmap(
        distance_matrices[1],
        ax=axes[1, 1],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Distance'}
    )
    axes[1, 1].set_title('d', loc='left', fontsize=20, fontweight='bold')
    axes[1, 1].set_xlabel(heatmap_label)
    axes[1, 1].set_ylabel(heatmap_label)
    # Remove xticks and xticklabels
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    #axes[1, 1].set_xticks(np.arange(len(cluster_annotations[1])))
    #axes[1, 1].set_xticklabels(cluster_annotations[1], rotation=90)
    #axes[1, 1].set_yticks(np.arange(len(cluster_annotations[1])))
    #axes[1, 1].set_yticklabels(cluster_annotations[1], rotation=0)

    plt.tight_layout()

    if save_fig:
        import os
        dataset_name = kwargs.get("dataset_name", "default")
        save_path = f"./figures/{dataset_name}/"
        fig_name = os.path.join(save_path, "comparison_tuvi_zanidm_zidm.pdf")
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')

    plt.show()

def plot_umap_and_distance_matrix(
        adata_1: AnnData | None,
        adata_2: AnnData | None,
        cell_embeddings_keys: List[str],
        transcriptomic_facet_keys: List[str],
        relevance_weight_keys: List[str] | None,
        distance_matrices: List[np.ndarray],
        cluster_annotations: np.ndarray,
        umap_color_display_key: str,
        model_type: str,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Plot annotation UMAPs and distance heatmaps for one or more
    transcriptomic facets, with optional modality-relevance-weight UMAPs.

    Layouts
    -------
    One facet without modality-relevance weights:
        Annotation UMAP | heatmap

    One facet with modality-relevance weights:
        Annotation UMAP | modality-relevance-weight UMAP | heatmap

    Multiple facets without modality-relevance weights:
        Row 1: annotation UMAPs
        Row 2: heatmaps

    Multiple facets with modality-relevance weights:
        Row 1: annotation UMAPs
        Row 2: modality-relevance-weight UMAPs
        Row 3: heatmaps
    """

    random_state = kwargs.get(
        "random_state",
        0
    )

    # ------------------------------------------------------------------
    # Typography for a 180 mm-wide Nature Methods figure
    # ------------------------------------------------------------------
    panel_label_fontsize = kwargs.get(
        "panel_label_fontsize",
        11
    )
    panel_label_y = kwargs.get(
        "panel_label_y",
        1.06
    )
    facet_title_fontsize = kwargs.get(
        "facet_title_fontsize",
        8
    )
    umap_axis_fontsize = kwargs.get(
        "umap_axis_fontsize",
        7
    )
    legend_fontsize = kwargs.get(
        "legend_fontsize",
        6.5
    )
    legend_title_fontsize = kwargs.get(
        "legend_title_fontsize",
        7
    )
    heatmap_bullet_fontsize = kwargs.get(
        "heatmap_bullet_fontsize",
        5
    )
    colorbar_label_fontsize = kwargs.get(
        "colorbar_label_fontsize",
        7
    )
    colorbar_tick_fontsize = kwargs.get(
        "colorbar_tick_fontsize",
        6
    )

    boundary_linewidth = kwargs.get(
        "boundary_linewidth",
        0.8
    )

    # ------------------------------------------------------------------
    # Modality-relevance weight UMAP settings
    # ------------------------------------------------------------------
    relevance_vmin = kwargs.get(
        "relevance_vmin",
        0.0
    )
    relevance_vmax = kwargs.get(
        "relevance_vmax",
        1.0
    )
    relevance_color_map = kwargs.get(
        "relevance_color_map",
        "viridis"
    )
    relevance_colorbar_gap_mm = kwargs.get(
        "relevance_colorbar_gap_mm",
        1.5
    )
    relevance_colorbar_width_mm = kwargs.get(
        "relevance_colorbar_width_mm",
        1.5
    )

    if relevance_colorbar_gap_mm < 0:
        raise ValueError(
            "relevance_colorbar_gap_mm must be non-negative."
        )

    if relevance_colorbar_width_mm <= 0:
        raise ValueError(
            "relevance_colorbar_width_mm must be greater than 0."
        )

    # ------------------------------------------------------------------
    # Validate facet-specific inputs
    # ------------------------------------------------------------------
    num_facets = len(
        transcriptomic_facet_keys
    )

    if num_facets == 0:
        raise ValueError(
            "At least one transcriptomic facet must be provided."
        )

    if not (
        len(cell_embeddings_keys)
        == len(distance_matrices)
        == num_facets
    ):
        raise ValueError(
            "cell_embeddings_keys, transcriptomic_facet_keys, and "
            "distance_matrices must have the same length."
        )

    has_relevance_weights = (
            relevance_weight_keys is not None
    )

    if has_relevance_weights:
        if len(relevance_weight_keys) != num_facets:
            raise ValueError(
                f"Expected {num_facets} relevance-weight keys, one "
                f"for each transcriptomic facet, but received "
                f"{len(relevance_weight_keys)}."
            )

        invalid_relevance_key_indices = [
            i
            for i, key in enumerate(
                relevance_weight_keys
            )
            if not isinstance(key, str) or not key
        ]

        if invalid_relevance_key_indices:
            raise ValueError(
                "Every relevance-weight key must be a non-empty "
                "string. Invalid entries were found at indices "
                f"{invalid_relevance_key_indices}."
            )

    heatmap_vmin = min(
        np.min(matrix)
        for matrix in distance_matrices
    )
    heatmap_vmax = max(
        np.max(matrix)
        for matrix in distance_matrices
    )

    alphabet_characters = list(
        "abcdefghijklmnopqrstuvwxyz"
    )

    if num_facets > len(alphabet_characters):
        raise ValueError(
            "A maximum of 26 labelled facets is supported."
        )

    # ------------------------------------------------------------------
    # Nature Methods figure dimensions
    # ------------------------------------------------------------------
    mm_per_inch = 25.4

    figure_width_mm = kwargs.get(
        "figure_width_mm",
        180.0
    )
    max_figure_height_mm = kwargs.get(
        "max_figure_height_mm",
        247.0
    )
    legend_height_mm = kwargs.get(
        "legend_height_mm",
        12.0
    )

    if figure_width_mm <= 0:
        raise ValueError(
            "figure_width_mm must be greater than 0."
        )

    if max_figure_height_mm <= 0:
        raise ValueError(
            "max_figure_height_mm must be greater than 0."
        )

    figure_width_in = (
        figure_width_mm / mm_per_inch
    )

    if num_facets == 1:
        number_of_plot_rows = 1
        number_of_plot_columns = (
            3 if has_relevance_weights else 2
        )

    else:
        number_of_plot_rows = (
            3 if has_relevance_weights else 2
        )
        number_of_plot_columns = num_facets

    default_figure_height_mm = (
        figure_width_mm
        * number_of_plot_rows
        / number_of_plot_columns
        + legend_height_mm
    )

    figure_height_mm = kwargs.get(
        "figure_height_mm",
        default_figure_height_mm
    )

    figure_height_mm = min(
        figure_height_mm,
        max_figure_height_mm
    )

    fig, axes = plt.subplots(
        nrows=number_of_plot_rows,
        ncols=number_of_plot_columns,
        figsize=(
            figure_width_in,
            figure_height_mm / mm_per_inch
        ),
        squeeze=False
    )

    cluster_annotations = np.asarray(
        cluster_annotations
    )

    # Store the modality-relevance axes and their colour bars so that the
    # colour bars can be repositioned after tight_layout.
    relevance_colorbar_pairs = []

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def get_color_from_legend_handle(handle):

        if hasattr(
            handle,
            "get_markerfacecolor"
        ):
            return handle.get_markerfacecolor()

        if hasattr(
            handle,
            "get_facecolor"
        ):
            color = handle.get_facecolor()

            if (
                np.ndim(color) > 1
                and len(color) > 0
            ):
                return color[0]

            return color

        if hasattr(
            handle,
            "get_color"
        ):
            return handle.get_color()

        return "black"

    def make_cluster_color_map(
            current_handles,
            current_labels
    ):

        return {
            str(label): get_color_from_legend_handle(
                handle
            )
            for handle, label in zip(
                current_handles,
                current_labels
            )
        }

    def get_cluster_centers_labels_and_boundaries(
            annotations
    ):

        boundary_positions = np.where(
            annotations[:-1]
            != annotations[1:]
        )[0] + 1

        starts = np.r_[
            0,
            boundary_positions
        ]
        ends = np.r_[
            boundary_positions,
            len(annotations)
        ]

        centers = (
            starts + ends
        ) / 2

        cluster_labels = annotations[
            starts
        ]

        return (
            centers,
            cluster_labels,
            boundary_positions
        )

    def format_custom_umap_coordinates(ax):
        """
        Add custom UMAP coordinate axes and reduce their typography.
        """
        number_of_existing_texts = len(
            ax.texts
        )

        plot_customized_UMAP_coordinates(
            ax,
            length=1.0
        )

        for text_artist in ax.texts[
            number_of_existing_texts:
        ]:
            text_artist.set_fontsize(
                umap_axis_fontsize
            )

        ax.xaxis.label.set_size(
            umap_axis_fontsize
        )
        ax.yaxis.label.set_size(
            umap_axis_fontsize
        )

        ax.tick_params(
            axis="both",
            labelsize=umap_axis_fontsize
        )

    def get_relevance_weight_title(
            transcriptomic_facet
    ):

        if transcriptomic_facet in [
            "GE",
            "S-GE",
            "P-GE",
            "GE-TU"
        ]:
            return "GE weight"

        if transcriptomic_facet in [
            "TU",
            "S-TU",
            "P-TU"
        ]:
            return "TU weight"

        raise ValueError(
            f"Cannot determine the relevance-weight title for "
            f"facet '{transcriptomic_facet}'."
        )

    def format_relevance_colorbar(ax):
        """
        Find and format the continuous colour bar created by Scanpy.
        """
        colorbar = None

        for collection in ax.collections:
            collection_colorbar = getattr(
                collection,
                "colorbar",
                None
            )

            if collection_colorbar is not None:
                colorbar = collection_colorbar
                break

        if colorbar is None:
            raise RuntimeError(
                "Scanpy did not create a colour bar for the "
                "relevance-weight UMAP."
            )

        colorbar.ax.tick_params(
            labelsize=colorbar_tick_fontsize,
            length=2,
            pad=1
        )

        colorbar.set_label(
            "Relevance",
            fontsize=colorbar_label_fontsize,
            labelpad=2
        )

        return colorbar

    def plot_distance_heatmap(
            ax,
            distance_matrix,
            annotations,
            cluster_color_map
    ):

        if (
            distance_matrix.shape[0]
            != distance_matrix.shape[1]
        ):
            raise ValueError(
                f"Expected a square distance matrix, "
                f"got shape {distance_matrix.shape}."
            )

        if (
            distance_matrix.shape[0]
            != len(annotations)
        ):
            raise ValueError(
                f"Distance matrix has shape "
                f"{distance_matrix.shape}, but "
                f"cluster_annotations has length "
                f"{len(annotations)}."
            )

        sns.heatmap(
            distance_matrix,
            ax=ax,
            cmap="viridis",
            vmin=heatmap_vmin,
            vmax=heatmap_vmax,
            square=True,
            cbar_kws={
                "label": "Distance",
                "fraction": 0.045,
                "pad": 0.025,
                "aspect": 30
            }
        )

        colorbar = ax.collections[0].colorbar

        if colorbar is not None:
            colorbar.ax.tick_params(
                labelsize=colorbar_tick_fontsize,
                length=2,
                pad=1
            )
            colorbar.set_label(
                "Distance",
                fontsize=colorbar_label_fontsize,
                labelpad=2
            )

        (
            centers,
            cluster_labels,
            boundary_positions
        ) = get_cluster_centers_labels_and_boundaries(
            annotations
        )

        for position in boundary_positions:
            ax.axhline(
                position,
                color="red",
                linestyle="--",
                linewidth=boundary_linewidth
            )
            ax.axvline(
                position,
                color="red",
                linestyle="--",
                linewidth=boundary_linewidth
            )

        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.set_xticks(
            centers
        )
        ax.set_yticks(
            centers
        )

        ax.set_xticklabels(
            ["●"] * len(cluster_labels),
            rotation=0,
            fontsize=heatmap_bullet_fontsize
        )
        ax.set_yticklabels(
            ["●"] * len(cluster_labels),
            rotation=0,
            fontsize=heatmap_bullet_fontsize
        )

        for tick_label, cluster_label in zip(
            ax.get_xticklabels(),
            cluster_labels
        ):
            tick_label.set_color(
                cluster_color_map.get(
                    str(cluster_label),
                    "black"
                )
            )

        for tick_label, cluster_label in zip(
            ax.get_yticklabels(),
            cluster_labels
        ):
            tick_label.set_color(
                cluster_color_map.get(
                    str(cluster_label),
                    "black"
                )
            )

        ax.tick_params(
            axis="both",
            length=0,
            pad=1
        )

    # ------------------------------------------------------------------
    # Plot facets
    # ------------------------------------------------------------------
    handles = None
    labels = None

    for i, transcriptomic_facet in enumerate(
        transcriptomic_facet_keys
    ):

        # --------------------------------------------------------------
        # Determine axes
        # --------------------------------------------------------------
        if num_facets == 1:
            annotation_umap_ax = axes[
                0,
                0
            ]

            if has_relevance_weights:
                relevance_umap_ax = axes[
                    0,
                    1
                ]
                heatmap_ax = axes[
                    0,
                    2
                ]
            else:
                relevance_umap_ax = None
                heatmap_ax = axes[
                    0,
                    1
                ]

        else:
            annotation_umap_ax = axes[
                0,
                i
            ]

            if has_relevance_weights:
                relevance_umap_ax = axes[
                    1,
                    i
                ]
                heatmap_ax = axes[
                    2,
                    i
                ]
            else:
                relevance_umap_ax = None
                heatmap_ax = axes[
                    1,
                    i
                ]

        # --------------------------------------------------------------
        # Select AnnData object
        # --------------------------------------------------------------
        if transcriptomic_facet in [
            "GE",
            "S-GE",
            "P-GE",
            "GE-TU"
        ]:
            adata = adata_1

        elif transcriptomic_facet in [
            "TU",
            "S-TU",
            "P-TU"
        ]:
            adata = adata_2

        else:
            raise ValueError(
                f"Invalid transcriptomic facet key: "
                f"{transcriptomic_facet}. Expected one of "
                "'GE', 'S-GE', 'P-GE', 'TU', 'S-TU', "
                "'P-TU', or 'GE-TU'."
            )

        if adata is None:
            raise ValueError(
                f"AnnData object is None for transcriptomic "
                f"facet '{transcriptomic_facet}'."
            )

        embedding_key = (
            cell_embeddings_keys[i]
        )

        if embedding_key not in adata.obsm:
            raise KeyError(
                f"Embedding key '{embedding_key}' is not present "
                f"in adata.obsm for facet "
                f"'{transcriptomic_facet}'."
            )

        if umap_color_display_key not in adata.obs:
            raise KeyError(
                f"UMAP colour key '{umap_color_display_key}' is not "
                f"present in adata.obs for facet "
                f"'{transcriptomic_facet}'."
            )

        if has_relevance_weights:
            relevance_weight_key = (
                relevance_weight_keys[i]
            )

            if relevance_weight_key not in adata.obs:
                raise KeyError(
                    f"Relevance-weight key "
                    f"'{relevance_weight_key}' is not present "
                    f"in adata.obs for facet "
                    f"'{transcriptomic_facet}'."
                )

        # Preserve pre-existing UMAP coordinates.
        had_existing_umap = (
            "X_umap" in adata.obsm
        )

        if had_existing_umap:
            existing_umap = (
                adata.obsm["X_umap"].copy()
            )
        else:
            existing_umap = None

        try:
            adata.obsm["X_umap"] = UMAP(
                n_components=2,
                random_state=random_state
            ).fit_transform(
                adata.obsm[embedding_key]
            )

            # ----------------------------------------------------------
            # Annotation UMAP
            # ----------------------------------------------------------
            sc.pl.umap(
                adata,
                color=umap_color_display_key,
                frameon=False,
                show=False,
                ax=annotation_umap_ax
            )

            annotation_umap_ax.set_title(
                transcriptomic_facet,
                fontsize=facet_title_fontsize,
                pad=2
            )

            format_custom_umap_coordinates(
                annotation_umap_ax
            )

            annotation_umap_ax.set_box_aspect(
                1
            )

            annotation_umap_ax.set_title(
                alphabet_characters[i],
                loc="left",
                fontsize=panel_label_fontsize,
                fontweight="bold",
                y=panel_label_y,
                pad=0
            )

            current_handles, current_labels = (
                annotation_umap_ax
                .get_legend_handles_labels()
            )

            annotation_legend = (
                annotation_umap_ax.get_legend()
            )

            if annotation_legend is not None:
                annotation_legend.remove()

            cluster_color_map = make_cluster_color_map(
                current_handles,
                current_labels
            )

            handles = current_handles
            labels = current_labels

            # ----------------------------------------------------------
            # Optional modality-relevance weight UMAP
            # ----------------------------------------------------------
            if has_relevance_weights:
                sc.pl.umap(
                    adata,
                    color=relevance_weight_key,
                    color_map=relevance_color_map,
                    vmin=relevance_vmin,
                    vmax=relevance_vmax,
                    frameon=False,
                    show=False,
                    ax=relevance_umap_ax
                )

                relevance_umap_ax.set_title(
                    get_relevance_weight_title(transcriptomic_facet),
                    fontsize=facet_title_fontsize,
                    pad=2
                )

                format_custom_umap_coordinates(
                    relevance_umap_ax
                )

                relevance_umap_ax.set_box_aspect(
                    1
                )

                relevance_colorbar = (
                    format_relevance_colorbar(relevance_umap_ax)
                )

                relevance_colorbar_pairs.append(
                    (
                        relevance_umap_ax,
                        relevance_colorbar
                    )
                )

        finally:
            # Restore any UMAP coordinates that existed before this call.
            if had_existing_umap:
                adata.obsm["X_umap"] = (
                    existing_umap
                )
            else:
                del adata.obsm["X_umap"]

        # --------------------------------------------------------------
        # Distance heatmap
        # --------------------------------------------------------------
        plot_distance_heatmap(
            heatmap_ax,
            distance_matrices[i],
            cluster_annotations,
            cluster_color_map
        )

    # ------------------------------------------------------------------
    # Shared categorical legend
    # ------------------------------------------------------------------
    legend_present = (
        bool(handles)
        and bool(labels)
    )

    if legend_present:
        maximum_legend_columns = kwargs.get(
            "legend_ncols",
            5
        )

        number_of_legend_columns = min(
            maximum_legend_columns,
            len(labels)
        )

        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                0.99
            ),
            ncol=number_of_legend_columns,
            frameon=False,
            fontsize=legend_fontsize,
            title="Cell type",
            title_fontsize=legend_title_fontsize,
            markerscale=0.6,
            handlelength=1.0,
            handletextpad=0.3,
            columnspacing=0.8,
            labelspacing=0.25,
            borderaxespad=0
        )

        if hasattr(
            legend,
            "_legend_box"
        ):
            legend._legend_box.align = "left"

    if legend_present:
        legend_fraction = (
            legend_height_mm
            / figure_height_mm
        )
        layout_top = (
            1.0 - legend_fraction
        )
    else:
        layout_top = 0.98

    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            layout_top
        ),
        pad=0.25,
        w_pad=0.5,
        h_pad=0.5
    )

    # ------------------------------------------------------------------
    # Reposition modality-relevance colour bars with a fixed physical gap
    # ------------------------------------------------------------------
    fig.canvas.draw()

    relevance_colorbar_gap_fraction = (
            relevance_colorbar_gap_mm
            / figure_width_mm
    )
    relevance_colorbar_width_fraction = (
            relevance_colorbar_width_mm
            / figure_width_mm
    )

    for (
            relevance_ax,
            relevance_colorbar
    ) in relevance_colorbar_pairs:

        parent_position = (
            relevance_ax.get_position()
        )
        colorbar_ax = (
            relevance_colorbar.ax
        )

        # Prevent Scanpy's axes-divider locator from restoring the
        # original colour-bar position during savefig.
        colorbar_ax.set_axes_locator(
            None
        )

        if hasattr(
            colorbar_ax,
            "set_box_aspect"
        ):
            colorbar_ax.set_box_aspect(
                None
            )

        colorbar_ax.set_position([
            parent_position.x1
            + relevance_colorbar_gap_fraction,
            parent_position.y0,
            relevance_colorbar_width_fraction,
            parent_position.height
        ])

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    if save_fig:
        import os

        dataset_name = kwargs.get(
            "dataset_name",
            "default"
        )
        file_suffix = kwargs.get(
            "file_suffix",
            "pdf"
        )
        tax_level = kwargs.get(
            "tax_level",
            "default"
        )

        save_path = (
            f"./figures/{dataset_name}/"
        )

        os.makedirs(
            save_path,
            exist_ok=True
        )

        fig_name = os.path.join(
            save_path,
            model_type
            + "_cell_embeddings_umap_heatmaps_"
            + tax_level
            + "."
            + file_suffix
        )

        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches=None,
            pad_inches=0
        )

    plt.show()


def plot_scvi_tuvi_cell_embedding_comparison(
        adata_objects: Tuple[AnnData, AnnData],
        evaluation_df: pd.DataFrame,
        color_annotation: str,
        likelihood_1: str = "ZINB",
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given a tuple of two AnnData objects where the first contains the inferred gene expression cell embeddings from scVI
    (with a NB or ZINB observation model) and the second the inferred transcript usage cell embeddings from tuVI (with
    DM, ZANIDM, and ZIDM observation model) and an evaluation dataframe, plot the UMAPs visualising the embeddings in 2D
    and plot the evaluation of the embeddings using biological conservation metrics (NMI, ARI, ASW, AvgBIO).

    :param adata_objects: tuple of AnnData objects
    :param evaluation_df: dataframe of evaluation dataframe
    :param color_annotation: annotation color
    :param likelihood_1: name of gene expression cell embedding observation model
    :param save_fig: save figure
    :param kwargs: keyword arguments
    """
    # Unpack the Tuple of AnnData objects
    adata_GE, adata_TU = adata_objects

    # Assert if .obsm[likelihood_GE + '_X_umap'] is present in adata_GE
    assert (likelihood_1 + '_X_umap') in adata_GE.obsm, f"The key '{likelihood_1 + '_X_umap'}' is not present in adata_GE.obsm. Please compute the UMAP coordinates for the gene expression cell embeddings using the '{likelihood_1}' observation model and add them to adata_GE.obsm['{likelihood_1}_X_umap'] before calling this function."

    # Assert if .obsm["DM_X_umap"], obsm["ZANIDM_X_umap"], and obsm["ZIDM_X_umap"] are present in adata_TU
    assert "DM_X_umap" in adata_TU.obsm, "The key 'DM_X_umap' is not present in adata_TU.obsm. Please compute the UMAP coordinates for the transcript usage cell embeddings using the 'DM' observation model and add them to adata_TU.obsm['DM_X_umap'] before calling this function."
    assert "ZANIDM_X_umap" in adata_TU.obsm, "The key 'ZANIDM_X_umap' is not present in adata_TU.obsm. Please compute the UMAP coordinates for the transcript usage cell embeddings using the 'ZANIDM' observation model and add them to adata_TU.obsm['ZANIDM_X_umap'] before calling this function."
    assert "ZIDM_X_umap" in adata_TU.obsm, "The key 'ZIDM_X_umap' is not present in adata_TU.obsm. Please compute the UMAP coordinates for the transcript usage cell embeddings using the 'ZIDM' observation model and add them to adata_TU.obsm['ZIDM_X_umap'] before calling this function."

    # Assert if color_annotation is present in adata_GE.obs and adata_TU.obs
    assert color_annotation in adata_GE.obs, f"The key '{color_annotation}' is not present in adata_GE.obs. Please ensure that the annotation used for coloring the UMAP plots is present in adata_GE.obs['{color_annotation}'] before calling this function."
    assert color_annotation in adata_TU.obs, f"The key '{color_annotation}' is not present in adata_TU.obs. Please ensure that the annotation used for coloring the UMAP plots is present in adata_TU.obs['{color_annotation}'] before calling this function."

    fig = plt.figure(figsize=(13, 16), dpi=300)
    gs = gridspec.GridSpec(4, 3, figure=fig)

    # UMAP of atlas using scGEVI-ZINB
    ax00 = fig.add_subplot(gs[0, 0])
    adata_GE.obsm['X_umap'] = adata_GE.obsm[likelihood_1 + '_X_umap'].copy()
    sc.pl.umap(adata_GE, color=color_annotation, legend_loc=None, frameon=False, show=False, ax=ax00)
    del adata_GE.obsm['X_umap']
    ax00.set_title('scVI-ZINB')
    ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')
    plot_customized_UMAP_coordinates(ax00, length=1.0)

    # UMAP of atlas using tuVI-DM
    ax01 = fig.add_subplot(gs[0, 1])
    adata_TU.obsm['X_umap'] = adata_TU.obsm['DM_X_umap'].copy()
    sc.pl.umap(adata_TU, color=color_annotation, frameon=False, show=False, ax=ax01)
    leg = ax01.get_legend()
    if leg:
        plt.setp(leg.get_texts(), fontsize=12)  # or a number like 8
        plt.setp(leg.get_title(), fontsize=12)
    del adata_TU.obsm['X_umap']
    ax01.set_title('tuVI-DM')
    ax01.set_title('b', loc='left', fontsize=20, fontweight='bold')
    plot_customized_UMAP_coordinates(ax01, length=1.0)

    # Modelling splicing
    ax10 = fig.add_subplot(gs[1:3, :3])

    try:
        img_resource = resources.files('crecerelle.default_figures').joinpath('crecerelle_splicing_tuVI.png')

        with resources.as_file(img_resource) as image_path:
            model_img = mpimg.imread(str(image_path))
            ax10.imshow(model_img)
    except (ImportError, FileNotFoundError):
        print("Warning: Splicing figure could not be loaded from package resources.")

    #image_path = "./default_figures/crecerelle_splicing_tuVI.png"
    #model_img = mpimg.imread(image_path)
    #ax10.imshow(model_img)
    ax10.axis('off')
    # ax10.set_title('a', loc='left', fontsize=20, fontweight='bold')

    # UMAP of atlas using tuVI-ZANIDM
    ax30 = fig.add_subplot(gs[3, 0])
    adata_TU.obsm['X_umap'] = adata_TU.obsm['ZANIDM_X_umap'].copy()
    sc.pl.umap(adata_TU, color=color_annotation, legend_loc=None, frameon=False, show=False, ax=ax30)
    del adata_TU.obsm['X_umap']
    ax30.set_title('tuVI-ZANIDM')
    ax30.set_title('f', loc='left', fontsize=20, fontweight='bold')
    plot_customized_UMAP_coordinates(ax30, length=1.0)

    # UMAP of atlas using tuVI-ZIDM
    ax31 = fig.add_subplot(gs[3, 1])
    adata_TU.obsm['X_umap'] = adata_TU.obsm['ZIDM_X_umap'].copy()
    sc.pl.umap(adata_TU, color=color_annotation, legend_loc=None, frameon=False, show=False, ax=ax31)
    del adata_TU.obsm['X_umap']
    ax31.set_title('tuVI-ZIDM')
    ax31.set_title('g', loc='left', fontsize=20, fontweight='bold')
    plot_customized_UMAP_coordinates(ax31, length=1.0)

    # Bar chart of biological conservation metrics
    # optimal leiden resolution for scVI-ZINB: res = 0.6 (40 clusters /  cell types)
    # optimal leiden resolution for tuVI-DM: res = 1.4 (32 clusters /  cell types)
    # optimal leiden resolution for tuVI-ZANIDM: res = 0.9 ( 38 clusters /  cell types)
    # optimal leiden resolution for tuVI-ZIDM: res =  0.9 ( 31 clusters /  cell types)

    ax32 = fig.add_subplot(gs[3, 2])
    sns.barplot(evaluation_df, x='Evaluation score', y='Score value', hue='Likelihood', ax=ax32)
    ax32.set_ylabel(" ")
    ax32.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax32.set_title('h', loc='left', fontsize=20, fontweight='bold')

    if save_fig:
        dataset_name = kwargs.get("dataset_name")
        file_suffix = kwargs.get("file_suffix", 'pdf')
        if dataset_name is not None:
            # plt.savefig("./figures/tabulaMuris/tuVI_splicing_modelling.png", dpi=300, bbox_inches='tight')
            fig_name = "./figures/" + dataset_name + "/tuVI_splicing_modelling." + file_suffix
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        else:
            fig_name = "./figures/tuVI_splicing_modelling." + file_suffix
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')

    plt.show()

def plot_deg_analysis(
        adata: AnnData,
        cluster_group: str,
        model_name: str,
        taxonomy_level: str | None,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given an Anndata object containing the results of the DEG analysis for a specific cell taxonomy level (e.g. tissue)
    and cluster group, plot the UMAPs of the clusters, the dotplot of the DEGs across cluster groups, and the top 6 DEGs
    for the cluster group specified. The DEG analysis should have been performed using the scVI or TRVI latent
    space. The chosen model and GE likelihood key (for TRVI also the embedding) should be represented in model_name
    (e.g. scVI_ZINB, TRVI_ZINB_ZIDM_shared, etc.). If the Anndata object was filtered to a cell taxonomy level
    (e.g. tissue (Heart, Brain_Non-Myeloid)) specify the cell taxonomy level name otherwise set taxonomy_level to None.
    The cluster_group specifies the Leiden cluster group for which the top 6 DEGs will be plotted. The figure is saved
    in the figures directory if save_fig is set to True.

    The default layout is 180 mm wide, has a maximum height of 247 mm,
    and uses equally sized square UMAP plotting areas.

    :param adata: AnnData object containing the results of the DEG analysis for a specific tissue and cluster group
    :param cluster_group: str, the Leiden cluster group for which the top 6 DEGs will be plotted
    :param model_name: str, the name of the model used for the DEG analysis (e.g. scVI_ZINB, TRVI_ZINB_ZIDM_shared, etc.)
    :param taxonomy_level: str, the name of the tissue for which the DEG analysis was performed, or None if the analysis was performed on all tissues
    :param save_fig: bool, whether to save the figure
    :param kwargs: additional keyword arguments for figure customization, including:
        - figsize: tuple, the size of the figure (width, height) in inches
        - dpi: int, the resolution of the figure in dots per inch
    """

    import os

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if "leiden" not in adata.obs:
        raise KeyError(
            "The key 'leiden' is not present in adata.obs. Perform "
            "Leiden clustering before calling this function."
        )

    if "rank_genes_groups" not in adata.uns:
        raise KeyError(
            "The key 'rank_genes_groups' is not present in adata.uns. "
            "Perform DEG analysis before calling this function."
        )

    cell_type_key = kwargs.get("cell_type_key")

    if not isinstance(cell_type_key, str) or not cell_type_key:
        raise ValueError(
            "Provide a non-empty cell-type key using "
            "cell_type_key='...'."
        )

    if cell_type_key not in adata.obs:
        raise KeyError(
            f"The cell-type key '{cell_type_key}' is not present "
            "in adata.obs."
        )

    ranked_genes_df = sc.get.rank_genes_groups_df(
        adata,
        group=cluster_group
    )

    if len(ranked_genes_df) < 6:
        raise ValueError(
            f"Only {len(ranked_genes_df)} ranked genes were available "
            f"for Leiden cluster '{cluster_group}'; six are required."
        )

    cluster_genes = (
        ranked_genes_df
        .head(6)["names"]
        .astype(str)
        .tolist()
    )

    missing_genes = [
        gene
        for gene in cluster_genes
        if gene not in adata.var_names
    ]

    if missing_genes:
        raise KeyError(
            "The following ranked genes are not present in "
            f"adata.var_names: {missing_genes}."
        )

    # ------------------------------------------------------------------
    # Typography
    # ------------------------------------------------------------------
    panel_label_fontsize = kwargs.get(
        "panel_label_fontsize",
        11
    )
    panel_label_y = kwargs.get(
        "panel_label_y",
        1.04
    )
    facet_title_fontsize = kwargs.get(
        "facet_title_fontsize",
        8
    )
    umap_axis_fontsize = kwargs.get(
        "umap_axis_fontsize",
        7
    )
    legend_fontsize = kwargs.get(
        "legend_fontsize",
        6.5
    )
    dotplot_tick_fontsize = kwargs.get(
        "dotplot_tick_fontsize",
        6
    )
    dotplot_label_fontsize = kwargs.get(
        "dotplot_label_fontsize",
        7
    )
    colorbar_label_fontsize = kwargs.get(
        "colorbar_label_fontsize",
        7
    )
    colorbar_tick_fontsize = kwargs.get(
        "colorbar_tick_fontsize",
        6
    )

    # ------------------------------------------------------------------
    # Cell-type legend settings
    # ------------------------------------------------------------------
    cell_type_legend_marker_size = kwargs.get(
        "cell_type_legend_marker_size",
        36.0
    )
    cell_type_legend_labelspacing = kwargs.get(
        "cell_type_legend_labelspacing",
        0.65
    )
    cell_type_legend_handletextpad = kwargs.get(
        "cell_type_legend_handletextpad",
        0.5
    )

    if cell_type_legend_marker_size <= 0:
        raise ValueError(
            "cell_type_legend_marker_size must be greater than 0."
        )

    if cell_type_legend_labelspacing < 0:
        raise ValueError(
            "cell_type_legend_labelspacing must be non-negative."
        )

    # ------------------------------------------------------------------
    # Dot-plot settings
    # ------------------------------------------------------------------
    dotplot_largest_dot = kwargs.get(
        "dotplot_largest_dot",
        60
    )
    dotplot_smallest_dot = kwargs.get(
        "dotplot_smallest_dot",
        0
    )
    dotplot_edge_linewidth = kwargs.get(
        "dotplot_edge_linewidth",
        0.35
    )

    if dotplot_largest_dot <= 0:
        raise ValueError(
            "dotplot_largest_dot must be greater than 0."
        )

    if dotplot_smallest_dot < 0:
        raise ValueError(
            "dotplot_smallest_dot must be non-negative."
        )

    # ------------------------------------------------------------------
    # Expression and colorbar settings
    # ------------------------------------------------------------------
    expression_color_map = kwargs.get(
        "expression_color_map",
        None
    )
    expression_colorbar_label = kwargs.get(
        "expression_colorbar_label",
        "Expression"
    )
    colorbar_gap_mm = kwargs.get(
        "colorbar_gap_mm",
        1.5
    )
    colorbar_width_mm = kwargs.get(
        "colorbar_width_mm",
        1.5
    )
    inter_umap_gap_mm = kwargs.get(
        "inter_umap_gap_mm",
        12.0
    )

    if colorbar_gap_mm < 0:
        raise ValueError(
            "colorbar_gap_mm must be non-negative."
        )

    if colorbar_width_mm <= 0:
        raise ValueError(
            "colorbar_width_mm must be greater than 0."
        )

    if inter_umap_gap_mm < 0:
        raise ValueError(
            "inter_umap_gap_mm must be non-negative."
        )

    # Use one expression scale across all six DEG UMAPs.
    all_values = adata[:, cluster_genes].X

    expression_vmin = kwargs.get(
        "expression_vmin",
        float(all_values.min())
    )
    expression_vmax = kwargs.get(
        "expression_vmax",
        float(all_values.max())
    )

    # ------------------------------------------------------------------
    # Nature Methods figure dimensions
    # ------------------------------------------------------------------
    mm_per_inch = 25.4

    figure_width_mm = kwargs.get(
        "figure_width_mm",
        180.0
    )
    max_figure_height_mm = kwargs.get(
        "max_figure_height_mm",
        247.0
    )
    figure_height_mm = kwargs.get(
        "figure_height_mm",
        max_figure_height_mm
    )

    if figure_width_mm <= 0:
        raise ValueError(
            "figure_width_mm must be greater than 0."
        )

    if max_figure_height_mm <= 0:
        raise ValueError(
            "max_figure_height_mm must be greater than 0."
        )

    if figure_height_mm <= 0:
        raise ValueError(
            "figure_height_mm must be greater than 0."
        )

    figure_height_mm = min(
        figure_height_mm,
        max_figure_height_mm
    )

    # ------------------------------------------------------------------
    # Physical panel spacing
    # ------------------------------------------------------------------
    top_to_dotplot_gap_mm = kwargs.get(
        "top_to_dotplot_gap_mm",
        5.0
    )
    dotplot_to_umap_gap_mm = kwargs.get(
        "dotplot_to_umap_gap_mm",
        14.0
    )
    deg_umap_row_gap_mm = kwargs.get(
        "deg_umap_row_gap_mm",
        8.0
    )
    dotplot_height_mm = kwargs.get(
        "dotplot_height_mm",
        45.0
    )

    if min(
        top_to_dotplot_gap_mm,
        dotplot_to_umap_gap_mm,
        deg_umap_row_gap_mm
    ) < 0:
        raise ValueError(
            "All physical gap settings must be non-negative."
        )

    if dotplot_height_mm <= 0:
        raise ValueError(
            "dotplot_height_mm must be greater than 0."
        )

    # ------------------------------------------------------------------
    # Grid widths
    # ------------------------------------------------------------------
    available_umap_width_mm = (
        figure_width_mm
        - 3 * (
            colorbar_gap_mm
            + colorbar_width_mm
        )
        - 2 * inter_umap_gap_mm
    )

    if available_umap_width_mm <= 0:
        raise ValueError(
            "The requested colorbar and inter-panel spacing leaves "
            "no room for the UMAP panels."
        )

    nominal_umap_width_mm = (
        available_umap_width_mm / 3
    )

    width_ratios = [
        nominal_umap_width_mm,
        colorbar_gap_mm,
        colorbar_width_mm,
        inter_umap_gap_mm,
        nominal_umap_width_mm,
        colorbar_gap_mm,
        colorbar_width_mm,
        inter_umap_gap_mm,
        nominal_umap_width_mm,
        colorbar_gap_mm,
        colorbar_width_mm
    ]

    height_ratios = [
        nominal_umap_width_mm,
        top_to_dotplot_gap_mm,
        dotplot_height_mm,
        dotplot_to_umap_gap_mm,
        nominal_umap_width_mm,
        deg_umap_row_gap_mm,
        nominal_umap_width_mm
    ]

    fig = plt.figure(
        figsize=(
            figure_width_mm / mm_per_inch,
            figure_height_mm / mm_per_inch
        ),
        dpi=300
    )

    gs = gridspec.GridSpec(
        nrows=7,
        ncols=11,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.0,
        hspace=0.0
    )

    # ------------------------------------------------------------------
    # Create axes
    # ------------------------------------------------------------------
    leiden_ax = fig.add_subplot(
        gs[0, 0]
    )
    cell_type_ax = fig.add_subplot(
        gs[0, 4]
    )
    dotplot_ax = fig.add_subplot(
        gs[2, :]
    )

    gene_positions = [
        (4, 0, 2),
        (4, 4, 6),
        (4, 8, 10),
        (6, 0, 2),
        (6, 4, 6),
        (6, 8, 10)
    ]

    gene_axes = []
    gene_colorbar_axes = []

    for row, umap_column, colorbar_column in gene_positions:
        gene_axes.append(
            fig.add_subplot(
                gs[row, umap_column]
            )
        )
        gene_colorbar_axes.append(
            fig.add_subplot(
                gs[row, colorbar_column]
            )
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def format_custom_umap_coordinates(ax):
        number_of_existing_texts = len(
            ax.texts
        )

        plot_customized_UMAP_coordinates(
            ax,
            length=1.0
        )

        for text_artist in ax.texts[
            number_of_existing_texts:
        ]:
            text_artist.set_fontsize(
                umap_axis_fontsize
            )

        ax.xaxis.label.set_size(
            umap_axis_fontsize
        )
        ax.yaxis.label.set_size(
            umap_axis_fontsize
        )

        ax.xaxis.labelpad = 1
        ax.yaxis.labelpad = 1

        ax.tick_params(
            axis="both",
            labelsize=umap_axis_fontsize
        )

        ax.set_box_aspect(
            1
        )
        ax.set_anchor(
            "C"
        )

    def set_panel_label(
            ax,
            label,
            y=None
    ):
        ax.set_title(
            label,
            loc="left",
            fontsize=panel_label_fontsize,
            fontweight="bold",
            y=panel_label_y if y is None else y,
            pad=0
        )

    def format_cell_type_legend(ax):
        """
        Rebuild the cell-type legend without a title and with explicit
        marker size and vertical spacing.
        """

        original_legend = ax.get_legend()

        if original_legend is None:
            return

        legend_handles = getattr(
            original_legend,
            "legend_handles",
            None
        )

        # Compatibility with older Matplotlib versions.
        if legend_handles is None:
            legend_handles = getattr(
                original_legend,
                "legendHandles",
                None
            )

        legend_labels = [
            text_artist.get_text()
            for text_artist in original_legend.get_texts()
        ]

        if legend_handles is None:
            legend_handles, fallback_labels = (
                ax.get_legend_handles_labels()
            )

            if not legend_labels:
                legend_labels = fallback_labels

        legend_handles = list(
            legend_handles
        )
        legend_labels = list(
            legend_labels
        )

        original_legend.remove()

        new_legend = ax.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(
                kwargs.get(
                    "cell_type_legend_x",
                    1.04
                ),
                kwargs.get(
                    "cell_type_legend_y",
                    0.5
                )
            ),
            frameon=False,
            fontsize=legend_fontsize,
            title=None,
            labelspacing=cell_type_legend_labelspacing,
            handletextpad=cell_type_legend_handletextpad,
            borderaxespad=0
        )

        # Give every categorical marker an explicit, compact size.
        final_legend_handles = getattr(
            new_legend,
            "legend_handles",
            None
        )

        if final_legend_handles is None:
            final_legend_handles = getattr(
                new_legend,
                "legendHandles",
                []
            )

        for legend_handle in final_legend_handles:
            if hasattr(
                legend_handle,
                "set_sizes"
            ):
                legend_handle.set_sizes([
                    cell_type_legend_marker_size
                ])

            elif hasattr(
                legend_handle,
                "set_markersize"
            ):
                legend_handle.set_markersize(
                    np.sqrt(
                        cell_type_legend_marker_size
                    )
                )

        # Keep the legend in the unused space to the right of panel b
        # without changing the dimensions of the UMAP axis.
        new_legend.set_in_layout(
            False
        )

        if hasattr(
            new_legend,
            "_legend_box"
        ):
            new_legend._legend_box.align = "left"

    def add_expression_colorbar(
            umap_ax,
            colorbar_ax
    ):
        mappable = None

        for collection in umap_ax.collections:
            values = collection.get_array()

            if (
                values is not None
                and np.size(values) > 0
            ):
                mappable = collection
                break

        if mappable is None:
            raise RuntimeError(
                "Could not identify the continuous expression "
                "mappable created by Scanpy."
            )

        colorbar = fig.colorbar(
            mappable,
            cax=colorbar_ax
        )

        colorbar.ax.tick_params(
            labelsize=colorbar_tick_fontsize,
            length=2,
            pad=1
        )

        if expression_colorbar_label:
            colorbar.set_label(
                expression_colorbar_label,
                fontsize=colorbar_label_fontsize,
                labelpad=1
            )

        return colorbar

    def normalize_dotplot_axes_result(
            dotplot_result,
            axes_before
    ):
        if isinstance(
            dotplot_result,
            dict
        ):
            return dotplot_result

        if (
            dotplot_result is not None
            and hasattr(
                dotplot_result,
                "get_axes"
            )
        ):
            result_axes = (
                dotplot_result.get_axes()
            )

            if isinstance(
                result_axes,
                dict
            ):
                return result_axes

        new_axes = [
            current_ax
            for current_ax in fig.axes
            if current_ax not in axes_before
        ]

        if new_axes:
            return {
                f"dotplot_ax_{index}": current_ax
                for index, current_ax in enumerate(
                    new_axes
                )
            }

        return {
            "mainplot_ax": dotplot_ax
        }

    # ------------------------------------------------------------------
    # Panel a: Leiden UMAP
    # ------------------------------------------------------------------
    sc.pl.umap(
        adata,
        color="leiden",
        title="",
        legend_loc="on data",
        legend_fontsize=legend_fontsize,
        frameon=False,
        show=False,
        ax=leiden_ax
    )

    format_custom_umap_coordinates(
        leiden_ax
    )

    # Same centered title style as the gene names in panel d.
    leiden_ax.set_title(
        "Leiden clusters",
        fontsize=facet_title_fontsize,
        pad=2
    )

    set_panel_label(
        leiden_ax,
        "a"
    )

    # ------------------------------------------------------------------
    # Panel b: cell-type UMAP
    # ------------------------------------------------------------------
    sc.pl.umap(
        adata,
        color=cell_type_key,
        title="",
        legend_loc="right margin",
        legend_fontsize=legend_fontsize,
        frameon=False,
        show=False,
        ax=cell_type_ax
    )

    format_custom_umap_coordinates(
        cell_type_ax
    )

    # Remove the legend title and use a centered UMAP title instead.
    format_cell_type_legend(
        cell_type_ax
    )

    cell_type_ax.set_title(
        "Cell type",
        fontsize=facet_title_fontsize,
        pad=2
    )

    set_panel_label(
        cell_type_ax,
        "b"
    )

    # ------------------------------------------------------------------
    # Panel c: DEG dot plot
    # ------------------------------------------------------------------
    axes_before_dotplot = set(
        fig.axes
    )

    dotplot_object = sc.pl.rank_genes_groups_dotplot(
        adata,
        groupby="leiden",
        standard_scale="var",
        n_genes=6,
        return_fig=True,
        show=False,
        ax=dotplot_ax
    )

    dotplot_object.style(
        smallest_dot=dotplot_smallest_dot,
        largest_dot=dotplot_largest_dot,
        dot_edge_lw=dotplot_edge_linewidth
    )

    dotplot_object.make_figure()

    dotplot_result = (
        dotplot_object.get_axes()
    )

    dotplot_axes_dict = (
        normalize_dotplot_axes_result(
            dotplot_result,
            axes_before_dotplot
        )
    )

    dotplot_related_axes = list(
        dict.fromkeys(
            dotplot_axes_dict.values()
        )
    )

    for current_ax in dotplot_related_axes:
        current_ax.tick_params(
            axis="both",
            labelsize=dotplot_tick_fontsize,
            length=2
        )

        current_ax.xaxis.label.set_size(
            dotplot_label_fontsize
        )
        current_ax.yaxis.label.set_size(
            dotplot_label_fontsize
        )
        current_ax.title.set_fontsize(
            dotplot_label_fontsize
        )

        for text_artist in current_ax.texts:
            text_artist.set_fontsize(
                dotplot_tick_fontsize
            )

    dotplot_ax.set_title(
        ""
    )

    dotplot_panel_axes = [
        current_ax
        for axis_name, current_ax
        in dotplot_axes_dict.items()
        if "legend" not in axis_name.lower()
    ]

    if not dotplot_panel_axes:
        dotplot_panel_axes = [
            dotplot_ax
        ]

    # ------------------------------------------------------------------
    # Panel d: six DEG expression UMAPs
    # ------------------------------------------------------------------
    colorbar_pairs = []

    for gene_index, (
        gene,
        gene_ax,
        colorbar_ax
    ) in enumerate(
        zip(
            cluster_genes,
            gene_axes,
            gene_colorbar_axes
        )
    ):
        sc.pl.umap(
            adata,
            color=gene,
            title=str(gene),
            color_map=expression_color_map,
            colorbar_loc=None,
            frameon=False,
            show=False,
            ax=gene_ax,
            vmin=expression_vmin,
            vmax=expression_vmax
        )

        gene_ax.set_title(
            str(gene),
            fontsize=facet_title_fontsize,
            pad=2
        )

        format_custom_umap_coordinates(
            gene_ax
        )

        expression_colorbar = (
            add_expression_colorbar(
                gene_ax,
                colorbar_ax
            )
        )

        colorbar_pairs.append(
            (
                gene_ax,
                expression_colorbar
            )
        )

        if gene_index == 0:
            set_panel_label(
                gene_ax,
                "d",
                y=1.04
            )

    # ------------------------------------------------------------------
    # Final layout
    # ------------------------------------------------------------------
    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.99
        ),
        pad=0.25,
        w_pad=0.0,
        h_pad=0.0
    )

    fig.canvas.draw()

    # Reposition expression colorbars with fixed physical dimensions.
    colorbar_gap_fraction = (
        colorbar_gap_mm
        / figure_width_mm
    )
    colorbar_width_fraction = (
        colorbar_width_mm
        / figure_width_mm
    )

    for gene_ax, colorbar in colorbar_pairs:
        parent_position = (
            gene_ax.get_position()
        )
        colorbar_ax = colorbar.ax

        colorbar_ax.set_axes_locator(
            None
        )
        colorbar_ax.set_in_layout(
            False
        )

        if hasattr(
            colorbar_ax,
            "set_box_aspect"
        ):
            colorbar_ax.set_box_aspect(
                None
            )

        colorbar_ax.set_position([
            parent_position.x1
            + colorbar_gap_fraction,
            parent_position.y0,
            colorbar_width_fraction,
            parent_position.height
        ])

    # Position panel label c relative to the rendered dot plot.
    panel_c_x = min(
        current_ax.get_position().x0
        for current_ax in dotplot_panel_axes
    )
    panel_c_y = max(
        current_ax.get_position().y1
        for current_ax in dotplot_panel_axes
    )

    panel_c_y_offset_mm = kwargs.get(
        "panel_c_y_offset_mm",
        1.0
    )

    panel_c_y += (
        panel_c_y_offset_mm
        / figure_height_mm
    )

    fig.text(
        panel_c_x,
        min(panel_c_y, 0.995),
        "c",
        fontsize=panel_label_fontsize,
        fontweight="bold",
        ha="left",
        va="bottom",
        transform=fig.transFigure
    )

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    if save_fig:
        dataset_name = kwargs.get(
            "dataset_name",
            "default"
        )
        file_suffix = kwargs.get(
            "file_suffix",
            "pdf"
        )

        save_path = os.path.join(
            "./figures",
            dataset_name
        )

        os.makedirs(
            save_path,
            exist_ok=True
        )

        if taxonomy_level is not None:
            filename = (
                f"deg_analysis_{model_name}_{taxonomy_level}_"
                f"clustergroup_{cluster_group}.{file_suffix}"
            )
        else:
            filename = (
                f"deg_analysis_{model_name}_"
                f"clustergroup_{cluster_group}.{file_suffix}"
            )

        fig_name = os.path.join(
            save_path,
            filename
        )

        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches=None,
            pad_inches=0
        )

    plt.show()

def rename_isoform_helper(intron_names: np.ndarray) -> np.ndarray:
    r"""
    Rename isoforms to "gene_name isoform_number" format, where isoform_number is assigned based on the order of
    appearance of the intron names for each gene.

    :param intron_names:
    """
    # Dictionary to store: { gene_name: { location_string: isoform_number } }
    gene_isoform_tracker = {}
    new_names = []

    for entry in intron_names:
        # Split only at the first underscore to separate gene from location
        if entry is not "0":
            gene, location = entry.split('_', 1)

            if gene not in gene_isoform_tracker:
                gene_isoform_tracker[gene] = {}

            # If we haven't seen this specific location for this gene yet, assign a new number
            if location not in gene_isoform_tracker[gene]:
                current_count = len(gene_isoform_tracker[gene]) + 1
                gene_isoform_tracker[gene][location] = current_count

            # Retrieve the isoform number and format the new name
            iso_id = gene_isoform_tracker[gene][location]
            new_names.append(f"{gene} iso {iso_id}")
        else:
            new_names.append(entry)

    intron_names_plot = np.array(new_names)

    return intron_names_plot

def dsg_dotplot_helper(
        adata: AnnData,
        cluster_groups: List[str],
        num_intron_group_markers: int = 3,
        **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Given an AnnData object containing the results of the DSG analysis for a specific cell taxonomy level (e.g. tissue)
    and cluster groups (e.g. ["0", "1", "2"] or ["EC", "OPC", "brain pericyte"]), calculate the mean psi value for each
    marker intron (DSG isoform), the number of cells where it is present. The mean psi values, the number of cells, and
    the intron names are returned as numpy arrays. The mean psi values and the number of cells are calculated for the
    top num_intron_group_markers DSGs for each cluster group specified in cluster_groups. The DSG analysis should have
    been performed using the tuVI or TRVI latent space and the results should be present in
    `adata.uns["rank_introns_groups"]`. The cluster groups specified in cluster_groups should be present in
    `adata.uns["rank_introns_groups"]["names"]`. The raw PSI scores should be present in `adata.layers["PSI_raw"]`.

    :param adata: AnnData object containing the results of the DSG analysis for a specific tissue and cluster groups
    :param cluster_groups: List of str, the Leiden cluster groups for which the mean psi values and number of cells will be calculated for the top DSGs
    :param num_intron_group_markers: int, the number of top DSGs to be considered for each cluster group specified in cluster_groups. Default is 3.
    :return:
    """

    assert "rank_introns_groups" in adata.uns, (
        "Key 'rank_introns_groups' is not present in adata.uns. "
        "Run a DSG analysis first"
    )

    assert "PSI_raw" in adata.layers, (
        "Key 'PSI_raw' is not present in adata.layers. "
        "Please add raw PSI scores to adata.layers['PSI_raw'] before running this function."
    )

    n_groups = len(cluster_groups)
    n_introns_total = num_intron_group_markers * n_groups

    intron_names = np.zeros(n_introns_total, dtype=object)

    # Fill intron_names according to the order in cluster_groups
    for i, test_group in enumerate(cluster_groups):
        intron_names_group = adata.uns["rank_introns_groups"]["names"][test_group][
            :num_intron_group_markers
        ]

        start = i * num_intron_group_markers
        end = (i + 1) * num_intron_group_markers
        intron_names[start:end] = intron_names_group

    psi_matrix = np.zeros((adata.shape[0], n_introns_total))

    for j, intron_name in enumerate(intron_names):
        ####### NEW TRIAL
        if intron_name in ("0", "", None) or intron_name not in adata.var_names:
            psi_matrix[:, j] = np.nan
            continue
        #######
        psi_values = adata[:, intron_name].layers["PSI_raw"]
        psi_matrix[:, j] = np.asarray(psi_values).reshape(-1)

    cluster_per_cell = adata.obs["leiden"].astype(str).to_numpy()

    mean_psi_groups_introns = np.zeros((n_groups, n_introns_total))
    num_cells_valid_matrix = np.zeros((n_groups, n_introns_total))

    # Fill rows according to the order in cluster_groups
    for i, test_group in enumerate(cluster_groups):
        cells_cluster_mask = cluster_per_cell == str(test_group)

        for j, intron_name in enumerate(intron_names):
            intron_psi_values = psi_matrix[cells_cluster_mask, j]

            num_cells_valid = np.sum(~np.isnan(intron_psi_values))

            #### NEW TRIAL
            if num_cells_valid == 0:
                mean_psi_values = np.nan
            else:
                mean_psi_values = np.nanmean(intron_psi_values)
            ####
            # OLD
            #mean_psi_values = np.nanmean(intron_psi_values)

            mean_psi_groups_introns[i, j] = mean_psi_values
            num_cells_valid_matrix[i, j] = num_cells_valid

    return mean_psi_groups_introns, num_cells_valid_matrix, intron_names

def plot_dsg_analysis(
        adata: AnnData,
        cluster_group: str,
        cluster_groups: List[str],
        model_name: str,
        taxonomy_level: str | None,
        rename_isoforms: bool = True,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given an Anndata object containing the results of the DSG analysis for a specific tissue and cluster group, plot the
    UMAPs of the clusters, the dotplot of the DSGs across cluster groups, and the top 3 DSG isoforms for the cluster
    group specified. The DSG analysis should have been performed using the tuVI or TRVI latent space. The chosen
    model and TU likelihood key should be represented in model_name (e.g. tuVI_ZIDM, TRVI_ZINB_ZIDM, etc.). If the
    Anndata object was filtered to a taxonomy level (e.g. tissue) specify the taxonomy level name otherwise set taxonomy
    level to None. The cluster_group specifies the Leiden cluster group for which the top 3 DSG isoforms will be
    plotted. The isoforms are renamed numbered according to location if rename_isoforms is set to True.The figure is
    saved in the figures directory if save_fig is set to True.

    The default layout is 180 mm wide and uses equally sized square
    UMAP plotting areas. Panel labels follow the order:
        a: Leiden clusters
        b: Cell type
        c: DSG dot plot
        d: Top three DSG isoforms

    :param adata: AnnData object containing the results of the DSG analysis for a specific tissue and cluster group
    :param cluster_group: str, the Leiden cluster group for which the top 3 DSG isoforms will be plotted
    :param cluster_groups: List of str, the Leiden cluster groups to be included in the dotplot of DSGs across cluster groups
    :param model_name: str, the name of the model used for the DSG analysis (e.g. tuVI_ZIDM, TRVI_ZINB_ZIDM, etc.)
    :param taxonomy_level: str or None, the taxonomy level to be plotted (e.g. tissue)
    :param rename_isoforms: bool, whether to rename the isoforms to "gene_name isoform_number" format, where isoform_number is assigned based on the order of appearance of the intron names for each gene
    :param save_fig: bool, whether to save the figure
    :param kwargs: Additional keyword arguments for customization, including:
        - psi_layer: str, the name of the layer in adata.layers containing the raw PSI scores (default: "PSI_raw")
        - cell_type_key: str, the key in adata.obs containing the cell type annotations (required)
        - num_intron_group_markers: int, the number of top DSG isoforms to be considered for each cluster group specified in cluster_groups (default: 3)
        - panel_label_fontsize: int, the font size for the panel labels (default: 11)
        - panel_label_y: float, the y position  of the panel labels (default: 1.04)
        - facet_title_fontsize: int, the font size for the facet titles (default: 8)
        - umap_axis_fontsize: int, the font size for the UMAP axis labels (default: 7)
        - legend_fontsize: int, the font size for the legends (default: 6.5)
        - dotplot_smallest_dot: float, the size of the smallest dot in the dotplot (default: 0.5)
        - dotplot_largest_dot: float, the size of the largest dot in the dotplot (default: 5.0)
        - dotplot_edge_linewidth: float, the linewidth of the edges of the dots in the dotplot (default: 0.5)
        - dotplot_tick_fontsize: int, the font size for the tick labels in the dotplot (default: 6)
        - dotplot_label_fontsize: int, the font size for the axis labels in the dotplot (default: 7)
        - expression_color_map: str, the colormap to be used for the expression UMAPs (default: "viridis")
        - expression_vmin: float, the minimum value for the expression color scale (default: None, which uses the minimum value in the data)
        - expression_vmax: float, the maximum value for the expression color scale (default: None, which uses the maximum value in the data)
        - expression_colorbar_label: str, the label for the expression colorbar (default: None, which uses no label)
        - colorbar_tick_fontsize: int, the font size for the colorbar tick labels (default: 6)
        - colorbar_label_fontsize: int, the font
    """

    import os
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if "leiden" not in adata.obs:
        raise KeyError(
            "The key 'leiden' is not present in adata.obs. Perform "
            "Leiden clustering before calling this function."
        )

    psi_layer = kwargs.get(
        "psi_layer",
        "PSI_raw"
    )

    if psi_layer not in adata.layers:
        raise KeyError(
            f"The PSI layer '{psi_layer}' is not present in "
            "adata.layers."
        )

    if "rank_introns_groups" not in adata.uns:
        raise KeyError(
            "The key 'rank_introns_groups' is not present in "
            "adata.uns. Perform DSG analysis before calling this "
            "function."
        )

    cell_type_key = kwargs.get(
        "cell_type_key"
    )

    if not isinstance(cell_type_key, str) or not cell_type_key:
        raise ValueError(
            "Provide a non-empty cell-type key using "
            "cell_type_key='...'."
        )

    if cell_type_key not in adata.obs:
        raise KeyError(
            f"The cell-type key '{cell_type_key}' is not present "
            "in adata.obs."
        )

    if not cluster_groups:
        raise ValueError(
            "cluster_groups must contain at least one cluster."
        )

    cluster_groups_array = np.asarray(
        cluster_groups
    ).astype(str)

    cluster_group_matches = np.flatnonzero(
        cluster_groups_array == str(cluster_group)
    )

    if len(cluster_group_matches) == 0:
        raise ValueError(
            f"cluster_group '{cluster_group}' is not present in "
            "cluster_groups."
        )

    if len(cluster_group_matches) > 1:
        raise ValueError(
            f"cluster_group '{cluster_group}' occurs more than once "
            "in cluster_groups."
        )

    # The dot plot may contain more than three markers per cluster,
    # but panel d displays the first three for the selected cluster.
    num_intron_group_markers = kwargs.get(
        "num_intron_group_markers",
        3
    )

    if num_intron_group_markers < 3:
        raise ValueError(
            "num_intron_group_markers must be at least 3 because "
            "panel d displays three isoform UMAPs."
        )

    # ------------------------------------------------------------------
    # Obtain DSG dot-plot data
    # ------------------------------------------------------------------
    (
        mean_psi_groups_introns,
        num_cells_valid_matrix,
        intron_names
    ) = dsg_dotplot_helper(
        adata=adata,
        cluster_groups=cluster_groups,
        num_intron_group_markers=num_intron_group_markers
    )

    mean_psi_groups_introns = np.asarray(
        mean_psi_groups_introns
    )
    num_cells_valid_matrix = np.asarray(
        num_cells_valid_matrix
    )
    intron_names = np.asarray(
        intron_names
    )

    expected_shape = (
        len(cluster_groups),
        len(intron_names)
    )

    if mean_psi_groups_introns.shape != expected_shape:
        raise ValueError(
            "mean_psi_groups_introns has shape "
            f"{mean_psi_groups_introns.shape}; expected "
            f"{expected_shape}."
        )

    if num_cells_valid_matrix.shape != expected_shape:
        raise ValueError(
            "num_cells_valid_matrix has shape "
            f"{num_cells_valid_matrix.shape}; expected "
            f"{expected_shape}."
        )

    if rename_isoforms:
        intron_names_plot = np.asarray(
            rename_isoform_helper(
                intron_names
            )
        )
    else:
        intron_names_plot = intron_names.astype(
            str
        )

    if len(intron_names_plot) != len(intron_names):
        raise ValueError(
            "rename_isoform_helper changed the number of isoform "
            "labels."
        )

    selected_group_index = int(
        cluster_group_matches[0]
    )

    selected_start_index = (
        num_intron_group_markers
        * selected_group_index
    )

    selected_intron_indices = np.arange(
        selected_start_index,
        selected_start_index + 3
    )

    if selected_intron_indices[-1] >= len(intron_names):
        raise IndexError(
            "The DSG dot-plot output does not contain three marker "
            f"isoforms for cluster '{cluster_group}'."
        )

    valid_selected_indices = [
        index
        for index in selected_intron_indices
        if str(intron_names[index]) != "0"
    ]

    if not valid_selected_indices:
        raise ValueError(
            f"No valid DSG marker isoforms were available for "
            f"cluster '{cluster_group}'."
        )

    # ------------------------------------------------------------------
    # Typography
    # ------------------------------------------------------------------
    panel_label_fontsize = kwargs.get(
        "panel_label_fontsize",
        11
    )
    panel_label_y = kwargs.get(
        "panel_label_y",
        1.04
    )
    facet_title_fontsize = kwargs.get(
        "facet_title_fontsize",
        8
    )
    umap_axis_fontsize = kwargs.get(
        "umap_axis_fontsize",
        7
    )
    legend_fontsize = kwargs.get(
        "legend_fontsize",
        6.5
    )
    dotplot_tick_fontsize = kwargs.get(
        "dotplot_tick_fontsize",
        6
    )
    dotplot_label_fontsize = kwargs.get(
        "dotplot_label_fontsize",
        7
    )
    colorbar_label_fontsize = kwargs.get(
        "colorbar_label_fontsize",
        7
    )
    colorbar_tick_fontsize = kwargs.get(
        "colorbar_tick_fontsize",
        6
    )

    # ------------------------------------------------------------------
    # Cell-type legend settings
    # ------------------------------------------------------------------
    cell_type_legend_marker_size = kwargs.get(
        "cell_type_legend_marker_size",
        36.0
    )
    cell_type_legend_labelspacing = kwargs.get(
        "cell_type_legend_labelspacing",
        0.65
    )
    cell_type_legend_handletextpad = kwargs.get(
        "cell_type_legend_handletextpad",
        0.5
    )

    if cell_type_legend_marker_size <= 0:
        raise ValueError(
            "cell_type_legend_marker_size must be greater than 0."
        )

    if cell_type_legend_labelspacing < 0:
        raise ValueError(
            "cell_type_legend_labelspacing must be non-negative."
        )

    # ------------------------------------------------------------------
    # DSG dot-plot settings
    # ------------------------------------------------------------------
    dotplot_row_spacing = kwargs.get(
        "dotplot_row_spacing",
        1.0
    )
    dotplot_largest_dot = kwargs.get(
        "dotplot_largest_dot",
        60.0
    )
    dotplot_edge_linewidth = kwargs.get(
        "dotplot_edge_linewidth",
        0.35
    )
    dotplot_alpha = kwargs.get(
        "dotplot_alpha",
        0.9
    )
    dotplot_color_map = kwargs.get(
        "dotplot_color_map",
        "Reds"
    )

    if dotplot_row_spacing <= 0:
        raise ValueError(
            "dotplot_row_spacing must be greater than 0."
        )

    if dotplot_largest_dot <= 0:
        raise ValueError(
            "dotplot_largest_dot must be greater than 0."
        )

    # ------------------------------------------------------------------
    # PSI settings
    # ------------------------------------------------------------------
    psi_vmin = kwargs.get(
        "psi_vmin",
        0.0
    )
    psi_vmax = kwargs.get(
        "psi_vmax",
        1.0
    )
    psi_color_map = kwargs.get(
        "psi_color_map",
        None
    )
    psi_colorbar_label = kwargs.get(
        "psi_colorbar_label",
        "PSI"
    )

    if psi_vmin >= psi_vmax:
        raise ValueError(
            "psi_vmin must be smaller than psi_vmax."
        )

    # ------------------------------------------------------------------
    # Colorbar and horizontal spacing
    # ------------------------------------------------------------------
    colorbar_gap_mm = kwargs.get(
        "colorbar_gap_mm",
        1.5
    )
    colorbar_width_mm = kwargs.get(
        "colorbar_width_mm",
        1.5
    )
    inter_umap_gap_mm = kwargs.get(
        "inter_umap_gap_mm",
        12.0
    )

    if colorbar_gap_mm < 0:
        raise ValueError(
            "colorbar_gap_mm must be non-negative."
        )

    if colorbar_width_mm <= 0:
        raise ValueError(
            "colorbar_width_mm must be greater than 0."
        )

    if inter_umap_gap_mm < 0:
        raise ValueError(
            "inter_umap_gap_mm must be non-negative."
        )

    # ------------------------------------------------------------------
    # Nature Methods figure width
    # ------------------------------------------------------------------
    mm_per_inch = 25.4

    figure_width_mm = kwargs.get(
        "figure_width_mm",
        180.0
    )
    max_figure_height_mm = kwargs.get(
        "max_figure_height_mm",
        247.0
    )

    if figure_width_mm <= 0:
        raise ValueError(
            "figure_width_mm must be greater than 0."
        )

    if max_figure_height_mm <= 0:
        raise ValueError(
            "max_figure_height_mm must be greater than 0."
        )

    # ------------------------------------------------------------------
    # Grid widths
    # ------------------------------------------------------------------
    available_umap_width_mm = (
        figure_width_mm
        - 3 * (
            colorbar_gap_mm
            + colorbar_width_mm
        )
        - 2 * inter_umap_gap_mm
    )

    if available_umap_width_mm <= 0:
        raise ValueError(
            "The requested colorbar and inter-panel spacing leaves "
            "no room for the UMAP panels."
        )

    nominal_umap_width_mm = (
        available_umap_width_mm / 3
    )

    width_ratios = [
        nominal_umap_width_mm,
        colorbar_gap_mm,
        colorbar_width_mm,
        inter_umap_gap_mm,
        nominal_umap_width_mm,
        colorbar_gap_mm,
        colorbar_width_mm,
        inter_umap_gap_mm,
        nominal_umap_width_mm,
        colorbar_gap_mm,
        colorbar_width_mm
    ]

    # ------------------------------------------------------------------
    # Vertical spacing and adaptive height
    # ------------------------------------------------------------------
    top_to_dotplot_gap_mm = kwargs.get(
        "top_to_dotplot_gap_mm",
        5.0
    )
    dotplot_height_mm = kwargs.get(
        "dotplot_height_mm",
        45.0
    )

    # The DSG/isoform labels can be longer than gene symbols.
    dotplot_to_umap_gap_mm = kwargs.get(
        "dotplot_to_umap_gap_mm",
        22.0
    )
    outer_vertical_padding_mm = kwargs.get(
        "outer_vertical_padding_mm",
        8.0
    )

    if min(
        top_to_dotplot_gap_mm,
        dotplot_to_umap_gap_mm,
        outer_vertical_padding_mm
    ) < 0:
        raise ValueError(
            "All physical gap settings must be non-negative."
        )

    if dotplot_height_mm <= 0:
        raise ValueError(
            "dotplot_height_mm must be greater than 0."
        )

    default_figure_height_mm = (
        2 * nominal_umap_width_mm
        + top_to_dotplot_gap_mm
        + dotplot_height_mm
        + dotplot_to_umap_gap_mm
        + outer_vertical_padding_mm
    )

    figure_height_mm = kwargs.get(
        "figure_height_mm",
        default_figure_height_mm
    )

    if figure_height_mm <= 0:
        raise ValueError(
            "figure_height_mm must be greater than 0."
        )

    figure_height_mm = min(
        figure_height_mm,
        max_figure_height_mm
    )

    height_ratios = [
        nominal_umap_width_mm,
        top_to_dotplot_gap_mm,
        dotplot_height_mm,
        dotplot_to_umap_gap_mm,
        nominal_umap_width_mm
    ]

    fig = plt.figure(
        figsize=(
            figure_width_mm / mm_per_inch,
            figure_height_mm / mm_per_inch
        ),
        dpi=300
    )

    gs = gridspec.GridSpec(
        nrows=5,
        ncols=11,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.0,
        hspace=0.0
    )

    # ------------------------------------------------------------------
    # Main axes
    # ------------------------------------------------------------------
    leiden_ax = fig.add_subplot(
        gs[0, 0]
    )
    cell_type_ax = fig.add_subplot(
        gs[0, 4]
    )

    # The custom DSG dot plot has its own legend column.
    dotplot_subgrid = gridspec.GridSpecFromSubplotSpec(
        nrows=1,
        ncols=2,
        subplot_spec=gs[2, :],
        width_ratios=[
            kwargs.get(
                "dotplot_width_ratio",
                4.0
            ),
            1.0
        ],
        wspace=kwargs.get(
            "dotplot_legend_wspace",
            0.12
        )
    )

    dotplot_ax = fig.add_subplot(
        dotplot_subgrid[0, 0]
    )
    dotplot_legend_ax = fig.add_subplot(
        dotplot_subgrid[0, 1]
    )

    isoform_axes = [
        fig.add_subplot(
            gs[4, 0]
        ),
        fig.add_subplot(
            gs[4, 4]
        ),
        fig.add_subplot(
            gs[4, 8]
        )
    ]

    isoform_colorbar_axes = [
        fig.add_subplot(
            gs[4, 2]
        ),
        fig.add_subplot(
            gs[4, 6]
        ),
        fig.add_subplot(
            gs[4, 10]
        )
    ]

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def format_custom_umap_coordinates(ax):
        number_of_existing_texts = len(
            ax.texts
        )

        plot_customized_UMAP_coordinates(
            ax,
            length=1.0
        )

        for text_artist in ax.texts[
            number_of_existing_texts:
        ]:
            text_artist.set_fontsize(
                umap_axis_fontsize
            )

        ax.xaxis.label.set_size(
            umap_axis_fontsize
        )
        ax.yaxis.label.set_size(
            umap_axis_fontsize
        )

        ax.xaxis.labelpad = 1
        ax.yaxis.labelpad = 1

        ax.tick_params(
            axis="both",
            labelsize=umap_axis_fontsize
        )

        ax.set_box_aspect(
            1
        )
        ax.set_anchor(
            "C"
        )

    def set_panel_label(
            ax,
            label,
            y=None
    ):
        ax.set_title(
            label,
            loc="left",
            fontsize=panel_label_fontsize,
            fontweight="bold",
            y=panel_label_y if y is None else y,
            pad=0
        )

    def format_cell_type_legend(ax):
        """
        Rebuild the cell-type legend without a title and with compact,
        non-overlapping markers.
        """

        original_legend = ax.get_legend()

        if original_legend is None:
            return

        legend_handles = getattr(
            original_legend,
            "legend_handles",
            None
        )

        if legend_handles is None:
            legend_handles = getattr(
                original_legend,
                "legendHandles",
                None
            )

        legend_labels = [
            text_artist.get_text()
            for text_artist in original_legend.get_texts()
        ]

        if legend_handles is None:
            legend_handles, fallback_labels = (
                ax.get_legend_handles_labels()
            )

            if not legend_labels:
                legend_labels = fallback_labels

        original_legend.remove()

        new_legend = ax.legend(
            list(legend_handles),
            list(legend_labels),
            loc="center left",
            bbox_to_anchor=(
                kwargs.get(
                    "cell_type_legend_x",
                    1.04
                ),
                kwargs.get(
                    "cell_type_legend_y",
                    0.5
                )
            ),
            frameon=False,
            fontsize=legend_fontsize,
            title=None,
            labelspacing=cell_type_legend_labelspacing,
            handletextpad=cell_type_legend_handletextpad,
            borderaxespad=0
        )

        final_legend_handles = getattr(
            new_legend,
            "legend_handles",
            None
        )

        if final_legend_handles is None:
            final_legend_handles = getattr(
                new_legend,
                "legendHandles",
                []
            )

        for legend_handle in final_legend_handles:
            if hasattr(
                legend_handle,
                "set_sizes"
            ):
                legend_handle.set_sizes([
                    cell_type_legend_marker_size
                ])

            elif hasattr(
                legend_handle,
                "set_markersize"
            ):
                legend_handle.set_markersize(
                    np.sqrt(
                        cell_type_legend_marker_size
                    )
                )

        new_legend.set_in_layout(
            False
        )

        if hasattr(
            new_legend,
            "_legend_box"
        ):
            new_legend._legend_box.align = "left"

    def add_psi_colorbar(
            umap_ax,
            colorbar_ax
    ):
        """Add a PSI colorbar without resizing the UMAP."""

        mappable = None

        for collection in umap_ax.collections:
            values = collection.get_array()

            if (
                values is not None
                and np.size(values) > 0
            ):
                mappable = collection
                break

        if mappable is None:
            raise RuntimeError(
                "Could not identify the continuous PSI mappable "
                "created by Scanpy."
            )

        colorbar = fig.colorbar(
            mappable,
            cax=colorbar_ax
        )

        colorbar.ax.tick_params(
            labelsize=colorbar_tick_fontsize,
            length=2,
            pad=1
        )

        if psi_colorbar_label:
            colorbar.set_label(
                psi_colorbar_label,
                fontsize=colorbar_label_fontsize,
                labelpad=1
            )

        return colorbar

    # ------------------------------------------------------------------
    # Panel a: Leiden UMAP
    # ------------------------------------------------------------------
    sc.pl.umap(
        adata,
        color="leiden",
        title="",
        legend_loc="on data",
        legend_fontsize=legend_fontsize,
        frameon=False,
        show=False,
        ax=leiden_ax
    )

    format_custom_umap_coordinates(
        leiden_ax
    )

    leiden_ax.set_title(
        "Leiden clusters",
        fontsize=facet_title_fontsize,
        pad=2
    )

    set_panel_label(
        leiden_ax,
        "a"
    )

    # ------------------------------------------------------------------
    # Panel b: cell-type UMAP
    # ------------------------------------------------------------------
    sc.pl.umap(
        adata,
        color=cell_type_key,
        title="",
        legend_loc="right margin",
        legend_fontsize=legend_fontsize,
        frameon=False,
        show=False,
        ax=cell_type_ax
    )

    format_custom_umap_coordinates(
        cell_type_ax
    )
    format_cell_type_legend(
        cell_type_ax
    )

    cell_type_ax.set_title(
        "Cell type",
        fontsize=facet_title_fontsize,
        pad=2
    )

    set_panel_label(
        cell_type_ax,
        "b"
    )

    # ------------------------------------------------------------------
    # Panel c: DSG dot plot
    # ------------------------------------------------------------------
    x_grid, y_grid = np.meshgrid(
        np.arange(
            len(intron_names)
        ),
        np.arange(
            len(cluster_groups)
        )
    )

    x_coordinates = x_grid.ravel()
    y_coordinates = (
        y_grid * dotplot_row_spacing
    ).ravel()

    color_values = (
        mean_psi_groups_introns.ravel()
    )
    size_values = np.nan_to_num(
        num_cells_valid_matrix.ravel(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    size_values = np.clip(
        size_values,
        a_min=0.0,
        a_max=None
    )

    size_legend_values = kwargs.get(
        "size_legend_values",
        [50, 100, 250, 500]
    )

    if not size_legend_values:
        raise ValueError(
            "size_legend_values must contain at least one value."
        )

    size_legend_values = np.asarray(
        size_legend_values,
        dtype=float
    )

    if np.any(size_legend_values <= 0):
        raise ValueError(
            "All size_legend_values must be greater than 0."
        )

    size_reference_maximum = max(
        float(
            np.max(
                size_values
            )
        ) if size_values.size else 0.0,
        float(
            np.max(
                size_legend_values
            )
        ),
        1.0
    )

    dot_size_scale = (
        dotplot_largest_dot
        / size_reference_maximum
    )

    scaled_dot_sizes = (
        size_values
        * dot_size_scale
    )

    psi_dotplot = dotplot_ax.scatter(
        x_coordinates,
        y_coordinates,
        c=color_values,
        s=scaled_dot_sizes,
        cmap=dotplot_color_map,
        edgecolors="black",
        linewidths=dotplot_edge_linewidth,
        alpha=dotplot_alpha,
        vmin=psi_vmin,
        vmax=psi_vmax
    )

    dotplot_ax.set_xticks(
        np.arange(
            len(intron_names_plot)
        )
    )
    dotplot_ax.set_xticklabels(
        intron_names_plot,
        rotation=90,
        ha="center",
        va="top",
        fontsize=dotplot_tick_fontsize
    )

    y_tick_positions = (
        np.arange(
            len(cluster_groups)
        )
        * dotplot_row_spacing
    )

    dotplot_ax.set_yticks(
        y_tick_positions
    )
    dotplot_ax.set_yticklabels(
        cluster_groups,
        fontsize=dotplot_tick_fontsize
    )

    dotplot_ax.set_ylim(
        -0.5 * dotplot_row_spacing,
        (
            len(cluster_groups)
            - 0.5
        ) * dotplot_row_spacing
    )
    dotplot_ax.set_xlim(
        -0.5,
        len(intron_names) - 0.5
    )

    dotplot_ax.tick_params(
        axis="both",
        labelsize=dotplot_tick_fontsize,
        length=2,
        pad=1
    )

    dotplot_ax.xaxis.label.set_size(
        dotplot_label_fontsize
    )
    dotplot_ax.yaxis.label.set_size(
        dotplot_label_fontsize
    )

    set_panel_label(
        dotplot_ax,
        "c"
    )

    # ------------------------------------------------------------------
    # DSG dot-plot legends
    # ------------------------------------------------------------------
    dotplot_legend_ax.set_xlim(
        -0.5,
        len(size_legend_values) - 0.5
    )
    dotplot_legend_ax.set_ylim(
        0,
        1
    )
    dotplot_legend_ax.axis(
        "off"
    )

    dotplot_legend_ax.set_title(
        "Number of cells\nin group",
        fontsize=dotplot_label_fontsize,
        pad=2
    )

    legend_x_positions = np.arange(
        len(size_legend_values)
    )

    scaled_legend_sizes = (
        size_legend_values
        * dot_size_scale
    )

    for x_position, marker_size, size_value in zip(
        legend_x_positions,
        scaled_legend_sizes,
        size_legend_values
    ):
        dotplot_legend_ax.scatter(
            x_position,
            0.72,
            s=marker_size,
            color="gray",
            edgecolors="black",
            linewidths=dotplot_edge_linewidth,
            alpha=dotplot_alpha
        )

        dotplot_legend_ax.plot(
            [
                x_position,
                x_position
            ],
            [
                0.49,
                0.57
            ],
            color="black",
            linewidth=0.8
        )

        dotplot_legend_ax.text(
            x_position,
            0.38,
            f"{size_value:g}",
            ha="center",
            va="center",
            fontsize=colorbar_tick_fontsize
        )

    dotplot_colorbar_ax = inset_axes(
        dotplot_legend_ax,
        width="90%",
        height="11%",
        loc="lower center",
        borderpad=0
    )

    dotplot_colorbar = fig.colorbar(
        psi_dotplot,
        cax=dotplot_colorbar_ax,
        orientation="horizontal"
    )

    dotplot_colorbar.set_ticks(
        kwargs.get(
            "dotplot_colorbar_ticks",
            [0.0, 0.5, 1.0]
        )
    )
    dotplot_colorbar.ax.tick_params(
        labelsize=colorbar_tick_fontsize,
        length=2,
        pad=1
    )
    dotplot_colorbar.ax.set_title(
        "Mean PSI value\nin group",
        fontsize=dotplot_label_fontsize,
        pad=4
    )

    # ------------------------------------------------------------------
    # Panel d: top three DSG isoform UMAPs
    # ------------------------------------------------------------------
    psi_colorbar_pairs = []
    panel_d_label_added = False

    for (
        selected_index,
        isoform_ax,
        colorbar_ax
    ) in zip(
        selected_intron_indices,
        isoform_axes,
        isoform_colorbar_axes
    ):
        intron_key = str(
            intron_names[
                selected_index
            ]
        )

        if intron_key == "0":
            isoform_ax.set_axis_off()
            colorbar_ax.set_axis_off()
            continue

        sc.pl.umap(
            adata,
            color=intron_key,
            layer=psi_layer,
            title="",
            color_map=psi_color_map,
            colorbar_loc=None,
            frameon=False,
            show=False,
            ax=isoform_ax,
            vmin=psi_vmin,
            vmax=psi_vmax
        )

        isoform_ax.set_title(
            str(
                intron_names_plot[
                    selected_index
                ]
            ),
            fontsize=facet_title_fontsize,
            pad=2
        )

        format_custom_umap_coordinates(
            isoform_ax
        )

        psi_colorbar = add_psi_colorbar(
            isoform_ax,
            colorbar_ax
        )

        psi_colorbar_pairs.append(
            (
                isoform_ax,
                psi_colorbar
            )
        )

        if not panel_d_label_added:
            set_panel_label(
                isoform_ax,
                "d"
            )
            panel_d_label_added = True

    # ------------------------------------------------------------------
    # Final layout
    # ------------------------------------------------------------------
    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.99
        ),
        pad=0.25,
        w_pad=0.0,
        h_pad=0.0
    )

    fig.canvas.draw()

    # Reposition PSI colorbars with fixed physical dimensions.
    colorbar_gap_fraction = (
        colorbar_gap_mm
        / figure_width_mm
    )
    colorbar_width_fraction = (
        colorbar_width_mm
        / figure_width_mm
    )

    for isoform_ax, colorbar in psi_colorbar_pairs:
        parent_position = (
            isoform_ax.get_position()
        )
        colorbar_ax = colorbar.ax

        colorbar_ax.set_axes_locator(
            None
        )
        colorbar_ax.set_in_layout(
            False
        )

        if hasattr(
            colorbar_ax,
            "set_box_aspect"
        ):
            colorbar_ax.set_box_aspect(
                None
            )

        colorbar_ax.set_position([
            parent_position.x1
            + colorbar_gap_fraction,
            parent_position.y0,
            colorbar_width_fraction,
            parent_position.height
        ])

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    if save_fig:
        dataset_name = kwargs.get(
            "dataset_name"
        )
        file_suffix = kwargs.get(
            "file_suffix",
            "pdf"
        )

        if dataset_name is None:
            save_path = "./figures"
        else:
            save_path = os.path.join(
                "./figures",
                dataset_name
            )

        os.makedirs(
            save_path,
            exist_ok=True
        )

        if taxonomy_level is not None:
            filename = (
                f"dsg_analysis_{model_name}_{taxonomy_level}_"
                f"clustergroup_{cluster_group}.{file_suffix}"
            )
        else:
            filename = (
                f"dsg_analysis_{model_name}_"
                f"clustergroup_{cluster_group}.{file_suffix}"
            )

        fig_name = os.path.join(
            save_path,
            filename
        )

        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches=None,
            pad_inches=0
        )

    plt.show()

def plot_marker_analysis_scvi_tuvi(
        adata_objects: Tuple[AnnData, AnnData],
        likelihoods: List[str],
        cluster_group_ids: Tuple[str, str],
        dsg_exp_df: pd.DataFrame,
        dsg_psi_df: pd.DataFrame,
        top_degs: List[str],
        top_dsgs: List[str],
        dsg_gene_names: List[str],
        umap_color_key_display: str,
        taxonomy_level: str,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given two AnnData objects containing the results of a differential gene expression or differential splicing analysis
    on cell embeddings inferred by scVI and tuVI respectively, analyse the top DEGs and DSGs for a specific cluster
    group and plot the UMAPs of the clusters, the dotplots of the DEGs and DSGs across cluster groups, and the top 3
    DEGs and DSGs for the cluster group specified. The DEG and DSG analyses should have been performed using the scVI or
    tuVI latent space. The chosen model and GE likelihood key (for tuVI also the embedding) should be represented in
    likelihoods (e.g. ["scVI_ZINB", "tuVI_ZIDM"]). If the Anndata objects were filtered to a cell taxonomy level
    (e.g. tissue (Heart, Brain_Non-Myeloid)) specify the cell taxonomy level name otherwise set taxonomy_level to None.
    The cluster_group_ids specifies the Leiden cluster group for which the top 3 DEGs and DSGs will be plotted. The
    figure is saved in the figures directory if save_fig is set to True.

    :param adata_objects:
    :param likelihoods:
    :param cluster_group_ids:
    :param dsg_exp_df:
    :param dsg_psi_df:
    :param top_degs:
    :param top_dsgs:
    :param dsg_gene_names:
    :param umap_color_key_display:
    :param taxonomy_level:
    :param save_fig:
    """
    # Unpack the Anndata objects
    adata_1, adata_2 = adata_objects

    # Unpack cluster group ids
    group_id_1, group_id_2 = cluster_group_ids
    dataset_name = kwargs.get("dataset_name", "default")

    fig = plt.figure(figsize=(16, 20), dpi=300)
    gs = gridspec.GridSpec(5, 4, figure=fig)

    # Add axis spanning [0,0], [0,1], [1,0], [1,1]

    # UMAP of gene expression data of selected tissue
    ax00 = fig.add_subplot(gs[:2, :2])
    adata_1.obsm['X_umap'] = adata_1.obsm[likelihoods[0] + '_X_umap'].copy()
    sc.pl.umap(adata_1, color=umap_color_key_display, legend_loc=None, frameon=False, show=False, ax=ax00)
    del adata_1.obsm['X_umap']
    ax00.set_title('scVI-ZINB')
    ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')
    plot_customized_UMAP_coordinates(ax00, length=1.0)

    # Plot UMAP of transcript usage data using ZIDM liklelihood
    ax01 = fig.add_subplot(gs[:2, 2:])
    adata_2.obsm['X_umap'] = adata_2.obsm[likelihoods[1] + "_X_umap"].copy()
    sc.pl.umap(
        adata_2,
        color=umap_color_key_display,
        legend_loc="right margin",  # temporarily create legend
        frameon=False,
        show=False,
        ax=ax01
    )

    # Extract handles and labels from the actual Scanpy legend
    legend = ax01.get_legend()

    if legend is not None:
        labels = [text.get_text() for text in legend.get_texts()]

        try:
            handles = legend.legend_handles
        except AttributeError:
            handles = legend.legendHandles

        legend.remove()
    else:
        handles, labels = [], []

    del adata_2.obsm['X_umap']
    ax01.set_title('tuVI-ZIDM')
    ax01.set_title('b', loc='left', fontsize=20, fontweight='bold')
    plot_customized_UMAP_coordinates(ax01, length=1.0)

    # Upsetplot of DE and DS genes
    ax20 = fig.add_subplot(gs[2, 0])
    upset_path = "./figures/"+ dataset_name + "/upsetplot_degs_" + group_id_1 + "_dsgs_" + group_id_2 + "_" + taxonomy_level + ".png"
    img = mpimg.imread(upset_path)
    ax20.imshow(img)
    ax20.axis('off')
    ax20.set_title('c', loc='left', fontsize=20, fontweight='bold')

    # Venn diagram of top DEGs and DSGs
    ax30 = fig.add_subplot(gs[3, 0])
    upset_path = "./figures/"+ dataset_name + "/ven_topdegs_" + group_id_1 + "_dsgs_" + group_id_2 + "_" + taxonomy_level + ".png"
    img = mpimg.imread(upset_path)
    ax30.imshow(img)
    ax30.axis('off')
    ax30.set_title('d', loc='left', fontsize=20, fontweight='bold')

    # UMAPs with DEGs
    # Find the global min/max across the three genes so the colorbar is valid for all
    all_values = adata_1[:, top_degs[:3]].X
    v_min, v_max = all_values.min(), all_values.max()

    # Plot 1: ax21 (No colorbar)
    ax21 = fig.add_subplot(gs[2, 1])
    adata_1.obsm['X_umap'] = adata_1.obsm[likelihoods[0] + '_X_umap'].copy()
    sc.pl.umap(adata_1, color=top_degs[0], frameon=False, show=False, ax=ax21,
               colorbar_loc=None, vmin=v_min, vmax=v_max)
    plot_customized_UMAP_coordinates(ax21, length=1.0)
    ax21.set_title('e', loc='left', fontsize=20, fontweight='bold')

    # Plot 2: ax22 (No colorbar)
    ax22 = fig.add_subplot(gs[2, 2])
    sc.pl.umap(adata_1, color=top_degs[1], frameon=False, show=False, ax=ax22,
               colorbar_loc=None, vmin=v_min, vmax=v_max)
    plot_customized_UMAP_coordinates(ax22, length=1.0)

    # Plot 3: ax23 (With colorbar)
    # Note: Renamed variable to ax23 for clarity, as you had 'ax22' twice
    ax23 = fig.add_subplot(gs[2, 3])
    sc.pl.umap(adata_1, color=top_degs[2], frameon=False, show=False, ax=ax23,
               vmin=v_min, vmax=v_max)
    plot_customized_UMAP_coordinates(ax23, length=1.0)

    # DSG plots for group id
    ax31 = fig.add_subplot(gs[3, 1])
    adata_2.obsm['X_umap'] = adata_2.obsm[likelihoods[1] + "_X_umap"].copy()
    sc.pl.umap(adata_2, color=top_dsgs[0], layer="PSI_raw", frameon=False, show=False, ax=ax31,
               colorbar_loc=None, vmin=0.0, vmax=1.0)
    ax31.set_title(' ')
    ax31.set_title(top_dsgs[0].split('_')[0])
    plot_customized_UMAP_coordinates(ax31, length=1.0)
    ax31.set_title('f', loc='left', fontsize=20, fontweight='bold')

    ax32 = fig.add_subplot(gs[3, 2])
    sc.pl.umap(adata_2, color=top_dsgs[1], layer="PSI_raw", frameon=False, show=False, ax=ax32,
               colorbar_loc=None, vmin=0.0, vmax=1.0)
    ax32.set_title(' ')
    ax32.set_title(top_dsgs[1].split('_')[0])
    plot_customized_UMAP_coordinates(ax32, length=1.0)

    ax33 = fig.add_subplot(gs[3, 3])
    sc.pl.umap(adata_2, color=top_dsgs[2], layer="PSI_raw", frameon=False, show=False, ax=ax33,
               vmin=0.0, vmax=1.0)
    ax33.set_title(' ')
    ax33.set_title(top_dsgs[2].split('_')[0])
    plot_customized_UMAP_coordinates(ax33, length=1.0)

    # DSG expression violin plots spanning 4,0 - 4,1
    ax40 = fig.add_subplot(gs[4, :2])

    sns.violinplot(
        data=dsg_exp_df,
        x='Gene',  # Genes on the x-axis
        y='Expression',  # Expression on the y-axis
        hue='Group',  # Color based on the group
        hue_order=['target', 'other'],  # To ensure no flipping of violins if target or other appears first in df
        split=True,  # Combine two groups into one violin
        inner='quartile',  # Show quartiles inside the violin
        palette={"target": '#E69F00', 'other': '#56B4E9'},  # Optional: specify colors
        cut=0,
        ax=ax40
    )

    ax40.set_title(' ')
    ax40.set_xlabel(' ')
    ax40.set_ylabel('Gene expression level')
    ax40.legend().remove()
    ax40.grid(axis='y', linestyle='--', alpha=0.7)
    ax40.set_title('g', loc='left', fontsize=20, fontweight='bold')

    # Start expression scale at 0, but leave some top whitespace
    expr_max = dsg_exp_df['Expression'].max()
    expr_pad = expr_max * 0.05

    ax40.set_ylim(-expr_pad, expr_max + expr_pad)

    # DSG PSI violin plots spanning 4,2 - 4,3
    ax42 = fig.add_subplot(gs[4, 2:])
    sns.violinplot(
        data=dsg_psi_df,
        x='Isoform',  # Genes on the x-axis
        y='PSI-score',  # Expression on the y-axis
        hue='Group',  # Color based on the group
        hue_order=['target', 'other'],  # To ensure no flipping of violins if target or other appears first in df
        split=True,  # Combine two groups into one violin
        inner='quartile',  # Show quartiles inside the violin
        palette={"target": '#E69F00', 'other': '#56B4E9'},  # Optional: specify colors
        cut=0,
        ax=ax42
    )

    ax42.yaxis.tick_right()
    ax42.yaxis.set_label_position("right")

    ax42.set_title(' ')
    ax42.set_xlabel(' ')
    ax42.set_xticklabels(dsg_gene_names)
    ax42.set_ylabel('PSI-score')

    # Leave visual whitespace while keeping PSI ticks from 0.0 to 1.0
    ax42.set_ylim(-0.03, 1.03)
    ax42.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax42.legend().remove()
    ax42.grid(axis='y', linestyle='--', alpha=0.7)

    ax42.set_title('h', loc='left', fontsize=20, fontweight='bold')

    # Cleanup
    #del adata_1.obsm['X_umap']
    #del adata_2.obsm['X_umap']

    # Increase vertical space between plots
    plt.subplots_adjust(hspace=0.5)

    # Legend placed above
    # Determine how many columns you want (e.g., 4 or 5)
    n_cols = 5

    # We use fig.legend to place it relative to the whole figure
    # bbox_to_anchor handles the positioning.
    # (0.1, 0.92, 0.5, 0.05) -> (x_start, y_start, width, height)
    # This places it above the left side of the figure.
    leg = fig.legend(
        handles,
        labels,
        loc='upper left',
        bbox_to_anchor=(0.05, 0.88, 0.85, 0.05),
        ncol=n_cols,
        mode="expand",  # This distributes the columns equally across the width provided
        borderaxespad=0.,
        frameon=False,
        fontsize=12,
        title="Cell type",
        title_fontsize=12
    )
    leg._legend_box.align = "left"  # Aligns title to the left of the legend block

    if save_fig:
        file_suffix = kwargs.get("file_suffix", 'pdf')
        fig_name = "./figures/tabulaMuris/cell_annotation_tuVI_" + taxonomy_level + "_clustergroup_ge_" + group_id_1 + "_clustergroup_tu_" + group_id_2 + "." + file_suffix
        fig.savefig(fig_name, dpi=300, bbox_inches="tight")

    plt.show()

def plot_differential_analysis_on_umap(
        adata,
        cluster_groups,
        tax_level,
        model_name,
        modality,
        renamed_isoforms=None,
        ref_umap_color="leiden",
        save_fig=True,
        **kwargs
):
    r"""
    Plot a reference UMAP in the first column and the top DEGs or DSGs
    for each cluster group in the remaining columns.

    The figure uses a 180 mm full-width Nature Methods layout with an
    adaptive height capped at 247 mm.

    :param adata: AnnData object containing differential-analysis results.
    :param cluster_groups: Leiden groups for which entities are plotted.
    :param tax_level: Tissue name, or None if all tissues were analysed.
    :param model_name: Model used for the differential analysis.
    :param modality: Either "GE" or "TU".
    :param renamed_isoforms: Optional flattened list of renamed isoforms.
    :param ref_umap_color: Key used to colour the reference UMAP.
    :param save_fig: Whether to save the figure.
    """

    if not cluster_groups:
        raise ValueError(
            "cluster_groups must contain at least one cluster."
        )

    assert "leiden" in adata.obs, (
        "The key 'leiden' is not present in adata.obs. Please perform "
        "Leiden clustering before calling this function."
    )

    assert ref_umap_color in adata.obs, (
        f"The reference UMAP colour key '{ref_umap_color}' is not "
        "present in adata.obs."
    )

    if modality == "GE":
        assert "rank_genes_groups" in adata.uns, (
            "The key 'rank_genes_groups' is not present in adata.uns. "
            "Please perform DEG analysis before calling this function."
        )

    elif modality == "TU":
        assert "PSI_raw" in adata.layers, (
            "The key 'PSI_raw' is not present in adata.layers. Please "
            "add the raw PSI scores to adata.layers['PSI_raw']."
        )

        assert "rank_introns_groups" in adata.uns, (
            "The key 'rank_introns_groups' is not present in adata.uns. "
            "Please perform DSG analysis before calling this function."
        )

    else:
        raise ValueError(
            "Invalid modality. Use 'GE' for gene expression or "
            "'TU' for transcript usage."
        )

    assert "X_umap" in adata.obsm, (
        "The key 'X_umap' is not present in adata.obsm. Please compute "
        "the UMAP coordinates before calling this function."
    )

    num_entities_per_group = kwargs.get(
        "num_entities_per_group",
        3
    )

    if num_entities_per_group < 1:
        raise ValueError(
            "num_entities_per_group must be at least 1."
        )

    # ------------------------------------------------------------------
    # Typography for a 180 mm-wide Nature Methods figure
    # ------------------------------------------------------------------
    panel_label_fontsize = kwargs.get(
        "panel_label_fontsize",
        11
    )
    panel_label_y = kwargs.get(
        "panel_label_y",
        1.05
    )
    entity_title_fontsize = kwargs.get(
        "entity_title_fontsize",
        7
    )
    umap_axis_fontsize = kwargs.get(
        "umap_axis_fontsize",
        7
    )
    leiden_label_fontsize = kwargs.get(
        "leiden_label_fontsize",
        6
    )
    legend_fontsize = kwargs.get(
        "legend_fontsize",
        6.5
    )
    legend_title_fontsize = kwargs.get(
        "legend_title_fontsize",
        7
    )
    colorbar_tick_fontsize = kwargs.get(
        "colorbar_tick_fontsize",
        6
    )
    colorbar_label_fontsize = kwargs.get(
        "colorbar_label_fontsize",
        7
    )

    # ------------------------------------------------------------------
    # Nature Methods figure dimensions
    # ------------------------------------------------------------------
    figure_width_mm = kwargs.get(
        "figure_width_mm",
        180.0
    )
    figure_max_height_mm = kwargs.get(
        "figure_max_height_mm",
        247.0
    )
    min_umap_row_height_mm = kwargs.get(
        "min_umap_row_height_mm",
        30.0
    )

    # Colour-bar dimensions and spacing.
    colorbar_width_factor = kwargs.get(
        "colorbar_width_factor",
        1.25
    )
    colorbar_gap_mm = kwargs.get(
        "colorbar_gap_mm",
        1.5
    )

    if figure_width_mm <= 0:
        raise ValueError(
            "figure_width_mm must be greater than 0."
        )

    if figure_max_height_mm <= 0:
        raise ValueError(
            "figure_max_height_mm must be greater than 0."
        )

    if colorbar_width_factor <= 0:
        raise ValueError(
            "colorbar_width_factor must be greater than 0."
        )

    if colorbar_gap_mm < 0:
        raise ValueError(
            "colorbar_gap_mm must be greater than or equal to 0."
        )

    # ------------------------------------------------------------------
    # Determine the space required by the shared legend
    # ------------------------------------------------------------------
    if ref_umap_color != "leiden":
        reference_categories = (
            adata.obs[ref_umap_color]
            .astype("category")
            .cat.remove_unused_categories()
            .cat.categories
        )

        number_of_reference_categories = len(
            reference_categories
        )

        default_legend_ncol = min(
            6,
            max(1, number_of_reference_categories)
        )

        requested_legend_ncol = kwargs.get(
            "ref_legend_ncol",
            default_legend_ncol
        )

        if requested_legend_ncol < 1:
            raise ValueError(
                "ref_legend_ncol must be at least 1."
            )

        number_of_legend_rows = int(
            np.ceil(
                number_of_reference_categories
                / requested_legend_ncol
            )
        )

        default_legend_height_mm = max(
            12.0,
            6.0 + 4.0 * number_of_legend_rows
        )

        legend_height_mm = kwargs.get(
            "legend_height_mm",
            default_legend_height_mm
        )

    else:
        requested_legend_ncol = 1
        legend_height_mm = 0.0

    mm_per_inch = 25.4
    figure_width_in = figure_width_mm / mm_per_inch

    # ------------------------------------------------------------------
    # Collect the top DEGs or DSGs
    # ------------------------------------------------------------------
    entity_names = np.empty(
        num_entities_per_group * len(cluster_groups),
        dtype=object
    )

    for i, test_group in enumerate(cluster_groups):

        if modality == "GE":
            entity_names_group = adata.uns[
                "rank_genes_groups"
            ]["names"][test_group][:num_entities_per_group]

        else:
            entity_names_group = adata.uns[
                "rank_introns_groups"
            ]["names"][test_group][:num_entities_per_group]

        if len(entity_names_group) < num_entities_per_group:
            raise ValueError(
                f"Only {len(entity_names_group)} ranked entities are "
                f"available for cluster {test_group}, but "
                f"{num_entities_per_group} were requested."
            )

        start_idx = i * num_entities_per_group
        end_idx = (i + 1) * num_entities_per_group

        entity_names[start_idx:end_idx] = entity_names_group

    if (
        modality == "TU"
        and renamed_isoforms is not None
        and len(renamed_isoforms) < len(entity_names)
    ):
        raise ValueError(
            f"renamed_isoforms contains {len(renamed_isoforms)} names, "
            f"but {len(entity_names)} names are required."
        )

    # ------------------------------------------------------------------
    # Adaptive figure height
    # ------------------------------------------------------------------
    num_rows = len(cluster_groups)
    num_cols = num_entities_per_group + 1

    ideal_umap_row_height_mm = min(
        45.0,
        0.95 * figure_width_mm / num_cols
    )

    available_umap_height_mm = (
        figure_max_height_mm - legend_height_mm
    )

    if available_umap_height_mm <= 0:
        raise ValueError(
            "The requested legend height leaves no space for UMAP rows."
        )

    umap_row_height_mm = min(
        ideal_umap_row_height_mm,
        available_umap_height_mm / num_rows
    )

    if umap_row_height_mm < min_umap_row_height_mm:
        max_rows_per_figure = max(
            1,
            int(
                available_umap_height_mm
                // min_umap_row_height_mm
            )
        )

        raise ValueError(
            f"{num_rows} UMAP rows cannot be displayed legibly within "
            f"{figure_width_mm:g} x {figure_max_height_mm:g} mm. "
            f"Plot at most {max_rows_per_figure} rows per figure by "
            "splitting cluster_groups into smaller subsets."
        )

    figure_height_mm = (
        legend_height_mm
        + num_rows * umap_row_height_mm
    )

    figure_height_in = figure_height_mm / mm_per_inch

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(
            figure_width_in,
            figure_height_in
        ),
        squeeze=False
    )

    handles = []
    labels = []
    alphabet = list("abcdefghijklmnopqrstuvwxyz")

    if num_rows > len(alphabet):
        raise ValueError(
            "A maximum of 26 labelled rows is supported."
        )

    def format_custom_umap_coordinates(ax):
        """
        Add the custom UMAP axes and reduce the size of any text created
        by plot_customized_UMAP_coordinates.
        """
        number_of_existing_texts = len(ax.texts)

        plot_customized_UMAP_coordinates(
            ax,
            length=1.0
        )

        # Resize text generated by the custom coordinate function.
        for text_artist in ax.texts[
                number_of_existing_texts:
        ]:
            text_artist.set_fontsize(
                umap_axis_fontsize
            )

        # Handle implementations that use standard axis labels.
        ax.xaxis.label.set_size(
            umap_axis_fontsize
        )
        ax.yaxis.label.set_size(
            umap_axis_fontsize
        )

        ax.tick_params(
            axis="both",
            labelsize=umap_axis_fontsize
        )

    # ------------------------------------------------------------------
    # Plot UMAP rows
    # ------------------------------------------------------------------
    for i, test_group in enumerate(cluster_groups):

        start_idx = i * num_entities_per_group
        end_idx = (i + 1) * num_entities_per_group

        group_entities = entity_names[start_idx:end_idx]

        # Use common colour limits within each cluster-group row.
        if modality == "GE":
            group_data = adata[:, group_entities].X
            group_vmin = float(group_data.min())
            group_vmax = float(group_data.max())

        else:
            group_vmin = 0.0
            group_vmax = 1.0

        # --------------------------------------------------------------
        # Reference UMAP
        # --------------------------------------------------------------
        reference_ax = axs[i, 0]

        if ref_umap_color != "leiden":
            sc.pl.umap(
                adata,
                color=ref_umap_color,
                legend_loc="right margin",
                frameon=False,
                show=False,
                ax=reference_ax
            )

            reference_legend = reference_ax.get_legend()

            if reference_legend is not None:
                if not handles:
                    labels = [
                        text.get_text()
                        for text in reference_legend.get_texts()
                    ]

                    try:
                        handles = reference_legend.legend_handles
                    except AttributeError:
                        handles = reference_legend.legendHandles

                reference_legend.remove()

        else:
            sc.pl.umap(
                adata,
                color=ref_umap_color,
                legend_loc="on data",
                legend_fontsize=leiden_label_fontsize,
                legend_fontoutline=1,
                frameon=False,
                show=False,
                ax=reference_ax
            )

        reference_ax.set_title("")

        reference_ax.set_title(
            alphabet[i],
            loc="left",
            fontsize=panel_label_fontsize,
            fontweight="bold",
            y=panel_label_y,
            pad=0
        )

        format_custom_umap_coordinates(
            reference_ax
        )

        # Ensure that all UMAP axes remain square.
        reference_ax.set_box_aspect(1)

        # --------------------------------------------------------------
        # Differential-analysis UMAPs
        # --------------------------------------------------------------
        for entity_idx, entity_name in enumerate(
                group_entities
        ):
            plot_col = entity_idx + 1
            ax = axs[i, plot_col]

            # Preserve the panel position but leave it blank.
            if str(entity_name) == "0":
                ax.set_axis_off()
                continue

            if modality == "GE":
                sc.pl.umap(
                    adata,
                    color=entity_name,
                    use_raw=False,
                    legend_loc=None,
                    frameon=False,
                    show=False,
                    ax=ax,
                    vmin=group_vmin,
                    vmax=group_vmax
                )

                display_name = entity_name

            else:
                sc.pl.umap(
                    adata,
                    color=entity_name,
                    layer="PSI_raw",
                    legend_loc=None,
                    frameon=False,
                    show=False,
                    ax=ax,
                    vmin=group_vmin,
                    vmax=group_vmax
                )

                if renamed_isoforms is not None:
                    renamed_idx = (
                        i * num_entities_per_group
                        + entity_idx
                    )

                    display_name = renamed_isoforms[
                        renamed_idx
                    ]

                else:
                    display_name = entity_name

            ax.set_title(
                str(display_name),
                fontsize=entity_title_fontsize,
                pad=2
            )

            format_custom_umap_coordinates(
                ax
            )

            # Ensure that all UMAP axes remain square and equally sized.
            ax.set_box_aspect(1)

    # ------------------------------------------------------------------
    # Shared legend for non-Leiden annotations
    # ------------------------------------------------------------------
    if handles:
        legend_title = kwargs.get(
            "ref_legend_title",
            ref_umap_color.replace(
                "_",
                " "
            ).capitalize()
        )

        legend_ncol = min(
            requested_legend_ncol,
            len(labels)
        )

        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=legend_ncol,
            borderaxespad=0.0,
            frameon=False,
            fontsize=legend_fontsize,
            title=legend_title,
            title_fontsize=legend_title_fontsize,
            markerscale=0.6,
            handlelength=1.0,
            handletextpad=0.3,
            columnspacing=0.8,
            labelspacing=0.25
        )

        if hasattr(legend, "_legend_box"):
            legend._legend_box.align = "left"

        legend_fraction = (
            legend_height_mm / figure_height_mm
        )

        fig.tight_layout(
            rect=(
                0.01,
                0.01,
                0.99,
                1.0 - legend_fraction
            ),
            pad=0.25,
            h_pad=0.4,
            w_pad=0.35
        )

    else:
        fig.tight_layout(
            rect=(
                0.01,
                0.01,
                0.99,
                0.98
            ),
            pad=0.25,
            h_pad=0.4,
            w_pad=0.35
        )

    # ------------------------------------------------------------------
    # Format and reposition Scanpy colour bars
    # ------------------------------------------------------------------
    fig.canvas.draw()

    # Convert the physical colour-bar gap into figure coordinates.
    colorbar_gap_fraction = (
        colorbar_gap_mm / figure_width_mm
    )

    for figure_ax in fig.axes:
        colorbar_object = getattr(
            figure_ax,
            "_colorbar",
            None
        )

        is_colorbar = (
            colorbar_object is not None
            or figure_ax.get_label() == "<colorbar>"
        )

        if not is_colorbar:
            continue

        # Compact colour-bar typography.
        figure_ax.tick_params(
            axis="both",
            labelsize=colorbar_tick_fontsize,
            length=2,
            pad=1
        )

        figure_ax.xaxis.label.set_size(
            colorbar_label_fontsize
        )
        figure_ax.yaxis.label.set_size(
            colorbar_label_fontsize
        )
        figure_ax.title.set_fontsize(
            colorbar_label_fontsize
        )

        figure_ax.xaxis.get_offset_text().set_fontsize(
            colorbar_tick_fontsize
        )
        figure_ax.yaxis.get_offset_text().set_fontsize(
            colorbar_tick_fontsize
        )

        colorbar_position = figure_ax.get_position()

        # Determine which UMAP owns this colour bar.
        parent_ax = None

        if colorbar_object is not None:
            mappable = getattr(
                colorbar_object,
                "mappable",
                None
            )

            if mappable is not None:
                parent_ax = getattr(
                    mappable,
                    "axes",
                    None
                )

        if parent_ax is not None:
            parent_position = parent_ax.get_position()

            # Place the colour bar to the right of its UMAP with a
            # fixed physical gap. The UMAP axis itself is unchanged.
            new_x0 = (
                parent_position.x1
                + colorbar_gap_fraction
            )

        else:
            # Fallback for Matplotlib versions where the parent UMAP
            # cannot be obtained from the colour-bar object.
            new_x0 = (
                colorbar_position.x0
                + colorbar_gap_fraction
            )

        new_width = (
            colorbar_position.width
            * colorbar_width_factor
        )

        # Prevent the outermost colour bar from leaving the canvas.
        maximum_right_position = 0.995
        available_width = (
            maximum_right_position - new_x0
        )

        new_width = min(
            new_width,
            available_width
        )

        if new_width <= 0:
            raise ValueError(
                "Insufficient horizontal space for the colour bar. "
                "Reduce colorbar_gap_mm or colorbar_width_factor."
            )

        # Prevent Scanpy's axes-divider locator from resetting the
        # manually assigned position during the final savefig draw.
        figure_ax.set_axes_locator(None)

        if hasattr(figure_ax, "set_box_aspect"):
            figure_ax.set_box_aspect(None)

        figure_ax.set_position([
            new_x0,
            colorbar_position.y0,
            new_width,
            colorbar_position.height
        ])

    # ------------------------------------------------------------------
    # Save at an exact physical width of 180 mm
    # ------------------------------------------------------------------
    if save_fig:
        import os

        diff_analysis = (
            "deg" if modality == "GE" else "dsg"
        )

        dataset_name = kwargs.get(
            "dataset_name",
            "default"
        )
        file_suffix = kwargs.get(
            "file_suffix",
            "pdf"
        )

        output_directory = (
            f"./figures/{dataset_name}/"
        )

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        filename = os.path.join(
            output_directory,
            f"{diff_analysis}_analysis_umaps_"
            f"{model_name}_{tax_level}.{file_suffix}"
        )

        fig.savefig(
            filename,
            dpi=300,
            bbox_inches=None,
            pad_inches=0
        )

        print(
            f"Figure saved to: {filename}"
        )

    plt.show()

def format_umap_axis(ax):
    """Keep the UMAP panel square without distorting the embedding."""
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_anchor("C")

def plot_deg_and_dsg_analysis_trvi(
        adata: Tuple[AnnData, AnnData],
        tax_level: str | None,
        model_name: str,
        rename_isoforms: bool = True,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given a tuple of two Anndata objects with the first one containing the results of the DEG analysis and the second
    one containing the results of the DSG analysis for a specific tissue, plot two UMAP of the specified cell embeddings
    with Leiden clusters and with the cell type annotation as reference, a dotplot of the top DEGs across Leiden clusters,
    and a dotplot of the top DSGs across Leiden clusters. The DEG and DSG analyses should have been performed using the
    TRVI latent space. The chosen model, GE/TU likelihood key, and embedding should be represented in model_name
    (e.g. TRVI_ZINB_ZIDM_shared, etc.). If the Anndata objects were filtered to a tissue specify the tissue name
    otherwise set tissue to None. The embedding parameter specifies which cell embedding was used for the clustering
    and differential analysis (e.g. "shared") and is used for plotting the UMAPs. The figure is saved in the figures
    directory if save_fig is set to True.

    Panel layout
    ------------
    a
        UMAP colored by Leiden cluster.
    b
        UMAP colored by cell type.
    c
        Dot plot of the top DEGs across Leiden clusters.
    d
        Dot plot of the top DSG isoforms across Leiden clusters.

    The default figure is 180 mm wide and follows the same typography,
    legend formatting, marker sizing, and spacing used by
    plot_deg_analysis and plot_dsg_analysis.

    :param adata: Tuple of two AnnData objects, the first one containing the results of the DEG analysis and the second one containing the results of the DSG analysis for a specific tissue
    :param tax_level: str, the name of the tissue for which the analysis was performed,
    or None if the analysis was performed on all tissues
    :param model_name: str, the name of the model used for the analysis (e.g. TRVI_ZINB_ZIDM_shared, etc.)
    :param rename_isoforms: bool, whether to rename the DSG isoforms to "gene_name isoform_number" format, where isoform_number is assigned based on the order of appearance of the intron names for each gene
    :param save_fig: bool, whether to save the figure
    """

    import os
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # ------------------------------------------------------------------
    # Validate AnnData objects
    # ------------------------------------------------------------------
    if not isinstance(adata, (tuple, list)) or len(adata) != 2:
        raise ValueError(
            "adata must contain exactly two AnnData objects: "
            "(adata_GE, adata_TU)."
        )

    adata_ge = adata[0]
    adata_tu = adata[1]

    for object_name, current_adata in [
        ("adata[0]", adata_ge),
        ("adata[1]", adata_tu)
    ]:
        if "leiden" not in current_adata.obs:
            raise KeyError(
                f"The key 'leiden' is not present in "
                f"{object_name}.obs."
            )

        if "X_umap" not in current_adata.obsm:
            raise KeyError(
                f"The key 'X_umap' is not present in "
                f"{object_name}.obsm."
            )

    if "rank_genes_groups" not in adata_ge.uns:
        raise KeyError(
            "The key 'rank_genes_groups' is not present in "
            "adata[0].uns."
        )

    if "rank_introns_groups" not in adata_tu.uns:
        raise KeyError(
            "The key 'rank_introns_groups' is not present in "
            "adata[1].uns."
        )

    psi_layer = kwargs.get(
        "psi_layer",
        "PSI_raw"
    )

    if psi_layer not in adata_tu.layers:
        raise KeyError(
            f"The PSI layer '{psi_layer}' is not present in "
            "adata[1].layers."
        )

    cell_type_key = kwargs.get(
        "cell_type_key"
    )

    if not isinstance(cell_type_key, str) or not cell_type_key:
        raise ValueError(
            "Provide a non-empty cell-type key using "
            "cell_type_key='...'."
        )

    if cell_type_key not in adata_ge.obs:
        raise KeyError(
            f"The cell-type key '{cell_type_key}' is not present "
            "in adata[0].obs."
        )

    dataset_name = kwargs.get(
        "dataset_name",
        "default"
    )

    # ------------------------------------------------------------------
    # Typography
    # ------------------------------------------------------------------
    panel_label_fontsize = kwargs.get(
        "panel_label_fontsize",
        11
    )
    panel_label_y = kwargs.get(
        "panel_label_y",
        1.04
    )
    facet_title_fontsize = kwargs.get(
        "facet_title_fontsize",
        8
    )
    umap_axis_fontsize = kwargs.get(
        "umap_axis_fontsize",
        7
    )
    legend_fontsize = kwargs.get(
        "legend_fontsize",
        6.5
    )
    dotplot_tick_fontsize = kwargs.get(
        "dotplot_tick_fontsize",
        6
    )
    dotplot_label_fontsize = kwargs.get(
        "dotplot_label_fontsize",
        7
    )
    colorbar_tick_fontsize = kwargs.get(
        "colorbar_tick_fontsize",
        6
    )

    # ------------------------------------------------------------------
    # Cell-type legend settings
    # ------------------------------------------------------------------
    cell_type_legend_marker_size = kwargs.get(
        "cell_type_legend_marker_size",
        36.0
    )
    cell_type_legend_labelspacing = kwargs.get(
        "cell_type_legend_labelspacing",
        0.65
    )
    cell_type_legend_handletextpad = kwargs.get(
        "cell_type_legend_handletextpad",
        0.5
    )

    if cell_type_legend_marker_size <= 0:
        raise ValueError(
            "cell_type_legend_marker_size must be greater than 0."
        )

    if cell_type_legend_labelspacing < 0:
        raise ValueError(
            "cell_type_legend_labelspacing must be non-negative."
        )

    # ------------------------------------------------------------------
    # DEG dot-plot settings
    # ------------------------------------------------------------------
    num_marker_genes = kwargs.get(
        "num_marker_genes",
        6
    )
    deg_dotplot_largest_dot = kwargs.get(
        "deg_dotplot_largest_dot",
        60.0
    )
    deg_dotplot_smallest_dot = kwargs.get(
        "deg_dotplot_smallest_dot",
        0.0
    )
    deg_dotplot_edge_linewidth = kwargs.get(
        "deg_dotplot_edge_linewidth",
        0.35
    )

    if num_marker_genes <= 0:
        raise ValueError(
            "num_marker_genes must be greater than 0."
        )

    if deg_dotplot_largest_dot <= 0:
        raise ValueError(
            "deg_dotplot_largest_dot must be greater than 0."
        )

    if deg_dotplot_smallest_dot < 0:
        raise ValueError(
            "deg_dotplot_smallest_dot must be non-negative."
        )

    # ------------------------------------------------------------------
    # DSG dot-plot settings
    # ------------------------------------------------------------------
    num_intron_group_markers = kwargs.get(
        "num_intron_group_markers",
        3
    )
    dsg_dotplot_row_spacing = kwargs.get(
        "dsg_dotplot_row_spacing",
        1.0
    )
    dsg_dotplot_largest_dot = kwargs.get(
        "dsg_dotplot_largest_dot",
        60.0
    )
    dsg_dotplot_edge_linewidth = kwargs.get(
        "dsg_dotplot_edge_linewidth",
        0.35
    )
    dsg_dotplot_alpha = kwargs.get(
        "dsg_dotplot_alpha",
        0.9
    )
    dsg_dotplot_color_map = kwargs.get(
        "dsg_dotplot_color_map",
        "Reds"
    )

    psi_vmin = kwargs.get(
        "psi_vmin",
        0.0
    )
    psi_vmax = kwargs.get(
        "psi_vmax",
        1.0
    )

    if num_intron_group_markers <= 0:
        raise ValueError(
            "num_intron_group_markers must be greater than 0."
        )

    if dsg_dotplot_row_spacing <= 0:
        raise ValueError(
            "dsg_dotplot_row_spacing must be greater than 0."
        )

    if dsg_dotplot_largest_dot <= 0:
        raise ValueError(
            "dsg_dotplot_largest_dot must be greater than 0."
        )

    if psi_vmin >= psi_vmax:
        raise ValueError(
            "psi_vmin must be smaller than psi_vmax."
        )

    # ------------------------------------------------------------------
    # Nature Methods figure dimensions
    # ------------------------------------------------------------------
    mm_per_inch = 25.4

    figure_width_mm = kwargs.get(
        "figure_width_mm",
        180.0
    )
    max_figure_height_mm = kwargs.get(
        "max_figure_height_mm",
        247.0
    )

    if figure_width_mm <= 0:
        raise ValueError(
            "figure_width_mm must be greater than 0."
        )

    if max_figure_height_mm <= 0:
        raise ValueError(
            "max_figure_height_mm must be greater than 0."
        )

    # Three equal horizontal regions are retained. The third region
    # accommodates the cell-type legend beside panel b.
    inter_umap_gap_mm = kwargs.get(
        "inter_umap_gap_mm",
        12.0
    )

    if inter_umap_gap_mm < 0:
        raise ValueError(
            "inter_umap_gap_mm must be non-negative."
        )

    available_umap_width_mm = (
        figure_width_mm
        - 2 * inter_umap_gap_mm
    )

    if available_umap_width_mm <= 0:
        raise ValueError(
            "inter_umap_gap_mm leaves no room for the UMAP panels."
        )

    nominal_umap_width_mm = (
        available_umap_width_mm / 3
    )

    width_ratios = [
        nominal_umap_width_mm,
        inter_umap_gap_mm,
        nominal_umap_width_mm,
        inter_umap_gap_mm,
        nominal_umap_width_mm
    ]

    # ------------------------------------------------------------------
    # Vertical spacing
    # ------------------------------------------------------------------
    top_to_deg_gap_mm = kwargs.get(
        "top_to_deg_gap_mm",
        5.0
    )
    deg_dotplot_height_mm = kwargs.get(
        "deg_dotplot_height_mm",
        45.0
    )

    # Space for the vertically oriented DEG names beneath panel c.
    deg_to_dsg_gap_mm = kwargs.get(
        "deg_to_dsg_gap_mm",
        18.0
    )
    dsg_dotplot_height_mm = kwargs.get(
        "dsg_dotplot_height_mm",
        45.0
    )

    # Space for the vertically oriented isoform names beneath panel d.
    bottom_label_space_mm = kwargs.get(
        "bottom_label_space_mm",
        22.0
    )
    outer_vertical_padding_mm = kwargs.get(
        "outer_vertical_padding_mm",
        8.0
    )

    if min(
        top_to_deg_gap_mm,
        deg_to_dsg_gap_mm,
        bottom_label_space_mm,
        outer_vertical_padding_mm
    ) < 0:
        raise ValueError(
            "All physical gap settings must be non-negative."
        )

    if deg_dotplot_height_mm <= 0:
        raise ValueError(
            "deg_dotplot_height_mm must be greater than 0."
        )

    if dsg_dotplot_height_mm <= 0:
        raise ValueError(
            "dsg_dotplot_height_mm must be greater than 0."
        )

    default_figure_height_mm = (
        nominal_umap_width_mm
        + top_to_deg_gap_mm
        + deg_dotplot_height_mm
        + deg_to_dsg_gap_mm
        + dsg_dotplot_height_mm
        + bottom_label_space_mm
        + outer_vertical_padding_mm
    )

    figure_height_mm = kwargs.get(
        "figure_height_mm",
        default_figure_height_mm
    )

    if figure_height_mm <= 0:
        raise ValueError(
            "figure_height_mm must be greater than 0."
        )

    figure_height_mm = min(
        figure_height_mm,
        max_figure_height_mm
    )

    height_ratios = [
        nominal_umap_width_mm,
        top_to_deg_gap_mm,
        deg_dotplot_height_mm,
        deg_to_dsg_gap_mm,
        dsg_dotplot_height_mm,
        bottom_label_space_mm
    ]

    fig = plt.figure(
        figsize=(
            figure_width_mm / mm_per_inch,
            figure_height_mm / mm_per_inch
        ),
        dpi=300
    )

    gs = gridspec.GridSpec(
        nrows=6,
        ncols=5,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.0,
        hspace=0.0
    )

    # ------------------------------------------------------------------
    # Main axes
    # ------------------------------------------------------------------
    leiden_ax = fig.add_subplot(
        gs[0, 0]
    )
    cell_type_ax = fig.add_subplot(
        gs[0, 2]
    )

    deg_dotplot_ax = fig.add_subplot(
        gs[2, :]
    )

    # Give the custom DSG dot plot its own legend column.
    dsg_dotplot_subgrid = gridspec.GridSpecFromSubplotSpec(
        nrows=1,
        ncols=2,
        subplot_spec=gs[4, :],
        width_ratios=[
            kwargs.get(
                "dsg_dotplot_width_ratio",
                4.0
            ),
            1.0
        ],
        wspace=kwargs.get(
            "dsg_dotplot_legend_wspace",
            0.12
        )
    )

    dsg_dotplot_ax = fig.add_subplot(
        dsg_dotplot_subgrid[0, 0]
    )
    dsg_legend_ax = fig.add_subplot(
        dsg_dotplot_subgrid[0, 1]
    )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def format_custom_umap_coordinates(ax):
        number_of_existing_texts = len(
            ax.texts
        )

        plot_customized_UMAP_coordinates(
            ax,
            length=1.0
        )

        for text_artist in ax.texts[
            number_of_existing_texts:
        ]:
            text_artist.set_fontsize(
                umap_axis_fontsize
            )

        ax.xaxis.label.set_size(
            umap_axis_fontsize
        )
        ax.yaxis.label.set_size(
            umap_axis_fontsize
        )

        ax.xaxis.labelpad = 1
        ax.yaxis.labelpad = 1

        ax.tick_params(
            axis="both",
            labelsize=umap_axis_fontsize
        )

        ax.set_box_aspect(
            1
        )
        ax.set_anchor(
            "C"
        )

    def set_panel_label(
            ax,
            label,
            y=None
    ):
        ax.set_title(
            label,
            loc="left",
            fontsize=panel_label_fontsize,
            fontweight="bold",
            y=panel_label_y if y is None else y,
            pad=0
        )

    def format_cell_type_legend(ax):
        """
        Rebuild the cell-type legend without a title and use compact,
        non-overlapping markers.
        """

        original_legend = ax.get_legend()

        if original_legend is None:
            return

        legend_handles = getattr(
            original_legend,
            "legend_handles",
            None
        )

        if legend_handles is None:
            legend_handles = getattr(
                original_legend,
                "legendHandles",
                None
            )

        legend_labels = [
            text_artist.get_text()
            for text_artist in original_legend.get_texts()
        ]

        if legend_handles is None:
            legend_handles, fallback_labels = (
                ax.get_legend_handles_labels()
            )

            if not legend_labels:
                legend_labels = fallback_labels

        original_legend.remove()

        new_legend = ax.legend(
            list(legend_handles),
            list(legend_labels),
            loc="center left",
            bbox_to_anchor=(
                kwargs.get(
                    "cell_type_legend_x",
                    1.04
                ),
                kwargs.get(
                    "cell_type_legend_y",
                    0.5
                )
            ),
            frameon=False,
            fontsize=legend_fontsize,
            title=None,
            labelspacing=cell_type_legend_labelspacing,
            handletextpad=cell_type_legend_handletextpad,
            borderaxespad=0
        )

        final_legend_handles = getattr(
            new_legend,
            "legend_handles",
            None
        )

        if final_legend_handles is None:
            final_legend_handles = getattr(
                new_legend,
                "legendHandles",
                []
            )

        for legend_handle in final_legend_handles:
            if hasattr(
                legend_handle,
                "set_sizes"
            ):
                legend_handle.set_sizes([
                    cell_type_legend_marker_size
                ])

            elif hasattr(
                legend_handle,
                "set_markersize"
            ):
                legend_handle.set_markersize(
                    np.sqrt(
                        cell_type_legend_marker_size
                    )
                )

        new_legend.set_in_layout(
            False
        )

        if hasattr(
            new_legend,
            "_legend_box"
        ):
            new_legend._legend_box.align = "left"

    def normalize_dotplot_axes_result(
            dotplot_result,
            axes_before
    ):
        if isinstance(
            dotplot_result,
            dict
        ):
            return dotplot_result

        if (
            dotplot_result is not None
            and hasattr(
                dotplot_result,
                "get_axes"
            )
        ):
            result_axes = (
                dotplot_result.get_axes()
            )

            if isinstance(
                result_axes,
                dict
            ):
                return result_axes

        new_axes = [
            current_ax
            for current_ax in fig.axes
            if current_ax not in axes_before
        ]

        if new_axes:
            return {
                f"dotplot_ax_{index}": current_ax
                for index, current_ax in enumerate(
                    new_axes
                )
            }

        return {
            "mainplot_ax": deg_dotplot_ax
        }

    # ------------------------------------------------------------------
    # Panel a: Leiden UMAP
    # ------------------------------------------------------------------
    sc.pl.umap(
        adata_ge,
        color="leiden",
        title="",
        legend_loc="on data",
        legend_fontsize=legend_fontsize,
        frameon=False,
        show=False,
        ax=leiden_ax
    )

    format_custom_umap_coordinates(
        leiden_ax
    )

    leiden_ax.set_title(
        "Leiden clusters",
        fontsize=facet_title_fontsize,
        pad=2
    )

    set_panel_label(
        leiden_ax,
        "a"
    )

    # ------------------------------------------------------------------
    # Panel b: cell-type UMAP
    # ------------------------------------------------------------------
    sc.pl.umap(
        adata_ge,
        color=cell_type_key,
        title="",
        legend_loc="right margin",
        legend_fontsize=legend_fontsize,
        frameon=False,
        show=False,
        ax=cell_type_ax
    )

    format_custom_umap_coordinates(
        cell_type_ax
    )
    format_cell_type_legend(
        cell_type_ax
    )

    cell_type_ax.set_title(
        "Cell type",
        fontsize=facet_title_fontsize,
        pad=2
    )

    set_panel_label(
        cell_type_ax,
        "b"
    )

    # ------------------------------------------------------------------
    # Panel c: DEG dot plot
    # ------------------------------------------------------------------
    deg_cluster_groups = (
        adata_ge.obs["leiden"]
        .unique()
        .tolist()
    )

    if (
        dataset_name == "tabulaMuris"
        and tax_level in [
            "Heart",
            "Brain_Non-Myeloid"
        ]
    ):
        deg_cluster_groups = (
            deg_cluster_groups[:5]
        )

    axes_before_deg_dotplot = set(
        fig.axes
    )

    deg_dotplot_object = (
        sc.pl.rank_genes_groups_dotplot(
            adata_ge,
            groups=deg_cluster_groups,
            groupby="leiden",
            standard_scale="var",
            n_genes=num_marker_genes,
            dendrogram=False,
            return_fig=True,
            show=False,
            ax=deg_dotplot_ax
        )
    )

    # Scanpy 1.11 requires marker sizing to be applied through
    # DotPlot.style rather than passed to rank_genes_groups_dotplot.
    deg_dotplot_object.style(
        smallest_dot=deg_dotplot_smallest_dot,
        largest_dot=deg_dotplot_largest_dot,
        dot_edge_lw=deg_dotplot_edge_linewidth
    )

    deg_dotplot_object.make_figure()

    deg_dotplot_result = (
        deg_dotplot_object.get_axes()
    )

    deg_dotplot_axes_dict = (
        normalize_dotplot_axes_result(
            deg_dotplot_result,
            axes_before_deg_dotplot
        )
    )

    deg_dotplot_related_axes = list(
        dict.fromkeys(
            deg_dotplot_axes_dict.values()
        )
    )

    for current_ax in deg_dotplot_related_axes:
        current_ax.tick_params(
            axis="both",
            labelsize=dotplot_tick_fontsize,
            length=2
        )

        current_ax.xaxis.label.set_size(
            dotplot_label_fontsize
        )
        current_ax.yaxis.label.set_size(
            dotplot_label_fontsize
        )
        current_ax.title.set_fontsize(
            dotplot_label_fontsize
        )

        for text_artist in current_ax.texts:
            text_artist.set_fontsize(
                dotplot_tick_fontsize
            )

    deg_dotplot_ax.set_title(
        ""
    )

    deg_dotplot_panel_axes = [
        current_ax
        for axis_name, current_ax
        in deg_dotplot_axes_dict.items()
        if "legend" not in axis_name.lower()
    ]

    if not deg_dotplot_panel_axes:
        deg_dotplot_panel_axes = [
            deg_dotplot_ax
        ]

    # ------------------------------------------------------------------
    # Panel d: DSG isoform dot plot
    # ------------------------------------------------------------------
    dsg_cluster_groups = (
        adata_tu.obs["leiden"]
        .unique()
        .tolist()
    )

    # Preserve the original tissue-specific exclusions.
    tabula_muris_exclusions = {
        "Brain_Non-Myeloid": {
            "5"
        },
        "Heart": {
            "5"
        },
        "Large_Intestine": {
            "5"
        },
        "Marrow": {
            "7"
        },
        "GAT": {
            "1"
        },
        "Skin": {
            "2"
        },
        "SCAT": {
            "4",
            "5"
        }
    }

    if dataset_name == "tabulaMuris":
        excluded_clusters = (
            tabula_muris_exclusions.get(
                tax_level,
                set()
            )
        )

        dsg_cluster_groups = [
            cluster
            for cluster in dsg_cluster_groups
            if str(cluster) not in excluded_clusters
        ]

    (
        mean_psi_groups_introns,
        num_cells_valid_matrix,
        intron_names
    ) = dsg_dotplot_helper(
        adata_tu,
        dsg_cluster_groups,
        num_intron_group_markers
    )

    mean_psi_groups_introns = np.asarray(
        mean_psi_groups_introns
    )
    num_cells_valid_matrix = np.asarray(
        num_cells_valid_matrix
    )
    intron_names = np.asarray(
        intron_names
    )

    expected_dsg_shape = (
        len(dsg_cluster_groups),
        len(intron_names)
    )

    if mean_psi_groups_introns.shape != expected_dsg_shape:
        raise ValueError(
            "mean_psi_groups_introns has shape "
            f"{mean_psi_groups_introns.shape}; expected "
            f"{expected_dsg_shape}."
        )

    if num_cells_valid_matrix.shape != expected_dsg_shape:
        raise ValueError(
            "num_cells_valid_matrix has shape "
            f"{num_cells_valid_matrix.shape}; expected "
            f"{expected_dsg_shape}."
        )

    if rename_isoforms:
        intron_names_plot = np.asarray(
            rename_isoform_helper(
                intron_names
            )
        )
    else:
        intron_names_plot = (
            intron_names.astype(str)
        )

    if len(intron_names_plot) != len(intron_names):
        raise ValueError(
            "rename_isoform_helper changed the number of DSG labels."
        )

    x_grid, y_grid = np.meshgrid(
        np.arange(
            len(intron_names)
        ),
        np.arange(
            len(dsg_cluster_groups)
        )
    )

    x_coordinates = (
        x_grid.ravel()
    )
    y_coordinates = (
        y_grid
        * dsg_dotplot_row_spacing
    ).ravel()

    color_values = (
        mean_psi_groups_introns.ravel()
    )
    size_values = np.nan_to_num(
        num_cells_valid_matrix.ravel(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    size_values = np.clip(
        size_values,
        a_min=0.0,
        a_max=None
    )

    size_legend_values = kwargs.get(
        "size_legend_values",
        [50, 100, 250, 500]
    )

    if not size_legend_values:
        raise ValueError(
            "size_legend_values must contain at least one value."
        )

    size_legend_values = np.asarray(
        size_legend_values,
        dtype=float
    )

    if np.any(size_legend_values <= 0):
        raise ValueError(
            "All size_legend_values must be greater than 0."
        )

    size_reference_maximum = max(
        float(
            np.max(
                size_values
            )
        ) if size_values.size else 0.0,
        float(
            np.max(
                size_legend_values
            )
        ),
        1.0
    )

    dsg_dot_size_scale = (
        dsg_dotplot_largest_dot
        / size_reference_maximum
    )

    scaled_dsg_dot_sizes = (
        size_values
        * dsg_dot_size_scale
    )

    psi_dotplot = dsg_dotplot_ax.scatter(
        x_coordinates,
        y_coordinates,
        c=color_values,
        s=scaled_dsg_dot_sizes,
        cmap=dsg_dotplot_color_map,
        edgecolors="black",
        linewidths=dsg_dotplot_edge_linewidth,
        alpha=dsg_dotplot_alpha,
        vmin=psi_vmin,
        vmax=psi_vmax
    )

    dsg_dotplot_ax.set_xticks(
        np.arange(
            len(intron_names_plot)
        )
    )
    dsg_dotplot_ax.set_xticklabels(
        intron_names_plot,
        rotation=90,
        ha="center",
        va="top",
        fontsize=dotplot_tick_fontsize
    )

    dsg_y_tick_positions = (
        np.arange(
            len(dsg_cluster_groups)
        )
        * dsg_dotplot_row_spacing
    )

    dsg_dotplot_ax.set_yticks(
        dsg_y_tick_positions
    )
    dsg_dotplot_ax.set_yticklabels(
        dsg_cluster_groups,
        fontsize=dotplot_tick_fontsize
    )

    dsg_dotplot_ax.set_ylim(
        -0.5 * dsg_dotplot_row_spacing,
        (
            len(dsg_cluster_groups)
            - 0.5
        ) * dsg_dotplot_row_spacing
    )
    dsg_dotplot_ax.set_xlim(
        -0.5,
        len(intron_names) - 0.5
    )

    dsg_dotplot_ax.tick_params(
        axis="both",
        labelsize=dotplot_tick_fontsize,
        length=2,
        pad=1
    )

    dsg_dotplot_ax.xaxis.label.set_size(
        dotplot_label_fontsize
    )
    dsg_dotplot_ax.yaxis.label.set_size(
        dotplot_label_fontsize
    )

    set_panel_label(
        dsg_dotplot_ax,
        "d"
    )

    # ------------------------------------------------------------------
    # DSG size legend and PSI colorbar
    # ------------------------------------------------------------------
    dsg_legend_ax.set_xlim(
        -0.5,
        len(size_legend_values) - 0.5
    )
    dsg_legend_ax.set_ylim(
        0,
        1
    )
    dsg_legend_ax.axis(
        "off"
    )

    dsg_legend_ax.set_title(
        "Number of cells\nin group",
        fontsize=dotplot_label_fontsize,
        pad=2
    )

    legend_x_positions = np.arange(
        len(size_legend_values)
    )

    scaled_legend_sizes = (
        size_legend_values
        * dsg_dot_size_scale
    )

    for (
        x_position,
        marker_size,
        size_value
    ) in zip(
        legend_x_positions,
        scaled_legend_sizes,
        size_legend_values
    ):
        dsg_legend_ax.scatter(
            x_position,
            0.72,
            s=marker_size,
            color="gray",
            edgecolors="black",
            linewidths=dsg_dotplot_edge_linewidth,
            alpha=dsg_dotplot_alpha
        )

        dsg_legend_ax.plot(
            [
                x_position,
                x_position
            ],
            [
                0.49,
                0.57
            ],
            color="black",
            linewidth=0.8
        )

        dsg_legend_ax.text(
            x_position,
            0.38,
            f"{size_value:g}",
            ha="center",
            va="center",
            fontsize=colorbar_tick_fontsize
        )

    dsg_colorbar_ax = inset_axes(
        dsg_legend_ax,
        width="90%",
        height="11%",
        loc="lower center",
        borderpad=0
    )

    dsg_colorbar = fig.colorbar(
        psi_dotplot,
        cax=dsg_colorbar_ax,
        orientation="horizontal"
    )

    dsg_colorbar.set_ticks(
        kwargs.get(
            "dsg_colorbar_ticks",
            [0.0, 0.5, 1.0]
        )
    )
    dsg_colorbar.ax.tick_params(
        labelsize=colorbar_tick_fontsize,
        length=2,
        pad=1
    )
    dsg_colorbar.ax.set_title(
        "Mean PSI value\nin group",
        fontsize=dotplot_label_fontsize,
        pad=4
    )

    # ------------------------------------------------------------------
    # Final layout
    # ------------------------------------------------------------------
    fig.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.99
        ),
        pad=0.25,
        w_pad=0.0,
        h_pad=0.0
    )

    fig.canvas.draw()

    # Position panel label c relative to Scanpy's rendered dot plot,
    # rather than its larger outer placeholder axis.
    panel_c_x = min(
        current_ax.get_position().x0
        for current_ax in deg_dotplot_panel_axes
    )
    panel_c_y = max(
        current_ax.get_position().y1
        for current_ax in deg_dotplot_panel_axes
    )

    panel_c_y_offset_mm = kwargs.get(
        "panel_c_y_offset_mm",
        1.0
    )

    panel_c_y += (
        panel_c_y_offset_mm
        / figure_height_mm
    )

    fig.text(
        panel_c_x,
        min(panel_c_y, 0.995),
        "c",
        fontsize=panel_label_fontsize,
        fontweight="bold",
        ha="left",
        va="bottom",
        transform=fig.transFigure
    )

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    if save_fig:
        file_suffix = kwargs.get(
            "file_suffix",
            "pdf"
        )

        save_path = os.path.join(
            "./figures",
            dataset_name
        )

        os.makedirs(
            save_path,
            exist_ok=True
        )

        if tax_level is not None:
            filename = (
                f"deg_dsg_analysis_{model_name}_{tax_level}."
                f"{file_suffix}"
            )
        else:
            filename = (
                f"deg_dsg_analysis_{model_name}.{file_suffix}"
            )

        fig_name = os.path.join(
            save_path,
            filename
        )

        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches=None,
            pad_inches=0
        )

    plt.show()

def plot_latent_space_benchmarking_trvi(
        adata_objects: Tuple[AnnData, AnnData],
        likelihood_keys: List[str],
        evaluation_df: pd.DataFrame,
        atlas_weight_dict: Dict[str, float],
        mean_weights_df: pd.DataFrame,
        relevance_weight_likelihood_key: str,
        seed: int | None,
        random_seeds: List[int],
        umap_color_display_key: str,
        sort_cell_type_key: str,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given two AnnData objects (gene expression and transcript usage) containing the inferred cell embeddings, plot the
    UMAPs of S-GE, S-TU, and GE-TU cell embeddings as well as the bar plots of atlas and cell type specific modality-
    relevance weights.

    :param adata_objects: Tuple of two AnnData objects, the first one containing the inferred cell embeddings for gene expression and the second one containing the inferred cell embeddings for transcript usage
    :param likelihood_keys: List of str, the likelihood keys for gene expression and transcript usage
    :param evaluation_df: pd.DataFrame, the evaluation dataframe of the cell embeddings on the atlas level
    :param atlas_weight_dict: Dict of atlas weight dict
    :param mean_weights_df: pandas dataframe of mean modality-relevance weights per cell type averaged across random seeds
    :param relevance_weight_likelihood_key: The key of the modality-relevance weight (either display gene expression or transcript usage relevance weights)
    :param seed: random seed for UMAPs
    :param random_seeds: List of random seeds
    :param umap_color_display_key: str
    :param sort_cell_type_key: str
    :param save_fig: bool
    """

    # Unpack the AnnData objects
    adata_GE, adata_TU = adata_objects
    likelihood_GE, likelihood_TU = likelihood_keys

    fig = plt.figure(figsize=(13, 20), dpi=300)
    gs = gridspec.GridSpec(5, 3, figure=fig)

    # Unpack the results from modality-relevance weight analysis
    total_mean_GE = atlas_weight_dict["total_mean_GE"]
    total_std_GE = atlas_weight_dict["total_std_GE"]
    total_mean_TU = atlas_weight_dict["total_mean_TU"]
    total_std_TU = atlas_weight_dict["total_std_TU"]

    # Reset index to make cell_type_key a column for plotting
    mean_weights_df = mean_weights_df.reset_index()

    likelihood_seeds_list = []

    for random_seed in random_seeds:
        likelihood_seeds_list.append(relevance_weight_likelihood_key + "_" + str(random_seed))

    mean_weights_transcriptomic_facet = mean_weights_df[likelihood_seeds_list].mean(axis=1)
    std_weights_transcriptomic_facet = mean_weights_df[likelihood_seeds_list].std(axis=1)

    cell_types = mean_weights_df["index"]
    cell_organ_system = mean_weights_df[sort_cell_type_key]

    # Plotting function to be moved into plotting_utils
    ax00 = fig.add_subplot(gs[:2, :2])

    from importlib import resources
    try:
        img_resource = resources.files('crecerelle.default_figures').joinpath('crecerelle_TRVI.png')

        with resources.as_file(img_resource) as image_path:
            model_img = mpimg.imread(str(image_path))
            ax00.imshow(model_img)
    except (ImportError, FileNotFoundError):
        print("Warning: TRVI architecture figure could not be loaded from package resources.")

    ax00.axis('off')
    ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')

    # UMAP S-GE
    ax02 = fig.add_subplot(gs[0, 2])
    adata_GE.obsm['X_umap'] = UMAP(n_components=2, random_state=seed).fit_transform(
        adata_GE.obsm[likelihood_GE + "_" + str(seed) + "_shared_latent_mean"])
    sc.pl.umap(
        adata_GE,
        color=umap_color_display_key,
        frameon=False,
        show=False,
        ax=ax02
    )

    # Extract handles and labels for the legend
    handles, labels = ax02.get_legend_handles_labels()

    # Remove the default legend that Scanpy might have drawn on ax02
    if ax02.get_legend():
        ax02.get_legend().remove()

    del adata_GE.obsm['X_umap']
    ax02.set_title('b', loc='left', fontsize=20, fontweight='bold')
    ax02.set_title(' ', loc='center', fontsize=20)
    plot_customized_UMAP_coordinates(ax02, length=1.0)

    # UMAP S-TU
    ax12 = fig.add_subplot(gs[1, 2])
    adata_TU.obsm['X_umap'] = UMAP(n_components=2, random_state=seed).fit_transform(
        adata_TU.obsm[likelihood_TU + "_" + str(seed) + "_shared_latent_mean"])
    sc.pl.umap(
        adata_TU,
        color=umap_color_display_key,
        legend_loc=None,
        frameon=False,
        show=False,
        ax=ax12
    )
    del adata_TU.obsm['X_umap']
    ax12.set_title('c', loc='left', fontsize=20, fontweight='bold')
    ax12.set_title(' ', loc='center', fontsize=20)
    plot_customized_UMAP_coordinates(ax12, length=1.0)

    # You need to lower the top of the subplots to make room for the new legend
    fig.subplots_adjust(top=0.88, hspace=0.5)

    # UMAP Joint
    ax22 = fig.add_subplot(gs[2:4, :2])
    adata_GE.obsm['X_umap'] = UMAP(n_components=2, random_state=seed).fit_transform(
        adata_GE.obsm[likelihood_GE + "_" + str(seed) + "_" + likelihood_TU + "_" + str(seed) + "_shared_latent_mean"])
    sc.pl.umap(
        adata_GE,
        color=umap_color_display_key,
        legend_loc=None,
        frameon=False,
        show=False,
        ax=ax22
    )

    del adata_GE.obsm['X_umap']
    ax22.set_title('d', loc='left', fontsize=20, fontweight='bold')
    ax22.set_title(' ', loc='center', fontsize=20)
    plot_customized_UMAP_coordinates(ax22, length=1.0)

    # Biological conservation scores (empty plot)
    ax30 = fig.add_subplot(gs[2, 2])
    sns.barplot(evaluation_df, x='Evaluation score', y='Score value', hue='Embedding', ax=ax30)

    ax30.set_ylabel(" ")
    # Adjust yscale to 0.0 to 1.0
    ax30.set_ylim(0.0, 1.0)
    # ax30.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax30.legend(loc='lower left', fontsize='small', frameon=True)
    ax30.set_title('e', loc='left', fontsize=20, fontweight='bold')

    # Modality weights on atlas level with colors GE: #317EC2 TU: #5AAA46
    colors = ['#317EC2', '#5AAA46']
    ax31 = fig.add_subplot(gs[3, 2])
    ax31.bar(
        x=["GE", "TU"],
        height=[total_mean_GE, total_mean_TU],
        yerr=[total_std_GE, total_std_TU],
        color=colors
    )
    # Adjust yscale to 0.0 to 1.0
    ax31.set_ylim(0.0, 1.0)
    ax31.set_title('f', loc='left', fontsize=20, fontweight='bold')

    # Weights per cell type bar chart plot
    ax40 = fig.add_subplot(gs[4, 0:3])
    ax40.bar(cell_types, mean_weights_transcriptomic_facet)
    ax40.errorbar(
        cell_types,
        mean_weights_transcriptomic_facet,
        yerr=std_weights_transcriptomic_facet,
        fmt="o",
        color="r"
    )
    ax40.axhline(y=0.5, color='k', linestyle='--')
    ax40.set_xticklabels(ax40.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    import matplotlib.transforms as transforms
    system_counts = cell_organ_system.value_counts(sort=False)
    boundaries = system_counts.cumsum()
    start_pos = 0

    # 1. Identify the boundaries
    system_counts = cell_organ_system.value_counts(sort=False)
    start_pos = 0

    # 2. Define the vertical position in "axes fraction"
    # 1.0 is the top of the plot. 1.05 starts just above the line.
    bracket_y = 1.05
    bracket_height = 0.03
    title_height = 1.1

    # Create a blended transform: x is data, y is axes fraction
    trans = transforms.blended_transform_factory(ax40.transData, ax40.transAxes)

    for system, count in system_counts.items():
        end_pos = start_pos + count - 1

        # Insert a newline character after '&'
        formatted_system = system.replace('&', '&\n')

        # Draw horizontal line
        ax40.plot([start_pos, end_pos], [bracket_y, bracket_y],
                  color='black', lw=1.5, clip_on=False, transform=trans)

        # Draw vertical ticks
        ax40.plot([start_pos, start_pos], [bracket_y, bracket_y - bracket_height],
                  color='black', lw=1.5, clip_on=False, transform=trans)
        ax40.plot([end_pos, end_pos], [bracket_y, bracket_y - bracket_height],
                  color='black', lw=1.5, clip_on=False, transform=trans)

        # Add the formatted text
        midpoint = (start_pos + end_pos) / 2
        ax40.text(midpoint, bracket_y + 0.01, formatted_system, transform=trans,
                  ha='center', va='bottom', fontsize=9, fontweight='bold',
                  clip_on=False, linespacing=0.9)  # linespacing tightens the wrap

        start_pos += count

    ax40.set_title('g', loc='left', fontsize=20, fontweight='bold', y=title_height)

    fig.subplots_adjust(top=0.8)

    # Increase space between ax30 and ax31
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    # plt.tight_layout()

    # Legend placed above
    # Determine how many columns you want (e.g., 4 or 5)
    n_cols = 4

    # We use fig.legend to place it relative to the whole figure
    # bbox_to_anchor handles the positioning.
    # (0.1, 0.92, 0.5, 0.05) -> (x_start, y_start, width, height)
    # This places it above the left side of the figure.
    leg = fig.legend(
        handles,
        labels,
        loc='upper left',
        bbox_to_anchor=(0.05, 0.81, 0.85, 0.05),
        ncol=n_cols,
        mode="expand",  # This distributes the columns equally across the width provided
        borderaxespad=0.,
        frameon=False,
        fontsize=12,
        title="Organ System",
        title_fontsize=12
    )
    leg._legend_box.align = "left"  # Aligns title to the left of the legend block

    if save_fig:
        file_suffix = kwargs.get("file_suffix", 'pdf')
        dataset_name = kwargs.get("dataset_name", "default")
        fig_name = "./figures/" + dataset_name + "/TRVI_benchmarking." + file_suffix
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')

def plot_cell_embeddings_analysis_trvi(
        adata_objects: Tuple[AnnData, AnnData],
        cluster_groups: List[str],
        likelihood_keys: List[str],
        cell_embeddings_selected: List[str],
        cell_embeddings_keys: List[str],
        seeds_selected: List[int],
        seed: int | None,
        isoforms_to_visualize: List[str] = None,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given two AnnData objects (gene expression and transcript usage) containing the inference results from TRVI, a
    specified group of Leiden clusters, likelihood keys, cell embeddings and their respective keys, as well as different
    random seeds, plot the latent spaces of the TRVI cell embeddings as UMAP, assess the robustness of the learnt
    relevance weights of GE and TU per cell type present, and plot the results of DEG and DSG analysis per cluster
    showing sets of marker genes and sets of marker isoforms. Optionally, cell-type specific isoform regulation can be
    visualised by projecting the PSI score of the isoform on the UMAP. The figure is saved in the figures directory
    if save_fig is set to True.

    :param adata_objects:
    :param cluster_groups:
    :param likelihood_keys:
    :param cell_embeddings_selected:
    :param cell_embeddings_keys:
    :param seeds_selected:
    :param seed:
    :param isoforms_to_visualize:
    :param save_fig:
    """
    adata_1 = adata_objects[0] # Gene expression
    adata_2 = adata_objects[1] # Transcript usage
    likelihood_key_1 = likelihood_keys[0] # Observation model of gene expression
    likelihood_key_2 = likelihood_keys[1] # Observation model of transcript usage

    # Unpack kwargs
    cell_type_key = kwargs.get("cell_type_key", "cell_ontology_class")
    dataset_name = kwargs.get("dataset_name", "default")
    tax_level = kwargs.get("tax_level", "atlas")
    num_intron_group_markers = kwargs.get("num_intron_group_markers", 3)
    num_marker_genes = kwargs.get("num_marker_genes", 3)
    rename_isoforms = kwargs.get("rename_isoforms", True)


    # Calculate the UMAP coordinates for the first two selected embeddings
    adata_1.obsm["X_umap"] = UMAP(n_components=2, random_state=seed).fit_transform(
        adata_1.obsm[cell_embeddings_keys[0]])
    adata_2.obsm["X_umap"] = UMAP(n_components=2, random_state=seed).fit_transform(
        adata_2.obsm[cell_embeddings_keys[1]])

    cell_weights_across_seeds_GE = []
    cell_weights_across_seeds_TU = []

    for seed_selected in seeds_selected:
        cell_weights_across_seeds_GE.append(likelihood_key_1 + "_" + str(seed_selected) + "_weighting")
        cell_weights_across_seeds_TU.append(likelihood_key_2 + "_" + str(seed_selected) + "_weighting")

    # Calculations for boxplots and violins

    # 1. Process GE weights
    df_ge_subset = adata_1.obs[[cell_type_key] + cell_weights_across_seeds_GE]
    df_ge_melted = df_ge_subset.melt(
        id_vars=cell_type_key,
        value_vars=cell_weights_across_seeds_GE,
        var_name='seed',
        value_name='weight'
    )
    df_ge_melted['Source'] = 'GE'  # Label for grouping

    # 2. Process TU weights
    df_tu_subset = adata_2.obs[[cell_type_key] + cell_weights_across_seeds_TU]
    df_tu_melted = df_tu_subset.melt(
        id_vars=cell_type_key,
        value_vars=cell_weights_across_seeds_TU,
        var_name='seed',
        value_name='weight'
    )
    df_tu_melted['Source'] = 'TU'  # Label for grouping

    # 3. Concatenate the two dataframes
    df_combined = pd.concat([df_ge_melted, df_tu_melted], axis=0)

    # 1. Create a combined dataframe for the ratios
    # We assume the indices (cell IDs) match between the two adata objects
    df_ratios = pd.DataFrame(index=adata_1.obs.index)
    df_ratios[cell_type_key] = adata_1.obs[cell_type_key]

    # 2. Calculate log(GE / TU) for each seed
    # We add a tiny epsilon to avoid division by zero or log(0)
    epsilon = 1e-10

    for i in seeds_selected:
        ge_col = f"{likelihood_key_1}_{i}_weighting"
        tu_col = f"{likelihood_key_2}_{i}_weighting"

        ratio_col = f"log_ratio_seed_{i}"

        # Calculate the log ratio: log(GE / TU)
        df_ratios[ratio_col] = np.log(
            (adata_1.obs[ge_col] + epsilon) /
            (adata_2.obs[tu_col] + epsilon)
        )

    # 3. Melt the dataframe to long format
    ratio_cols = [f"log_ratio_seed_{i}" for i in seeds_selected]
    df_ratios_melted = df_ratios.melt(
        id_vars=cell_type_key,
        value_vars=ratio_cols,
        var_name='seed',
        value_name='log_weight_ratio'
    )

    mean_psi_groups_introns, num_cells_valid_matrix, intron_names = dsg_dotplot_helper(
        adata_2,
        cluster_groups,
        num_intron_group_markers
    )

    # DSG PLOT
    # Create a grid of x, y coordinates
    X, Y = np.meshgrid(range(len(intron_names)), range(len(cluster_groups)))
    x_coords = X.flatten()
    #y_coords = Y.flatten()

    color_values = mean_psi_groups_introns.flatten()
    size_values = num_cells_valid_matrix.flatten()

    # Rename isoforms to "gene_name isoform_number" format, where isoform_number is assigned based on the order of appearance of the intron names for each gene.
    if rename_isoforms:
        intron_names_plot = rename_isoform_helper(intron_names)
    else:
        intron_names_plot = intron_names

    # Plotting
    has_isoform_row = isoforms_to_visualize is not None

    fig = plt.figure(
        figsize=(22, 18 if has_isoform_row else 15),
        dpi=300
    )

    gs = gridspec.GridSpec(
        5 if has_isoform_row else 4,
        5,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 0.75],
        height_ratios=(
            [1, 1, 0.95, 1.30, 1]
            if has_isoform_row
            else [1, 1, 0.95, 1.30]
        )
    )

    modality_palette = {"GE": "#5da5da", "TU": "#faa43a"}

    # UMAP S-GE
    ax00 = fig.add_subplot(gs[0, 0])
    sc.pl.umap(
        adata_1,
        color=cell_type_key,
        ax=ax00,
        legend_loc=None,
        show=False,
        frameon=False,
        title=cell_embeddings_selected[0]
    )
    del adata_1.obsm["X_umap"]
    plot_customized_UMAP_coordinates(ax00, length=1.0)
    ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')

    # UMAP S-TU
    ax10 = fig.add_subplot(gs[1, 0])
    sc.pl.umap(
        adata_2,
        color=cell_type_key,
        ax=ax10,
        legend_loc=None,
        show=False,
        frameon=False,
        title=cell_embeddings_selected[1]
    )
    del adata_2.obsm["X_umap"]
    plot_customized_UMAP_coordinates(ax10, length=1.0)

    # UMAP joint
    sc.pp.neighbors(adata_1, use_rep=cell_embeddings_keys[2], random_state=seed)
    adata_1.obsm["X_umap"] = UMAP(n_components=2, random_state=seed).fit_transform(
        adata_1.obsm[cell_embeddings_keys[2]])
    ax11 = fig.add_subplot(gs[0:2, 1:3])
    sc.pl.umap(
        adata_1,
        color=cell_type_key,
        ax=ax11,
        show=False,
        frameon=False,
        title=cell_embeddings_selected[2]
    )

    # Extract handles and labels for the legend
    handles, labels = ax11.get_legend_handles_labels()

    # Remove the default legend that Scanpy might have drawn on ax02
    if ax11.get_legend():
        ax11.get_legend().remove()

    plot_customized_UMAP_coordinates(ax11, length=1.0)

    # Relevance weights bar plot
    ax20 = fig.add_subplot(gs[2, 0])

    cell_weights_1_mean = adata_1.obs.groupby(cell_type_key)[
        f"{likelihood_key_1}_weighting"].mean().reset_index()
    cell_weights_2_mean = adata_2.obs.groupby(cell_type_key)[
        f"{likelihood_key_2}_weighting"].mean().reset_index()

    cell_weights_1_mean.columns = [cell_type_key, "Weighting"]
    cell_weights_1_mean["Modality"] = "GE"
    cell_weights_2_mean.columns = [cell_type_key, "Weighting"]
    cell_weights_2_mean["Modality"] = "TU"

    weights_df = pd.concat([cell_weights_1_mean, cell_weights_2_mean], axis=0)

    sns.barplot(
        data=weights_df,
        x=cell_type_key,
        y="Weighting",
        hue="Modality",
        ax=ax20,
        palette=modality_palette
    )
    ax20.axhline(y=0.5, color='k', linestyle='--')
    # remove legend
    ax20.get_legend().remove()

    # REMOVE X-AXIS LABELS
    ax20.set_xticklabels([])
    ax20.set_xlabel("")
    ax20.set_ylabel("Relevance")

    # --- CREATE CUSTOM CELL TYPE LEGEND ON THE RIGHT ---
    # Extract the color palette used in the AnnData
    categories = adata_1.obs[cell_type_key].cat.categories
    colors = adata_1.uns[cell_type_key + "_colors"]
    cell_type_palette = dict(zip(categories, colors))

    # The x-positions for the bars in a categorical Seaborn plot are 0, 1, 2, ...
    x_positions = range(len(categories))

    # Create "bullet point" markers for the legend
    legend_handles = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=8, label=cat)
        for cat, color in zip(categories, colors)
    ]
    ax20.set_title('c', loc='left', fontsize=20, fontweight='bold')

    # UMAP displaying ZINB weighting
    ax03 = fig.add_subplot(gs[0, 3])
    sc.pl.umap(
        adata_1,
        color=f"{likelihood_key_1}_weighting",
        ax=ax03,
        show=False,
        frameon=False,
        title="GE weight",
        size=10,
        vmin=0.0,
        vmax=1.0,
        colorbar_loc=None
    )
    plot_customized_UMAP_coordinates(ax03, length=1.0)
    ax03.set_title('b', loc='left', fontsize=20, fontweight='bold')

    # UMAP displaying ZIDM weighting
    ax13 = fig.add_subplot(gs[1, 3])
    adata_2.obsm["X_umap"] = adata_1.obsm["X_umap"].copy()
    sc.pl.umap(
        adata_2,
        color=f"{likelihood_key_2}_weighting",
        ax=ax13,
        show=False,
        frameon=False,
        title="TU weight",
        size=10,
        vmin=0.0,
        vmax=1.0,
        colorbar_loc=None
    )

    plot_customized_UMAP_coordinates(ax13, length=1.0)

    # Dedicated GridSpec cells for the two colorbars of the relevance weights
    ge_cbar_host = fig.add_subplot(gs[0, 4])
    tu_cbar_host = fig.add_subplot(gs[1, 4])

    ge_cbar_host.axis("off")
    tu_cbar_host.axis("off")

    # Explicit positions within the reserved cells
    ge_cax = ge_cbar_host.inset_axes(
        [0.15, 0.10, 0.10, 0.80]
    )

    tu_cax = tu_cbar_host.inset_axes(
        [0.15, 0.10, 0.10, 0.80]
    )

    ge_cbar = fig.colorbar(
        ax03.collections[0],
        cax=ge_cax
    )

    tu_cbar = fig.colorbar(
        ax13.collections[0],
        cax=tu_cax
    )

    for cbar in [ge_cbar, tu_cbar]:
        cbar.set_ticks([0.0, 0.5, 1.0])

    ge_cbar.set_label("Relevance")
    tu_cbar.set_label("Relevance")

    # Violinplot
    ax21 = fig.add_subplot(gs[2, 1:3])
    sns.violinplot(
        data=df_ratios_melted,
        x=cell_type_key,
        y='log_weight_ratio',
        palette=cell_type_palette,  # 'vlag' is great for centered data (diverging colors)
        inner='quartile',
        legend=False,
        ax=ax21
    )

    ax21.axhline(0, color='red', linestyle='--', alpha=0.6)
    ax21.yaxis.tick_right()
    ax21.yaxis.set_label_position("right")
    ax21.set_xticklabels([])
    ax21.set_xlabel("")
    ax21.set_ylabel(r'$\log\left(\frac{GE\ Weight}{TU\ Weight}\right)$')
    ax21.set_title('d', loc='left', fontsize=20, fontweight='bold')

    for ax in [ax20, ax21]:
        ax.scatter(x_positions, [-0.1] * len(categories), c=colors, s=80, transform=ax.get_xaxis_transform(),
                   clip_on=False, zorder=5)

    # Dotplot of DEGs
    deg_cluster_groups = adata_1.obs["leiden"].unique().tolist()

    if tax_level == "Heart" and dataset_name == "tabulaMuris":
        deg_cluster_groups = deg_cluster_groups[:5]
    elif tax_level == "Brain_Non-Myeloid" and dataset_name == "tabulaMuris":
        deg_cluster_groups = deg_cluster_groups[:5]

    ax30 = fig.add_subplot(gs[3, 0:2])
    sc.pl.rank_genes_groups_dotplot(
        adata_1,
        groups=deg_cluster_groups,
        groupby="leiden",
        standard_scale="var",
        n_genes=num_marker_genes,
        show=False,
        dendrogram=False,
        ax=ax30,
        return_fig=False
    )
    ax30.set_title('e', loc='left', fontsize=20, fontweight='bold')

    # Dotplot of DSGs
    ax33 = fig.add_subplot(gs[3, 2:4]) # prev [5, :3]
    row_spacing = 0.75

    y_coords = (Y * row_spacing).flatten()

    psi_dotplot = ax33.scatter(
        x_coords,
        y_coords,
        c=color_values,
        s=size_values,
        cmap='Reds',
        edgecolors='k',
        alpha=0.9,
        vmin=0.0,
        vmax=1.0
    )

    # X axis
    ax33.set_xticks(range(len(intron_names_plot)))
    ax33.set_xticklabels(intron_names_plot, rotation=90, ha='right', fontsize=12)

    # Y axis: ticks must match compressed row positions
    y_tick_positions = np.arange(len(cluster_groups)) * row_spacing
    ax33.set_yticks(y_tick_positions)
    ax33.set_yticklabels(cluster_groups)

    # Ensure first and last row dots are fully visible
    ax33.set_ylim(-0.5 * row_spacing, (len(cluster_groups) - 0.5) * row_spacing)

    # Optional: also give a little horizontal margin so edge dots are fully visible
    ax33.set_xlim(-0.5, len(intron_names) - 0.5)
    ax33.set_title('f', loc='left', fontsize=20, fontweight='bold')

    # Custom dot-size legend
    size_legend_values = [50, 100, 250, 500]

    legend_ax = fig.add_subplot(gs[3, 4])# prev [5,3]
    legend_ax.set_xlim(-0.5, len(size_legend_values) - 0.5)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    legend_ax.set_title(
        "Number of cells\nin group",
        fontsize=10,
        pad=2
    )

    legend_x_positions = np.arange(len(size_legend_values))

    for x, s in zip(legend_x_positions, size_legend_values):
        legend_ax.scatter(
            x,
            0.72,
            s=s,
            color="gray",
            edgecolors="k",
            alpha=0.9
        )

        legend_ax.plot(
            [x, x],
            [0.47, 0.57],
            color="black",
            linewidth=1.3
        )

        legend_ax.text(
            x,
            0.35,
            str(s),
            ha="center",
            va="center",
            fontsize=9
        )

    # Horizontal colorbar inside the same reserved legend column
    cax = legend_ax.inset_axes(
        [0.05, 0.06, 0.90, 0.09]
    )

    cbar = fig.colorbar(
        psi_dotplot,
        cax=cax,
        orientation="horizontal"
    )

    cbar.set_ticks([0.00, 0.50, 1.00])
    cbar.ax.set_title("Mean PSI value\nin group", fontsize=10, pad=6)

    # Legend placed above
    n_cols = 5

    # We use fig.legend to place it relative to the whole figure
    # bbox_to_anchor handles the positioning.
    # (0.1, 0.92, 0.5, 0.05) -> (x_start, y_start, width, height)
    # This places it above the left side of the figure.
    leg = fig.legend(
        handles,
        labels,
        loc='upper left',
        bbox_to_anchor=(0.05, 0.965, 0.85, 0.03), #(0.05, 0.88, 0.85, 0.05),
        ncol=n_cols,
        mode="expand",  # This distributes the columns equally across the width provided
        borderaxespad=0.,
        frameon=False,
        fontsize=12,
        title="Cell type",
        title_fontsize=12
    )
    leg._legend_box.align = "left"  # Aligns title to the left of the legend block

    if isoforms_to_visualize is not None:
        assert "PSI_raw" in adata_2.layers, (
            "The key 'PSI_raw' is not present in adata.layers. "
            "Please add the raw PSI scores to "
            "adata.layers['PSI_raw'] before calling this function."
        )

        letters = ["g", "h"]
        renamed_isoforms = rename_isoform_helper(
            isoforms_to_visualize
        )

        for i, isoform in enumerate(isoforms_to_visualize):
            assert isoform in adata_2.var_names, (
                f"Isoform {isoform} not found in adata_2.var_names."
            )

            ax_iso = fig.add_subplot(gs[4, i])

            sc.pl.umap(
                adata_2,
                color=isoform,
                layer="PSI_raw",
                legend_loc=None,
                colorbar_loc=None,
                frameon=False,
                show=False,
                ax=ax_iso,
                vmin=0.0,
                vmax=1.0
            )

            # Add a separate colorbar directly beside this UMAP
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax_iso)

            isoform_cax = divider.append_axes(
                "right",
                size="4%",
                pad=0.08
            )

            isoform_cbar = fig.colorbar(
                ax_iso.collections[0],
                cax=isoform_cax
            )

            isoform_cbar.set_ticks([0.0, 0.5, 1.0])
            isoform_cbar.set_label("PSI")
            isoform_cbar.ax.tick_params(labelsize=9)

            ax_iso.set_title(
                renamed_isoforms[i],
                fontsize=12
            )
            ax_iso.set_title(
                letters[i],
                loc="left",
                fontsize=20,
                fontweight="bold"
            )

    plt.subplots_adjust(
        top=0.91,
        bottom=0.07,
        left=0.06,
        right=0.98,
        hspace=0.60,
        wspace=0.50
    )

    if save_fig:
        fig.canvas.draw()

        file_suffix = kwargs.get("file_suffix", 'pdf')
        fig_name = "./figures/" + dataset_name + "/clustering_TRVI_" + tax_level + "." + file_suffix
        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.25
        )

def plot_deg_dsg_upsetplot(
        ranked_genes_df: pd.DataFrame,
        ranked_spl_introns_df: pd.DataFrame,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given two dataframes ranked_spl_introns_df and ranked_genes_df containing the results of the differential expression
    and differential splicing analysis for a specific cluster, plot an upset plot to visualize the shared and unique
    significantly differentially expressed genes and differentially spliced genes. The figure is saved in the figures
    directory if save_fig is set to True.

    :param ranked_genes_df: pd.DataFrame, the dataframe containing the results of the differential expression analysis for a specific cluster, with a column "names" containing the gene names of the differentially expressed genes
    :param ranked_spl_introns_df: pd.DataFrame, the dataframe containing the results of the differential splicing analysis for a specific cluster, with a column "gene_names" containing the gene names of the differentially spliced genes
    :param save_fig: bool, whether to save the figure
    """

    sig_dsg_genes_group = np.unique(ranked_spl_introns_df["gene_names"].to_numpy())
    if "0" in sig_dsg_genes_group:
        sig_dsg_genes_group = np.delete(sig_dsg_genes_group, np.where(sig_dsg_genes_group == "0"))
    sig_deg_genes_group = ranked_genes_df["names"].to_numpy()

    dsg_only = np.setdiff1d(sig_dsg_genes_group, sig_deg_genes_group)
    deg_only = np.setdiff1d(sig_deg_genes_group, sig_dsg_genes_group)
    dsg_and_deg = np.intersect1d(sig_dsg_genes_group, sig_deg_genes_group)

    num_dsg_only = len(dsg_only)
    num_deg_only = len(deg_only)
    num_dsg_and_deg = len(dsg_and_deg)

    index_arrays = [
        np.array([True, False, True]),
        np.array([True, True, False])
    ]

    upset_names = ["# DEG", "# DSG"]

    multi_index_array = pd.MultiIndex.from_arrays(index_arrays, names=upset_names)
    upset_values = np.array([num_dsg_and_deg, num_dsg_only, num_deg_only])
    upset_series = pd.Series(upset_values, index=multi_index_array)

    upsetplot.plot(upset_series)

    # Print the exact numbers
    print(f"Number of differentially expressed genes: {num_deg_only}")
    print(f"Number of differentially spliced genes: {num_dsg_only}")
    print(f"Number of differentially expressed and differentially spliced genes: {num_dsg_and_deg}")
    print(f"The differentially expressed and spliced genes are: {dsg_and_deg}")

    # Unpack kwargs for saving if present
    dataset_name = kwargs.get("dataset_name", "dataset")
    group_id_GE = kwargs.get("group_id_GE", "GE")
    group_id_TU = kwargs.get("group_id_TU", "TU")
    tax_level = kwargs.get("tax_level", "all_tissues")

    if save_fig:
        fig_name = "./figures/"+ dataset_name +"/upsetplot_degs_" + group_id_GE + "_dsgs_" + group_id_TU + "_" + tax_level + ".png"
        plt.savefig(
            fig_name,
            dpi=300)

def plot_deg_dsg_venn_diagramm(
        ranked_genes_df: pd.DataFrame,
        ranked_spl_introns_df: pd.DataFrame,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given two dataframes ranked_genes_df and ranked_spl_introns_df containing the results of the differential expression
    and differential splicing analysis for a specific cluster, plot a Venn diagram to visualize the shared and unique
    significantly differentially expressed genes and differentially spliced genes. The set of DEGs is sized according to
    the size of the DSG set since there are less DSGs. The figure is saved in the figures directory if save_fig is set
    to True.

    :param ranked_genes_df: pd.DataFrame, the dataframe containing the results of the differential expression analysis for a specific cluster,
    :param ranked_spl_introns_df: pd.DataFrame, the dataframe containing the results of the differential splicing analysis for a specific cluster,
    :param save_fig: bool, whether to save the figure
    """
    # 1. Prepare sets
    set_dsg = set(ranked_spl_introns_df["gene_names"])
    if "0" in set_dsg:
        set_dsg.remove("0")
    set_deg = set(ranked_genes_df["names"].iloc[:len(set_dsg)].to_numpy())

    # 2. Create the plot
    plt.figure(figsize=(10, 8))
    v = venn2(subsets=[set_deg, set_dsg], set_labels=('Top DEGs', 'DSGs'))

    # --- CONFIGURATION ---
    FONT_SIZE = 18

    # 3. Format Helper
    def format_names_multi_col(name_list, cols=2):
        names = sorted(list(name_list))
        if not names: return ""
        num_names = len(names)
        rows = (num_names + cols - 1) // cols
        lines = []
        for r in range(rows):
            row_names = names[r::rows]
            lines.append("    ".join(row_names))
        return "\n".join(lines)

    # 4. Update Labels and Centering
    # We use x_offset to push labels toward the true center of the circles
    ids = ['10', '01', '11']
    subsets = [set_deg - set_dsg, set_dsg - set_deg, set_deg & set_dsg]

    for vid, s_data in zip(ids, subsets):
        lbl = v.get_label_by_id(vid)
        if lbl:
            num_cols = 2 if vid != '11' else 1
            lbl.set_text(format_names_multi_col(s_data, cols=num_cols))
            lbl.set_fontsize(FONT_SIZE)
            lbl.set_va('center')
            lbl.set_ha('center')

            # --- MANUAL POSITION ADJUSTMENT ---
            curr_pos = lbl.get_position()
            if vid == '10':  # Left Circle (DEGs)
                # Move slightly right (+x) or left (-x) to center in the circle
                lbl.set_position((curr_pos[0] + 0.05, curr_pos[1]))
            elif vid == '01':  # Right Circle (DSGs)
                # Move slightly left (-x) or right (+x)
                lbl.set_position((curr_pos[0] - 0.05, curr_pos[1]))
            elif vid == '11':  # Intersection
                # Usually stays at x=0, but you can nudge y if needed
                lbl.set_position((curr_pos[0], curr_pos[1]))

    # 5. Set Titles
    for label in v.set_labels:
        if label: label.set_fontsize(FONT_SIZE + 2)

    plt.gca().set_axis_off()

    # Unpack kwargs for saving if present
    dataset_name = kwargs.get("dataset_name", "dataset")
    group_id_GE = kwargs.get("group_id_GE", "GE")
    group_id_TU = kwargs.get("group_id_TU", "TU")
    tax_level = kwargs.get("tax_level", "all_tissues")

    if save_fig:
        fig_name = "./figures/" + dataset_name + "/ven_topdegs_" + group_id_GE + "_dsgs_" + group_id_TU + "_" + tax_level + ".png"
        plt.savefig(fig_name, dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.show()

def plot_deg_dsg_upsetplot_venn_across_taxonomy_level(
        cluster_group_pairs: List[List[str]],
        taxonomy_level: str,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Plot UpSet plots and Venn diagrams for DEG–DSG comparisons, with two
    comparisons per row.

    :param cluster_group_pairs: List of cluster-group pairs
    :param taxonomy_level: Taxonomy level analysed
    :param save_fig: Whether to save the figure
    """
    dataset_name = kwargs.get("dataset_name", "default")
    file_suffix = kwargs.get("file_suffix", "pdf")

    # Two comparisons per row, with an UpSet plot and Venn diagram each
    n_pairs_per_row = 2
    n_rows = (len(cluster_group_pairs) + n_pairs_per_row - 1) // n_pairs_per_row

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=4,
        figsize=(16, n_rows * 4),
        squeeze=False,
        gridspec_kw={
            "width_ratios": [1.3, 1, 1.3, 1],
            "wspace": 0.10,
            "hspace": 0.15,
        }
    )

    for i, cluster_group_pair in enumerate(cluster_group_pairs):
        group_id_1 = cluster_group_pair[0]
        group_id_2 = cluster_group_pair[1]

        # Determine the row and first column of this comparison
        row = i // n_pairs_per_row
        first_col = (i % n_pairs_per_row) * 2

        upset_ax = axs[row, first_col]
        venn_ax = axs[row, first_col + 1]

        # UpSet plot
        upset_path = (
            f"./figures/{dataset_name}/"
            f"upsetplot_degs_{group_id_1}_dsgs_{group_id_2}_"
            f"{taxonomy_level}.png"
        )
        upset_img = mpimg.imread(upset_path)
        upset_ax.imshow(upset_img)
        upset_ax.axis("off")

        # Venn diagram
        venn_path = (
            f"./figures/{dataset_name}/"
            f"ven_topdegs_{group_id_1}_dsgs_{group_id_2}_"
            f"{taxonomy_level}.png"
        )
        venn_img = mpimg.imread(venn_path)
        venn_ax.imshow(venn_img)
        venn_ax.axis("off")

        # Panel label on the UpSet plot
        import string
        upset_ax.set_title(
            f"{string.ascii_lowercase[i]}",
            loc="left",
            fontsize=20,
            fontweight="bold",
            pad=5
        )

    # Hide unused axes when there is an odd number of comparisons
    total_axes_needed = len(cluster_group_pairs) * 2
    for flat_index, ax in enumerate(axs.flat):
        if flat_index >= total_axes_needed:
            ax.axis("off")

    if save_fig:
        fig_name = (
            f"./figures/{dataset_name}/"
            f"deg_dsg_upsetplots_venns_{taxonomy_level}.{file_suffix}"
        )

        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1
        )

    plt.show()

def plot_expression_psi_distribution_violins_across_taxonomy_level(
        distribution_exp_df_list: List[pd.DataFrame],
        distribution_psi_df_list: List[pd.DataFrame],
        taxonomy_level: str,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Plot gene-expression and PSI distributions for DEGs and DSGs across
    a specified taxonomy level.

    Each row contains:

    - Left: gene-expression violin plots.
    - Right: corresponding PSI-score violin plots.

    The figure uses a 180 mm full-width Nature Methods layout with an
    adaptive height capped at 247 mm.

    :param distribution_exp_df_list:
        List of DataFrames containing gene-expression distributions.
    :param distribution_psi_df_list:
        List of DataFrames containing PSI-score distributions.
    :param taxonomy_level:
        Taxonomy level or tissue represented in the figure.
    :param save_fig:
        Whether to save the figure.
    """

    if not distribution_exp_df_list:
        raise ValueError(
            "distribution_exp_df_list must contain at least one DataFrame."
        )

    if len(distribution_exp_df_list) != len(
            distribution_psi_df_list
    ):
        raise ValueError(
            "distribution_exp_df_list and distribution_psi_df_list "
            "must have the same length."
        )

    dataset_name = kwargs.get(
        "dataset_name",
        "default"
    )

    # ------------------------------------------------------------------
    # Typography for a 180 mm-wide Nature Methods figure
    # ------------------------------------------------------------------
    panel_label_fontsize = kwargs.get(
        "panel_label_fontsize",
        11
    )
    panel_label_y = kwargs.get(
        "panel_label_y",
        1.04
    )
    axis_label_fontsize = kwargs.get(
        "axis_label_fontsize",
        7
    )
    x_tick_fontsize = kwargs.get(
        "x_tick_fontsize",
        7
    )
    y_tick_fontsize = kwargs.get(
        "y_tick_fontsize",
        6
    )
    x_tick_rotation = kwargs.get(
        "x_tick_rotation",
        0
    )

    # Line widths
    frame_linewidth = kwargs.get(
        "frame_linewidth",
        0.6
    )
    violin_linewidth = kwargs.get(
        "violin_linewidth",
        0.6
    )
    grid_linewidth = kwargs.get(
        "grid_linewidth",
        0.5
    )
    grid_alpha = kwargs.get(
        "grid_alpha",
        0.5
    )

    # ------------------------------------------------------------------
    # Nature Methods figure dimensions
    # ------------------------------------------------------------------
    mm_per_inch = 25.4

    figure_width_mm = kwargs.get(
        "figure_width_mm",
        180.0
    )
    figure_max_height_mm = kwargs.get(
        "figure_max_height_mm",
        247.0
    )
    ideal_row_height_mm = kwargs.get(
        "violin_row_height_mm",
        42.0
    )
    minimum_row_height_mm = kwargs.get(
        "minimum_row_height_mm",
        32.0
    )

    if figure_width_mm <= 0:
        raise ValueError(
            "figure_width_mm must be greater than 0."
        )

    if figure_max_height_mm <= 0:
        raise ValueError(
            "figure_max_height_mm must be greater than 0."
        )

    if ideal_row_height_mm <= 0:
        raise ValueError(
            "violin_row_height_mm must be greater than 0."
        )

    if frame_linewidth <= 0:
        raise ValueError(
            "frame_linewidth must be greater than 0."
        )

    if violin_linewidth <= 0:
        raise ValueError(
            "violin_linewidth must be greater than 0."
        )

    number_of_rows = len(
        distribution_exp_df_list
    )

    available_row_height_mm = (
        figure_max_height_mm / number_of_rows
    )

    actual_row_height_mm = min(
        ideal_row_height_mm,
        available_row_height_mm
    )

    if actual_row_height_mm < minimum_row_height_mm:
        maximum_rows_per_figure = max(
            1,
            int(
                figure_max_height_mm
                // minimum_row_height_mm
            )
        )

        raise ValueError(
            f"{number_of_rows} violin rows cannot be displayed legibly "
            f"within {figure_width_mm:g} x "
            f"{figure_max_height_mm:g} mm. Plot at most "
            f"{maximum_rows_per_figure} rows per figure by splitting "
            "the input lists into smaller subsets."
        )

    figure_height_mm = (
        number_of_rows * actual_row_height_mm
    )

    fig, axs = plt.subplots(
        nrows=number_of_rows,
        ncols=2,
        figsize=(
            figure_width_mm / mm_per_inch,
            figure_height_mm / mm_per_inch
        ),
        squeeze=False
    )

    alphabet = list(
        "abcdefghijklmnopqrstuvwxyz"
    )

    if number_of_rows > len(alphabet):
        raise ValueError(
            "A maximum of 26 labelled rows is supported."
        )

    target_palette = {
        "target": "#E69F00",
        "other": "#56B4E9"
    }

    for i, (exp_df, psi_df) in enumerate(
            zip(
                distribution_exp_df_list,
                distribution_psi_df_list
            )
    ):
        expression_ax = axs[i, 0]
        psi_ax = axs[i, 1]

        # --------------------------------------------------------------
        # Gene-expression violin plots
        # --------------------------------------------------------------
        sns.violinplot(
            data=exp_df,
            x="Gene",
            y="Expression",
            hue="Group",
            hue_order=[
                "target",
                "other"
            ],
            split=True,
            inner="quartile",
            palette=target_palette,
            cut=0,
            linewidth=violin_linewidth,
            ax=expression_ax
        )

        expression_ax.set_title("")
        expression_ax.set_xlabel("")
        expression_ax.set_ylabel(
            "Gene expression level",
            fontsize=axis_label_fontsize,
            labelpad=2
        )

        expression_ax.tick_params(
            axis="x",
            labelsize=x_tick_fontsize,
            labelrotation=x_tick_rotation,
            pad=1
        )
        expression_ax.tick_params(
            axis="y",
            labelsize=y_tick_fontsize,
            length=2,
            pad=1
        )

        expression_legend = (
            expression_ax.get_legend()
        )

        if expression_legend is not None:
            expression_legend.remove()

        expression_ax.grid(
            axis="y",
            linestyle="--",
            linewidth=grid_linewidth,
            alpha=grid_alpha
        )
        expression_ax.set_axisbelow(True)

        expression_max = float(
            exp_df["Expression"].max()
        )

        expression_padding = max(
            abs(expression_max) * 0.05,
            1e-6
        )

        expression_ax.set_ylim(
            -expression_padding,
            expression_max + expression_padding
        )

        # One panel label for the complete expression/PSI row.
        expression_ax.set_title(
            alphabet[i],
            loc="left",
            fontsize=panel_label_fontsize,
            fontweight="bold",
            y=panel_label_y,
            pad=0
        )

        dsg_gene_names = (
            exp_df["Gene"]
            .drop_duplicates()
            .astype(str)
            .to_numpy()
        )

        # --------------------------------------------------------------
        # PSI-score violin plots
        # --------------------------------------------------------------
        sns.violinplot(
            data=psi_df,
            x="Isoform",
            y="PSI-score",
            hue="Group",
            hue_order=[
                "target",
                "other"
            ],
            split=True,
            inner="quartile",
            palette=target_palette,
            cut=0,
            linewidth=violin_linewidth,
            ax=psi_ax
        )

        psi_ax.set_title("")
        psi_ax.set_xlabel("")

        # Display the PSI y-axis and ticks on the left.
        psi_ax.yaxis.tick_left()
        psi_ax.yaxis.set_label_position(
            "left"
        )

        psi_ax.set_ylabel(
            "PSI-score",
            fontsize=axis_label_fontsize,
            labelpad=2
        )

        psi_ax.tick_params(
            axis="x",
            labelsize=x_tick_fontsize,
            labelrotation=x_tick_rotation,
            pad=1
        )
        psi_ax.tick_params(
            axis="y",
            labelsize=y_tick_fontsize,
            length=2,
            pad=1
        )

        # Replace isoform identifiers with their corresponding gene names.
        psi_tick_positions = (
            psi_ax.get_xticks()
        )

        if len(psi_tick_positions) != len(
                dsg_gene_names
        ):
            raise ValueError(
                f"Row {i} contains {len(psi_tick_positions)} PSI "
                f"categories but {len(dsg_gene_names)} gene names."
            )

        psi_ax.set_xticks(
            psi_tick_positions
        )
        psi_ax.set_xticklabels(
            dsg_gene_names,
            fontsize=x_tick_fontsize,
            rotation=x_tick_rotation
        )

        psi_ax.set_ylim(
            -0.03,
            1.03
        )
        psi_ax.set_yticks([
            0.0,
            0.25,
            0.5,
            0.75,
            1.0
        ])

        psi_legend = psi_ax.get_legend()

        if psi_legend is not None:
            psi_legend.remove()

        psi_ax.grid(
            axis="y",
            linestyle="--",
            linewidth=grid_linewidth,
            alpha=grid_alpha
        )
        psi_ax.set_axisbelow(True)

        # --------------------------------------------------------------
        # Reduce axis-frame and tick widths
        # --------------------------------------------------------------
        for ax in (
                expression_ax,
                psi_ax
        ):
            for spine in ax.spines.values():
                spine.set_linewidth(
                    frame_linewidth
                )

            ax.tick_params(
                axis="both",
                width=frame_linewidth
            )

    # Leave enough space between columns for the PSI y-axis label.
    fig.tight_layout(
        rect=(
            0.01,
            0.01,
            0.99,
            0.99
        ),
        pad=0.3,
        h_pad=0.8,
        w_pad=1.2
    )

    # ------------------------------------------------------------------
    # Save at an exact physical width of 180 mm
    # ------------------------------------------------------------------
    if save_fig:
        import os

        file_suffix = kwargs.get(
            "file_suffix",
            "pdf"
        )

        output_directory = (
            f"./figures/{dataset_name}/"
        )

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        fig_name = os.path.join(
            output_directory,
            f"deg_dsg_violins_"
            f"{taxonomy_level}.{file_suffix}"
        )

        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches=None,
            pad_inches=0
        )

        print(
            f"Figure saved to: {fig_name}"
        )

    plt.show()

def plot_shared_and_unique_enrichment_terms_upsetplot(
        deg_terms: int,
        dsg_terms: int,
        shared_pathways: int,
        tissue: str | None,
        dataset_name: str,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given the number of significantly enriched pathwats for DEGs, DSGs, and both, plot an upset plot to visualize the
    shared and unique significantly enriched functional terms for DEGs and DSGs. The figure is saved in the figures directory
    if save_fig is set to True.

    :param deg_terms: int, the number of significantly enriched pathways for DEGs
    :param dsg_terms: int, the number of significantly enriched pathways for DSGs
    :param shared_pathways: int, the number of significantly enriched pathways for both DEGs and DSGs
    :param tissue: str, the name of the tissue for which the analysis was performed, or None if the analysis was performed on all tissues
    :param dataset_name: str, the name of the dataset (e.g. "tabulaMuris") for which the analysis was performed, used for saving the figure
    :param save_fig: bool, whether to save the figure
    """

    # Each list inside the first argument represents a set "membership"
    upset_data = upsetplot.from_memberships(
        [
            ['DEG terms'],  # Only in DEG
            ['DSG terms'],  # Only in DSG
            ['DEG terms', 'DSG terms']  # In both (Overlapping)
        ],
        data=[deg_terms, dsg_terms, shared_pathways]
    )

    # 3. Create and show the plot
    upset = upsetplot.UpSet(upset_data, subset_size='sum', show_counts=True)
    upset.plot()

    if save_fig:
        # Add unpacking of kwargs dataset_name
        if tissue is not None:
            plt.savefig(f"./figures/{dataset_name}/upset_plot_deg_dsg_terms_{tissue}.png", bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f"./figures/{dataset_name}/upset_plot_deg_dsg_terms.png", bbox_inches='tight', dpi=300)

    plt.show()

def plot_enrichment_term_barchart(
    enrichment_results_df: pd.DataFrame,
    title: str = "Functional enrichment analysis",
    top_n: int | None = 15,
    sources: str | list[str] | tuple[str, ...] | None = None,
    order_by: str = "dataframe",
    show_rank: bool = True,
    pvalue_scale: str = "log",
    pvalue_label: str = "p-value",
    plot_term_id: bool = True,
    ax: plt.Axes | None = None,
    **kwargs
):
    """
    Plot functional-enrichment terms as a horizontal bar plot.

    :param enrichment_results_df: DataFrame containing the enrichment results. Must contain the following columns:
        - ``source``: The source of the enrichment term (e.g., "GO:BP").
        - ``native``: The native identifier of the enrichment term (e.g., "GO:0008150").
        - ``name``: The name of the enrichment term (e.g., "biological_process").
        - ``p_value``: The p-value associated with the enrichment term.
        - ``intersection_size``: The number of genes associated with the enrichment term.
    :param title: Title of the plot.
    :param top_n: Number of terms to display. Use ``None`` to display every term.
    :param sources: One source (e.g. ``"GO:BP"``), several sources, or ``None`` for all.
    :param order_by: ``"dataframe"`` preserves the input row ranking; ``"p_value"`` orders by significance; and ``"intersection_size"`` orders by gene number.
    :param show_rank: Prefix each y-axis label with its original 1-based dataframe rank.
    :param pvalue_scale: Color scale for p-values: ``"log"`` or ``"linear"``.
    :param pvalue_label: Label for the p-value colorbar.
    :param plot_term_id: Whether to include the term ID in the y-axis labels.
    :param ax: matplotlib axes object
    :param kwargs: Additional keyword arguments. Can include ``p_max`` to set the maximum p-value for color scaling.

    """
    from matplotlib import colors, ticker

    required = {"source", "native", "name", "p_value", "intersection_size"}
    missing = required.difference(enrichment_results_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    plot_df = enrichment_results_df.copy()
    plot_df["dataframe_rank"] = range(1, len(plot_df) + 1)

    if isinstance(sources, str):
        plot_df = plot_df.loc[plot_df["source"].eq(sources)].copy()
    elif sources is not None:
        plot_df = plot_df.loc[plot_df["source"].isin(sources)].copy()

    plot_df["p_value"] = pd.to_numeric(plot_df["p_value"], errors="coerce")
    plot_df["intersection_size"] = pd.to_numeric(
        plot_df["intersection_size"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["native", "name", "p_value", "intersection_size"])
    plot_df = plot_df.loc[plot_df["p_value"] > 0].copy()

    if plot_df.empty:
        raise ValueError(f"No terms remain after filtering for {sources!r}.")

    valid_orderings = {"dataframe", "p_value", "intersection_size"}
    if order_by not in valid_orderings:
        raise ValueError(f"order_by must be one of {sorted(valid_orderings)}")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be a positive integer or None")
    if pvalue_scale not in {"log", "linear"}:
        raise ValueError("pvalue_scale must be 'log' or 'linear'")

    # Select terms according to the requested ranking.
    if top_n is not None:
        if order_by == "dataframe":
            plot_df = plot_df.head(top_n)
        elif order_by == "p_value":
            plot_df = plot_df.nsmallest(top_n, "p_value")
        else:
            plot_df = plot_df.nlargest(top_n, "intersection_size")

    # barh draws the final row at the top, so sort in the reverse display order.
    if order_by == "dataframe":
        plot_df = plot_df.sort_values("dataframe_rank", ascending=False)
    elif order_by == "p_value":
        plot_df = plot_df.sort_values("p_value", ascending=False)
    else:
        plot_df = plot_df.sort_values("intersection_size", ascending=True)

    if plot_term_id:
        plot_df["term_label"] = (plot_df["native"].astype(str))
    else:
        plot_df["term_label"] = (
                plot_df["native"].astype(str) + " " + plot_df["name"].astype(str)
        )

    if show_rank:
        plot_df["term_label"] = (
            plot_df["dataframe_rank"].astype(str)
            + ". "
            + plot_df["term_label"]
        )
    # Do this if kwargs["p_max"] exists
    p_max = kwargs.get("p_max", plot_df["p_value"].max())
    p_min = plot_df["p_value"].min()
    if p_min == p_max:
        norm = colors.Normalize(vmin=p_min * 0.99, vmax=p_max * 1.01)
    elif pvalue_scale == "linear":
        norm = colors.Normalize(vmin=p_min, vmax=p_max)
    else:
        norm = colors.LogNorm(vmin=p_min, vmax=p_max)

    cmap = plt.colormaps["RdYlBu"]  # low p = red; high p = blue
    bar_colors = cmap(norm(plot_df["p_value"].to_numpy()))

    fig_height = max(4.5, 0.52 * len(plot_df) + 1.5)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(8, fig_height),
            constrained_layout=True,
        )
    else:
        fig = ax.figure

    ax.barh(
        plot_df["term_label"],
        plot_df["intersection_size"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.7,
    )

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Gene number")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="x", linestyle=(0, (1, 4)), color="0.65", linewidth=0.9)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.20)
    cbar = fig.colorbar(sm, cax=cax)

    cbar.set_label(pvalue_label)

    if isinstance(norm, colors.LogNorm):
        cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    cbar.ax.invert_yaxis()

    return fig, ax

def plot_enrichment_term_set_comparison(
        dataset_name: str,
        tax_level_list: List[str],
        sources: None | str = None,
        save_fig: bool = True,
        **kwargs
):
    r"""
    After having run enrichment term analysis based on a set of DSGs and DEGs, this function plots the results of the
    enrichment term analysis for each taxonomy level (e.g. tissue) in a grid of bar plots. The first column shows the
    enrichment terms for DSGs, while the second column shows the enrichment terms for overlapping DEGs and DSGs. The
    figure is saved in the figures directory if save_fig is set to True.

    :param dataset_name: str
    :param tax_level_list: List[str]
    :param sources: List[str]
    :param save_fig: bool
    """
    fig, axes = plt.subplots(len(tax_level_list), 2, figsize=(18, 6 * len(tax_level_list)))
    for i, tax_level in enumerate(tax_level_list):
        # Assess if file exists
        assert os.path.exists(f"./data/{dataset_name}/dsg_unique_terms_df_{tax_level}.csv"), f"File not found: ./data/{dataset_name}/dsg_unique_terms_df_{tax_level}.csv"
        assert os.path.exists(f"./data/{dataset_name}/overlapping_terms_df_{tax_level}.csv"), f"File not found: ./data/{dataset_name}/overlapping_terms_df_{tax_level}.csv"

        enrichment_results_dsg_unique_df = pd.read_csv(
            "./data/" + dataset_name + "/dsg_unique_terms_df_" + tax_level + ".csv")

        enrichment_results_overlapping_df = pd.read_csv(
            "./data/" + dataset_name + "/overlapping_terms_df_" + tax_level + ".csv")

        plot_enrichment_term_barchart(
            enrichment_results_dsg_unique_df,
            title=f"DSG enrichment terms for {tax_level}",
            top_n=25,
            sources=sources,  # None = all sources; e.g. "GO:BP" = GO biological process only
            order_by="dataframe",
            show_rank=False,
            pvalue_scale="log",  # use "linear" for a linear p-value color scale
            pvalue_label="adjusted p-value",
            plot_term_id=True,
            ax=axes[i, 0],
            p_max=0.05
        )

        plot_enrichment_term_barchart(
            enrichment_results_overlapping_df,
            title=f"DEG and DSG enrichment terms for {tax_level}",
            top_n=25,
            sources=sources,  # None = all sources; e.g. "GO:BP" = GO biological process only
            order_by="dataframe",
            show_rank=False,
            pvalue_scale="log",  # use "linear" for a linear p-value color scale
            pvalue_label="adjusted p-value",
            plot_term_id=True,
            ax=axes[i, 1],
            p_max=0.05
        )

    axes[0, 0].set_title('a', loc='left', fontsize=20, fontweight='bold', y=1.15)
    axes[0, 1].set_title('b', loc='left', fontsize=20, fontweight='bold', y=1.15)

    plt.tight_layout()

    if save_fig:
        # Convert sources into a filename-safe string
        file_suffix = kwargs.get("file_suffix", 'pdf')
        if sources is None:
            sources_name = "all_sources"
        elif isinstance(sources, str):
            sources_name = sources.replace(":", "_")
        else:
            sources_name = "_".join(
                str(source).replace(":", "_") for source in sources
            )

        joint_tax_levels = "_".join(str(tax_level) for tax_level in tax_level_list)

        fig_name = (
            f"./figures/{dataset_name}/"
            f"enrichment_{sources_name}_{joint_tax_levels}.{file_suffix}"
        )

        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.2,
        )

    plt.show()


@dataclass(frozen=True)
class GOFigurePlotResult:
    r"""
    Objects and tables produced by :func:`plot_functional_enrichment_bubble`. This is part of the wrapper function for
    GO-Figure plotting https://gitlab.com/evogenlab/GO-Figure

    """

    ax: Any
    plot_df: pd.DataFrame
    cluster_members: pd.DataFrame
    colorbar: Any | None
    size_legend: Any | None
    term_legend: Any | None
    extra_artists: tuple[Any, ...]


def _resolve_gofigure_script(gofigure_path: str | Path) -> Path:
    r"""
    Resolve a GO-Figure repository path or gofigure.py path. This is part of the wrapper function for
    GO-Figure plotting https://gitlab.com/evogenlab/GO-Figure

    """
    path = Path(gofigure_path).expanduser()
    if path.is_dir():
        path = path / "gofigure.py"
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find GO-Figure at {path}. Clone it with:\n"
            "git clone https://gitlab.com/evogenlab/GO-Figure.git"
        )
    if not (path.parent / "data" / "go.obo").is_file():
        raise FileNotFoundError(
            f"GO-Figure's data directory was not found beside {path}."
        )
    return path


@lru_cache(maxsize=4)
def _load_gofigure(gofigure_script_string: str) -> tuple[Any, ...]:
    r"""
    Load GO-Figure's functions and ontology resources without its CLI main. This is part of the wrapper function for
    GO-Figure plotting https://gitlab.com/evogenlab/GO-Figure

    """
    gofigure_script = Path(gofigure_script_string)
    code = gofigure_script.read_text(encoding="utf-8")
    main_marker = "\nparser = read_arguments()"
    if main_marker not in code:
        raise RuntimeError(
            "The installed GO-Figure version has an unexpected entry point."
        )

    definitions = code.split(main_marker, 1)[0]
    # Preserve the active Matplotlib backend and Python's global warning system.
    definitions = definitions.replace("matplotlib.use('Agg')", "")
    definitions = definitions.replace("warnings.warn = warn", "")

    api = ModuleType(f"gofigure_{abs(hash(gofigure_script_string))}")
    api.__file__ = str(gofigure_script)
    try:
        exec(compile(definitions, str(gofigure_script), "exec"), api.__dict__)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "A GO-Figure dependency is missing. Install the dependencies with:\n"
            "pip install numpy pandas matplotlib seaborn scikit-learn adjustText"
        ) from error

    logger = logging.getLogger(api.__name__)
    logger.addHandler(logging.NullHandler())
    api.logger = logger
    api.logger_std = logger

    script_path = str(gofigure_script.parent)
    ic_dict, frequency_dict = api.read_IC(script_path)
    description_dict, namespace_dict, obsolete_dict, alt_dict = api.read_OBO(
        script_path
    )
    parents_dict, children_dict = api.read_parents_children(script_path)

    return (
        api,
        ic_dict,
        frequency_dict,
        description_dict,
        namespace_dict,
        obsolete_dict,
        alt_dict,
        parents_dict,
        children_dict,
    )

def plot_functional_enrichment_bubble(
    enrichment_results_df: pd.DataFrame,
    ax: Any,
    *,
    gofigure_path: str | Path = "./GO-Figure",
    source: str = "GO:BP",
    title: str | None = "Functional enrichment analysis",
    top_n: int | None = None,
    id_col: str = "native",
    pvalue_col: str = "p_value",
    gene_number_col: str = "intersection_size",
    source_col: str = "source",
    colour_by: str = "p_value",
    tissue_col: str = "tissue",
    tissue_palette: str = "tab20",
    tissue_legend_location: str = "upper left",
    tissue_legend_bbox_to_anchor: tuple[float, float] | None = (1.05, 1.0),
    similarity_cutoff: float = 1.0,
    palette: str = "Reds_r",
    size_range: str = "medium",
    label_n: int | None = None,
    description_limit: int = 60,
    random_state: int | None = 42,
    alpha: float = 0.5,
    pvalue_label: str = "Adjusted p-value",
    add_colorbar: bool = True,
    colorbar_ax: Any | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    add_size_legend: bool = True,
    size_legend_title: str = "Gene number",
    size_legend_location: str = "upper left",
    size_legend_bbox_to_anchor: tuple[float, float] | None = (1.65, 1.0),
    size_legend_labelspacing: float = 4.0,
    size_legend_handletextpad: float = 0.8,
    add_term_legend: bool = True,
    term_legend_location: str = "upper center",
    term_legend_bbox_to_anchor: tuple[float, float] = (0.5, -0.15),
    term_legend_columns: int = 2,
    scatter_kwargs: dict[str, Any] | None = None,
) -> GOFigurePlotResult:
    r"""Plot ontology-based functional enrichment bubbles on ``ax``.

    GO-Figure supplies the ontology, Lin semantic similarity, redundancy
    clustering, representative selection, and multidimensional scaling. This
    function only draws the resulting data on the supplied Matplotlib axis. It
    never creates, saves, shows, or closes a figure. It is a wrapper for https://gitlab.com/evogenlab/GO-Figure.

    Bubble area represents ``intersection_size``. Bubble colour represents
    either the adjusted p-value in ``p_value`` or the tissue in ``tissue``, as
    selected with ``colour_by``. GO-Figure defines separate semantic spaces for
    biological process, molecular function, and cellular component terms;
    select one with ``source``.

    :params enrichment_results_df: pandas.DataFrame Enrichment results containing ``native``, ``p_value``, ``intersection_size``, and ``source`` by default.
    :params ax: Existing Matplotlib axis on which the plot is drawn.
    :params gofigure_path: Path to the cloned GO-Figure repository or its ``gofigure.py`` file.
    :params source: str the enrichment term identifier `"GO:BP"``, ``"GO:MF"``, or ``"GO:CC"``.
    :params title: str title for the plot
    :params top_n: int | None Number of terms to display. Use ``None`` to display every term.
    :params id_col: str Column name for the enrichment term identifier.
    :params pvalue_col: str Column name for the enrichment term p-value.
    :params gene_number_col: str Column name for the enrichment term gene number.
    :params source_col: str Column name for the enrichment term source.
    :params colour_by: str Colour bubbles by ``"p_value"`` or ``"tissue"``.
    :params tissue_col: str Column containing the tissue annotation.
    :params tissue_palette: str Categorical palette used for tissue colours.
    :params tissue_legend_location: str Location of the tissue legend.
    :params tissue_legend_bbox_to_anchor: tuple[float, float]
    :params similarity_cutoff: float GO-Figure redundancy threshold. ``1.0`` only combines terms with Lin similarity exactly equal to one; ``0.5`` is GO-Figure's default.
    :params palette: str palette to use for plotting
    :params size_range: str range of size to use for plotting
    :params label_n: int | None Number of representative bubbles to number. ``None`` labels all.
    :params description_limit: int
    :params random_state: int
    :params alpha: float
    :params pvalue_label: str
    :params add_colorbar: bool
    :params colorbar_ax: Matplotlib axis on which the plot is drawn.
    :params colorbar_kwargs: dict
    :params add_size_legend: bool
    :params size_legend_title: str
    :params size_legend_location: str
    :params size_legend_bbox_to_anchor: tuple[float, float]
    :params size_legend_labelspacing: float
    :params size_legend_handletextpad: float
    :params add_term_legend: bool
    :params term_legend_location: str
    :params term_legend_bbox_to_anchor: tuple[float, float]
    :params term_legend_columns: int
    :params scatter_kwargs: dict[str, Any] | None

    Returns
    -------
    GOFigurePlotResult
        Plot data, cluster membership, supporting artists, and the supplied
        axis. ``extra_artists`` can be passed to a later ``fig.savefig`` call.

    Examples
    --------
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> result = plot_functional_enrichment_bubble(
    ...     enrichment_df, ax, gofigure_path="./GO-Figure", source="GO:BP"
    ... )
    >>> fig.savefig(
    ...     "enrichment.pdf", dpi=300, bbox_inches="tight", pad_inches=0.3,
    ...     bbox_extra_artists=result.extra_artists,
    ... )
    """
    if ax is None or not hasattr(ax, "scatter"):
        raise TypeError("ax must be an existing Matplotlib axis.")

    namespace_map = {
        "GO:BP": "biological_process",
        "GO:MF": "molecular_function",
        "GO:CC": "cellular_component",
    }
    if source not in namespace_map:
        raise ValueError("source must be 'GO:BP', 'GO:MF', or 'GO:CC'.")
    if colour_by not in {"p_value", "tissue"}:
        raise ValueError("colour_by must be 'p_value' or 'tissue'.")
    if not 0 <= similarity_cutoff <= 1:
        raise ValueError("similarity_cutoff must be between 0 and 1.")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be a positive integer or None.")
    if label_n is not None and label_n < 0:
        raise ValueError("label_n must be non-negative or None.")
    if size_range not in {"small", "medium", "big"}:
        raise ValueError("size_range must be 'small', 'medium', or 'big'.")
    if size_legend_handletextpad < 0:
        raise ValueError("size_legend_handletextpad must be non-negative.")
    if term_legend_columns < 1:
        raise ValueError("term_legend_columns must be at least 1.")

    required = {id_col, pvalue_col, gene_number_col, source_col}
    if colour_by == "tissue":
        required.add(tissue_col)
    missing = sorted(required.difference(enrichment_results_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    terms = enrichment_results_df.copy()
    identifiers = terms[id_col].astype("string").str.strip()
    terms = terms[
        identifiers.str.fullmatch(r"GO:\d{7}", na=False)
        & terms[source_col].eq(source)
    ].copy()
    terms[id_col] = terms[id_col].astype(str).str.strip()
    terms[pvalue_col] = pd.to_numeric(terms[pvalue_col], errors="coerce")
    terms[gene_number_col] = pd.to_numeric(
        terms[gene_number_col], errors="coerce"
    )
    terms = terms.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[id_col, pvalue_col, gene_number_col]
    )
    if colour_by == "tissue":
        terms = terms.dropna(subset=[tissue_col])
        terms[tissue_col] = terms[tissue_col].astype(str)
    terms = terms[
        (terms[pvalue_col] > 0) & (terms[gene_number_col] > 0)
    ].sort_values(pvalue_col, kind="stable")

    if top_n is not None:
        terms = terms.head(top_n)
    if terms.empty:
        raise ValueError(f"No valid {source} terms remain after filtering.")
    if terms[id_col].duplicated().any():
        duplicates = terms.loc[terms[id_col].duplicated(), id_col].tolist()
        raise ValueError(f"GO identifiers must be unique; duplicates: {duplicates}")

    gofigure_script = _resolve_gofigure_script(gofigure_path)
    (
        api,
        ic_dict,
        frequency_dict,
        description_dict,
        namespace_dict,
        obsolete_dict,
        alt_dict,
        parents_dict,
        children_dict,
    ) = _load_gofigure(str(gofigure_script))

    input_dict = {
        row[id_col]: [
            row[id_col],
            str(float(row[gene_number_col])),
            str(float(row[pvalue_col])),
        ]
        for _, row in terms.iterrows()
    }

    # GO-Figure 1.0.1 stores several runtime settings as module globals.
    api.obsolete_dict = obsolete_dict
    api.parents_dict = parents_dict
    api.children_dict = children_dict
    api.ic_dict = ic_dict
    api.top_level = None
    api.name_changes = None
    api.sum_user = False
    api.sort_by = "pval"

    go_dict = api.create_GO_dict(
        input_dict,
        "standard-plus",
        namespace_map[source],
        namespace_dict,
        ic_dict,
        frequency_dict,
        99999.0,
        alt_dict,
    )
    if not go_dict:
        raise ValueError(
            f"GO-Figure found no {source} identifiers in its bundled ontology."
        )

    semantic_clusters = api.create_clusters(
        go_dict,
        parents_dict,
        children_dict,
        ic_dict,
        similarity_cutoff,
        None,
    )
    cluster_dict = api.create_clusterdict(semantic_clusters, description_dict)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        plot_df, semantic_clusters = api.create_df(
            cluster_dict,
            go_dict,
            description_dict,
            "pval",
            "user",
            description_limit,
            "exhaustive",
            "numbered",
            random_state,
            semantic_clusters,
        )
    if plot_df.empty:
        raise ValueError("GO-Figure did not produce any representative terms.")

    cluster_rows: list[dict[str, Any]] = []
    for representative, members in semantic_clusters.items():
        seen: set[str] = set()
        for member, pvalue, user_value, information_content, frequency in members:
            if member in seen:
                continue
            seen.add(member)
            cluster_rows.append(
                {
                    "cluster_representative": representative,
                    "cluster_member": member,
                    "cluster_member_description": description_dict[member],
                    "p_value": float(pvalue),
                    "intersection_size": float(user_value),
                    "information_content": float(information_content),
                    "frequency": float(frequency),
                }
            )
    cluster_members = pd.DataFrame(cluster_rows)

    size_limits = {
        "small": (200.0, 2000.0),
        "medium": (400.0, 4000.0),
        "big": (600.0, 6000.0),
    }[size_range]
    reference_sizes = plot_df["size"].to_numpy(dtype=float)
    reference_min, reference_max = reference_sizes.min(), reference_sizes.max()

    def size_to_area(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if reference_max > reference_min:
            return size_limits[0] + (
                (values - reference_min) / (reference_max - reference_min)
            ) * (size_limits[1] - size_limits[0])
        return np.full(values.shape, np.mean(size_limits), dtype=float)

    plot_df = plot_df.copy()
    plot_df["bubble_area"] = size_to_area(reference_sizes)

    from matplotlib import rcParams
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FormatStrFormatter

    cmap = None
    norm = None
    tissue_colours = {}
    if colour_by == "p_value":
        cmap = ListedColormap(api.sns.color_palette(palette))
        colour_values = plot_df["colour"].to_numpy(dtype=float)
        colour_min, colour_max = colour_values.min(), colour_values.max()
        if np.isclose(colour_min, colour_max):
            delta = abs(colour_min) * 0.05 or 0.05
            colour_min, colour_max = colour_min - delta, colour_max + delta
        norm = Normalize(vmin=colour_min, vmax=colour_max)
    else:
        tissue_by_term = terms.set_index(id_col)[tissue_col]
        plot_df[tissue_col] = plot_df["representative"].map(tissue_by_term)
        if plot_df[tissue_col].isna().any():
            raise ValueError("A plotted GO term has no tissue annotation.")
        tissues = sorted(plot_df[tissue_col].unique())
        tissue_colours = dict(
            zip(
                tissues,
                api.sns.color_palette(tissue_palette, n_colors=len(tissues)),
            )
        )

    draw_df = plot_df.iloc[::-1]
    scatter_options: dict[str, Any] = {
        "s": draw_df["bubble_area"],
        "edgecolor": "black",
        "linewidth": 1.0,
        "alpha": alpha,
        "zorder": 2,
    }
    if colour_by == "p_value":
        scatter_options.update(c=draw_df["colour"], cmap=cmap, norm=norm)
    else:
        scatter_options["c"] = draw_df[tissue_col].map(tissue_colours)
    if scatter_kwargs:
        scatter_options.update(scatter_kwargs)
    ax.scatter(draw_df["x"], draw_df["y"], **scatter_options)

    displayed_labels = len(plot_df) if label_n is None else min(
        label_n, len(plot_df)
    )
    for number, row in plot_df.iloc[:displayed_labels].iterrows():
        ax.text(
            row["x"],
            row["y"],
            str(number),
            ha="center",
            va="center",
            fontsize="small",
            fontweight="semibold",
            color="black",
            zorder=3,
        )

    ax.set_xlabel("Semantic space X")
    ax.set_ylabel("Semantic space Y")
    if title is not None:
        ax.set_title(title)

    colorbar = None
    if add_colorbar and colour_by == "p_value":
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        kwargs = dict(colorbar_kwargs or {})
        if colorbar_ax is not None:
            kwargs["cax"] = colorbar_ax
            colorbar = ax.figure.colorbar(mappable, **kwargs)
        else:
            kwargs.setdefault("pad", 0.03)
            kwargs.setdefault("fraction", 0.055)
            colorbar = ax.figure.colorbar(mappable, ax=ax, **kwargs)
        colorbar.set_label(pvalue_label)
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))

    term_legend = None
    if add_term_legend and displayed_labels:
        term_labels = plot_df.iloc[:displayed_labels]["legend"].astype(str)
        empty_handles = [
            Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)
            for _ in term_labels
        ]
        term_legend = ax.legend(
            empty_handles,
            term_labels,
            bbox_to_anchor=term_legend_bbox_to_anchor,
            loc=term_legend_location,
            ncol=term_legend_columns,
            frameon=False,
            handlelength=0,
            handletextpad=0,
            fontsize="small",
        )
        ax.add_artist(term_legend)

    tissue_legend = None
    if colour_by == "tissue":
        tissue_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=colour,
                markeredgecolor="black",
                markersize=8,
                alpha=alpha,
                label=tissue,
            )
            for tissue, colour in tissue_colours.items()
        ]
        tissue_legend = ax.legend(
            handles=tissue_handles,
            title="Tissue",
            loc=tissue_legend_location,
            bbox_to_anchor=tissue_legend_bbox_to_anchor,
            frameon=True,
        )
        ax.add_artist(tissue_legend)

    size_legend = None
    if add_size_legend:
        size_values = np.unique(
            np.rint(np.quantile(reference_sizes, [0.0, 0.5, 1.0])).astype(int)
        )
        legend_areas = size_to_area(size_values.astype(float))
        size_handles = [
            ax.scatter(
                [],
                [],
                s=float(area),
                facecolors="none",
                edgecolors="black",
                linewidths=1.0,
            )
            for area in legend_areas
        ]

        # A scatter marker is not clipped to Matplotlib's default legend
        # handle box.  Make that box at least as wide as the largest marker so
        # every numeric label starts beyond the bubble and on the same line.
        legend_fontsize = FontProperties(
            size=rcParams["legend.fontsize"]
        ).get_size_in_points()
        marker_diameter = float(np.sqrt(legend_areas.max()))
        handlelength = max(
            float(rcParams["legend.handlelength"]),
            marker_diameter / legend_fontsize,
        )
        size_legend = ax.legend(
            size_handles,
            [str(value) for value in size_values],
            title=size_legend_title,
            loc=size_legend_location,
            bbox_to_anchor=size_legend_bbox_to_anchor,
            frameon=True,
            scatterpoints=1,
            markerscale=1.0,
            labelspacing=size_legend_labelspacing,
            handlelength=handlelength,
            handletextpad=size_legend_handletextpad,
            handleheight=4.0,
            borderpad=1.0,
        )
        size_legend.set_in_layout(True)

    extra_artists = tuple(
        artist
        for artist in (term_legend, tissue_legend, size_legend)
        if artist is not None
    )
    return GOFigurePlotResult(
        ax=ax,
        plot_df=plot_df,
        cluster_members=cluster_members,
        colorbar=colorbar,
        size_legend=size_legend,
        term_legend=term_legend,
        extra_artists=extra_artists,
    )

def plot_umap_latent_space_trvi(
        adata_objects: Tuple[AnnData, AnnData],
        observation_model_keys: List[str],
        seeds: List[str],
        color_keys: List[str],
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given a tuple of AnnData objects and model parameters, plot all five TRVI latent spaces for one or more model
    seeds. Each seed occupies a 2 x 6 block:

        P-GE | S-GE | GE-TU
        P-TU | S-TU | GE-TU

    For multiple seeds, these blocks are stacked vertically in the
    order given by ``seeds``.

    When one seed is supplied, the five latent-space UMAPs receive
    panel labels a-e. When multiple seeds are supplied, each complete
    seed block receives one panel label on its upper-left P-GE axis.

    Parameters
    ----------
    :param adata_objects: Tuple containing the GE and TU AnnData objects.
    :param observation_model_keys: Two observation-model names:
        [GE observation model, TU observation model].
        Supported GE models are NB and ZINB. Supported TU models are
        DM, ZIDM, and ZANIDM.
    :param seeds: Model seeds as strings, for example ["0"] or ["0", "5", "8"].
        The seed is used to identify the latent-space keys and as the
        Torch, NumPy, and UMAP random seed.
    :param color_keys: Two observation keys used to color the GE and TU UMAPs:
        [GE color key, TU color key]. GE-TU uses the GE color key.
    :param save_fig: Whether to save the figure.

    Keyword arguments
    -----------------
    umap_n_neighbors
        Number of UMAP neighbours. Default: 15.
    umap_min_dist
        UMAP minimum distance. Default: 0.1.
    figure_width_mm
        Figure width in millimetres. Default: 180.
    figure_height_mm
        Total figure height. By default, 90 mm per seed plus the
        space required for the horizontal legend and its gap when
        one or two seeds are displayed.
    max_figure_height_mm
        Maximum figure height. Default: 247 mm.
    seed_panel_hspace
        Vertical spacing between complete seed panels when more than
        one seed is plotted. Default: 0.14.
    multi_seed_figure_left, multi_seed_figure_right
        Left and right margins for a multi-seed figure in relative
        figure coordinates. Defaults: 0.035 and 0.965.
    multi_seed_figure_bottom, multi_seed_figure_top
        Bottom and top margins for a multi-seed figure in relative
        figure coordinates. Defaults: 0.03 and 0.975.
    legend_title
        Title of the shared categorical legend. Default: "Cell type".
    horizontal_legend_height_mm
        Height reserved below the plots for the shared legend when
        one or two seeds are displayed. Default: 14 mm.
    horizontal_legend_gap_mm
        Space between the UMAP panels and the horizontal legend.
        Default: 8 mm for one seed and 3 mm for two seeds.
    horizontal_legend_ncol
        Number of legend columns for one or two seeds. By default,
        up to four columns are used; additional entries wrap into
        further rows.
    vertical_legend_ncol
        Number of legend columns when more than two seeds are
        displayed. Default: 1.
    vertical_legend_gap
        Relative horizontal gap between the GE-TU UMAP and the
        full-height legend column. Default: 0.018.
    """

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not isinstance(adata_objects, (tuple, list)) or len(adata_objects) != 2:
        raise ValueError(
            "adata_objects must contain exactly two AnnData objects: "
            "(adata_GE, adata_TU)."
        )

    adata_ge, adata_tu = adata_objects

    if (
        not isinstance(observation_model_keys, (tuple, list))
        or len(observation_model_keys) != 2
    ):
        raise ValueError(
            "observation_model_keys must contain exactly two entries: "
            "[GE observation model, TU observation model]."
        )

    if not all(
        isinstance(model, str) and model.strip()
        for model in observation_model_keys
    ):
        raise TypeError(
            "Both entries in observation_model_keys must be non-empty strings."
        )

    ge_observation_model = observation_model_keys[0].strip().upper()
    tu_observation_model = observation_model_keys[1].strip().upper()

    valid_ge_models = {"NB", "ZINB"}
    valid_tu_models = {"DM", "ZIDM", "ZANIDM"}

    if ge_observation_model not in valid_ge_models:
        raise ValueError(
            f"Unsupported GE observation model '{ge_observation_model}'. "
            f"Expected one of {sorted(valid_ge_models)}."
        )

    if tu_observation_model not in valid_tu_models:
        raise ValueError(
            f"Unsupported TU observation model '{tu_observation_model}'. "
            f"Expected one of {sorted(valid_tu_models)}."
        )

    if not isinstance(seeds, (tuple, list)) or len(seeds) == 0:
        raise ValueError(
            "seeds must be a non-empty list of strings, for example "
            "['0'] or ['0', '5', '8']."
        )

    seed_specs = []
    used_seed_strings = set()
    used_seed_values = set()

    for seed in seeds:
        if not isinstance(seed, str) or not seed.strip():
            raise TypeError(
                "Every entry in seeds must be a non-empty string."
            )

        seed_string = seed.strip()

        try:
            seed_value = int(seed_string)
        except ValueError as error:
            raise ValueError(
                f"Seed '{seed_string}' cannot be converted to an integer."
            ) from error

        # NumPy and sklearn/UMAP RandomState require this range.
        if seed_value < 0 or seed_value > (2**32 - 1):
            raise ValueError(
                f"Seed '{seed_string}' is outside the supported range "
                f"0 to {2**32 - 1}."
            )

        if seed_string in used_seed_strings or seed_value in used_seed_values:
            raise ValueError(
                f"Seed '{seed_string}' occurs more than once in seeds."
            )

        used_seed_strings.add(seed_string)
        used_seed_values.add(seed_value)
        seed_specs.append((seed_string, seed_value))

    if not isinstance(color_keys, (tuple, list)) or len(color_keys) != 2:
        raise ValueError(
            "color_key must contain exactly two entries: "
            "[GE color key, TU color key]."
        )

    color_key_ge, color_key_tu = color_keys

    if color_key_ge not in adata_ge.obs:
        raise KeyError(
            f"The GE color key '{color_key_ge}' is not present in "
            "adata_objects[0].obs."
        )

    if color_key_tu not in adata_tu.obs:
        raise KeyError(
            f"The TU color key '{color_key_tu}' is not present in "
            "adata_objects[1].obs."
        )

    # ------------------------------------------------------------------
    # Identify the five latent spaces for each seed
    # ------------------------------------------------------------------
    latent_settings_by_seed = {}
    latent_keys_by_seed = {}

    for seed_string, _ in seed_specs:
        escaped_seed = re.escape(seed_string)
        escaped_ge_model = re.escape(ge_observation_model)
        escaped_tu_model = re.escape(tu_observation_model)

        latent_settings = {
            "P-GE": {
                "adata": adata_ge,
                "color_key": color_key_ge,
                "pattern": re.compile(
                    rf"^{escaped_ge_model}_{escaped_seed}_"
                    rf"private_latent_mean$"
                )
            },
            "P-TU": {
                "adata": adata_tu,
                "color_key": color_key_tu,
                "pattern": re.compile(
                    rf"^{escaped_tu_model}_{escaped_seed}_"
                    rf"private_latent_mean$"
                )
            },
            "S-GE": {
                "adata": adata_ge,
                "color_key": color_key_ge,
                "pattern": re.compile(
                    rf"^{escaped_ge_model}_{escaped_seed}_"
                    rf"shared_latent_mean$"
                )
            },
            "S-TU": {
                "adata": adata_tu,
                "color_key": color_key_tu,
                "pattern": re.compile(
                    rf"^{escaped_tu_model}_{escaped_seed}_"
                    rf"shared_latent_mean$"
                )
            },
            "GE-TU": {
                "adata": adata_ge,
                "color_key": color_key_ge,
                "pattern": re.compile(
                    rf"^{escaped_ge_model}_{escaped_seed}_"
                    rf"{escaped_tu_model}_{escaped_seed}_"
                    rf"shared_latent_mean$"
                )
            }
        }

        latent_settings_by_seed[seed_string] = latent_settings
        latent_keys_by_seed[seed_string] = {}

        for latent_space_name, settings in latent_settings.items():
            matching_keys = [
                key
                for key in settings["adata"].obsm.keys()
                if settings["pattern"].fullmatch(str(key))
            ]

            if len(matching_keys) == 0:
                raise KeyError(
                    f"No latent-space key was found for seed "
                    f"'{seed_string}' and latent space "
                    f"'{latent_space_name}'. Expected a key matching "
                    f"'{settings['pattern'].pattern}'."
                )

            if len(matching_keys) > 1:
                raise ValueError(
                    f"Multiple latent-space keys matched seed "
                    f"'{seed_string}' and latent space "
                    f"'{latent_space_name}': {matching_keys}."
                )

            latent_keys_by_seed[seed_string][
                latent_space_name
            ] = matching_keys[0]

    # ------------------------------------------------------------------
    # Calculate reproducible UMAP coordinates for each seed
    # ------------------------------------------------------------------
    umap_n_neighbors = kwargs.get("umap_n_neighbors", 15)
    umap_min_dist = kwargs.get("umap_min_dist", 0.1)

    if umap_n_neighbors <= 0:
        raise ValueError(
            "umap_n_neighbors must be greater than 0."
        )

    if umap_min_dist < 0:
        raise ValueError(
            "umap_min_dist must be non-negative."
        )

    latent_umap_coordinates = {}

    for seed_string, seed_value in seed_specs:
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)

        latent_umap_coordinates[seed_string] = {}

        for latent_space_name, latent_space_key in (
            latent_keys_by_seed[seed_string].items()
        ):
            current_adata = latent_settings_by_seed[
                seed_string
            ][latent_space_name]["adata"]

            latent_values = np.asarray(
                current_adata.obsm[latent_space_key]
            )

            if latent_values.ndim != 2:
                raise ValueError(
                    f"Latent-space key '{latent_space_key}' has shape "
                    f"{latent_values.shape}; a two-dimensional matrix "
                    "is required."
                )

            if latent_values.shape[0] != current_adata.n_obs:
                raise ValueError(
                    f"Latent-space key '{latent_space_key}' contains "
                    f"{latent_values.shape[0]} observations, but its "
                    f"AnnData object contains {current_adata.n_obs}."
                )

            latent_umap_coordinates[
                seed_string
            ][latent_space_name] = UMAP(
                n_components=2,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                random_state=seed_value
            ).fit_transform(latent_values)

    # ------------------------------------------------------------------
    # Nature Methods dimensions and typography
    # ------------------------------------------------------------------
    mm_per_inch = 25.4
    number_of_seeds = len(seed_specs)

    shared_categorical_legend = (
        "weighting" not in str(color_key_ge).lower()
    )
    use_horizontal_shared_legend = (
        shared_categorical_legend
        and number_of_seeds <= 2
    )
    use_vertical_shared_legend = (
        shared_categorical_legend
        and number_of_seeds > 2
    )

    horizontal_legend_height_mm = kwargs.get(
        "horizontal_legend_height_mm",
        14.0
    )
    horizontal_legend_gap_mm = kwargs.get(
        "horizontal_legend_gap_mm",
        (
            8.0
            if number_of_seeds == 1
            else 3.0
        )
    )

    if horizontal_legend_height_mm <= 0:
        raise ValueError(
            "horizontal_legend_height_mm must be greater than 0."
        )

    if horizontal_legend_gap_mm < 0:
        raise ValueError(
            "horizontal_legend_gap_mm must be non-negative."
        )

    horizontal_legend_reserved_mm = (
        horizontal_legend_height_mm
        + horizontal_legend_gap_mm
        if use_horizontal_shared_legend
        else 0.0
    )

    figure_width_mm = kwargs.get("figure_width_mm", 180.0)
    figure_height_mm = kwargs.get(
        "figure_height_mm",
        90.0 * number_of_seeds
        + horizontal_legend_reserved_mm
    )
    max_figure_height_mm = kwargs.get(
        "max_figure_height_mm",
        247.0
    )

    if figure_width_mm <= 0:
        raise ValueError(
            "figure_width_mm must be greater than 0."
        )

    if figure_height_mm <= 0:
        raise ValueError(
            "figure_height_mm must be greater than 0."
        )

    if max_figure_height_mm <= 0:
        raise ValueError(
            "max_figure_height_mm must be greater than 0."
        )

    figure_height_mm = min(
        figure_height_mm,
        max_figure_height_mm
    )

    latent_title_fontsize = kwargs.get(
        "latent_title_fontsize",
        8
    )
    panel_label_fontsize = kwargs.get(
        "panel_label_fontsize",
        11
    )
    panel_label_y = kwargs.get(
        "panel_label_y",
        1.04
    )
    umap_axis_fontsize = kwargs.get(
        "umap_axis_fontsize",
        7
    )
    legend_fontsize = kwargs.get(
        "legend_fontsize",
        6.5
    )
    legend_title_fontsize = kwargs.get(
        "legend_title_fontsize",
        8
    )
    colorbar_label_fontsize = kwargs.get(
        "colorbar_label_fontsize",
        7
    )
    colorbar_tick_fontsize = kwargs.get(
        "colorbar_tick_fontsize",
        6
    )

    cell_type_legend_marker_size = kwargs.get(
        "cell_type_legend_marker_size",
        36.0
    )
    cell_type_legend_labelspacing = kwargs.get(
        "cell_type_legend_labelspacing",
        0.65
    )
    cell_type_legend_handletextpad = kwargs.get(
        "cell_type_legend_handletextpad",
        0.5
    )

    weighting_vmin = kwargs.get("weighting_vmin", 0.0)
    weighting_vmax = kwargs.get("weighting_vmax", 1.0)

    if weighting_vmin >= weighting_vmax:
        raise ValueError(
            "weighting_vmin must be smaller than weighting_vmax."
        )

    # ------------------------------------------------------------------
    # Create the stacked seed layout
    # ------------------------------------------------------------------
    fig = plt.figure(
        figsize=(
            figure_width_mm / mm_per_inch,
            figure_height_mm / mm_per_inch
        ),
        dpi=300
    )

    axes_by_seed = {}
    grid_wspace = kwargs.get("grid_wspace", 0.22)
    grid_hspace = kwargs.get("grid_hspace", 0.22)

    horizontal_legend_height_fraction = (
        horizontal_legend_height_mm / figure_height_mm
        if use_horizontal_shared_legend
        else 0.0
    )
    horizontal_legend_reserved_fraction = (
        horizontal_legend_reserved_mm / figure_height_mm
        if use_horizontal_shared_legend
        else 0.0
    )

    if number_of_seeds == 1:
        single_seed_figure_left = kwargs.get(
            "single_seed_figure_left",
            0.035
        )
        single_seed_figure_right = kwargs.get(
            "single_seed_figure_right",
            0.965
        )
        single_seed_figure_bottom = kwargs.get(
            "single_seed_figure_bottom",
            (
                horizontal_legend_reserved_fraction + 0.015
                if use_horizontal_shared_legend
                else 0.03
            )
        )
        single_seed_figure_top = kwargs.get(
            "single_seed_figure_top",
            0.975
        )

        gs = gridspec.GridSpec(
            nrows=2,
            ncols=6,
            figure=fig,
            width_ratios=[1] * 6,
            height_ratios=[1] * 2,
            left=single_seed_figure_left,
            right=single_seed_figure_right,
            bottom=single_seed_figure_bottom,
            top=single_seed_figure_top,
            wspace=grid_wspace,
            hspace=grid_hspace
        )

        seed_string = seed_specs[0][0]

        axes_by_seed[seed_string] = {
            "P-GE": fig.add_subplot(gs[0, 0]),
            "P-TU": fig.add_subplot(gs[1, 0]),
            "S-GE": fig.add_subplot(gs[0, 1]),
            "S-TU": fig.add_subplot(gs[1, 1]),
            "GE-TU": fig.add_subplot(gs[:2, 2:])
        }

    else:
        # Use a separate 2 x 6 subgrid for every seed. This permits
        # a larger gap between complete seed panels without also
        # increasing the spacing between the UMAPs within each panel.
        multi_seed_figure_left = kwargs.get(
            "multi_seed_figure_left",
            0.035
        )
        multi_seed_figure_right = kwargs.get(
            "multi_seed_figure_right",
            (
                0.72
                if use_vertical_shared_legend
                else 0.965
            )
        )
        multi_seed_figure_bottom = kwargs.get(
            "multi_seed_figure_bottom",
            (
                horizontal_legend_reserved_fraction + 0.015
                if use_horizontal_shared_legend
                else 0.03
            )
        )
        multi_seed_figure_top = kwargs.get(
            "multi_seed_figure_top",
            0.975
        )
        seed_panel_hspace = kwargs.get(
            "seed_panel_hspace",
            0.14
        )

        if not (
            0 <= multi_seed_figure_left
            < multi_seed_figure_right <= 1
        ):
            raise ValueError(
                "The multi-seed left and right margins must satisfy "
                "0 <= left < right <= 1."
            )

        if not (
            0 <= multi_seed_figure_bottom
            < multi_seed_figure_top <= 1
        ):
            raise ValueError(
                "The multi-seed bottom and top margins must satisfy "
                "0 <= bottom < top <= 1."
            )

        if seed_panel_hspace < 0:
            raise ValueError(
                "seed_panel_hspace must be non-negative."
            )

        seed_panel_grid = gridspec.GridSpec(
            nrows=number_of_seeds,
            ncols=1,
            figure=fig,
            left=multi_seed_figure_left,
            right=multi_seed_figure_right,
            bottom=multi_seed_figure_bottom,
            top=multi_seed_figure_top,
            hspace=seed_panel_hspace
        )

        for seed_index, (seed_string, _) in enumerate(seed_specs):
            seed_grid = seed_panel_grid[
                seed_index,
                0
            ].subgridspec(
                nrows=2,
                ncols=6,
                width_ratios=[1] * 6,
                height_ratios=[1] * 2,
                wspace=grid_wspace,
                hspace=grid_hspace
            )

            axes_by_seed[seed_string] = {
                "P-GE": fig.add_subplot(seed_grid[0, 0]),
                "P-TU": fig.add_subplot(seed_grid[1, 0]),
                "S-GE": fig.add_subplot(seed_grid[0, 1]),
                "S-TU": fig.add_subplot(seed_grid[1, 1]),
                "GE-TU": fig.add_subplot(seed_grid[:2, 2:])
            }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def format_custom_umap_coordinates(ax, anchor="C"):
        number_of_existing_texts = len(ax.texts)

        plot_customized_UMAP_coordinates(
            ax,
            length=1.0
        )

        for text_artist in ax.texts[number_of_existing_texts:]:
            text_artist.set_fontsize(umap_axis_fontsize)

        ax.xaxis.label.set_size(umap_axis_fontsize)
        ax.yaxis.label.set_size(umap_axis_fontsize)
        ax.xaxis.labelpad = 1
        ax.yaxis.labelpad = 1

        ax.tick_params(
            axis="both",
            labelsize=umap_axis_fontsize
        )

        ax.set_box_aspect(1)
        ax.set_anchor(anchor)

    def find_scanpy_colorbar(ax):
        for collection in ax.collections:
            colorbar = getattr(collection, "colorbar", None)

            if colorbar is not None:
                return colorbar

        return None

    def format_scanpy_colorbar(ax, colorbar_label=None):
        colorbar = find_scanpy_colorbar(ax)

        if colorbar is None:
            return

        colorbar.ax.tick_params(
            labelsize=colorbar_tick_fontsize,
            length=2,
            pad=1
        )

        if colorbar_label:
            colorbar.set_label(
                colorbar_label,
                fontsize=colorbar_label_fontsize,
                labelpad=2
            )

    shared_legend_handles = None
    shared_legend_labels = None

    def extract_categorical_legend(ax):
        """Extract legend entries from one UMAP and remove its legend."""
        original_legend = ax.get_legend()

        if original_legend is None:
            return None, None

        legend_handles = getattr(
            original_legend,
            "legend_handles",
            None
        )

        if legend_handles is None:
            legend_handles = getattr(
                original_legend,
                "legendHandles",
                None
            )

        legend_labels = [
            text_artist.get_text()
            for text_artist in original_legend.get_texts()
        ]

        if legend_handles is None:
            legend_handles, fallback_labels = (
                ax.get_legend_handles_labels()
            )

            if not legend_labels:
                legend_labels = fallback_labels

        original_legend.remove()

        return list(legend_handles), list(legend_labels)

    def plot_latent_umap(
            seed_string,
            latent_space_name,
            collect_categorical_legend=False
    ):
        nonlocal shared_legend_handles
        nonlocal shared_legend_labels

        current_settings = latent_settings_by_seed[
            seed_string
        ][latent_space_name]

        current_adata = current_settings["adata"]
        current_color_key = current_settings["color_key"]
        current_ax = axes_by_seed[
            seed_string
        ][latent_space_name]

        is_weighting_key = (
            "weighting" in str(current_color_key).lower()
        )

        had_existing_umap = "X_umap" in current_adata.obsm

        if had_existing_umap:
            existing_umap = current_adata.obsm[
                "X_umap"
            ].copy()
        else:
            existing_umap = None

        try:
            current_adata.obsm["X_umap"] = (
                latent_umap_coordinates[
                    seed_string
                ][latent_space_name]
            )

            plot_kwargs = {
                "color": current_color_key,
                "title": "",
                "frameon": False,
                "show": False,
                "ax": current_ax
            }

            if is_weighting_key:
                plot_kwargs.update({
                    "vmin": weighting_vmin,
                    "vmax": weighting_vmax
                })
            else:
                plot_kwargs["legend_loc"] = (
                    "right margin"
                    if collect_categorical_legend
                    else None
                )

            sc.pl.umap(
                current_adata,
                **plot_kwargs
            )

        finally:
            if had_existing_umap:
                current_adata.obsm["X_umap"] = existing_umap
            else:
                del current_adata.obsm["X_umap"]

        current_ax.set_title(
            latent_space_name,
            loc="center",
            fontsize=latent_title_fontsize,
            pad=2
        )

        format_custom_umap_coordinates(
            current_ax,
            anchor=(
                "W"
                if latent_space_name == "GE-TU"
                else "C"
            )
        )

        if is_weighting_key:
            if latent_space_name in {
                "P-GE",
                "S-GE",
                "GE-TU"
            }:
                colorbar_label = kwargs.get(
                    "ge_colorbar_label",
                    "GE weight"
                )
            else:
                colorbar_label = kwargs.get(
                    "tu_colorbar_label",
                    "TU weight"
                )

            format_scanpy_colorbar(
                current_ax,
                colorbar_label=colorbar_label
            )

        elif collect_categorical_legend:
            (
                shared_legend_handles,
                shared_legend_labels
            ) = extract_categorical_legend(current_ax)

    def make_panel_label(index):
        """Return a, b, ..., z, aa, ab, ... for a zero-based index."""
        label = ""
        current_index = index + 1

        while current_index > 0:
            current_index, remainder = divmod(
                current_index - 1,
                26
            )
            label = chr(ord("a") + remainder) + label

        return label

    def add_shared_categorical_legend():
        """Add one figure-level categorical legend for all seeds."""
        if not shared_legend_handles or not shared_legend_labels:
            return None

        legend_title = kwargs.get(
            "legend_title",
            "Cell type"
        )
        legend_columnspacing = kwargs.get(
            "legend_columnspacing",
            1.4
        )
        legend_handles_to_plot = list(shared_legend_handles)
        legend_labels_to_plot = list(shared_legend_labels)

        if use_horizontal_shared_legend:
            legend_left = kwargs.get(
                "horizontal_legend_left",
                plot_content_left
            )
            legend_right = kwargs.get(
                "horizontal_legend_right",
                plot_content_right
            )
            legend_bottom = kwargs.get(
                "horizontal_legend_bottom",
                0.005
            )
            legend_top = kwargs.get(
                "horizontal_legend_top",
                horizontal_legend_height_fraction + 0.005
            )

            if not (
                0 <= legend_left < legend_right <= 1
                and 0 <= legend_bottom < legend_top <= 1
            ):
                raise ValueError(
                    "The horizontal legend coordinates must lie "
                    "between 0 and 1 and define a positive area."
                )

            legend_ncol = int(kwargs.get(
                "horizontal_legend_ncol",
                min(len(shared_legend_labels), 4)
            ))

            if legend_ncol <= 0:
                raise ValueError(
                    "horizontal_legend_ncol must be greater than 0."
                )

            # Matplotlib fills multi-row legends column by column.
            # Reorder entries so they appear row by row instead. For
            # five entries and four columns, the fifth entry is placed
            # beneath the first one, as in the requested layout.
            if len(legend_labels_to_plot) > legend_ncol:
                number_of_legend_rows = int(np.ceil(
                    len(legend_labels_to_plot) / legend_ncol
                ))
                row_major_indices = []

                for column_index in range(legend_ncol):
                    for row_index in range(number_of_legend_rows):
                        item_index = (
                            row_index * legend_ncol
                            + column_index
                        )

                        if item_index < len(legend_labels_to_plot):
                            row_major_indices.append(item_index)

                legend_handles_to_plot = [
                    legend_handles_to_plot[index]
                    for index in row_major_indices
                ]
                legend_labels_to_plot = [
                    legend_labels_to_plot[index]
                    for index in row_major_indices
                ]

            legend_ax = fig.add_axes([
                legend_left,
                legend_bottom,
                legend_right - legend_left,
                legend_top - legend_bottom
            ])
            legend_ax.set_axis_off()

            shared_legend = legend_ax.legend(
                legend_handles_to_plot,
                legend_labels_to_plot,
                title=legend_title,
                title_fontsize=legend_title_fontsize,
                loc="upper left",
                bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                bbox_transform=legend_ax.transAxes,
                mode=(
                    "expand"
                    if legend_ncol > 1
                    else None
                ),
                ncol=legend_ncol,
                frameon=False,
                fontsize=legend_fontsize,
                labelspacing=cell_type_legend_labelspacing,
                handletextpad=cell_type_legend_handletextpad,
                columnspacing=legend_columnspacing,
                borderaxespad=0
            )

        else:
            # This axis spans the complete height of the seed grid;
            # it is deliberately independent of individual seed rows.
            legend_left = kwargs.get(
                "vertical_legend_left",
                plot_content_right
                + kwargs.get(
                    "vertical_legend_gap",
                    0.018
                )
            )
            legend_right = kwargs.get(
                "vertical_legend_right",
                0.975
            )
            legend_bottom = kwargs.get(
                "vertical_legend_bottom",
                multi_seed_figure_bottom
            )
            legend_top = kwargs.get(
                "vertical_legend_top",
                multi_seed_figure_top
            )

            if not (
                0 <= legend_left < legend_right <= 1
                and 0 <= legend_bottom < legend_top <= 1
            ):
                raise ValueError(
                    "The vertical legend coordinates must lie "
                    "between 0 and 1 and define a positive area."
                )

            legend_ncol = int(kwargs.get(
                "vertical_legend_ncol",
                1
            ))

            if legend_ncol <= 0:
                raise ValueError(
                    "vertical_legend_ncol must be greater than 0."
                )

            legend_ax = fig.add_axes([
                legend_left,
                legend_bottom,
                legend_right - legend_left,
                legend_top - legend_bottom
            ])
            legend_ax.set_axis_off()

            shared_legend = legend_ax.legend(
                legend_handles_to_plot,
                legend_labels_to_plot,
                title=legend_title,
                title_fontsize=legend_title_fontsize,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.0),
                bbox_transform=legend_ax.transAxes,
                ncol=legend_ncol,
                frameon=False,
                fontsize=legend_fontsize,
                labelspacing=cell_type_legend_labelspacing,
                handletextpad=cell_type_legend_handletextpad,
                columnspacing=legend_columnspacing,
                borderaxespad=0
            )

        final_legend_handles = getattr(
            shared_legend,
            "legend_handles",
            None
        )

        if final_legend_handles is None:
            final_legend_handles = getattr(
                shared_legend,
                "legendHandles",
                []
            )

        for legend_handle in final_legend_handles:
            if hasattr(legend_handle, "set_sizes"):
                legend_handle.set_sizes([
                    cell_type_legend_marker_size
                ])
            elif hasattr(legend_handle, "set_markersize"):
                legend_handle.set_markersize(
                    np.sqrt(cell_type_legend_marker_size)
                )

        shared_legend.set_in_layout(False)

        if hasattr(shared_legend, "_legend_box"):
            shared_legend._legend_box.align = "left"

        return shared_legend

    # ------------------------------------------------------------------
    # Plot every seed block
    # ------------------------------------------------------------------
    latent_space_order = [
        "P-GE",
        "P-TU",
        "S-GE",
        "S-TU",
        "GE-TU"
    ]

    for seed_index, (seed_string, _) in enumerate(seed_specs):
        for latent_space_name in latent_space_order:
            plot_latent_umap(
                seed_string=seed_string,
                latent_space_name=latent_space_name,
                collect_categorical_legend=(
                    shared_categorical_legend
                    and seed_index == 0
                    and latent_space_name == "GE-TU"
                )
            )

    # ------------------------------------------------------------------
    # Add panel labels
    # ------------------------------------------------------------------
    if number_of_seeds == 1:
        seed_string = seed_specs[0][0]

        single_seed_panel_labels = {
            "P-GE": "a",
            "P-TU": "b",
            "S-GE": "c",
            "S-TU": "d",
            "GE-TU": "e"
        }

        for latent_space_name, panel_label in (
            single_seed_panel_labels.items()
        ):
            axes_by_seed[
                seed_string
            ][latent_space_name].set_title(
                panel_label,
                loc="left",
                fontsize=panel_label_fontsize,
                fontweight="bold",
                y=panel_label_y,
                pad=0
            )

    else:
        # For multiple seeds, label only the upper-left corner of
        # each complete two-row seed block.
        for seed_index, (seed_string, _) in enumerate(seed_specs):
            axes_by_seed[
                seed_string
            ]["P-GE"].set_title(
                make_panel_label(seed_index),
                loc="left",
                fontsize=panel_label_fontsize,
                fontweight="bold",
                y=panel_label_y,
                pad=0
            )

    # ------------------------------------------------------------------
    # Final layout
    # ------------------------------------------------------------------

    # For multiple seeds, the outer and inter-panel spacing is set
    # explicitly by seed_panel_grid. Calling tight_layout here would
    # reintroduce the large white margins visible in the prior figure.

    # Force Matplotlib to apply the square box aspects before reading
    # the final UMAP positions. Legends and separators are aligned to
    # these rendered axes rather than to the wider GridSpec slots.
    fig.canvas.draw()

    plot_content_left = min(
        axes_by_seed[seed_string][latent_space_name]
        .get_position().x0
        for seed_string, _ in seed_specs
        for latent_space_name in ("P-GE", "P-TU")
    )
    plot_content_right = max(
        axes_by_seed[seed_string]["GE-TU"]
        .get_position().x1
        for seed_string, _ in seed_specs
    )

    # ------------------------------------------------------------------
    # Add dashed separators between seed panels
    # ------------------------------------------------------------------
    from matplotlib.lines import Line2D
    if number_of_seeds > 1:
        separator_color = kwargs.get(
            "seed_separator_color",
            "#808080"
        )
        separator_linewidth = kwargs.get(
            "seed_separator_linewidth",
            0.7
        )
        separator_linestyle = kwargs.get(
            "seed_separator_linestyle",
            (0, (4, 3))
        )
        separator_xmin = kwargs.get(
            "seed_separator_xmin",
            plot_content_left
        )
        separator_xmax = kwargs.get(
            "seed_separator_xmax",
            plot_content_right
        )

        for seed_index in range(number_of_seeds - 1):
            upper_seed = seed_specs[seed_index][0]
            lower_seed = seed_specs[seed_index + 1][0]

            # Lowest edge of the upper seed panel.
            upper_panel_bottom = min(
                ax.get_position().y0
                for ax in axes_by_seed[upper_seed].values()
            )

            # Highest edge of the following seed panel.
            lower_panel_top = max(
                ax.get_position().y1
                for ax in axes_by_seed[lower_seed].values()
            )

            # Place the separator in the centre of the available gap.
            separator_y = (
                upper_panel_bottom + lower_panel_top
            ) / 2

            separator = Line2D(
                [separator_xmin, separator_xmax],
                [separator_y, separator_y],
                transform=fig.transFigure,
                color=separator_color,
                linewidth=separator_linewidth,
                linestyle=separator_linestyle,
                solid_capstyle="butt",
                zorder=1000,
                clip_on=False
            )

            fig.add_artist(separator)

    # Add the shared legend only after the UMAP axes and separator
    # positions have been finalised.
    if shared_categorical_legend:
        add_shared_categorical_legend()

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    if save_fig:
        dataset_name = kwargs.get(
            "dataset_name",
            "default"
        )
        tax_level = kwargs.get(
            "tax_level",
            "atlas_level"
        )
        file_suffix = kwargs.get(
            "file_suffix",
            "pdf"
        )

        save_path = os.path.join(
            "./figures",
            dataset_name
        )
        os.makedirs(save_path, exist_ok=True)

        fig_name = os.path.join(
            save_path,
            f"umap_latent_space_trvi_"
            f"{tax_level}.{file_suffix}"
        )

        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches=None,
            pad_inches=0
        )

    plt.show()

def center_enrichment_panel(
        fig,
        main_ax,
        panel_axes
):
    r"""
    This is a helper function plot_transcript_usage_cell_states_and_pathways_trvi to center the bubble-plot axis in
    the available figure width.

    Any auxiliary axes created by plot_functional_enrichment_bubble, such as
    its colorbar axis, are shifted by the same amount so that the panel layout
    remains intact.

    Parameters
    :param fig:
    :param main_ax:
    :param panel_axes:
    """
    fig.canvas.draw()

    main_position = main_ax.get_position()
    main_center_x = (main_position.x0 + main_position.x1) / 2
    available_center_x = (fig.subplotpars.left + fig.subplotpars.right) / 2
    horizontal_shift = available_center_x - main_center_x

    for panel_ax in panel_axes:
        position = panel_ax.get_position()
        panel_ax.set_position([
            position.x0 + horizontal_shift,
            position.y0,
            position.width,
            position.height,
        ])

def plot_transcript_usage_cell_states_and_pathways_trvi(
        adata_objects: Tuple[AnnData, AnnData],
        enrichment_results_dsg_unique_df_all: pd.DataFrame,
        enrichment_results_overlapping_df_all: pd.DataFrame,
        tax_level_list: List[str],
        embedding: str,
        seed: int,
        cell_org_hierarchy_dictionary: Dict[str, str],
        save_fig: bool = True ,
        **kwargs
):
    r"""
    Given two Anndata objects containing the gene expression and transcript usage data, the enrichment result dataframes
    inferred only from DSGs and inferred by both DEGs and DSGs, a list of four taxonomy levels (i.e. tissues for Tabula
    Muris), the selected TRVI embedding (i.e. GE-TU is recommended), a random seed, and a cell organisation
    hierarchy dictionary (maps taxonomy levels i.e. tissues in Tabula Muris to organ systems), plot three UMAPs in the
    first row showing the the different cell type groups (tissues in Tabula Muris), and the GE and TU relevance weights
    per cell. The second row contains a dot plot of the aggregated mean cell type GE and TU weight across cell types and
    cell type groups (tissues). The third row contains the aggregated GE and TU weights per cell type group (tissues).
    Then, for the four cell type groups (taxonomy levels / tissues) selected the enrichment results based on DSGs and
    DEGs are compared via upset plots. Finally the determined DSG-associated biological pathways and DEG and DSG
    associated pathways are plotted as bubble enrichemt plots. This requires an installation of GO-Figure
    (https://gitlab.com/evogenlab/GO-Figure.). If save_fig is set to True, the figure panel is saved.

    :param adata_objects: Tuple of two AnnData objects containing gene expression and transcript usage data
    :param enrichment_results_dsg_unique_df_all: DataFrame containing enrichment results for unique DSGs
    :param enrichment_results_overlapping_df_all: DataFrame containing enrichment results for overlapping DEGs and DSGs
    :param tax_level_list: List of taxonomy levels (tissues in Tabula Muris)
    :param embedding: Selected TRVI embedding
    :param seed: Random seed for reproducibility
    :param cell_org_hierarchy_dictionary:
    :param save_fig: boolean by default True
    """

    # Unpack adata_objects
    adata_1 = adata_objects[0].copy() # gene expression data
    adata_2 = adata_objects[1].copy() # transcript usage data

    # Construct modality relevance weight keys
    weighting_key_GE = "ZINB_" + str(seed) + "_weighting"
    weighting_key_TU = "ZIDM_" + str(seed) + "_weighting"

    # Unpack kwargs
    cell_type_groups_key = kwargs.get("cell_type_groups_key", "cell_type_group") # Default is for Tabula Muris
    cell_type_key = kwargs.get("cell_type_key", "cell_ontology_class") # Default is for Tabula Muris
    dataset_name = kwargs.get("dataset_name", "default")
    umap_dot_size = kwargs.get("umap_dot_size", 12)

    fig = plt.figure(figsize=(30, 82), dpi=300)

    # Separate the existing panels from the enrichment section
    outer_gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[48, 32],
        hspace=0.08,
    )

    # Existing panels a-e
    gs = outer_gs[0].subgridspec(
        6,
        4,
        height_ratios=[1, 1, 1, 1, 1, 1],
        wspace=0.8,
        hspace=1.1,
    )

    # Three equally sized columns used only by the first UMAP row
    umap_gs = gs[0:2, :].subgridspec(
        1,
        3,
        wspace=0.45,
    )

    # UMAP color bars
    umap_cbar_length = 0.85  # 85% of the UMAP-axis height
    umap_cbar_width = 0.035
    umap_cbar_pad = 0.03
    umap_cbar_y = (1 - umap_cbar_length) / 2

    # Cell type groups legend below the first UMAP
    cell_type_groups_columns = 4
    cell_type_groups_legend_y = -0.12

    # a UMAP of all_cell_type_groups

    adata_1.obsm["X_umap"] = UMAP(n_components=2, random_state=seed).fit_transform(adata_1.obsm[embedding])
    adata_2.obsm["X_umap"] = adata_1.obsm["X_umap"].copy()

    ax00 = fig.add_subplot(umap_gs[0, 0])

    sc.pl.umap(
        adata_1,
        color=cell_type_groups_key,
        ax=ax00,
        size=umap_dot_size,
        show=False,
        frameon=False
    )
    plot_customized_UMAP_coordinates(ax00, length=1.0)

    ax00.set_box_aspect(1)
    ax00.set_anchor("C")

    cell_type_groups_legend = ax00.get_legend()
    if cell_type_groups_legend is not None:
        cell_type_groups_legend.set_loc("upper center")
        cell_type_groups_legend.set_bbox_to_anchor((0.5, cell_type_groups_legend_y))
        cell_type_groups_legend.set_ncols(cell_type_groups_columns)
        cell_type_groups_legend.set_title("Tissue")
        cell_type_groups_legend.set_frame_on(False)
        cell_type_groups_legend.set_in_layout(True)

    ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')

    # b UMAP of GE-TU atlas with GE Weights plotted
    ax01 = fig.add_subplot(umap_gs[0, 1])

    sc.pl.umap(
        adata_1,
        color=weighting_key_GE,
        ax=ax01,
        size=umap_dot_size,
        show=False,
        frameon=False,
        legend_loc=None,
        colorbar_loc=None,
        vmin=0.0,
        vmax=1.0,
    )
    plot_customized_UMAP_coordinates(ax01, length=1.0)

    ax01.set_box_aspect(1)
    ax01.set_anchor("C")

    cbar_ax01 = ax01.inset_axes([
        1.0 + umap_cbar_pad,
        umap_cbar_y,
        umap_cbar_width,
        umap_cbar_length,
    ])

    fig.colorbar(ax01.collections[0], cax=cbar_ax01)
    ax01.set_title('b', loc='left', fontsize=20, fontweight='bold')
    ax01.set_title("GE weight", loc='center', fontsize=16)

    # c UMAP of GE-TU atlas with TU Weights plotted
    ax02 = fig.add_subplot(umap_gs[0, 2])
    sc.pl.umap(
        adata_2,
        color=weighting_key_TU,
        ax=ax02,
        size=umap_dot_size,
        show=False,
        frameon=False,
        legend_loc=None,
        colorbar_loc=None,
        vmin=0.0,
        vmax=1.0,
    )

    plot_customized_UMAP_coordinates(ax02, length=1.0)

    ax02.set_box_aspect(1)
    ax02.set_anchor("C")

    cbar_ax02 = ax02.inset_axes([
        1.0 + umap_cbar_pad,
        umap_cbar_y,
        umap_cbar_width,
        umap_cbar_length,
    ])

    fig.colorbar(ax02.collections[0], cax=cbar_ax02)

    ax02.set_title('c', loc='left', fontsize=20, fontweight='bold')
    ax02.set_title("TU weight", loc='center', fontsize=16)

    # d Dot plot
    all_cell_type_groups = list(adata_1.obs[cell_type_groups_key].unique())

    single_cell_weightings_1 = adata_1.obs[weighting_key_GE].to_numpy()
    single_cell_weightings_2 = 1 - single_cell_weightings_1
    single_cell_labels = adata_1.obs[cell_type_key].to_numpy()
    unique_cell_labels = np.unique(single_cell_labels)

    single_cell_weightings = np.zeros((len(unique_cell_labels), len(all_cell_type_groups) * 2))
    num_cells_GETU_per_cell_type_group = np.zeros((len(unique_cell_labels), len(all_cell_type_groups) * 2))
    cell_cell_type_group = adata_1.obs[cell_type_groups_key].to_numpy()

    for i, cell_type_group in enumerate(all_cell_type_groups):
        cell_type_group_indices = np.argwhere(cell_cell_type_group == cell_type_group)
        sc_w1_t = single_cell_weightings_1[cell_type_group_indices]
        sc_w2_t = single_cell_weightings_2[cell_type_group_indices]
        sc_labels_t = single_cell_labels[cell_type_group_indices]

        for j, label in enumerate(unique_cell_labels):
            label_indices = np.argwhere(sc_labels_t == label)
            if label_indices.size == 0: continue

            idx = label_indices[:, 0]
            single_cell_weightings[j, i * 2] = np.mean(sc_w1_t[idx])
            single_cell_weightings[j, i * 2 + 1] = np.mean(sc_w2_t[idx])
            num_cells_GETU_per_cell_type_group[j, i * 2] = len(idx)
            num_cells_GETU_per_cell_type_group[j, i * 2 + 1] = len(idx)

    num_cells_GETU_per_cell_type_group_relative = num_cells_GETU_per_cell_type_group / np.max(num_cells_GETU_per_cell_type_group)

    # DOT PLOT START

    # 1. Logic for naming and primary organ system assignment
    cell_system_counts_df = adata_1.obs.value_counts([cell_type_key, "organ_system"])

    cell_mapping = {}
    for label in unique_cell_labels:
        organ_systems_cell_present = cell_system_counts_df[label]
        primary_system = organ_systems_cell_present.index[0]
        display_name = label + "*" if len(organ_systems_cell_present) > 1 else label
        cell_mapping[label] = (primary_system, display_name)

    # Sort the cell types based on their assigned Primary Organ System
    sorted_cell_labels = sorted(
        unique_cell_labels,
        key=lambda x: (cell_mapping[x][0], x)
    )

    label_to_idx = {label: i for i, label in enumerate(unique_cell_labels)}
    new_row_indices = [label_to_idx[label] for label in sorted_cell_labels]
    x_display_labels = [cell_mapping[label][1] for label in sorted_cell_labels]

    # 2. Sort Tissues (Y-axis)
    sorted_cell_type_groups = sorted(all_cell_type_groups, key=lambda x: (cell_org_hierarchy_dictionary.get(x, "Unknown"), x))
    new_col_indices = []
    for t in sorted_cell_type_groups:
        orig_idx = all_cell_type_groups.index(t)
        new_col_indices.extend([orig_idx * 2, orig_idx * 2 + 1])

    # 3. Apply reordering to data
    ordered_weightings = single_cell_weightings[new_row_indices, :][:, new_col_indices]
    ordered_sizes = num_cells_GETU_per_cell_type_group_relative[new_row_indices, :][:, new_col_indices]

    weights_transposed = ordered_weightings.T
    sizes_transposed = ordered_sizes.T

    X_grid, Y_grid = np.meshgrid(range(weights_transposed.shape[1]), range(weights_transposed.shape[0]))
    x_coords = X_grid.flatten()
    y_coords = Y_grid.flatten()

    # 4. Create the Subplot
    # INCREASE FIGURE WIDTH HERE IF NEEDED: fig.set_figwidth(20)
    ax_dot = fig.add_subplot(gs[2:4, 0:])

    scatter = ax_dot.scatter(
        x=x_coords, y=y_coords,
        s=sizes_transposed.flatten() * 700,
        c=weights_transposed.flatten(),
        cmap='Reds',
        edgecolors='k',
        alpha=0.9,
        vmin=0.0,
        vmax=1.0
    )

    # 5. Styling Ticks and Labels (INCREASED SPACING)
    ax_dot.yaxis.tick_right()
    ax_dot.yaxis.set_label_position("right")

    # Use a steeper rotation (60 or 90) and smaller font if the list is very long
    ax_dot.set_xticks(np.arange(len(sorted_cell_labels)))
    ax_dot.set_xticklabels(x_display_labels, rotation=45, ha='right', fontsize=10)
    # Use tick_params to add physical padding between the axis and the labels
    ax_dot.tick_params(axis='x', which='major', pad=10)

    y_labels = ["GE", "TU"] * len(sorted_cell_type_groups)
    ax_dot.set_yticks(np.arange(len(y_labels)))
    ax_dot.set_yticklabels(y_labels, fontsize=10)

    # Add Tissue Names to the right
    for i, tissue_name in enumerate(sorted_cell_type_groups):
        center_y = i * 2 + 0.5
        ax_dot.text(1.04, center_y, tissue_name, ha='left', va='center',
                    fontsize=11, fontweight='bold', transform=ax_dot.get_yaxis_transform())
        if i < len(sorted_cell_type_groups) - 1:
            ax_dot.axhline(y=i * 2 + 1.5, color='black', linestyle='--', linewidth=1, alpha=0.3)

    # 6. Color bar ABOVE the dot plot
    # Position the colorbar relative to ax_dot so that it remains outside the
    # dot plot even when the surrounding GridSpec is adjusted.
    cbar_ax = ax_dot.inset_axes(
        [0.35, 1.10, 0.30, 0.035],
        transform=ax_dot.transAxes,
    )

    cbar_ax.set_in_layout(True)
    cbar_ax.set_clip_on(False)

    cbar = fig.colorbar(
        scatter,
        cax=cbar_ax,
        orientation="horizontal",
    )

    cbar.set_label("Relevance", fontsize=12, labelpad=8)
    cbar.set_ticks([0.0, 0.5, 1.0])

    cbar_ax.xaxis.set_ticks_position("top")
    cbar_ax.xaxis.set_label_position("top")

    ax_dot.set_title(
        "d",
        loc="left",
        fontsize=20,
        fontweight="bold",
        # pad=30,
    )

    # Formatting Grid and Background
    for j in range(len(sorted_cell_labels) - 1):
        ax_dot.axvline(x=j + 0.5, color='lightgrey', linestyle=':', linewidth=0.8, alpha=0.5)

    ax_dot.grid(False)
    ax_dot.set_facecolor('white')

    # DOT PLOT END

    # e Bar chart of weights and DEGs across tissus
    """
    cell_type_group_GE_weighting_df = adata_1.obs.groupby(cell_type_groups_key)[weighting_key_GE].mean().to_frame()
    cell_type_group_TU_weighting_df = adata_2.obs.groupby(cell_type_groups_key)[weighting_key_TU].mean().to_frame()

    cell_type_group_GE_weighting_df.rename(columns={weighting_key_GE: "GE weight"}, inplace=True)
    cell_type_group_TU_weighting_df.rename(columns={weighting_key_TU: "TU weight"}, inplace=True)

    cell_type_group_weighting_df = pd.concat([cell_type_group_GE_weighting_df, cell_type_group_TU_weighting_df], axis=1)
    cell_type_group_weighting_df = cell_type_group_weighting_df.reindex(sorted_cell_type_groups)

    plot_df = cell_type_group_weighting_df.reset_index().rename(columns={'index': cell_type_groups_key})
    plot_df = plot_df.melt(id_vars=cell_type_groups_key, var_name='Weight Type', value_name='Weight')
    """
    from .utils import create_cell_type_groups_relevance_weight_df
    plot_df = create_cell_type_groups_relevance_weight_df(adata_objects=(adata_1, adata_2),
                                                          weighting_keys=[weighting_key_GE, weighting_key_TU],
                                                          cell_type_groups_key=cell_type_groups_key,
                                                          cell_org_hierarchy_dictionary=cell_org_hierarchy_dictionary,
                                                          save_cell_type_groups_relevance_weight_df=True,
                                                          dataset_name=dataset_name)

    ax40 = fig.add_subplot(gs[4, :])

    sns.barplot(
        data=plot_df,
        x=cell_type_groups_key,
        y='Weight',
        hue='Weight Type',
        ax=ax40,
        palette='muted'
    )

    # 3. Refine the aesthetics
    ax40.set_ylim(0, 1.0)
    ax40.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Threshold (0.5)')

    # Adjust x-axis labels for readability
    ax40.set_xticklabels(ax40.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax40.set_xlabel(cell_type_groups_key, fontsize=10)
    ax40.set_ylabel('Weight Value', fontsize=10)

    # Move legend to avoid overlapping bars
    ax40.legend(
        title='Weight Type',
        loc='upper left',  # The anchor point on the legend box itself
        bbox_to_anchor=(1.05, 1),  # Places the legend just outside the right border
        fontsize='small',
        borderaxespad=0.  # Removes padding between the axes and the legend
    )
    ax40.set_title('e', loc='left', fontsize=20, fontweight='bold', pad=40)

    # f Enrichment terms upsetplots for three or four taxonomy level (cell_type_group) members (e.g. Brain_Non-Myeloid, Heart, Marrow, GAT)
    import matplotlib.image as mpimg
    ax50 = fig.add_subplot(gs[5, 0])
    upset_path = "./figures/" + dataset_name + "/upset_plot_deg_dsg_pathways_" + tax_level_list[0] + ".png"
    img = mpimg.imread(upset_path)
    ax50.imshow(img)
    ax50.axis('off')
    ax50.set_title(tax_level_list[0], loc='center', fontsize=12)

    ax50.set_title('f', loc='left', fontsize=20, fontweight='bold', y=1.15)

    ax51 = fig.add_subplot(gs[5, 1])
    upset_path = "./figures/" + dataset_name + "/upset_plot_deg_dsg_terms_" + tax_level_list[1] + ".png"
    img = mpimg.imread(upset_path)
    ax51.imshow(img)
    ax51.axis('off')
    ax51.set_title(tax_level_list[1], loc='center', fontsize=12)

    ax52 = fig.add_subplot(gs[5, 2])
    upset_path = "./figures/" + dataset_name + "/upset_plot_deg_dsg_terms_" + tax_level_list[2] + ".png"
    img = mpimg.imread(upset_path)
    ax52.imshow(img)
    ax52.axis('off')
    ax52.set_title(tax_level_list[2], loc='center', fontsize=12)

    ax53 = fig.add_subplot(gs[5, 3])
    upset_path = "./figures/" + dataset_name + "/upset_plot_deg_dsg_terms_" + tax_level_list[3] + ".png"
    img = mpimg.imread(upset_path)
    ax53.imshow(img)
    ax53.axis('off')
    ax53.set_title(tax_level_list[3], loc='center', fontsize=12)

    # Enrichment panels stacked vertically:
    # plot g
    # space for legend g
    # plot h
    # space for the longer legend h
    enrichment_gs = outer_gs[1].subgridspec(
        5,
        1,
        height_ratios=[
            2.4,  # panel f
            2.6,  # legend f
            0.6,  # explicit gap between panels
            2.4,  # panel g
            5.2,  # longer legend g
        ],
        hspace=0.04,
    )

    legend_space_f = fig.add_subplot(enrichment_gs[1, 0])
    legend_space_f.axis("off")

    panel_spacer = fig.add_subplot(enrichment_gs[2, 0])
    panel_spacer.axis("off")

    legend_space_g = fig.add_subplot(enrichment_gs[4, 0])
    legend_space_g.axis("off")

    # Panel g
    ax60 = fig.add_subplot(enrichment_gs[0, 0])
    ax60.set_box_aspect(1)
    ax60.set_anchor("C")

    # Record existing axes so that newly created colorbar axes can be identified
    axes_before_f = set(fig.axes)

    result_dsg_enrichment = plot_functional_enrichment_bubble(
        enrichment_results_dsg_unique_df_all,
        ax=ax60,
        gofigure_path="./GO-Figure",
        source="GO:BP",
        title="DSG-unique functional enrichment results",
        colour_by=cell_type_groups_key  # If not set colour_by = "p_value"
    )

    ax60.set_box_aspect(1)
    ax60.set_anchor("C")

    panel_f_axes = [ax60] + [
        ax for ax in fig.axes
        if ax not in axes_before_f
    ]

    ax60.set_title(
        "g",
        loc="left",
        fontsize=20,
        fontweight="bold",
        y=1.03,
    )

    # Panel h
    ax62 = fig.add_subplot(enrichment_gs[3, 0])
    ax62.set_box_aspect(1)
    ax62.set_anchor("C")

    axes_before_g = set(fig.axes)

    result_overlap_enrichment = plot_functional_enrichment_bubble(
        enrichment_results_overlapping_df_all,
        ax=ax62,
        gofigure_path="./GO-Figure",
        source="GO:BP",
        title="DSG-DEG functional enrichment results",
        colour_by=cell_type_groups_key  # If not set colour_by = "p_value"
    )

    ax62.set_box_aspect(1)
    ax62.set_anchor("C")

    panel_g_axes = [ax62] + [
        ax for ax in fig.axes
        if ax not in axes_before_g
    ]

    ax62.set_title(
        "h",
        loc="left",
        fontsize=20,
        fontweight="bold",
        y=1.03,
    )

    from matplotlib.legend import Legend

    for ax in (ax60, ax62):
        for artist in ax.get_children():
            if isinstance(artist, Legend):
                artist.set_clip_on(False)
                artist.set_in_layout(True)

    # Apply the final figure margins before centering the enrichment panels.
    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.98,
        bottom=0.02,
    )

    # Center the main bubble axes in their respective rows. The associated
    # enrichment colorbar axes are moved by the same amount.
    center_enrichment_panel(fig, ax60, panel_f_axes)
    center_enrichment_panel(fig, ax62, panel_g_axes)

    fig.canvas.draw()

    if save_fig:
        file_suffix = kwargs.pop("file_suffix", "pdf")
        fig_name = "./figures/" + dataset_name + "/figure6." + file_suffix
        fig.savefig(
            fig_name,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
        )



def plot_latent_spaces_MMVAEplus(
        adata_objects: Tuple[ad.AnnData, ad.AnnData],
        tissue: str,
        model_name: str,
        num_cols: int = 2,
        num_rows: int = 3,
        figsize: int = 4,
        wspace: float = 0.5,
        show_plot: bool = False,
        save_fig: bool = True,
        save_pgf: bool = False
) -> None:
    r"""
    Given two AnnData objects containing the gene expression and transcript usage data of a specific tissue,
    plot UMAPs of the shared and private latent spaces of the MMVAE+. As reference the UMAPs with PCA embeddings are
    given

    :param adata_objects:
    :param tissue:
    :param model_name:
    :param num_cols:
    :param num_rows:
    :param figsize:
    :param wspace:
    :param show_plot:
    :param save_fig:
    :param save_pgf:

    """

    adata_GE_linear, adata_TU_linear = adata_objects
    adata_GE_nonlinear = adata_GE_linear.copy()
    adata_TU_nonlinear = adata_TU_linear.copy()
    adata_shared = adata_GE_linear.copy()

    # Compute neighbourhood graph for linear embeddings of gene expression data
    sc.pp.pca(adata_GE_linear)  # default is 50 PCs
    sc.pp.neighbors(adata_GE_linear)
    sc.tl.umap(adata_GE_linear)

    # Compute neighbourhood graph for nonlinear embeddings of gene expression data
    adata_GE_nonlinear.obsm["X_umap"] = UMAP(n_components=2).fit_transform(adata_GE_nonlinear.obsm["MMVAE_latent"])

    # Compute neighbourhood graph for linear embeddings of transcript usage data
    sc.pp.pca(adata_TU_linear, layer="psi")  # default is 50 PCs
    sc.pp.neighbors(adata_TU_linear)
    sc.tl.umap(adata_TU_linear)

    # Compute neighbourhood graph for nonlinear embeddings of transcript usage data
    adata_TU_nonlinear.obsm["X_umap"] = UMAP(n_components=2).fit_transform(adata_TU_nonlinear.obsm["MMVAE_latent"])

    # Compute neighbourhood graph for shared nonlinear embeddings of gene expression and transcript usage
    adata_shared.obsm["X_umap"] = UMAP(n_components=2).fit_transform(adata_shared.obsm["MMVAE_shared_latent"])

    # Plot UMAPs
    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(num_cols * figsize + figsize * wspace * (num_cols - 1), num_rows * figsize),
    )

    plt.subplots_adjust(wspace=wspace)

    sc.pl.umap(adata_GE_linear, color='cell_ontology_class', title="GE PCA Embedding", legend_loc=None, ax=axs[0, 0],
               show=False, frameon=False)
    sc.pl.umap(adata_GE_nonlinear, color='cell_ontology_class', title="GE MMVAE+ Embedding", ax=axs[0, 1],
               show=False, frameon=False)
    sc.pl.umap(adata_TU_linear, color='cell_ontology_class', title="TU PCA Embedding", legend_loc=None,
               ax=axs[1, 0], show=False, frameon=False)
    sc.pl.umap(adata_TU_nonlinear, color='cell_ontology_class', title="TU MMVAE+ Embedding", legend_loc=None,
               ax=axs[1, 1], show=False, frameon=False)
    axs[2, 0].axis("off")
    sc.pl.umap(adata_shared, color='cell_ontology_class', title="Shared MMVAE+ Embedding", legend_loc=None,
               ax=axs[2, 1], show=False, frameon=False)

    # Customize UMAPs with coordinate system
    x_bottom = 0.05
    y_bottom = 0.05
    length = 0.1
    skip_axs = [-2]

    plot_customized_UMAP_coordinates(axes = axs, skip_axes=skip_axs, x_bottom=x_bottom, y_bottom=y_bottom, length=length)

    fig.suptitle(f"Comparison of Shared and Private Latent Spaces")

    if save_fig:
        fig.savefig("./figures/tabulaMuris/umap_embeddings_" + model_name + "_" + tissue + ".png", dpi=300,
                    bbox_inches='tight')

    if show_plot:
        plt.show()

    if save_pgf:

        fig, axs = plt.subplots()
        sc.pl.umap(adata_GE_nonlinear, color='cell_ontology_class', title="GE MMVAE+ Embedding", ax=axs, show=False, legend_loc=None)
        # Hide the axes
        axs.set_xticks([])
        axs.set_yticks([])
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.set_xlabel('')
        axs.set_ylabel('')
        axs.set_title('')
        fig.savefig("./figures/tabulaMuris/umap_embeddings_GE_MMVAE_" + model_name + "_" + tissue + ".png")

        fig, axs = plt.subplots()
        sc.pl.umap(adata_TU_nonlinear, color='cell_ontology_class', title="TU MMVAE+ Embedding", ax=axs, show=False, legend_loc=None)
        # Hide the axes
        axs.set_xticks([])
        axs.set_yticks([])
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.set_xlabel('')
        axs.set_ylabel('')
        axs.set_title('')
        fig.savefig("./figures/tabulaMuris/umap_embeddings_TU_MMVAE_" + model_name + "_" + tissue + ".png")

        fig, axs = plt.subplots()
        sc.pl.umap(adata_shared, color='cell_ontology_class', title="Shared MMVAE+ Embedding", ax=axs, show=False, legend_loc=None)
        # Hide the axes
        axs.set_xticks([])
        axs.set_yticks([])
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.set_xlabel('')
        axs.set_ylabel('')
        axs.set_title('')
        fig.savefig("./figures/tabulaMuris/umap_embeddings_shared_MMVAE_" + model_name + "_" + tissue + ".png")


def compare_linear_nonlinear_embeddings_GE_TU(
        adata_objects: Tuple[ad.AnnData, ad.AnnData],
        tissue: str,
        beta: float = 1.0,
        num_cols: int = 2,
        num_rows: int = 2,
        figsize: int = 4,
        wspace: float = 0.5,
        show_plot: bool = False,
        save_fig: bool = True,
) -> None:
    r"""
    Given two AnnData objects containing the gene expression and transcript usage data of a specific tissue, plot UMAPs with the linear
    and non-linear embeddings of the data

    :param adata_objects:
    :param tissue:
    :param beta:
    :param num_cols:
    :param num_rows:
    :param figsize:
    :param wspace:
    :param show_plot:
    :param save_fig:
    """
    adata_GE_linear, adata_TU_linear = adata_objects
    adata_GE_nonlinear = adata_GE_linear.copy()
    adata_TU_nonlinear = adata_TU_linear.copy()

    # Compute neighbourhood graph for linear embeddings of gene expression data
    sc.pp.pca(adata_GE_linear)  # default is 50 PCs
    sc.pp.neighbors(adata_GE_linear)
    sc.tl.umap(adata_GE_linear)

    # Compute neighbourhood graph for nonlinear embeddings of gene expression data TO DO CHECK IF VAE LATENT IS USED
    adata_GE_nonlinear.obsm["X_umap"] = UMAP(n_components=2).fit_transform(adata_GE_nonlinear.obsm["VAE_latent"])

    # Compute neighbourhood graph for linear embeddings of transcript usage data
    sc.pp.pca(adata_TU_linear)  # default is 50 PCs
    sc.pp.neighbors(adata_TU_linear)
    sc.tl.umap(adata_TU_linear)

    # Compute neighbourhood graph for nonlinear embeddings of transcript usage data
    adata_TU_nonlinear.obsm["X_umap"] = UMAP(n_components=2).fit_transform(adata_TU_nonlinear.obsm["VAE_latent"])

    # Plot UMAPs
    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(num_cols * figsize + figsize * wspace * (num_cols - 1), num_rows * figsize),
    )

    plt.subplots_adjust(wspace=wspace)

    sc.pl.umap(adata_GE_linear, color='cell_ontology_class', title="GE PCA Embedding", legend_loc=None, ax=axs[0, 0],
               show=False)
    sc.pl.umap(adata_GE_nonlinear, color='cell_ontology_class', title="GE VAE Embedding", ax=axs[0, num_cols - 1],
               show=False)
    sc.pl.umap(adata_TU_linear, color='cell_ontology_class', title="TU PCA Embedding", legend_loc=None,
               ax=axs[num_rows - 1, 0], show=False)
    sc.pl.umap(adata_TU_nonlinear, color='cell_ontology_class', title="TU VAE Embedding", legend_loc=None,
               ax=axs[num_rows - 1, num_cols - 1], show=False)

    fig.suptitle(f"Effect of Non-linear Embeddings for GE and TU")

    if save_fig:
        beta_str = f"{beta:.1f}"
        beta_str = beta_str.replace(".", "")
        while len(beta_str) < 3:
            beta_str = "0" + beta_str
        fig.savefig("./figures/tabulaMuris/umap_embeddings_GE_TU_VAE_" + beta_str + "_" + tissue + ".png", dpi=300,
                    bbox_inches='tight')

    if show_plot:
        plt.show()


def compare_UMAP_GE_TU_clustering(
        adata_objects: Tuple[ad.AnnData, ad.AnnData],
        tissue: str,
        beta: float,
        resolutions: np.ndarray,
        clustering_alg: str = "leiden",
        num_cols: int = 2,
        num_rows: int = 2,
        figsize: int = 4,
        wspace: float = 0.5,
        show_plot: bool = True,
        save_fig: bool = True,
) -> None:
    r"""
    Given two AnnData objects with UMAP obsm, plot the two UMAPs with reference cell ontology annotation as well as
    two UMAPs with the choice of clustering algorithm

    :param adata_objects:
    :param tissue:
    :param beta:
    :param resolutions:
    :param clustering_alg:
    :param num_cols:
    :param num_rows:
    :param figsize:
    :param wspace:
    :param show_plot:
    :param save_fig:
    """
    adata1, adata2 = adata_objects

    # Calculate the neighbourhood graph
    sc.pp.neighbors(adata1, use_rep="X_umap")
    sc.pp.neighbors(adata2, use_rep="X_umap")

    # Determine optimal clustering
    res1, nmi_score1 = scib.me.cluster_optimal_resolution(
        adata1,
        cluster_key="cluster",
        resolutions=resolutions,
        label_key="cell_ontology_class"
    )
    res2, nmi_score2 = scib.me.cluster_optimal_resolution(
        adata2,
        cluster_key="cluster",
        resolutions=resolutions,
        label_key="cell_ontology_class"
    )

    nmi_score1 = np.round(nmi_score1, 2)
    nmi_score2 = np.round(nmi_score2, 2)

    # Clustering algorithm
    if clustering_alg == "leiden":
        sc.tl.leiden(adata1, resolution=res1, flavor="igraph", n_iterations=2)
        sc.tl.leiden(adata2, resolution=res2, flavor="igraph", n_iterations=2)
    elif clustering_alg == "louvain":
        sc.tl.louvain(adata1, resolution=res1, flavor="igraph")
        sc.tl.louvain(adata2, resolution=res2, flavor="igraph")
    else:
        raise ValueError("Only Leiden or Louvain can be used as clustering algoriths")

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(num_cols * figsize + figsize * wspace * (num_cols - 1), num_rows * figsize),
    )

    plt.subplots_adjust(wspace=wspace)

    sc.pl.umap(adata1, color='cell_ontology_class', title="GE VAE Embedding", legend_loc=None, ax=axs[0, 0], show=False)
    sc.pl.umap(adata2, color='cell_ontology_class', title="TU VAE Embedding", ax=axs[0, num_cols - 1], show=False)
    sc.pl.umap(adata1, color=[clustering_alg], title="NMI: " + str(nmi_score1), ax=axs[num_rows - 1, 0], show=False)
    sc.pl.umap(adata2, color=[clustering_alg], title="NMI: " + str(nmi_score2), ax=axs[num_rows - 1, num_cols - 1],
               show=False)

    fig.suptitle(f"UMAP of GE and TU for $\\beta$ = {beta}")

    if show_plot:
        plt.show()

    if save_fig:
        beta_str = f"{beta:.1f}"
        beta_str = beta_str.replace(".", "")
        while len(beta_str) < 3:
            beta_str = "0" + beta_str
        fig.savefig("./figures/tabulaMuris/umap_comparison_GE_TU_VAE_" + beta_str + "_" + tissue + ".png", dpi=300,
                    bbox_inches='tight')


def compare_UMAP_MMVAEplus_clustering(
        adata_objects: Tuple[ad.AnnData, ad.AnnData, ad.AnnData],
        tissue: str,
        resolutions: np.ndarray,
        model_name: str,
        clustering_alg: str = "leiden",
        num_cols: int = 3,
        num_rows: int = 2,
        figsize: int = 4,
        wspace: float = 0.5,
        show_plot: bool = True,
        save_fig: bool = True,
        save_tikz: bool = False
) -> None:
    r"""
    Given three adata objects with UMAP obsm, plot the two UMAPs with reference cell ontology annotation as well as
    two UMAPs with the choice of clustering algorithm

    :param adata_objects:
    :param tissue:
    :param resolutions:
    :param model_name:
    :param clustering_alg:
    :param num_cols:
    :param num_rows:
    :param figsize:
    :param wspace:
    :param show_plot:
    :param save_fig:
    :param save_tikz:
    """
    adata1, adata2, adata3 = adata_objects

    # Calculate the neighbourhood graph
    sc.pp.neighbors(adata1, use_rep="X_umap")
    sc.pp.neighbors(adata2, use_rep="X_umap")
    sc.pp.neighbors(adata3, use_rep="X_umap")

    # Determine optimal clustering
    res1, nmi_score1 = scib.me.cluster_optimal_resolution(
        adata1,
        cluster_key="cluster",
        resolutions=resolutions,
        label_key="cell_ontology_class"
    )
    res2, nmi_score2 = scib.me.cluster_optimal_resolution(
        adata2,
        cluster_key="cluster",
        resolutions=resolutions,
        label_key="cell_ontology_class"
    )
    res3, nmi_score3 = scib.me.cluster_optimal_resolution(
        adata3,
        cluster_key="cluster",
        resolutions=resolutions,
        label_key="cell_ontology_class"
    )

    nmi_score1 = np.round(nmi_score1, 2)
    nmi_score2 = np.round(nmi_score2, 2)
    nmi_score3 = np.round(nmi_score3, 2)

    # Clustering algorithm
    if clustering_alg == "leiden":
        sc.tl.leiden(adata1, resolution=res1, flavor="igraph", n_iterations=2)
        sc.tl.leiden(adata2, resolution=res2, flavor="igraph", n_iterations=2)
        sc.tl.leiden(adata3, resolution=res3, flavor="igraph", n_iterations=2)
    elif clustering_alg == "louvain":
        sc.tl.louvain(adata1, resolution=res1, flavor="igraph")
        sc.tl.louvain(adata2, resolution=res2, flavor="igraph")
        sc.tl.louvain(adata3, resolution=res3, flavor="igraph")
    else:
        raise ValueError("Only Leiden or Louvain can be used as clustering algorithms")

    cell_types_1 = adata1.obs["cell_ontology_class"].to_numpy()
    cell_types_2 = adata2.obs["cell_ontology_class"].to_numpy()
    cell_types_3 = adata3.obs["cell_ontology_class"].to_numpy()

    assert np.all(cell_types_1 == cell_types_2), "Cells are not the same"
    assert np.all(cell_types_1 == cell_types_3), "Cells are not the same"

    cell_type_clusters = cell_types_1.reshape(-1, 1)

    for adata in adata_objects:
        cells_clusters = adata.obs[["cell_ontology_class", clustering_alg]]
        cell_indices = cells_clusters.index
        cell_types = adata.obs["cell_ontology_class"].unique().tolist()
        cells_majority_cluster = np.zeros(cells_clusters.shape[0], dtype=int)

        for cell_type in cell_types:
            filtered_cells_clusters = cells_clusters[cells_clusters["cell_ontology_class"] == cell_type]
            cluster_list = filtered_cells_clusters[clustering_alg].to_numpy().astype(int)
            cluster_cand, cluster_votes = np.unique(cluster_list, return_counts=True)
            majority_cand = np.argmax(cluster_votes)
            filtered_cell_indices = np.argwhere(cell_indices.isin(filtered_cells_clusters.index)).squeeze().astype(int)

            cells_majority_cluster[filtered_cell_indices] = cluster_cand[majority_cand]

        cells_majority_cluster = cells_majority_cluster.astype(int).reshape(-1, 1)
        cells_individual_cluster = cells_clusters[clustering_alg].to_numpy().astype(int).reshape(-1, 1)

        cell_type_clusters = np.concatenate(
            (cell_type_clusters, cells_majority_cluster, cells_individual_cluster),
            axis=-1
        )

    # Recovered cell type through clustering
    cell_type_recovered_GE = (cell_type_clusters[:, 1] == cell_type_clusters[:, 2]).reshape(-1, 1)
    cell_type_recovered_TU = (cell_type_clusters[:, 3] == cell_type_clusters[:, 4]).reshape(-1, 1)
    cell_type_recovered_GETU = (cell_type_clusters[:, 5] == cell_type_clusters[:, 6]).reshape(-1, 1)

    cell_type_recovered_boolean = np.concatenate(
        (cell_type_recovered_GE, cell_type_recovered_TU, cell_type_recovered_GETU),
        axis=-1
    )

    recovered_none = np.sum(
        ~cell_type_recovered_boolean[:, 0] & ~cell_type_recovered_boolean[:, 1] & ~cell_type_recovered_boolean[:, 2]
    )
    recovered_GETU_only = np.sum(
        ~cell_type_recovered_boolean[:, 0] & ~cell_type_recovered_boolean[:, 1] & cell_type_recovered_boolean[:, 2]
    )
    recovered_TU_only = np.sum(
        ~cell_type_recovered_boolean[:, 0] & cell_type_recovered_boolean[:, 1] & ~cell_type_recovered_boolean[:, 2]
    )
    recovered_TU_GETU = np.sum(
        ~cell_type_recovered_boolean[:, 0] & cell_type_recovered_boolean[:, 1] & cell_type_recovered_boolean[:, 2]
    )
    recovered_GE_only = np.sum(
        cell_type_recovered_boolean[:, 0] & ~cell_type_recovered_boolean[:, 1] & ~cell_type_recovered_boolean[:, 2]
    )
    recovered_GE_GETU = np.sum(
        cell_type_recovered_boolean[:, 0] & ~cell_type_recovered_boolean[:, 1] & cell_type_recovered_boolean[:, 2]
    )
    recovered_GE_TU = np.sum(
        cell_type_recovered_boolean[:, 0] & cell_type_recovered_boolean[:, 1] & ~cell_type_recovered_boolean[:, 2]
    )
    recovered_GE_TU_GETU = np.sum(
        cell_type_recovered_boolean[:, 0] & cell_type_recovered_boolean[:, 1] & cell_type_recovered_boolean[:, 2]
    )

    cell_type_recovered_array = np.array([
        recovered_none, recovered_GETU_only, recovered_TU_only, recovered_TU_GETU,
        recovered_GE_only, recovered_GE_GETU, recovered_GE_TU, recovered_GE_TU_GETU
    ])

    index_arrays = [
        np.array([False, False, False, False, True, True, True, True]),
        np.array([False, False, True, True, False, False, True, True]),
        np.array([False, True, False, True, False, True, False, True]),
    ]

    modality_name = ["GE", "TU", "Shared"]

    multi_index_array = pd.MultiIndex.from_arrays(index_arrays, names=modality_name)

    cell_clusters_upset = pd.Series(cell_type_recovered_array, index=multi_index_array)

    # Plotting of UMAPs
    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(num_cols * figsize + figsize * wspace * (num_cols - 1), num_rows * figsize),
        dpi=300
    )

    plt.subplots_adjust(wspace=wspace)

    sc.pl.umap(adata1, color='cell_ontology_class', title="GE MMVAE+ Embedding", legend_loc=None, ax=axs[0, 0],
               show=False, frameon=False)
    sc.pl.umap(adata2, color='cell_ontology_class', title="TU MMVAE+ Embedding", legend_loc=None, ax=axs[0, 1],
               show=False, frameon=False)
    sc.pl.umap(adata3, color='cell_ontology_class', title="Shared MMVAE+ Embedding", ax=axs[0, 2], show=False, frameon=False)
    sc.pl.umap(adata1, color=[clustering_alg], title="NMI: " + str(nmi_score1), legend_loc=None, ax=axs[1, 0],
               show=False, frameon=False)
    sc.pl.umap(adata2, color=[clustering_alg], title="NMI: " + str(nmi_score2), legend_loc=None, ax=axs[1, 1],
               show=False, frameon=False)
    sc.pl.umap(adata3, color=[clustering_alg], title="NMI: " + str(nmi_score3), legend_loc=None, ax=axs[1, 2],
               show=False, frameon=False)

    # Customize UMAPs with coordinate system
    x_bottom = 0.05
    y_bottom = 0.05
    length = 0.1
    skip_axs = None
    plot_customized_UMAP_coordinates(axes=axs, skip_axes=skip_axs, x_bottom=x_bottom, y_bottom=y_bottom, length=length)

    fig.suptitle(f"UMAP of Private and Shared Embeddings")

    if show_plot:
        plt.show()

    if save_fig:
        fig.savefig("./figures/tabulaMuris/umap_clustering_" + model_name + "_" + tissue + ".png", dpi=300,
                    bbox_inches='tight')
    if save_tikz:
        fig.savefig("./figures/tabulaMuris/umap_clustering_" + model_name + "_" + tissue + ".pgf")

    # Plotting of Upset Plots
    upset_fig = plt.figure(figsize=(figsize, figsize))

    # Generate the upsetplot
    plot(
        cell_clusters_upset,
        show_counts=True,  # Show counts on top of bars
    )

    # Add title to the upset plot
    plt.suptitle("Cell Type Recovery by Modality", fontsize=10)

    if save_fig:
        plt.savefig("./figures/tabulaMuris/upsetplot_" + model_name + "_" + tissue + ".png", dpi=300,
                    bbox_inches='tight')

    if save_tikz:
        plt.savefig("./figures/tabulaMuris/upsetplot_" + model_name + "_" + tissue + ".pgf")


def plot_gene_expression_transcript_usage(
        adata: Tuple[AnnData, AnnData],
        tissue: str,
        genes: None | List[str],
        cell_types: None | List[str],
        transcript: int = 0,
        save_fig: bool=True
) -> None:
    r"""
    Given two AnnData objects where the first one contains the gene expression data and the second one the transcript
    usage, the gene expression levels and transcript usage levels of all genes present or the specified genes and
    transcripts are compared first through the UMAPs of the gene expression, transcript usage, shared latent embeddings,
    and second through two heatmaps. The left hand heatmap shows on the horizontal axis the PSI score for the individual
    introns of all intron groups present in the data and on the vertical axis the cells present filtered to the
    specified tissue and optionally even the cell type. Similarly, the right hand heatmap shows the gene expression
    levels for the same cells and genes. The genes are plotted on the horizontal axis and the cells on the vertical
    axis.

    :param adata: Tuple of AnnData objects, the first one containing the gene expression data and the second one the transcript usage data
    :param tissue: str, the name of the tissue
    :param genes: List of str, the genes to be filtered for
    :param transcript: int, the index of the transcript to be filtered for
    :param cell_types: List of str, the cell types to be filtered for
    :return: None

    """

    # Check if the input is a tuple of two AnnData objects
    if not isinstance(adata, tuple) or len(adata) != 2:
        raise ValueError("Input must be a tuple of two AnnData objects")
    if not isinstance(adata[0], AnnData) or not isinstance(adata[1], AnnData):
        raise ValueError("Both elements of the tuple must be AnnData objects")

    adata_1, adata_2 = adata

    # Check if specified tissue is present in the data
    assert tissue in np.unique(adata_1.obs["tissue"].to_numpy()), "Specified tissue not found in gene expression data"
    assert tissue in np.unique(adata_2.obs["tissue"].to_numpy()), "Specified tissue not found in transcript usage data"

    # Filter the data for the specified tissue
    adata_1 = adata_1[adata_1.obs["tissue"] == tissue]
    adata_2 = adata_2[adata_2.obs["tissue"] == tissue]

    # Check if genes are specified
    if genes is not None:
        if not isinstance(genes, list):
            raise ValueError("genes must be a list of strings")
        # Check if the specified genes are present in the data
        if not all(gene in np.unique(adata_1.var_names.to_numpy()) for gene in genes):
            raise ValueError("Specified genes not found in gene expression data")
        if not all(gene in np.unique(adata_2.var["gene_name"].to_numpy()) for gene in genes):
            raise ValueError("Specified genes not found in transcript usage data")

    # Check if cell types are specified
    if cell_types is not None:
        if not isinstance(cell_types, list):
            raise ValueError("cell_types must be a list of strings")
        # Check if the specified cell types are present in the data
        if not all(cell_type in np.unique(adata_1.obs["cell_ontology_class"].to_numpy()) for cell_type in cell_types):
            raise ValueError("Specified cell types not found in gene expression data")
        if not all(cell_type in np.unique(adata_2.obs["cell_ontology_class"].to_numpy()) for cell_type in cell_types):
            raise ValueError("Specified cell types not found in transcript usage data")

    # Check if anndata objects contain the latent embeddings
    if "MMVAE_latent" not in adata_1.obsm.keys():
        raise ValueError("MMVAE_latent not found in gene expression data. Infer the latent embeddings first.")
    if "MMVAE_latent" not in adata_2.obsm.keys():
        raise ValueError("MMVAE_latent not found in transcript usage data. Infer the latent embeddings first.")
    if "MMVAE_shared_latent" not in adata_1.obsm.keys():
        raise ValueError("MMVAE_shared_latent not found in gene expression data. Infer the shared latent embeddings first.")

    # Anndata objects of specified tissue for UMAP projections
    adata_1_tissue = adata_1.copy()
    adata_2_tissue = adata_2.copy()
    adata_shared_tissue = adata_1.copy()

    # Compute neighbourhood graph for different latent embeddings
    adata_1_tissue.obsm["X_umap"] = UMAP(n_components=2).fit_transform(adata_1_tissue.obsm["MMVAE_latent"])
    adata_2_tissue.obsm["X_umap"] = UMAP(n_components=2).fit_transform(adata_2_tissue.obsm["MMVAE_latent"])
    adata_shared_tissue.obsm["X_umap"] = UMAP(n_components=2).fit_transform(adata_shared_tissue.obsm["MMVAE_shared_latent"])

    # Filter the data for the specified cell types if provided
    if cell_types is not None:
        adata_1 = adata_1[adata_1.obs["cell_ontology_class"].isin(cell_types)]
        adata_2 = adata_2[adata_2.obs["cell_ontology_class"].isin(cell_types)]

    # Filter out intron groups with too few cells
    adata_2 = filter_min_cells_per_feature(adata_2, 100)
    adata_2 = filter_min_cells_per_intron_group(adata_2, 100)

    # Check if cells are the same in both datasets
    assert np.all(adata_1.obs_names == adata_2.obs_names), "Cells are not the same in both datasets"

    # Filter out genes not corresponding to intron groups present in the transcript usage data
    intron_group_gene_ids = np.unique(adata_2.var["gene_id"].to_numpy())
    intron_group_genes = adata_1.var["gene_id"].isin(intron_group_gene_ids).to_numpy()
    adata_1 = adata_1[:, intron_group_genes]

    # If genes are provided, filter the gene expression data for the specified genes
    if genes is not None:
        gene_indices = np.array([gene in genes for gene in adata_1.var_names.to_numpy()])
        intron_group_specified_genes = np.array([gene in genes for gene in adata_2.var["gene_name"].to_numpy()])
        adata_1 = adata_1[:, gene_indices]
        adata_2 = adata_2[:, intron_group_specified_genes]

    # Extract gene expression levels and psi scores
    gene_expression_matrix = adata_1.X.toarray()
    psi_matrix = adata_2.layers["psi"]

    # Extract gene names and intron group names
    gene_names = adata_1.var_names.to_numpy()
    intron_group_names = adata_2.var_names.to_numpy()

    # Extract cell names
    if cell_types is None:
        cell_names = adata_1.obs["cell_ontology_class"].to_numpy()
    else:
        cell_names = adata_1.obs_names

    # Sort the matrices and cell_names by cell_ontology_class
    if cell_types is None:
        gene_expression_matrix = gene_expression_matrix[np.argsort(cell_names)]
        psi_matrix = psi_matrix[np.argsort(cell_names), :]
        cell_names = cell_names[np.argsort(cell_names)]

    # Sort the matrices by gene names and intron group names
    gene_expression_matrix = gene_expression_matrix[:, np.argsort(gene_names)]
    psi_matrix = psi_matrix[:, np.argsort(intron_group_names)]
    gene_names = gene_names[np.argsort(gene_names)]
    intron_group_names = intron_group_names[np.argsort(intron_group_names)]

    # Filter for specified transcript if provided
    if transcript < 0 or transcript >= len(intron_group_names):
         raise ValueError(f"Invalid transcript index. The acceptable range is 0 to {len(intron_group_names) - 1}.")

    transcript_idx = np.argwhere(adata_2_tissue.var_names.to_numpy() == intron_group_names[transcript]).item()
    psi_values = adata_2_tissue.layers["psi"][:, transcript_idx]
    adata_2_tissue.obs[intron_group_names[transcript] + '_psi'] = psi_values

    # Create a figure with two x three subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=300)
    """
    # TO DO: Check if sup title is necessary for paper figure
    if genes is None:
        if cell_types is None:
            fig.suptitle(f"Gene Expression and Transcript Usage in {tissue}")
        else:
            fig.suptitle(f"Gene Expression and Transcript Usage in {tissue} - Cell Types: {', '.join(cell_types)}")
    else:
        if cell_types is None:
            fig.suptitle(f"Gene Expression and Transcript Usage in {tissue} - Genes: {', '.join(genes)}")
        else:
            fig.suptitle(f"Gene Expression and Transcript Usage in {tissue} - Genes: {', '.join(genes)} - Cell Types: {', '.join(cell_types)}")
    """

    if len(genes) == 1:
        sc.pl.umap(adata_1_tissue, color=genes[0], title=genes[0], legend_loc=None, ax=axes[0, 0],
                   show=False, frameon=False)
        sc.pl.umap(adata_2_tissue, color=intron_group_names[transcript] + '_psi', title=intron_group_names[transcript],
                   legend_loc=None, ax=axes[0, 1], show=False, frameon=False)
        sc.pl.umap(adata_shared_tissue, color='cell_ontology_class', title=" ", ax=axes[0, 2], show=False, frameon=False)
    else:
        sc.pl.umap(adata_1_tissue, color='cell_ontology_class', title="Gene Expression Latent", legend_loc=None,
                   ax=axes[0, 0], show=False, frameon=False)
        sc.pl.umap(adata_2_tissue, color='cell_ontology_class', title="Transcript Usage Latent", legend_loc=None,
                   ax=axes[0, 1], show=False, frameon=False)
        sc.pl.umap(adata_shared_tissue, color='cell_ontology_class', title="Shared Latent", ax=axes[0, 2],
                   show=False, frameon=False)

    # Customize UMAPs with coordinate system
    x_bottom = 0.0
    y_bottom = 0.0
    length = 0.1
    skip_axes = [3,4,5]
    plot_customized_UMAP_coordinates(axes=axes, skip_axes=skip_axes, x_bottom=x_bottom, y_bottom=y_bottom,
                                         length=length)

    # Plot the gene expression heatmap
    im1 = axes[1, 0].imshow(gene_expression_matrix, aspect='auto')
    axes[1, 0].set_title("Gene Expression")
    axes[1, 0].set_xlabel("Genes")
    axes[1, 0].set_ylabel("Cells")
    axes[1, 0].set_xticks(np.arange(len(gene_names)))
    axes[1, 0].set_xticklabels(gene_names, rotation=90)
    axes[1, 0].set_yticks([])
    fig.colorbar(im1, ax=axes[1, 0], orientation='vertical')

    # Plot the transcript usage heatmap
    im2 = axes[1, 1].imshow(psi_matrix, aspect='auto')
    axes[1, 1].set_title("Transcript Usage")
    axes[1, 1].set_xlabel("Introns")
    axes[1, 1].set_xticks(np.arange(len(intron_group_names)))
    axes[1, 1].set_xticklabels(intron_group_names, rotation=90)
    axes[1, 1].set_yticks([])
    fig.colorbar(im2, ax=axes[1, 1], orientation='vertical')
    #plt.tight_layout()

    # Hide the last subplot
    axes[1, 2].axis("off")

    plt.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust the top to make room for the suptitle

    # Save the figure
    if save_fig:
        if genes is None:
            if cell_types is None:
                plt.savefig(f"./figures/tabulaMuris/gene_expression_transcript_usage_{tissue}.pdf", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"./figures/tabulaMuris/gene_expression_transcript_usage_{tissue}_{'_'.join(cell_types)}.pdf", dpi=300, bbox_inches='tight')
        else:
            if cell_types is None:
                plt.savefig(f"./figures/tabulaMuris/gene_expression_transcript_usage_{tissue}_{'_'.join(genes)}.pdf", dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f"./figures/tabulaMuris/gene_expression_transcript_usage_{tissue}_{'_'.join(genes)}_{'_'.join(cell_types)}.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot_data_imputation(
        adata: Tuple[AnnData, AnnData],
        tissue: str,
        cell_types: None | List[str],
        save_fig: bool=True
) -> None:
    r"""
    Given a tuple of two AnnData objects, the first one contains gene expression data and the second one transcript
    usage data. Both AnnData objects contain the original data but as layers also the imputed data, and the
    cross-modally imputed data. For the gene expression data, the original gene expression levels, the imputed and the
    cross-modally imputed are plotted as three heatmaps. For the transcript usage data, the psi scores of the original
    data, the imputed and the cross-modally imputed are plotted as three heatmaps. The fidelity of the imputation is
    quantitively compared through bar charts of each modality plotting the reconstruction error of the imputed and
    cross-modally imputed data.

    :param adata: Tuple of AnnData objects, the first one containing the gene expression data and the second one the transcript usage data
    :param tissue: str, the name of the tissue
    :param cell_types: List of str, the cell types to be filtered for
    :param save_fig: bool, whether to save the figure or not
    :return: None
    """

    # Check if the input is a tuple of two AnnData objects
    if not isinstance(adata, tuple) or len(adata) != 2:
        raise ValueError("Input must be a tuple of two AnnData objects")
    if not isinstance(adata[0], AnnData) or not isinstance(adata[1], AnnData):
        raise ValueError("Both elements of the tuple must be AnnData objects")
    adata_1, adata_2 = adata

    # Check if specified tissue is present in the data
    assert tissue in np.unique(adata_1.obs["tissue"].to_numpy()), "Specified tissue not found in gene expression data"
    assert tissue in np.unique(adata_2.obs["tissue"].to_numpy()), "Specified tissue not found in transcript usage data"

    # Filter the data for the specified tissue
    adata_1 = adata_1[adata_1.obs["tissue"] == tissue]
    adata_2 = adata_2[adata_2.obs["tissue"] == tissue]

    # Check if cell types are specified
    if cell_types is not None:
        if not isinstance(cell_types, list):
            raise ValueError("cell_types must be a list of strings")
        # Check if the specified cell types are present in the data
        if not all(cell_type in np.unique(adata_1.obs["cell_ontology_class"].to_numpy()) for cell_type in cell_types):
            raise ValueError("Specified cell types not found in gene expression data")
        if not all(cell_type in np.unique(adata_2.obs["cell_ontology_class"].to_numpy()) for cell_type in cell_types):
            raise ValueError("Specified cell types not found in transcript usage data")

    # Check if AnnData objects contain imputed and cross-modally imputed data and for transcript usage also psi
    if "imputed" not in adata_1.layers.keys():
        raise ValueError("imputed not found in gene expression data. Infer the imputed data first.")
    if "imputed" not in adata_2.layers.keys():
        raise ValueError("imputed not found in transcript usage data. Infer the imputed data first.")
    if "psi" not in adata_2.layers.keys():
        raise ValueError("psi not found in transcript usage data. Infer the psi scores first.")
    if "imputed_cross_modal" not in adata_1.layers.keys():
        raise ValueError("cross_modality_imputed not found in gene expression data. Infer the cross-modally imputed data first.")
    if "imputed_cross_modal" not in adata_2.layers.keys():
        raise ValueError("cross_modality_imputed not found in transcript usage data. Infer the cross-modally imputed data first.")

    # Check if cells are the same in both datasets
    assert np.all(adata_1.obs_names == adata_2.obs_names), "Cells are not the same in both datasets"

    # Filter the data for the specified cell types if provided
    if cell_types is not None:
        # TO DO LATER
        pass

    # Extract gene expression levels and psi scores
    gene_expression_matrix = adata_1.X.toarray()
    imputed_gene_expression_matrix = adata_1.layers["imputed"].toarray()
    cross_modality_imputed_gene_expression_matrix = adata_1.layers["imputed_cross_modal"].toarray()
    psi_matrix = adata_2.layers["psi"]
    imputed_psi_matrix = adata_2.layers["imputed"]
    cross_modality_imputed_psi_matrix = adata_2.layers["imputed_cross_modal"]

    # Extract gene names and intron group names
    gene_names = adata_1.var_names.to_numpy()
    intron_group_names = adata_2.var_names.to_numpy()

    # Extract cell names
    if cell_types is None:
        cell_names = adata_1.obs["cell_ontology_class"].to_numpy()
    else:
        cell_names = adata_1.obs_names

    # Sort the matrices by gene names and intron group names
    gene_expression_matrix = gene_expression_matrix[:, np.argsort(gene_names)]
    psi_matrix = psi_matrix[:, np.argsort(intron_group_names)]
    imputed_gene_expression_matrix = imputed_gene_expression_matrix[:, np.argsort(gene_names)]
    imputed_psi_matrix = imputed_psi_matrix[:, np.argsort(intron_group_names)]
    gene_names = gene_names[np.argsort(gene_names)]
    intron_group_names = intron_group_names[np.argsort(intron_group_names)]

    # Create a figure with two x three subplots where each subplot is a heatmap with a colorbar
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=300)

    # TO DO: Extend subplots to 2x4 and add bar plots for reconstruction error

    # Plot the gene expression heatmap
    im1 = axes[0, 0].imshow(gene_expression_matrix, aspect='auto')
    axes[0, 0].set_title("Gene Expression")
    axes[0, 0].set_xlabel("Genes")
    axes[0, 0].set_ylabel("Cells")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')

    # Plot the imputed gene expression heatmap
    im2 = axes[0, 1].imshow(imputed_gene_expression_matrix, aspect='auto')
    axes[0, 1].set_title("Imputed")
    axes[0, 1].set_xlabel("Genes")
    axes[0, 1].set_ylabel("Cells")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')

    # Plot the cross-modally imputed gene expression heatmap
    im3 = axes[0, 2].imshow(cross_modality_imputed_gene_expression_matrix, aspect='auto')
    axes[0, 2].set_title("Cross-Modally Imputed")
    axes[0, 2].set_xlabel("Genes")
    axes[0, 2].set_ylabel("Cells")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    fig.colorbar(im3, ax=axes[0, 2], orientation='vertical')

    # Plot the transcript usage heatmap
    im4 = axes[1, 0].imshow(psi_matrix, aspect='auto')
    axes[1, 0].set_title("Transcript Usage")
    axes[1, 0].set_xlabel("Introns")
    axes[1, 0].set_ylabel("Cells")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    fig.colorbar(im4, ax=axes[1, 0], orientation='vertical')

    # Plot the imputed transcript usage heatmap
    im5 = axes[1, 1].imshow(imputed_psi_matrix, aspect='auto')
    axes[1, 1].set_title("Imputed")
    axes[1, 1].set_xlabel("Introns")
    axes[1, 1].set_ylabel("Cells")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    fig.colorbar(im5, ax=axes[1, 1], orientation='vertical')

    # Plot the cross-modally imputed transcript usage heatmap
    im6 = axes[1, 2].imshow(cross_modality_imputed_psi_matrix, aspect='auto')
    axes[1, 2].set_title("Cross-Modally Imputed")
    axes[1, 2].set_xlabel("Introns")
    axes[1, 2].set_ylabel("Cells")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    fig.colorbar(im6, ax=axes[1, 2], orientation='vertical')

    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    if save_fig:
        if cell_types is None:
            plt.savefig(f"./figures/tabulaMuris/data_imputation_{tissue}.pdf", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"./figures/tabulaMuris/data_imputation_{tissue}_{'_'.join(cell_types)}.pdf", dpi=300, bbox_inches='tight')

    plt.show()

def create_histogram_bins(
        array: np.ndarray,
        bin_size: int | float = 250,
        count_data_modality: bool = True,
        proportions: bool = True,
        zero_inflation_included: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Create histogram bins for a given array. The function first determines the minimum and maximum values of the array,
    then creates bins of size `bin_size` between these two values. If `zero_inflation_included` is set to True, the
    function also includes a bin for zero values. The function returns the bin labels and the bin proportions or counts
    in each bin.

    :param array: Input array for which to create histogram bins
    :param bin_size: Size of each bin
    :param count_data_modality: If True, the bin labels will be formatted as integers, otherwise as floats
    :param proportions: If True, return proportions of counts in each bin, otherwise return counts
    :param zero_inflation_included: If True, include a bin for zero values
    :return: Tuple of bin labels and bin proportions or counts
    :rtype: Tuple[np.ndarray, np.ndarray]

    Example:
    >>> array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> bin_labels, bin_proportions = create_histogram_bins(array, bin_size=2, proportions=True, zero_inflation_included=True)
    >>> print(bin_labels)
    >>> # Output: [0, 2, 4, 6, 8, 10]
    >>> print(bin_proportions)
    >>> # Output: [0.1, 0.1, 0.1, 0.1, 0.1]
    """

    # Create list of counts for bins and labels
    counts_in_bins = []
    bin_labels = []

    # Determine the minimum and maximum values of the array (min_value is always zero for gene expression data)
    min_value = np.min(array)
    max_value = np.max(array)

    # If zero inflation is included, add a bin for zero values
    if zero_inflation_included:

        if count_data_modality:
            bin_labels.append(str(int(min_value)) + "-" + str(int(min_value)))
        else:
            bin_labels.append(str(min_value) + " - " + str(min_value))

        counts_in_bins.append(np.sum((array >= 0) & (array < 0.5)))
        min_value = 0.5  # Set the minimum value to the second smallest value

    else:
        # Zero inflation is not included, so the zero values are not included
        if count_data_modality:
            min_value = 0.5
        else:
            min_value = np.log(1 + 0.5)

    # Create bins of size `bin_size` between the minimum and maximum values

    while min_value <= max_value:

        # Calculate the upper bound of the current bin
        current_upper_bound = min_value + bin_size

        # Adjust the last bin to ensure it captures all remaining values up to max_val
        if current_upper_bound >= max_value:
            current_upper_bound = max_value
            # If min_value is already past max_val (and it's not the first bin, i.e., not 1),
            # then we've covered all values, so break to prevent adding an empty bin.
            if min_value > max_value and min_value != 1:
                break

        # Count elements whose values fall within the current range

        cells_in_range = np.sum((array >= min_value) & (array <= current_upper_bound))
        counts_in_bins.append(cells_in_range)

        # Create the label for the bin
        if count_data_modality:
            bin_labels.append(f"{int(min_value)}-{int(current_upper_bound)}")
        else:
            bin_lower_bound = str(np.round(min_value, 2))
            bin_upper_bound = str(np.round(current_upper_bound, 2))
            bin_label = bin_lower_bound + " - " + bin_upper_bound
            bin_labels.append(bin_label)

        min_value += bin_size  # Move to the start of the next bin

    # Convert counts to proportions if required
    if proportions:
        total_counts = np.sum(counts_in_bins)
        if total_counts > 0:
            counts_in_bins = np.array(counts_in_bins) / total_counts
        else:
            counts_in_bins = np.zeros_like(counts_in_bins)

    # Convert bin labels to a numpy array
    bin_labels = np.array(bin_labels)

    return bin_labels, np.array(counts_in_bins, dtype=float)

def distance_matrix(cell_embeddings: np.ndarray, cell_types: np.ndarray, metric: str="euclidean") -> Tuple[np.ndarray,np.ndarray]:
    unique_cell_types = np.sort(np.unique(cell_types))
    num_cell_types = len(unique_cell_types)

    ordered_cell_embeddings = []
    ordered_cell_types = []

    for cell_type in unique_cell_types:
        # Get the indices where the current cell_type matches
        indices = np.where(cell_types == cell_type)[0]
        # Append the embeddings for this cell type
        ordered_cell_embeddings.append(cell_embeddings[indices])
        # Append the corresponding cell type labels
        ordered_cell_types.extend([cell_type] * len(indices))

    # Concatenate the list of arrays into a single NumPy array
    ordered_cell_embeddings = np.vstack(ordered_cell_embeddings)
    ordered_cell_types = np.array(ordered_cell_types)

    condensed_distance_matrix = pdist(ordered_cell_embeddings, metric='euclidean')
    distance_matrix = squareform(condensed_distance_matrix)

    return distance_matrix, ordered_cell_types

def _validate_square_matrix(matrix, name, expected_size=None):
    r"""
    Return *matrix* as a float array after validating its dimensions.
    """
    array = np.asarray(matrix, dtype=float)

    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(
            f"{name} must be a square two-dimensional matrix; "
            f"received shape {array.shape}."
        )

    if expected_size is not None and array.shape != (expected_size, expected_size):
        raise ValueError(
            f"{name} must have shape ({expected_size}, {expected_size}); "
            f"received {array.shape}."
        )

    return array


def _make_lineage_color_dict(
    lineage_mapping,
    cluster_annotations,
    palette=None,
):
    r"""
    Create one lineage palette shared by the heatmaps and network.
    """
    missing_annotations = [
        annotation
        for annotation in cluster_annotations
        if annotation not in lineage_mapping
    ]

    if missing_annotations:
        raise KeyError(
            "lineage_mapping does not contain the following cell types: "
            f"{missing_annotations}."
        )

    lineage_order = list(
        dict.fromkeys(
            lineage_mapping[annotation]
            for annotation in cluster_annotations
        )
    )

    if isinstance(palette, Mapping):
        missing_lineages = [
            lineage for lineage in lineage_order if lineage not in palette
        ]

        if missing_lineages:
            raise KeyError(
                "The supplied lineage palette does not contain colours for: "
                f"{missing_lineages}."
            )

        color_dict = {
            lineage: to_rgba(palette[lineage])
            for lineage in lineage_order
        }
    else:
        if palette is None:
            palette = "tab10" if len(lineage_order) <= 10 else "tab20"

        colors = sns.color_palette(
            palette,
            n_colors=len(lineage_order),
        )
        color_dict = {
            lineage: to_rgba(color)
            for lineage, color in zip(lineage_order, colors)
        }

    return lineage_order, color_dict


def _create_matrix_panel_axes(
    fig,
    subplot_spec,
    panel_label,
    title,
    panel_label_fontsize,
    title_fontsize,
):
    r"""
    Create equally sized matrix, strip and colorbar axes in one grid cell.
    """
    panel_grid = subplot_spec.subgridspec(
        nrows=3,
        ncols=3,
        height_ratios=[0.14, 0.045, 0.815],
        width_ratios=[0.045, 0.875, 0.080],
        hspace=0.06,
        wspace=0.10,
    )

    title_ax = fig.add_subplot(panel_grid[0, :])
    top_strip_ax = fig.add_subplot(panel_grid[1, 1])
    left_strip_ax = fig.add_subplot(panel_grid[2, 0])
    matrix_ax = fig.add_subplot(panel_grid[2, 1])
    colorbar_ax = fig.add_subplot(panel_grid[2, 2])

    title_ax.set_axis_off()
    title_ax.text(
        0.0,
        0.5,
        panel_label,
        transform=title_ax.transAxes,
        ha="left",
        va="center",
        fontsize=panel_label_fontsize,
        fontweight="bold",
    )
    title_ax.text(
        0.5,
        0.5,
        title,
        transform=title_ax.transAxes,
        ha="center",
        va="center",
        fontsize=title_fontsize,
    )

    return {
        "title": title_ax,
        "top_strip": top_strip_ax,
        "left_strip": left_strip_ax,
        "matrix": matrix_ax,
        "colorbar": colorbar_ax,
    }


def _format_colorbar(
    heatmap_ax,
    label,
    label_fontsize,
    tick_fontsize,
):
    r"""
    Apply the common typography to a seaborn heatmap colorbar.
    """
    if not heatmap_ax.collections:
        return None

    colorbar = heatmap_ax.collections[0].colorbar

    if colorbar is None:
        return None

    colorbar.set_label(
        label,
        fontsize=label_fontsize,
        labelpad=2,
    )
    colorbar.ax.tick_params(
        labelsize=tick_fontsize,
        length=2,
        pad=1,
    )

    return colorbar


def _align_matrix_annotation_axes(panel_axes, strip_gap_mm=0.8):
    r"""
    Align lineage strips and colorbar to the rendered heatmap rectangle.

    ``square=True`` can shrink and centre the matrix axis inside its GridSpec
    slot. The auxiliary axes do not undergo the same aspect correction. This
    helper therefore reads the final matrix position and explicitly matches the
    strip width/height and colorbar height to it.
    """
    matrix_ax = panel_axes["matrix"]
    left_strip_ax = panel_axes["left_strip"]
    top_strip_ax = panel_axes["top_strip"]
    colorbar_ax = panel_axes["colorbar"]
    fig = matrix_ax.figure

    fig.canvas.draw()

    matrix_position = matrix_ax.get_position()
    left_position = left_strip_ax.get_position()
    top_position = top_strip_ax.get_position()
    colorbar_position = colorbar_ax.get_position()

    figure_width_mm = fig.get_figwidth() * 25.4
    figure_height_mm = fig.get_figheight() * 25.4
    horizontal_gap = strip_gap_mm / figure_width_mm
    vertical_gap = strip_gap_mm / figure_height_mm

    left_strip_ax.set_position([
        matrix_position.x0 - horizontal_gap - left_position.width,
        matrix_position.y0,
        left_position.width,
        matrix_position.height,
    ])
    top_strip_ax.set_position([
        matrix_position.x0,
        matrix_position.y1 + vertical_gap,
        matrix_position.width,
        top_position.height,
    ])
    colorbar_ax.set_position([
        colorbar_position.x0,
        matrix_position.y0,
        colorbar_position.width,
        matrix_position.height,
    ])


def _adjust_network_label_positions(
    ax,
    label_artists,
    node_positions,
    *,
    iterations=250,
    padding_points=1.2,
    max_displacement_points=20.0,
    pull_strength=0.012,
    move_fraction=0.70,
    draw_connectors=True,
    connector_min_distance_points=4.0,
    connector_color="#707070",
    connector_alpha=0.55,
    connector_linewidth=0.35,
):
    r"""
    Repel overlapping node labels in display coordinates.

    Working in display coordinates accounts for the true rendered width of
    labels such as ``EC (coronary)``. The optimisation is deterministic and
    dependency-free, and optional connector lines retain the association with
    nodes whose labels have moved appreciably.
    """
    if not label_artists:
        return {}, []

    if iterations < 0:
        raise ValueError("node_label_adjust_iterations must be non-negative.")
    if padding_points < 0 or max_displacement_points < 0:
        raise ValueError(
            "Label padding and maximum displacement must be non-negative."
        )
    if not 0 < move_fraction <= 1:
        raise ValueError("node_label_move_fraction must lie in (0, 1].")
    if not 0 <= pull_strength < 1:
        raise ValueError("node_label_pull_strength must lie in [0, 1).")

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points_to_pixels = fig.dpi / 72.0
    padding_pixels = padding_points * points_to_pixels
    maximum_displacement_pixels = max_displacement_points * points_to_pixels
    connector_distance_pixels = (
        connector_min_distance_points * points_to_pixels
    )

    nodes = list(label_artists)
    texts = [label_artists[node] for node in nodes]
    bounding_boxes = [text.get_window_extent(renderer) for text in texts]
    label_widths = np.asarray([box.width for box in bounding_boxes])
    label_heights = np.asarray([box.height for box in bounding_boxes])

    initial_label_display_positions = np.asarray([
        ax.transData.transform(text.get_position())
        for text in texts
    ])
    current_positions = initial_label_display_positions.copy()
    axes_box = ax.get_window_extent(renderer)

    for iteration_index in range(int(iterations)):
        number_of_overlaps = 0

        for first_index in range(len(nodes) - 1):
            for second_index in range(first_index + 1, len(nodes)):
                delta = current_positions[first_index] - current_positions[second_index]
                required_x_distance = (
                    0.5
                    * (
                        label_widths[first_index]
                        + label_widths[second_index]
                    )
                    + padding_pixels
                )
                required_y_distance = (
                    0.5
                    * (
                        label_heights[first_index]
                        + label_heights[second_index]
                    )
                    + padding_pixels
                )
                overlap_x = required_x_distance - abs(delta[0])
                overlap_y = required_y_distance - abs(delta[1])

                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                number_of_overlaps += 1

                # Resolve the overlap along the axis requiring less movement.
                if overlap_y <= overlap_x:
                    if delta[1] == 0:
                        direction = 1.0 if (first_index + iteration_index) % 2 else -1.0
                    else:
                        direction = np.sign(delta[1])
                    shift = 0.5 * (overlap_y + padding_pixels)
                    current_positions[first_index, 1] += (
                        move_fraction * direction * shift
                    )
                    current_positions[second_index, 1] -= (
                        move_fraction * direction * shift
                    )
                else:
                    if delta[0] == 0:
                        direction = 1.0 if (first_index + iteration_index) % 2 else -1.0
                    else:
                        direction = np.sign(delta[0])
                    shift = 0.5 * (overlap_x + padding_pixels)
                    current_positions[first_index, 0] += (
                        move_fraction * direction * shift
                    )
                    current_positions[second_index, 0] -= (
                        move_fraction * direction * shift
                    )

        current_positions += pull_strength * (
            initial_label_display_positions - current_positions
        )

        # Keep labels close enough to their nodes that the graph remains easy
        # to read, and ensure that no label is pushed outside the graph axis.
        from_initial = current_positions - initial_label_display_positions
        distances = np.linalg.norm(from_initial, axis=1)
        outside_radius = distances > maximum_displacement_pixels

        if np.any(outside_radius) and maximum_displacement_pixels > 0:
            from_initial[outside_radius] *= (
                maximum_displacement_pixels
                / distances[outside_radius, None]
            )
            current_positions[outside_radius] = (
                initial_label_display_positions[outside_radius]
                + from_initial[outside_radius]
            )

        current_positions[:, 0] = np.clip(
            current_positions[:, 0],
            axes_box.x0 + 0.5 * label_widths + padding_pixels,
            axes_box.x1 - 0.5 * label_widths - padding_pixels,
        )
        current_positions[:, 1] = np.clip(
            current_positions[:, 1],
            axes_box.y0 + 0.5 * label_heights + padding_pixels,
            axes_box.y1 - 0.5 * label_heights - padding_pixels,
        )

        if number_of_overlaps == 0:
            break

    inverse_transform = ax.transData.inverted()
    final_label_positions = {}
    connector_artists = []

    for node, text, display_position in zip(
        nodes,
        texts,
        current_positions,
    ):
        final_position = inverse_transform.transform(display_position)
        text.set_position(final_position)
        text.set_horizontalalignment("center")
        text.set_verticalalignment("center")
        text.set_clip_on(True)
        final_label_positions[node] = tuple(final_position)

        node_display_position = ax.transData.transform(node_positions[node])
        distance_from_node = np.linalg.norm(
            display_position - node_display_position
        )

        if draw_connectors and distance_from_node >= connector_distance_pixels:
            connector = Line2D(
                [node_positions[node][0], final_position[0]],
                [node_positions[node][1], final_position[1]],
                color=connector_color,
                alpha=connector_alpha,
                linewidth=connector_linewidth,
                solid_capstyle="round",
                zorder=1.5,
                clip_on=True,
            )
            ax.add_line(connector)
            connector_artists.append(connector)

    return final_label_positions, connector_artists


def plot_seed_similarity_matrix(
    seed_similarity_matrix: np.ndarray,
    median_pairwise_seed_similarity: float=None,
    lower_pairwise_seed_similarity: float=None,
    upper_pairwise_seed_similarity: float=None,
    *,
    ax,
    cbar_ax,
    seeds=None,
    cmap="viridis",
    vmin=-1.0,
    vmax=1.0,
    axis_fontsize=7,
    tick_fontsize=6,
    colorbar_label_fontsize=7,
    colorbar_tick_fontsize=6,
    verbose=True,
) -> Dict[str, float | plt.Axes | mcolorbar.Colorbar]:
    r"""
    Plot the seed-similarity heatmap into supplied axes.

    This axis-aware version deliberately does not create, lay out or show a
    figure. Those operations are owned by the composite plotting function.

    :param seed_similarity_matrix: Square matrix of pairwise seed similarities.
    :param median_pairwise_seed_similarity: Optional pre-computed median of the pairwise seed similarities. If not provided, it will be computed from the matrix.
    :param lower_pairwise_seed_similarity: Optional pre-computed minimum of the pairwise seed similarities. If not provided, it will be computed from the matrix.
    :param upper_pairwise_seed_similarity: Optional pre-computed maximum of the pairwise seed similarities. If not provided, it will be computed from the matrix.
    :param ax: Matplotlib axis for the heatmap.
    :param cbar_ax: Matplotlib axis for the colorbar.
    :param seeds: Optional list of seed labels. If not provided, default labels will be generated.
    :param cmap: Colormap for the heatmap.
    :param vmin: Minimum value for the heatmap color scale.
    :param vmax: Maximum value for the heatmap color scale.
    :param axis_fontsize: Font size for the axis labels.
    :param tick_fontsize: Font size for the tick labels.
    :param colorbar_label_fontsize: Font size for the colorbar label.
    :param colorbar_tick_fontsize: Font size for the colorbar tick labels.
    :param verbose: If True, print the median and range of pairwise seed similarities.
    :return: Dictionary containing the heatmap axis, colorbar, and computed statistics.
    """
    matrix = _validate_square_matrix(
        seed_similarity_matrix,
        "seed_similarity_matrix",
    )
    number_of_seeds = matrix.shape[0]

    if seeds is None:
        seed_labels = [str(index) for index in range(number_of_seeds)]
    else:
        if len(seeds) != number_of_seeds:
            raise ValueError(
                "The number of seed labels must equal the dimensions of "
                "seed_similarity_matrix."
            )
        seed_labels = [str(seed) for seed in seeds]

    upper_triangle = matrix[np.triu_indices(number_of_seeds, k=1)]
    finite_pairwise_values = upper_triangle[np.isfinite(upper_triangle)]

    if finite_pairwise_values.size:
        if median_pairwise_seed_similarity is None:
            median_pairwise_seed_similarity = float(
                np.median(finite_pairwise_values)
            )
        if lower_pairwise_seed_similarity is None:
            lower_pairwise_seed_similarity = float(
                np.min(finite_pairwise_values)
            )
        if upper_pairwise_seed_similarity is None:
            upper_pairwise_seed_similarity = float(
                np.max(finite_pairwise_values)
            )

    if verbose and median_pairwise_seed_similarity is not None:
        print(
            "The median pairwise seed correlation is: "
            f"{median_pairwise_seed_similarity:.4f}"
        )

    if (
        verbose
        and lower_pairwise_seed_similarity is not None
        and upper_pairwise_seed_similarity is not None
    ):
        print(
            "The range of the pairwise seed correlation is: "
            f"{lower_pairwise_seed_similarity:.4f} - "
            f"{upper_pairwise_seed_similarity:.4f}"
        )

    sns.heatmap(
        matrix,
        ax=ax,
        cbar_ax=cbar_ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        xticklabels=seed_labels,
        yticklabels=seed_labels,
        linewidths=0,
        rasterized=True,
    )

    ax.set_xlabel("Seed", fontsize=axis_fontsize, labelpad=2)
    ax.set_ylabel("Seed", fontsize=axis_fontsize, labelpad=2)
    ax.tick_params(
        axis="both",
        labelsize=tick_fontsize,
        length=0,
        pad=1,
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0 if number_of_seeds <= 12 else 90,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_box_aspect(1)

    colorbar = _format_colorbar(
        ax,
        "Spearman correlation",
        colorbar_label_fontsize,
        colorbar_tick_fontsize,
    )

    return {
        "heatmap_ax": ax,
        "colorbar": colorbar,
        "median_pairwise_seed_similarity": median_pairwise_seed_similarity,
        "lower_pairwise_seed_similarity": lower_pairwise_seed_similarity,
        "upper_pairwise_seed_similarity": upper_pairwise_seed_similarity,
    }

def plot_neighbour_retention(
    retention_df: pd.DataFrame,
    abbrev_dict: Dict[str, str] | None = None,
    lineage_mapping: Dict[str, str] | None = None,
    k: int = 5,
    figsize: Tuple[float, float] = (14, 5.5),
) -> Tuple[plt.Figure, plt.Axes]:
    r"""
    Plots cell-type specific neighbor retention scores as a dot plot with an expected chance baseline.

    :param retention_df: DataFrame output from compute_neighbor_retention.
    :param abbrev_dict: Dictionary mapping full cell names to abbreviations (e.g., TABULA_MURIS_CELL_TYPE_ABBREVIATION_DICT).
    :param lineage_mapping: Optional dictionary mapping cell types to lineage groups for dot coloring.
    :param k: Neighborhood size integer used in score column lookup.
    :param figsize: Tuple defining figure dimensions.
    :return: Matplotlib figure and axes objects for further customization or saving.
    """
    df = retention_df.copy()

    # 1. Map full names to abbreviations
    if abbrev_dict is not None:
        df["display_name"] = df["cell_type"].map(
            lambda x: abbrev_dict.get(x, x)
        )
    else:
        df["display_name"] = df["cell_type"]

    score_col = (
        f"retention_score_k{k}"
        if f"retention_score_k{k}" in df.columns
        else "retention_score"
    )
    baseline = (
        df["expected_baseline"].iloc[0]
        if "expected_baseline" in df.columns
        else k / (len(df) - 1)
    )

    # 2. Setup Figure
    fig, ax = plt.subplots(figsize=figsize)

    # 3. Plot dots with optional lineage coloring
    if lineage_mapping is not None:
        df["lineage"] = df["cell_type"].map(
            lambda x: lineage_mapping.get(x, "Other")
        )
        unique_lineages = list(dict.fromkeys(df["lineage"]))
        palette = sns.color_palette("tab10", len(unique_lineages))
        color_dict = dict(zip(unique_lineages, palette))

        sns.scatterplot(
            data=df,
            x="display_name",
            y=score_col,
            hue="lineage",
            palette=color_dict,
            s=70,
            zorder=3,
            ax=ax,
        )
        ax.legend(
            title="Cell Lineages",
            title_fontproperties={"weight": "bold"},
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=False,
        )
    else:
        sns.scatterplot(
            data=df,
            x="display_name",
            y=score_col,
            color="#1f77b4",
            s=70,
            zorder=3,
            ax=ax,
        )

    # 4. Add horizontal dashed line for expected baseline
    ax.axhline(
        y=baseline,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        zorder=2,
        label=f"Chance Baseline ({baseline:.3f})",
    )

    if lineage_mapping is None:
        ax.legend(frameon=False, loc="upper right")

    # 5. Axis formatting
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(
        df["display_name"], rotation=90, fontsize=8, ha="center"
    )
    ax.set_xlabel("Cell Type", fontsize=11, fontweight="bold")
    ax.set_ylabel(
        f"Retention Score ($R_i^{{({k})}}$)", fontsize=11, fontweight="bold"
    )
    ax.set_title(
        f"Neighbor Retention Score per Cell Type (k = {k})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(-0.02, max(1.0, df[score_col].max() + 0.05))
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.tight_layout()
    return fig, ax


def plot_heatmap_panel(
    panel_axes,
    matrix,
    cbar_label,
    lineage_colors,
    *,
    cmap="viridis",
    vmin=None,
    vmax=None,
    display_labels=None,
    show_cell_type_labels=False,
    tick_fontsize=4.5,
    colorbar_label_fontsize=7,
    colorbar_tick_fontsize=6,
    annotation_strip_gap_mm=0.8,
):
    r"""
    Plot one distance-like matrix with matching lineage annotation strips.
    """
    ax = panel_axes["matrix"]
    cbar_ax = panel_axes["colorbar"]
    left_strip_ax = panel_axes["left_strip"]
    top_strip_ax = panel_axes["top_strip"]

    number_of_cell_types = matrix.shape[0]
    if display_labels is None:
        display_labels = [str(index) for index in range(number_of_cell_types)]

    ticklabels = display_labels if show_cell_type_labels else False

    sns.heatmap(
        matrix,
        ax=ax,
        cbar_ax=cbar_ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        linewidths=0,
        rasterized=True,
    )

    ax.tick_params(
        axis="both",
        labelsize=tick_fontsize,
        length=0,
        pad=1,
    )

    if show_cell_type_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_box_aspect(1)

    left_strip = lineage_colors.reshape(-1, 1, 4)
    top_strip = lineage_colors.reshape(1, -1, 4)

    left_strip_ax.imshow(
        left_strip,
        aspect="auto",
        interpolation="nearest",
    )
    top_strip_ax.imshow(
        top_strip,
        aspect="auto",
        interpolation="nearest",
    )
    left_strip_ax.set_axis_off()
    top_strip_ax.set_axis_off()

    colorbar = _format_colorbar(
        ax,
        cbar_label,
        colorbar_label_fontsize,
        colorbar_tick_fontsize,
    )

    _align_matrix_annotation_axes(
        panel_axes,
        strip_gap_mm=annotation_strip_gap_mm,
    )

    return {
        "heatmap_ax": ax,
        "colorbar": colorbar,
    }


def plot_consensus_distance_matrix(
    consensus_distance_matrix: np.ndarray,
    interquartile_range_matrix: np.ndarray,
    lineage_mapping: Dict[str, str],
    cluster_annotations: np.ndarray,
    *,
    consensus_panel_axes,
    iqr_panel_axes,
    color_dict,
    cluster_relabeling_dict: Dict[str, str] = None,
    consensus_cmap="viridis",
    iqr_cmap="viridis",
    consensus_vmin=None,
    consensus_vmax=None,
    iqr_vmin=0.0,
    iqr_vmax=None,
    show_cell_type_labels=False,
    tick_fontsize=4.5,
    colorbar_label_fontsize=7,
    colorbar_tick_fontsize=6,
    annotation_strip_gap_mm=0.8,
) -> Dict[str, Dict[str, plt.Axes]]:
    r"""
    Plot the consensus-distance and IQR matrices into supplied panels.

    :param consensus_distance_matrix: A square matrix of consensus distances between cell types.
    :param interquartile_range_matrix: A square matrix of interquartile ranges of distances between cell types.
    :param lineage_mapping: A dictionary mapping cell types to their respective lineages
    :param cluster_annotations: An array of cell type annotations corresponding to the rows/columns of the distance and IQR matrices.
    :param consensus_panel_axes: A dictionary of axes for the consensus distance heatmap panel.
    :param iqr_panel_axes: A dictionary of axes for the interquartile range heatmap panel.
    :param color_dict: A dictionary mapping lineages to colors for annotation strips.
    :param cluster_relabeling_dict: An optional dictionary for relabeling cell types in the plot.
    :param consensus_cmap: Colormap for the consensus distance heatmap.
    :param iqr_cmap: Colormap for the interquartile range heatmap.
    :param consensus_vmin: Minimum value for the consensus distance heatmap color scale.
    :param consensus_vmax: Maximum value for the consensus distance heatmap color scale.
    :param iqr_vmin: Minimum value for the interquartile range heatmap color scale.
    :param iqr_vmax: Maximum value for the interquartile range heatmap color scale.
    :param show_cell_type_labels: Whether to display cell type labels on the heatmaps.
    :param tick_fontsize: Font size for the heatmap ticks.
    :param colorbar_label_fontsize: Font size for the colorbar labels.
    :param colorbar_tick_fontsize: Font size for the colorbar ticks.
    :param annotation_strip_gap_mm: Gap in millimeters between the heatmap and the annotation strips.
    :return: A dictionary containing the axes for the consensus distance and interquartile range

    """
    number_of_cell_types = len(cluster_annotations)
    consensus_matrix = _validate_square_matrix(
        consensus_distance_matrix,
        "consensus_distance_matrix",
        expected_size=number_of_cell_types,
    )
    iqr_matrix = _validate_square_matrix(
        interquartile_range_matrix,
        "interquartile_range_matrix",
        expected_size=number_of_cell_types,
    )

    lineage_colors = np.asarray(
        [
            color_dict[lineage_mapping[annotation]]
            for annotation in cluster_annotations
        ]
    )

    if cluster_relabeling_dict is None:
        display_labels = [str(label) for label in cluster_annotations]
    else:
        display_labels = [
            str(cluster_relabeling_dict.get(label, label))
            for label in cluster_annotations
        ]

    consensus_result = plot_heatmap_panel(
        panel_axes=consensus_panel_axes,
        matrix=consensus_matrix,
        cbar_label="Median normalised distance",
        lineage_colors=lineage_colors,
        cmap=consensus_cmap,
        vmin=consensus_vmin,
        vmax=consensus_vmax,
        display_labels=display_labels,
        show_cell_type_labels=show_cell_type_labels,
        tick_fontsize=tick_fontsize,
        colorbar_label_fontsize=colorbar_label_fontsize,
        colorbar_tick_fontsize=colorbar_tick_fontsize,
        annotation_strip_gap_mm=annotation_strip_gap_mm,
    )

    iqr_result = plot_heatmap_panel(
        panel_axes=iqr_panel_axes,
        matrix=iqr_matrix,
        cbar_label="Interquartile range",
        lineage_colors=lineage_colors,
        cmap=iqr_cmap,
        vmin=iqr_vmin,
        vmax=iqr_vmax,
        display_labels=display_labels,
        show_cell_type_labels=show_cell_type_labels,
        tick_fontsize=tick_fontsize,
        colorbar_label_fontsize=colorbar_label_fontsize,
        colorbar_tick_fontsize=colorbar_tick_fontsize,
        annotation_strip_gap_mm=annotation_strip_gap_mm,
    )

    return {
        "consensus": consensus_result,
        "interquartile_range": iqr_result,
    }


def _scale_edge_widths(
    weights,
    minimum_support,
    maximum_support,
    minimum_width,
    maximum_width,
) -> np.ndarray:
    r"""
    Map edge-support values linearly onto plotting widths.
    """
    weights = np.asarray(weights, dtype=float)

    if weights.size == 0:
        return np.asarray([], dtype=float)

    if maximum_support <= minimum_support:
        return np.full(weights.shape, maximum_width, dtype=float)

    scaled = (
        (weights - minimum_support)
        / (maximum_support - minimum_support)
    )
    scaled = np.clip(scaled, 0.0, 1.0)

    return minimum_width + scaled * (maximum_width - minimum_width)


def plot_consensus_neighbourhood_topology(
    consensus_distance_matrix: np.ndarray,
    edge_support: np.ndarray,
    lineage_mapping: Dict[str, str],
    cluster_annotations: np.ndarray,
    *,
    ax,
    color_dict,
    cluster_relabeling_dict: Dict[str, str]=None,
    edge_support_threshold=0.7,
    node_size=75,
    node_alpha=0.90,
    edge_alpha=0.45,
    edge_color="#707070",
    minimum_edge_width=0.45,
    maximum_edge_width=2.5,
    node_label_fontsize=5.5,
    axis_fontsize=7,
    tick_fontsize=6,
    title_fontsize=8,
    panel_label_fontsize=11,
    show_grid=True,
    topology_margin=0.10,
    adjust_node_labels=True,
    node_label_adjust_iterations=500,
    node_label_repel_padding_points=1.2,
    node_label_max_displacement_points=35.0,
    node_label_pull_strength=0.003,
    node_label_move_fraction=0.80,
    draw_label_connectors=True,
    label_connector_min_distance_points=6.0,
    verbose=True,
) -> Dict[str, Any]:
    r"""
    Plot the consensus cell-type neighbourhood topology on an existing axis.

    :param consensus_distance_matrix: A square matrix of consensus distances between cell types.
    :param edge_support: A square matrix of edge support values between cell types.
    :param lineage_mapping: A dictionary mapping cell types to their respective lineages.
    :param cluster_annotations: An array of cell type annotations corresponding to the rows/columns of the distance and support matrices.
    :param ax: The matplotlib axis on which to plot the topology.
    :param color_dict: A dictionary mapping lineages to colors for node coloring.
    :param cluster_relabeling_dict: An optional dictionary for relabeling cell types in the plot.
    :param edge_support_threshold: Minimum edge support required to draw an edge.
    :param node_size: Size of the nodes in the plot.
    :param node_alpha: Transparency of the nodes.
    :param edge_alpha: Transparency of the edges.
    :param edge_color: Color of the edges.
    :param minimum_edge_width: Minimum width of the edges.
    :param maximum_edge_width: Maximum width of the edges.
    :param node_label_fontsize: Font size of the node labels.
    :param axis_fontsize: Font size of the axis labels.
    :param tick_fontsize: Font size of the axis ticks.
    :param title_fontsize: Font size of the plot title.
    :param panel_label_fontsize: Font size of the panel label.
    :param show_grid: Whether to show grid lines on the plot.
    :param topology_margin: Margin around the topology plot.
    :param adjust_node_labels: Whether to adjust node labels to avoid overlap.
    :param node_label_adjust_iterations: Number of iterations for label adjustment.
    :param node_label_repel_padding_points: Padding for label repulsion in points.
    :param node_label_max_displacement_points: Maximum displacement for labels in points.
    :param node_label_pull_strength: Strength of the pull towards original label positions.
    :param node_label_move_fraction: Fraction of movement allowed for label adjustment.
    :param draw_label_connectors: Whether to draw connectors from nodes to their labels.
    :param label_connector_min_distance_points: Minimum distance for drawing label connectors in points.
    :param verbose: Whether to print additional information during plotting.
    :return: A dictionary containing the graph object, node positions, and label positions.
    """
    number_of_cell_types = len(cluster_annotations)
    consensus_matrix = _validate_square_matrix(
        consensus_distance_matrix,
        "consensus_distance_matrix",
        expected_size=number_of_cell_types,
    )
    support_matrix = _validate_square_matrix(
        edge_support,
        "edge_support",
        expected_size=number_of_cell_types,
    )

    if not 0.0 <= edge_support_threshold <= 1.0:
        raise ValueError(
            "edge_support_threshold must lie between 0 and 1."
        )

    if not np.allclose(
        support_matrix,
        support_matrix.T,
        equal_nan=True,
        atol=1e-8,
    ):
        raise ValueError("edge_support must be symmetric.")

    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(consensus_matrix)
    variance_explained = pca.explained_variance_ratio_ * 100.0

    if verbose:
        print(
            "Variance explained by PC1 and PC2: "
            f"{variance_explained[0]:.2f}% and "
            f"{variance_explained[1]:.2f}% "
            f"({variance_explained.sum():.2f}% total)."
        )

    positions = {
        label: (pca_results[index, 0], pca_results[index, 1])
        for index, label in enumerate(cluster_annotations)
    }

    graph = nx.Graph()
    graph.add_nodes_from(cluster_annotations)

    retained_edge_weights = []
    for row_index in range(number_of_cell_types):
        for column_index in range(row_index + 1, number_of_cell_types):
            weight = support_matrix[row_index, column_index]

            if np.isfinite(weight) and weight >= edge_support_threshold:
                graph.add_edge(
                    cluster_annotations[row_index],
                    cluster_annotations[column_index],
                    weight=float(weight),
                )
                retained_edge_weights.append(float(weight))

    if retained_edge_weights:
        maximum_retained_support = max(retained_edge_weights)
        edge_widths = _scale_edge_widths(
            retained_edge_weights,
            edge_support_threshold,
            maximum_retained_support,
            minimum_edge_width,
            maximum_edge_width,
        )
    else:
        maximum_retained_support = edge_support_threshold
        edge_widths = np.asarray([], dtype=float)

    node_colors = [
        color_dict[lineage_mapping[node]]
        for node in graph.nodes()
    ]

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        width=edge_widths.tolist(),
        alpha=edge_alpha,
        edge_color=edge_color,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_colors,
        node_size=node_size,
        alpha=node_alpha,
        linewidths=0.25,
        edgecolors="white",
    )

    ax.margins(x=topology_margin, y=topology_margin)

    y_values = pca_results[:, 1]
    y_range = float(np.ptp(y_values))
    label_offset = 0.015 * y_range if y_range > 0 else 0.02
    label_positions = {
        node: (x_coordinate, y_coordinate + label_offset)
        for node, (x_coordinate, y_coordinate) in positions.items()
    }
    display_labels = {
        node: (
            cluster_relabeling_dict.get(node, node)
            if cluster_relabeling_dict is not None
            else node
        )
        for node in cluster_annotations
    }

    label_artists = nx.draw_networkx_labels(
        graph,
        label_positions,
        labels=display_labels,
        ax=ax,
        font_size=node_label_fontsize,
        font_weight="medium",
        horizontalalignment="center",
        verticalalignment="center",
        clip_on=True,
    )

    if adjust_node_labels:
        (
            adjusted_label_positions,
            label_connector_artists,
        ) = _adjust_network_label_positions(
            ax=ax,
            label_artists=label_artists,
            node_positions=positions,
            iterations=node_label_adjust_iterations,
            padding_points=node_label_repel_padding_points,
            max_displacement_points=(
                node_label_max_displacement_points
            ),
            pull_strength=node_label_pull_strength,
            move_fraction=node_label_move_fraction,
            draw_connectors=draw_label_connectors,
            connector_min_distance_points=(
                label_connector_min_distance_points
            ),
            connector_color=edge_color,
        )
    else:
        adjusted_label_positions = {
            node: text_artist.get_position()
            for node, text_artist in label_artists.items()
        }
        label_connector_artists = []

    ax.set_xlabel(
        f"PC1 ({variance_explained[0]:.1f}%)",
        fontsize=axis_fontsize,
        labelpad=2,
    )
    ax.set_ylabel(
        f"PC2 ({variance_explained[1]:.1f}%)",
        fontsize=axis_fontsize,
        labelpad=2,
    )
    ax.tick_params(
        axis="both",
        labelsize=tick_fontsize,
        length=2,
        pad=1,
    )
    ax.set_title(
        "Cell-type neighbourhood topology",
        loc="center",
        fontsize=title_fontsize,
        pad=4,
    )
    ax.set_title(
        "d",
        loc="left",
        fontsize=panel_label_fontsize,
        fontweight="bold",
        pad=4,
    )

    if show_grid:
        ax.grid(
            True,
            linestyle=":",
            linewidth=0.45,
            alpha=0.35,
            zorder=0,
        )
    else:
        ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    return {
        "graph": graph,
        "positions": positions,
        "retained_edge_weights": retained_edge_weights,
        "maximum_retained_support": maximum_retained_support,
        "variance_explained": variance_explained,
        "minimum_edge_width": minimum_edge_width,
        "maximum_edge_width": maximum_edge_width,
        "label_artists": label_artists,
        "adjusted_label_positions": adjusted_label_positions,
        "label_connector_artists": label_connector_artists,
    }


def _add_lineage_legend(
    ax,
    lineage_order,
    color_dict,
    *,
    title_fontsize,
    legend_fontsize,
    marker_size,
    labelspacing,
    handletextpad,
    ncol,
):
    r"""
    Add the shared lineage legend to its dedicated axis.
    """
    ax.set_axis_off()
    handles = [
        mpatches.Patch(
            facecolor=color_dict[lineage],
            edgecolor="none",
            label=str(lineage),
        )
        for lineage in lineage_order
    ]

    legend = ax.legend(
        handles=handles,
        title="Cell lineages",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=ax.transAxes,
        ncol=ncol,
        frameon=False,
        fontsize=legend_fontsize,
        title_fontsize=title_fontsize,
        labelspacing=labelspacing,
        handlelength=marker_size,
        handleheight=marker_size,
        handletextpad=handletextpad,
        columnspacing=1.2,
        borderaxespad=0,
    )

    if hasattr(legend, "_legend_box"):
        legend._legend_box.align = "left"

    return legend


def _add_edge_support_legend(
    ax,
    topology_result,
    edge_support_threshold,
    *,
    title_fontsize,
    legend_fontsize,
    edge_color,
    edge_alpha,
):
    r"""
    Add representative retained-edge widths to a dedicated legend axis.
    """
    ax.set_axis_off()
    retained_weights = topology_result["retained_edge_weights"]

    if not retained_weights:
        ax.text(
            0.0,
            1.0,
            "Edge support",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=title_fontsize,
        )
        ax.text(
            0.0,
            0.82,
            f"No edges with support >= {edge_support_threshold:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
        )
        return None

    maximum_support = topology_result["maximum_retained_support"]
    sample_weights = np.unique(
        np.linspace(edge_support_threshold, maximum_support, num=3)
    )
    sample_widths = _scale_edge_widths(
        sample_weights,
        edge_support_threshold,
        maximum_support,
        topology_result["minimum_edge_width"],
        topology_result["maximum_edge_width"],
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=edge_color,
            linewidth=float(width),
            alpha=edge_alpha,
            label=f"{weight:.2f}",
        )
        for weight, width in zip(sample_weights, sample_widths)
    ]

    legend = ax.legend(
        handles=handles,
        title="Edge support",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=ax.transAxes,
        frameon=False,
        fontsize=legend_fontsize,
        title_fontsize=title_fontsize,
        labelspacing=0.9,
        handlelength=2.8,
        handletextpad=0.8,
        borderaxespad=0,
    )

    if hasattr(legend, "_legend_box"):
        legend._legend_box.align = "left"

    return legend


def plot_seed_stability_and_consensus_topology(
    seed_similarity_matrix: np.ndarray,
    consensus_distance_matrix: np.ndarray,
    interquartile_range_matrix: np.ndarray,
    edge_support: np.ndarray,
    lineage_mapping: Dict[str, str],
    cluster_annotations: np.ndarray,
    seeds: List[int] = None,
    median_pairwise_seed_similarity: float = None,
    lower_pairwise_seed_similarity: float = None,
    upper_pairwise_seed_similarity: float = None,
    cluster_relabeling_dict: Dict[str, str] = None,
    edge_support_threshold: float = 0.7,
    save_fig: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, Dict[str, plt.Axes], Dict[str, Any]]:
    r"""
    Plot seed stability and consensus cell-type topology as one panel based on the the seed similarity matrix, the
    consensus distance matrix, the interquartile range matrix, and the edge-support matrix.

    The composite layout is::

        a seed similarity | b consensus distance | c interquartile range
        d neighbourhood topology (spans two columns) | lineage legend
        d neighbourhood topology (spans two columns) | edge-support legend

    :param seed_similarity_matrix: Square seed-by-seed Spearman-correlation matrix.
    :param consensus_distance_matrix: Square cell-type consensus distance matrix. Its ordering must match
        ``cluster_annotations``.
    :param interquartile_range_matrix: Square cell-type IQR matrix in the same order as the consensus matrix.
    :param edge_support: Symmetric cell-type edge-support matrix in the same order as the consensus matrix.
    :param lineage_mapping: Mapping from every cell type in ``cluster_annotations`` to a lineage.
    :param cluster_annotations: Ordered cell-type labels for the cell-type matrices.
    :param seeds: Optional display labels for the seed-similarity matrix.
    :param median_pairwise_seed_similarity: Optional pre-computed median pairwise seed similarity.
    :param lower_pairwise_seed_similarity: Optional pre-computed lower bound of pairwise seed similarity.
    :param upper_pairwise_seed_similarity: Optional pre-computed upper bound of pairwise seed similarity.
    :param cluster_relabeling_dict: Optional mapping from original cell-type labels to display labels for the consensus
        and IQR matrices, and the neighbourhood topology.
    :param edge_support_threshold: Minimum support required to draw an edge. Default: 0.7.
    :param save_fig: Whether to save the figure.
    :return: Matplotlib figure, named axes and calculated plotting results.

    Other Parameters
    ----------------
    figure_width_mm
        Figure width in millimetres. Default: 180.
    figure_height_mm
        Figure height in millimetres. Default: 160.
    max_figure_height_mm
        Maximum allowed figure height. Default: 247.
    figure_right_margin_mm
        Physical space reserved to the right of panel c for its colorbar tick
        labels and vertical title. Default: 12 mm. The complete figure width
        remains 180 mm.
    show_cell_type_labels
        Show cell-type labels around panels b and c. Default: False.
    annotation_strip_gap_mm
        Physical gap between the lineage strips and heatmaps. Default: 0.8 mm.
    adjust_node_labels
        Repel overlapping labels in panel d. Default: True.
    node_label_max_displacement_points
        Maximum distance through which a label may be moved. Default: 35 pt.
    lineage_palette
        Seaborn palette name or mapping from lineages to colours.
    dataset_name, tax_level, file_suffix
        Components used to construct the saved filename.
    """
    cluster_annotations = list(cluster_annotations)

    if len(cluster_annotations) == 0:
        raise ValueError("cluster_annotations must not be empty.")

    if len(set(cluster_annotations)) != len(cluster_annotations):
        raise ValueError("cluster_annotations must contain unique labels.")

    seed_matrix = _validate_square_matrix(
        seed_similarity_matrix,
        "seed_similarity_matrix",
    )

    number_of_cell_types = len(cluster_annotations)

    consensus_matrix = _validate_square_matrix(
        consensus_distance_matrix,
        "consensus_distance_matrix",
        expected_size=number_of_cell_types,
    )

    iqr_matrix = _validate_square_matrix(
        interquartile_range_matrix,
        "interquartile_range_matrix",
        expected_size=number_of_cell_types,
    )

    support_matrix = _validate_square_matrix(
        edge_support,
        "edge_support",
        expected_size=number_of_cell_types,
    )

    if seeds is not None and len(seeds) != seed_matrix.shape[0]:
        raise ValueError(
            "The number of entries in seeds must equal the dimensions of "
            "seed_similarity_matrix."
        )

    lineage_order, color_dict = _make_lineage_color_dict(
        lineage_mapping=lineage_mapping,
        cluster_annotations=cluster_annotations,
        palette=kwargs.get("lineage_palette"),
    )

    mm_per_inch = 25.4

    figure_width_mm = float(
        kwargs.get("figure_width_mm", 180.0)
    )
    figure_height_mm = float(
        kwargs.get("figure_height_mm", 160.0)
    )
    max_figure_height_mm = float(
        kwargs.get("max_figure_height_mm", 247.0)
    )

    if figure_width_mm <= 0 or figure_height_mm <= 0:
        raise ValueError(
            "Figure width and height must be greater than zero."
        )

    if max_figure_height_mm <= 0:
        raise ValueError(
            "max_figure_height_mm must be greater than zero."
        )

    figure_height_mm = min(
        figure_height_mm,
        max_figure_height_mm,
    )

    figure_right_margin_mm = float(
        kwargs.get("figure_right_margin_mm", 12.0)
    )

    if not 0 <= figure_right_margin_mm < figure_width_mm:
        raise ValueError(
            "figure_right_margin_mm must be non-negative and smaller than "
            "figure_width_mm."
        )

    # Reserve space inside the fixed-width canvas for the tick labels and
    # vertical description of the rightmost colorbar.
    default_figure_right = (
        1.0 - figure_right_margin_mm / figure_width_mm
    )

    panel_label_fontsize = kwargs.get(
        "panel_label_fontsize",
        11,
    )
    title_fontsize = kwargs.get(
        "title_fontsize",
        8,
    )
    axis_fontsize = kwargs.get(
        "axis_fontsize",
        7,
    )
    tick_fontsize = kwargs.get(
        "tick_fontsize",
        6,
    )
    matrix_tick_fontsize = kwargs.get(
        "matrix_tick_fontsize",
        4.5,
    )
    legend_fontsize = kwargs.get(
        "legend_fontsize",
        6.5,
    )
    legend_title_fontsize = kwargs.get(
        "legend_title_fontsize",
        8,
    )
    colorbar_label_fontsize = kwargs.get(
        "colorbar_label_fontsize",
        7,
    )
    colorbar_tick_fontsize = kwargs.get(
        "colorbar_tick_fontsize",
        6,
    )

    fig = plt.figure(
        figsize=(
            figure_width_mm / mm_per_inch,
            figure_height_mm / mm_per_inch,
        ),
        dpi=300,
    )

    outer_grid = gridspec.GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        width_ratios=kwargs.get(
            "width_ratios",
            [1.0, 1.0, 1.0],
        ),
        height_ratios=kwargs.get(
            "height_ratios",
            [1.0, 0.82, 0.82],
        ),
        left=kwargs.get(
            "figure_left",
            0.035,
        ),
        right=kwargs.get(
            "figure_right",
            default_figure_right,
        ),
        bottom=kwargs.get(
            "figure_bottom",
            0.055,
        ),
        top=kwargs.get(
            "figure_top",
            0.975,
        ),
        wspace=kwargs.get(
            "outer_wspace",
            0.24,
        ),
        hspace=kwargs.get(
            "outer_hspace",
            0.24,
        ),
    )

    seed_panel_axes = _create_matrix_panel_axes(
        fig=fig,
        subplot_spec=outer_grid[0, 0],
        panel_label="a",
        title="Seed similarity",
        panel_label_fontsize=panel_label_fontsize,
        title_fontsize=title_fontsize,
    )

    consensus_panel_axes = _create_matrix_panel_axes(
        fig=fig,
        subplot_spec=outer_grid[0, 1],
        panel_label="b",
        title="Consensus distance",
        panel_label_fontsize=panel_label_fontsize,
        title_fontsize=title_fontsize,
    )

    iqr_panel_axes = _create_matrix_panel_axes(
        fig=fig,
        subplot_spec=outer_grid[0, 2],
        panel_label="c",
        title="Interquartile range",
        panel_label_fontsize=panel_label_fontsize,
        title_fontsize=title_fontsize,
    )

    # Panel a has no lineage strips, but its blank strip axes reserve the same
    # physical space as panels b and c so all three matrices remain equal.
    seed_panel_axes["left_strip"].set_axis_off()
    seed_panel_axes["top_strip"].set_axis_off()

    topology_ax = fig.add_subplot(
        outer_grid[1:, :2]
    )
    lineage_legend_ax = fig.add_subplot(
        outer_grid[1, 2]
    )
    edge_legend_ax = fig.add_subplot(
        outer_grid[2, 2]
    )

    seed_result = plot_seed_similarity_matrix(
        seed_similarity_matrix=seed_matrix,
        median_pairwise_seed_similarity=(
            median_pairwise_seed_similarity
        ),
        lower_pairwise_seed_similarity=(
            lower_pairwise_seed_similarity
        ),
        upper_pairwise_seed_similarity=(
            upper_pairwise_seed_similarity
        ),
        ax=seed_panel_axes["matrix"],
        cbar_ax=seed_panel_axes["colorbar"],
        seeds=seeds,
        cmap=kwargs.get(
            "seed_similarity_cmap",
            "viridis",
        ),
        vmin=kwargs.get(
            "seed_similarity_vmin",
            -1.0,
        ),
        vmax=kwargs.get(
            "seed_similarity_vmax",
            1.0,
        ),
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        colorbar_label_fontsize=colorbar_label_fontsize,
        colorbar_tick_fontsize=colorbar_tick_fontsize,
        verbose=kwargs.get(
            "verbose",
            True,
        ),
    )

    matrix_results = plot_consensus_distance_matrix(
        consensus_distance_matrix=consensus_matrix,
        interquartile_range_matrix=iqr_matrix,
        lineage_mapping=lineage_mapping,
        cluster_annotations=cluster_annotations,
        consensus_panel_axes=consensus_panel_axes,
        iqr_panel_axes=iqr_panel_axes,
        color_dict=color_dict,
        cluster_relabeling_dict=cluster_relabeling_dict,
        consensus_cmap=kwargs.get(
            "consensus_cmap",
            "viridis",
        ),
        iqr_cmap=kwargs.get(
            "iqr_cmap",
            "viridis",
        ),
        consensus_vmin=kwargs.get(
            "consensus_vmin",
        ),
        consensus_vmax=kwargs.get(
            "consensus_vmax",
        ),
        iqr_vmin=kwargs.get(
            "iqr_vmin",
            0.0,
        ),
        iqr_vmax=kwargs.get(
            "iqr_vmax",
        ),
        show_cell_type_labels=kwargs.get(
            "show_cell_type_labels",
            False,
        ),
        tick_fontsize=matrix_tick_fontsize,
        colorbar_label_fontsize=colorbar_label_fontsize,
        colorbar_tick_fontsize=colorbar_tick_fontsize,
        annotation_strip_gap_mm=kwargs.get(
            "annotation_strip_gap_mm",
            0.8,
        ),
    )

    edge_color = kwargs.get(
        "edge_color",
        "#707070",
    )
    edge_alpha = kwargs.get(
        "edge_alpha",
        0.45,
    )

    topology_result = plot_consensus_neighbourhood_topology(
        consensus_distance_matrix=consensus_matrix,
        edge_support=support_matrix,
        lineage_mapping=lineage_mapping,
        cluster_annotations=cluster_annotations,
        ax=topology_ax,
        color_dict=color_dict,
        cluster_relabeling_dict=cluster_relabeling_dict,
        edge_support_threshold=edge_support_threshold,
        node_size=kwargs.get(
            "node_size",
            75,
        ),
        node_alpha=kwargs.get(
            "node_alpha",
            0.90,
        ),
        edge_alpha=edge_alpha,
        edge_color=edge_color,
        minimum_edge_width=kwargs.get(
            "minimum_edge_width",
            0.45,
        ),
        maximum_edge_width=kwargs.get(
            "maximum_edge_width",
            2.5,
        ),
        node_label_fontsize=kwargs.get(
            "node_label_fontsize",
            5.5,
        ),
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
        title_fontsize=title_fontsize,
        panel_label_fontsize=panel_label_fontsize,
        show_grid=kwargs.get(
            "show_grid",
            True,
        ),
        topology_margin=kwargs.get(
            "topology_margin",
            0.10,
        ),
        adjust_node_labels=kwargs.get(
            "adjust_node_labels",
            True,
        ),
        node_label_adjust_iterations=kwargs.get(
            "node_label_adjust_iterations",
            500,
        ),
        node_label_repel_padding_points=kwargs.get(
            "node_label_repel_padding_points",
            1.2,
        ),
        node_label_max_displacement_points=kwargs.get(
            "node_label_max_displacement_points",
            35.0,
        ),
        node_label_pull_strength=kwargs.get(
            "node_label_pull_strength",
            0.003,
        ),
        node_label_move_fraction=kwargs.get(
            "node_label_move_fraction",
            0.80,
        ),
        draw_label_connectors=kwargs.get(
            "draw_label_connectors",
            True,
        ),
        label_connector_min_distance_points=kwargs.get(
            "label_connector_min_distance_points",
            6.0,
        ),
        verbose=kwargs.get(
            "verbose",
            True,
        ),
    )

    lineage_legend_ncol = int(
        kwargs.get(
            "lineage_legend_ncol",
            1 if len(lineage_order) <= 10 else 2,
        )
    )

    if lineage_legend_ncol <= 0:
        raise ValueError(
            "lineage_legend_ncol must be greater than zero."
        )

    lineage_legend = _add_lineage_legend(
        ax=lineage_legend_ax,
        lineage_order=lineage_order,
        color_dict=color_dict,
        title_fontsize=legend_title_fontsize,
        legend_fontsize=legend_fontsize,
        marker_size=kwargs.get(
            "lineage_legend_marker_size",
            0.8,
        ),
        labelspacing=kwargs.get(
            "lineage_legend_labelspacing",
            0.45,
        ),
        handletextpad=kwargs.get(
            "lineage_legend_handletextpad",
            0.5,
        ),
        ncol=lineage_legend_ncol,
    )

    edge_legend = _add_edge_support_legend(
        ax=edge_legend_ax,
        topology_result=topology_result,
        edge_support_threshold=edge_support_threshold,
        title_fontsize=legend_title_fontsize,
        legend_fontsize=legend_fontsize,
        edge_color=edge_color,
        edge_alpha=edge_alpha,
    )

    # Reapply the strip alignment after every artist has been added. This is
    # important for interactive backends that defer aspect-ratio calculations
    # until the first complete canvas draw.
    fig.canvas.draw()

    _align_matrix_annotation_axes(
        consensus_panel_axes,
        strip_gap_mm=kwargs.get(
            "annotation_strip_gap_mm",
            0.8,
        ),
    )

    _align_matrix_annotation_axes(
        iqr_panel_axes,
        strip_gap_mm=kwargs.get(
            "annotation_strip_gap_mm",
            0.8,
        ),
    )

    axes = {
        "seed_similarity": seed_panel_axes,
        "consensus_distance": consensus_panel_axes,
        "interquartile_range": iqr_panel_axes,
        "neighbourhood_topology": topology_ax,
        "lineage_legend": lineage_legend_ax,
        "edge_support_legend": edge_legend_ax,
    }

    results = {
        "seed_similarity": seed_result,
        "distance_matrices": matrix_results,
        "topology": topology_result,
        "lineage_color_dict": color_dict,
        "lineage_legend": lineage_legend,
        "edge_support_legend": edge_legend,
    }

    if save_fig:
        dataset_name = kwargs.get(
            "dataset_name",
            "default",
        )
        tax_level = kwargs.get(
            "tax_level",
            "atlas_level",
        )
        file_suffix = kwargs.get(
            "file_suffix",
            "pdf",
        )
        figure_name = kwargs.get(
            "figure_name",
            (
                f"seed_stability_consensus_topology_"
                f"{tax_level}.{file_suffix}"
            ),
        )

        save_directory = os.path.join(
            "./figures",
            str(dataset_name),
        )
        os.makedirs(
            save_directory,
            exist_ok=True,
        )

        figure_path = os.path.join(
            save_directory,
            figure_name,
        )

        fig.savefig(
            figure_path,
            dpi=300,
            bbox_inches=None,
            pad_inches=0,
        )

        results["figure_path"] = figure_path

    if kwargs.get("show", True):
        plt.show()

    return fig, axes, results

def cell_classification_dataframe(
        cell_type: str,
        classifier: str,
        classification_dict
) -> pd.DataFrame:
    r"""
    TO DO: Later add marker gene classifier her
    """
    # Extract classification metrics of the embedding-cell type classifier for the cell type given
    accuracy_emb_cell_classifier = classification_dict[classifier]["roc_auc"][cell_type]["accuracy"]
    auroc_emb_cell_classifier = classification_dict[classifier]["roc_auc"][cell_type]["auroc"]
    f1_emb_cell_classifier = classification_dict[classifier]["roc_auc"][cell_type]["f1"]

    # Extract the classification metrics of the marker gene-cell type classifier for the cell type given
    accuracy_mg_cell_classifier = classification_dict["marker_genes"]["roc_auc"][cell_type]["accuracy"]
    auroc_mg_cell_classifier = classification_dict["marker_genes"]["roc_auc"][cell_type]["auroc"]
    f1_mg_cell_classifier = classification_dict["marker_genes"]["roc_auc"][cell_type]["f1"]


    cell_classification_metrics_values = np.array([
        accuracy_emb_cell_classifier,
        auroc_emb_cell_classifier,
        f1_emb_cell_classifier,
        accuracy_mg_cell_classifier,
        auroc_mg_cell_classifier,
        f1_mg_cell_classifier
    ])
    cell_classification_metrics_keys = np.array(["accuracy", "auroc", "f1"] * 2)
    emb_classifier_name = np.array([classifier] * 3)
    mg_classifier_name = np.array(["marker_genes"] * 3)

    classifier_type_list = np.concatenate((emb_classifier_name, mg_classifier_name))

    # Create dataframe
    cell_classification_eval_df = pd.DataFrame({
        "Embedding": classifier_type_list,
        "Classification Metric": cell_classification_metrics_keys,
        "Value": cell_classification_metrics_values
    })

    return cell_classification_eval_df

def plot_transcript_usage_data_distribution_analysis(
        adata: AnnData,
        gene_name: str,
        intron_group_name: str,
        zero_inflation_included: bool = True,
        likelihood_keys: List[str] = ["DM", "ZIDM"],
        tissue = None,
        cluster_eval_df: pd.DataFrame = None,
        save_fig: bool = True
) -> None:
        r"""
        Given an AnnData object containing transcript usage data and a specified gene and intron group, plot the
        distribution of non-zero intron counts across cells per intron as bar chart. Optionally, the structural zeros (cells
        where all introns of the group have zero counts) can be included in the bar chart. Then, the distribution of the
        likelihood of each intron across cells is plotted as a histogram for each intron participating in the intron group.
        The charts are plotted in a (2 x intron group size + 1) - grid. The charts are saved as PDF file.

        :param adata: AnnData object containing the transcript usage data
        :param gene_name: str, the name of the gene to be analyzed
        :param intron_group_name: str, the name of the intron group to be analyzed
        :param zero_inflation_included: bool, whether to include the structural zeros in the bar chart
        :param save_fig: bool, whether to save the figure as a PDF file
        :return: None
        """
        # Check if the input is an AnnData object
        if not isinstance(adata, AnnData):
            raise ValueError("Input must be an AnnData object")

        # Check if gene name is present in the AnnData object
        if gene_name not in adata.var["gene_name"].to_numpy():
            raise ValueError(f"Gene {gene_name} not found in the AnnData object")

        # Check if intron group name is present in the AnnData object
        if intron_group_name not in adata.var["intron_group"].to_numpy():
            raise ValueError(f"Intron group {intron_group_name} not found in the AnnData object")

        # Check if cluster_eval_df contains matching likelihoods
        if cluster_eval_df is not None:
            for likelihood_key in likelihood_keys:
                if likelihood_key not in cluster_eval_df["Likelihood"].to_numpy():
                    raise ValueError(f"Likelihood {likelihood_key} not found in the cluster evaluation dataframe")

        # Check label_key for cell types
        if "cell_ontology_class" in adata.obs.keys():
            cell_label_key = "cell_ontology_class"
        elif "subclass_label" in adata.obs.keys():
            cell_label_key = "subclass_label"
        else:
            raise ValueError(
                "No cell type label found in adata.obs. Please provide a cell type label in adata.obs with key 'cell_ontology_class' or 'subclass_label'")

        # Extract the full intron names
        intron_names = adata.var_names[adata.var["intron_group"] == intron_group_name].tolist()

        # Remove gene from intron group name and get intron group type (+ or -)
        intron_group_name = intron_group_name.replace(gene_name + "_", "")
        intron_group_type = intron_group_name[-1]

        # Extract intron counts
        all_intron_names = adata.var_names.to_numpy()
        intron_names_indices = []
        for intron_name in intron_names:
            intron_idx = np.argwhere(all_intron_names == intron_name)
            intron_names_indices.append(intron_idx.item())

        # Adjust naming of intron
        if intron_group_type == "+":
            intron_names = [s.split(':')[1] for s in intron_names]
            intron_names = [s.split("-")[0] for s in intron_names]
        elif intron_group_type == "-":
            intron_names = [s.split(':')[1] for s in intron_names]
            intron_names = [s.split("-")[-1] for s in intron_names]

        non_zero_intron_names = intron_names.copy()

        intron_counts = adata.layers["counts"].toarray()[:, intron_names_indices]
        intron_group_size = intron_counts.shape[-1]

        # Counts per intron across cells, optionally structural zeros are counted as well as additional entry
        full_zero_intron_group_counts = np.all(intron_counts == 0, axis=1).sum()

        if zero_inflation_included:
            intron_group_counts = np.arange(intron_group_size + 1)
            intron_group_counts[-1] = full_zero_intron_group_counts

            for intron in np.arange(intron_group_size):
                intron_group_counts[intron] = len(np.argwhere(intron_counts[:, intron] > 0))

            non_zero_intron_names.append("Structural 0")
        else:
            intron_group_counts = np.arange(intron_group_size)

            for intron in np.arange(intron_group_size):
                intron_group_counts[intron] = np.count_nonzero(intron_counts[:, intron])

        # Likelihood distribution of introns across cells TO DO: for each likelihood key
        count_recon_keys = ["counts"]
        for likelihood_key in likelihood_keys:
            if likelihood_key + "_reconstructions" not in adata.layers.keys():
                raise ValueError(f"{likelihood_key}_reconstructions not found in the AnnData object. "
                                 f"Please provide the correct likelihood key.")

            count_recon_keys.append(likelihood_key + "_reconstructions")

        count_recon_dict = {}

        for key in count_recon_keys:
            if key == "counts":
                intron_counts = adata.layers["counts"].toarray()[:, intron_names_indices]
            else:
                intron_counts = np.round(adata.layers[key][:, intron_names_indices])

            # Determine the cells where intron counts are present
            present_intron_counts = intron_counts[np.all(intron_counts == 0, axis=1) == False, :]
            intron_group_sums = np.sum(present_intron_counts, axis=1)
            intron_likelihoods = present_intron_counts / intron_group_sums.reshape(-1, 1)
            count_recon_dict[key] = intron_likelihoods.copy()

        # Create histogram proportions TO DO: for each likelihood key
        intron_histogram_dict = {}
        for intron in np.arange(intron_group_size):
            proportions_data, bin_labels_data = create_histogram_transcript_usage(count_recon_dict["counts"][:, intron],
                                                                                  bin_size=0.1)
            proportions_data = proportions_data / np.sum(proportions_data)
            proportions = proportions_data.copy()
            bin_labels = bin_labels_data.copy()
            data_type = np.full(proportions_data.shape[0], "Data", dtype=object)

            for likelihood_key in likelihood_keys:

                proportions_recons, bin_labels_recons = create_histogram_transcript_usage(
                    count_recon_dict[likelihood_key + "_reconstructions"][:, intron],
                    bin_size=0.1
                )

                proportions_recons = proportions_recons / np.sum(proportions_recons)
                proportions = np.concatenate((proportions, proportions_recons))
                bin_labels = np.concatenate((bin_labels, bin_labels_recons))

                data_type = np.concatenate(
                    (
                        data_type,
                        np.full(proportions_recons.shape[0], likelihood_key, dtype=object)
                    )
                )

            intron_histogram_dict["Intron " + str(intron.item())] = {"bin_labels": bin_labels.copy(),
                                                                     "proportions": proportions.copy(),
                                                                     "Likelihood": data_type.copy()}
        # Plotting
        if cluster_eval_df is not None:
            # TO DO: add another row for  second likelihood key if present
            #fig, axes = plt.subplots(3, intron_group_size + 1, figsize=(18, 12), dpi=300)
            fig = plt.figure(figsize=(20, 16), dpi=300)
            num_col = np.max((intron_group_size, len(likelihood_keys)))
            gs = gridspec.GridSpec(3, num_col + 1, figure=fig)
        else:
            # TO DO: add another row for  second likelihood key if present
            fig, axes = plt.subplots(2, intron_group_size + 1, figsize=(18, 12), dpi=300)
            fig = plt.figure(figsize=(20, 16), dpi=300)
            gs = gridspec.GridSpec(3, intron_group_size + 1, figure=fig)
        #fig.suptitle(f"Transcript Usage Data Distribution Analysis for Intron Group {intron_group_name} in {gene_name}",fontsize=12)

        # Non-zero intron counts bar chart for specified intron group
        ax00 = fig.add_subplot(gs[0, 0])
        #ax00.set_title(f"Non-zero Intron Counts Across Cells", fontsize=10)
        ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')
        ax00.grid(axis='y', alpha=0.75, linestyle='--')
        ax00.bar(np.array(non_zero_intron_names), intron_group_counts, color='#004d40')
        ax00.set_xlabel("Introns")
        ax00.set_ylabel("Non-zero counts")
        ax00.tick_params(axis='both')
        ax00.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
        plt.setp(ax00.get_xticklabels(), ha='right')

        custom_greens = [
            '#004d40',  # Very Dark Teal Green
            '#00897b',  # Medium Teal Green
            '#4db6ac',  # Light Mint Green
            '#89e5dd'  # Very Light Mint Green
        ]

        for intron_idx, intron_name in enumerate(intron_names):
            # Distribution of likelihood of intron across cells
            ax_intron = fig.add_subplot(gs[0, intron_idx + 1])
            intron_histogram_df = pd.DataFrame(intron_histogram_dict["Intron " + str(intron_idx)])
            #ax_intron.set_title(f"Distribution of Intron {intron_name} Across Cells",fontsize=10)
            if intron_idx == 0:
                ax_intron.set_title('b', loc='left', fontsize=20, fontweight='bold')
            sns.barplot(
                intron_histogram_df,
                x='bin_labels',
                y='proportions',
                hue="Likelihood",
                ax=ax_intron,
                palette=custom_greens
            )

            ax_intron.grid(axis='y', alpha=0.75, linestyle='--')
            ax_intron.set_xlabel("Likelihood")
            ax_intron.set_ylabel("Proportion of cells")
            ax_intron.tick_params(axis='both')
            ax_intron.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
            plt.setp(ax_intron.get_xticklabels(), ha='right')

        ax2022_full_row = fig.add_subplot(gs[1, :])
        image_path = "./figures/crecerelle_tuVI.png"
        prob_model_img = mpimg.imread(image_path)
        ax2022_full_row.imshow(prob_model_img)
        ax2022_full_row.axis("off")
        ax2022_full_row.set_title('c', loc='left', fontsize=20, fontweight='bold')

        """
        # Load images to display in third row
        image_path_vae = "./figures/vae_generative_model_tu.png"
        image_path_eq_classic = "./figures/vae_generative_model_tu_eq_classic.jpg"
        image_path_eq_datacentric = "./figures/vae_generative_model_tu_eq_datacentric.jpg"
        prob_model_img = mpimg.imread(image_path_vae)
        eq_classic_img = mpimg.imread(image_path_eq_classic)
        eq_datacentric_img = mpimg.imread(image_path_eq_datacentric)

        # Display the equations of the classic view in the first column of the third row
        axes[1, 0].imshow(eq_classic_img)
        axes[1, 0].set_title("Dirichlet-Multinomial Likelihood Parameterisation", fontsize=10)
        axes[1, 0].axis('off')

        # Display the probabilistic model image in the second column of the third row
        axes[1, 1].imshow(prob_model_img)
        axes[1, 1].set_title("Probabilistic Model", fontsize=10)
        axes[1, 1].axis('off')

        # Display the equations of the classic view in the third column of the third row
        axes[1, 2].imshow(eq_datacentric_img)
        axes[1, 2].set_title("ZIDM Likelihood Parameterisation", fontsize=10)
        axes[1, 2].axis('off')

        # Hide plots in axes[1,:]
        for ax in axes[1, 1:]:
            ax.axis("off")
        
        """

        if cluster_eval_df is not None:

            umap_axes = [fig.add_subplot(gs[2, col]) for col in range(1, len(likelihood_keys) + 1)]


            # Customize UMAPs with coordinate system
            x_bottom = 0.05
            y_bottom = 0.05
            length = 0.1

            if tissue is not None:
                for i, likelihood_key in enumerate(likelihood_keys):
                    # Set the UMAP coordinates
                    adata.obsm["X_umap"] = adata.obsm[likelihood_key + "_X_umap"].copy()
                    sc.pl.umap(adata[adata.obs.tissue == tissue], color=cell_label_key, title=' ',
                               ax=umap_axes[i],
                               show=False,
                               frameon=False)
                    del adata.obsm["X_umap"]

                # Remove legened
                if len(likelihood_keys) == 3:
                    umap_axes[0].get_legend().remove()
                    umap_axes[1].get_legend().remove()
                else:
                    umap_axes[0].get_legend().remove()

            else:
                for i, likelihood_key in enumerate(likelihood_keys):
                    # Set the UMAP coordinates

                    adata.obsm["X_umap"] = adata.obsm[likelihood_key + "_X_umap"].copy()
                    sc.pl.umap(adata, color=cell_label_key, title=' ', ax=umap_axes[i], show=False,
                               frameon=False)

                    del adata.obsm["X_umap"]

                # Remove legend
                if len(likelihood_keys) == 3:
                    umap_axes[0].get_legend().remove()
                    umap_axes[1].get_legend().remove()
                else:
                    umap_axes[0].get_legend().remove()

            if len(likelihood_keys) == 3:
                umap_axes[0].set_title('e', loc='left', fontsize=20, fontweight='bold')
                umap_axes[1].set_title('f', loc='left', fontsize=20, fontweight='bold')
                umap_axes[2].set_title('g', loc='left', fontsize=20, fontweight='bold')
            else:
                umap_axes[0].set_title('e', loc='left', fontsize=20, fontweight='bold')
                umap_axes[1].set_title('f', loc='left', fontsize=20, fontweight='bold')

            plot_customized_UMAP_coordinates(
                axes=umap_axes, skip_axes=None, x_bottom=x_bottom, y_bottom=y_bottom, length=length
            )

            #evaluation_df = evaluate_embedding_clustering(adata, tissue, likelihood_keys)

            # Display the evaluation metrics as bar chart if predictions are provided
            ax21 = fig.add_subplot(gs[2, 0])
            ax21.set_title('d', loc='left', fontsize=20, fontweight='bold')
            sns.barplot(cluster_eval_df, x='Evaluation score', y='Score value', hue='Likelihood', ax=ax21, palette=custom_greens[1:])
            ax21.set_xlabel("Evaluation score")
            ax21.set_ylabel("Value")
            ax21.tick_params(axis='both')
            ax21.grid(axis='y', alpha=0.75, linestyle='--')
            ax21.get_legend().remove()

        # Adjust layout
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        plt.tight_layout()

        # Save the figure
        if save_fig:
            if zero_inflation_included:
                plt.savefig(
                    f"./figures/tabulaMuris/transcript_usage_data_distribution_{gene_name}_{intron_group_name}_with_zero_inflation.pdf",
                    dpi=300, bbox_inches='tight')
            else:
                plt.savefig(
                    f"./figures/tabulaMuris/transcript_usage_data_distribution_{gene_name}_{intron_group_name}_without_zero_inflation.pdf",
                    dpi=300, bbox_inches='tight')

        plt.show()

def create_histogram_transcript_usage(intron_likelihood: np.ndarray, bin_size: float=0.1) -> Tuple[np.ndarray, List[str]]:
    r"""
    Create histogram bins for the likelihood of introns across cells. The function creates bins of size `bin_size`
    between 0.0 and 1.0, ensuring that the last bin does not exceed 1.0 by more than a small tolerance. It returns the
    proportions of counts in each bin and the corresponding bin labels formatted to two decimal places.

    :param intron_likelihood: np.ndarray, the likelihood of introns across cells
    :param bin_size: float, the size of each bin
    :return: Tuple of proportions of counts in each bin and the corresponding bin labels
    :rtype: Tuple[np.ndarray, List[str]]
    """
    bin_size = 0.1
    bins = np.arange(0.0, 1.0 + bin_size, bin_size)
    if bins[-1] > 1.0 + 1e-9: # Check if the last bin is significantly over 1.0
        bins = bins[:-1] # Remove the last bin if it's too far beyond 1.0
    bins = np.append(bins, 1.0) # Always include 1.0 as the very last edge

    bins = np.unique(np.round(bins, decimals=5))
    bin_labels = []
    for i in range(len(bins) - 1):
        # Format labels to two decimal places for clarity
        lower_bound = f"{bins[i]:.2f}"
        upper_bound = f"{bins[i+1]:.2f}"
        bin_labels.append(f"{lower_bound}-{upper_bound}")

    proportions, _ = np.histogram(intron_likelihood, bins=bins, density=True)

    return proportions, bin_labels


def plot_gene_expression_data_distribution_analysis(
        adata: AnnData,
        gene_name: str,
        likelihood_keys: List[str] = ["NB", "Gaussian"],
        bin_sizes: list = [500, 0.8, 500, 0.8],
        tissue =  None,
        cluster_eval_df: pd.DataFrame = None,
        save_fig: bool = True
) -> None:
    r"""
    Given an AnnData object containing gene expression data both as counts and as expression levels (normalized and
    log(1+x)-transformed counts) and a specified gene, plot the data points sorted according to their magnitude as bar
    chart and plot the underlying distribution of the data as two histograms. The first takes the zero-inflation into
    account and the second discards it. These three charts are plotted for both the counts and the expression levels.
    They are plotted in a 2x3 grid where the first row contains the counts and the second row the expression levels.
    The first column contains the bar chart, the second column the histogram with zero-inflation and the third column
    the histogram without zero-inflation. In the third row the probabilistic model and the equations are plotted. If
    predictions (predictions = True) are provided in the AnnData object. Then, it also contains an additional layer
    "reconstructions" as well as an additional observation "nll", and obsm "latent_mean", and "X_umap". These will then
    be plotted in an additional fourth row. The overall plot containing a 3x3 or 4x3 grid of charts is saved as a
    PDF file.

    :param adata: AnnData object containing the gene expression data
    :param gene_name: str, the name of the gene to be analyzed
    :param likelihood_keys: list of names of likelihood keys to be used for predictions
    :param bin_sizes: list, the size of the bins for the histograms
    :param tissue: None, if provided the UMAP of the latent space of the embeddings of that tissue is shown
    :param save_fig: bool, whether to save the figure as a PDF file
    :return: None
    """

    # Check if the input is an AnnData object
    if not isinstance(adata, AnnData):
        raise ValueError("Input must be an AnnData object")
    if gene_name not in adata.var_names:
        raise ValueError(f"Gene {gene_name} not found in the AnnData object")

    # Check if the AnnData object contains the necessary count layer and the normalized log1p data
    if "counts" not in adata.layers.keys():
        raise ValueError("Counts not found in the AnnData object. Please provide the counts layer.")
    #if "log1p" not in adata.uns.keys():
    #    raise ValueError("Gene expression data must be normalized and log(1+x)-transformed. Please run `sc.pp.normalize_total` and `sc.pp.log1p` first.")

    # Check if provided tissue is contained in AnnData object
    if tissue is not None:
        if tissue not in adata.obs.tissue.to_numpy():
            raise ValueError("Provided tissue type is not contained in data")

    # Check if cluster_eval_df contains matching likelihoods
    if cluster_eval_df is not None:
        for likelihood_key in likelihood_keys:
            if likelihood_key not in cluster_eval_df["Likelihood"].to_numpy():
                raise ValueError(f"Likelihood {likelihood_key} not found in the cluster evaluation dataframe")

    # Check label_key for cell types
    if "cell_ontology_class" in adata.obs.keys():
        cell_label_key = "cell_ontology_class"
    elif "subclass_label" in adata.obs.keys():
        cell_label_key = "subclass_label"
    else:
        raise ValueError(
            "No cell type label found in adata.obs. Please provide a cell type label in adata.obs with key 'cell_ontology_class' or 'subclass_label'")

    # Check if the AnnData object contains predictions and evaluation metrics
    if cluster_eval_df is not None:
        for key in likelihood_keys:
            if key + "_reconstructions" not in adata.layers.keys():
                raise ValueError("AnnData object does not contain reconstructions. Please run predictions first")
            if key + "_nll" not in adata.obs:
                raise ValueError("AnnData object does not contain negative log-likelihoods for cells. Please run predictions first.")
            if key + "_latent_mean" not in adata.obsm:
                raise  ValueError("AnnData object does not contain latent mean embeddings. Please run predictions first.")
            if key + "_X_umap" not in adata.obsm:
                raise ValueError("Anndata object does not contain UMAP embeddings. Please run predictions first.")

    # TO DO: create a loop for reconstructions and originals
    plotting_keys = ["counts", "expression"]
    for likelihood_key in likelihood_keys:
        plotting_keys.append(likelihood_key + "_reconstructions") # "recon_Gaussian" "recon_NB"
    plotting_dict = {}

    # Likelihood keys Gaussian are treated as expression levels, likelihood keys NB and ZINB as counts

    # TO DO adjust such that for Gaussian_reconstructiuons, gene counts (in reality expression levels) are used
    for key in plotting_keys:

        key_dict = {}

        # Sort the data points according to their magnitude for the bar charts and create ordered cell indices
        if (key == "counts") or (key.endswith("_reconstructions") and "NB" in key) or (key.endswith("_reconstructions") and "ZINB" in key):

            gene_counts = adata.layers[key][:, adata.var_names == gene_name]

            # PROBLEM: If data are gene expression levels and not count
            if isinstance(gene_counts, np.ndarray):
                gene_counts = gene_counts.flatten()
            else:
                gene_counts = gene_counts.toarray().flatten()

            # These are the sorted gene counts
            key_dict["Sorted gene counts"] = np.sort(gene_counts)
            num_cells = gene_counts.shape[0]
            key_dict["Cell indices"] = np.linspace(0, num_cells - 1, num_cells)

            # Create bins and labels for the histograms with and without zero-inflation
            key_dict["Gene count bin labels ZI"], key_dict["Gene count bins ZI"] = create_histogram_bins(
                gene_counts,
                count_data_modality=True,
                bin_size=bin_sizes[0],
                proportions=True,
                zero_inflation_included=True
            )

            key_dict["Gene count bin labels"], key_dict["Gene count bins"] = create_histogram_bins(
                gene_counts,
                count_data_modality=True,
                bin_size=bin_sizes[2],
                proportions=True,
                zero_inflation_included=False
            )
        elif (key == "expression") or (key.endswith("_reconstructions") and "Gaussian" in key):
            # These are the original gene expression levels
            if key == "expression":
                gene_expression = adata.X[:, adata.var_names == gene_name].toarray().flatten()
            else:
                gene_expression = adata.layers[key][:, adata.var_names == gene_name].flatten()

            key_dict["Sorted gene expression"] = np.sort(gene_expression)
            num_cells = gene_expression.shape[0]
            key_dict["Cell indices"] = np.linspace(0, num_cells - 1, num_cells)

            key_dict["Gene expression bin labels ZI"], key_dict["Gene expression bins ZI"] = create_histogram_bins(
                gene_expression,
                count_data_modality=False,
                bin_size=bin_sizes[1],
                proportions=True,
                zero_inflation_included=True
            )

            key_dict["Gene expression bin labels"], key_dict["Gene expression bins"] = create_histogram_bins(
                gene_expression,
                count_data_modality=False,
                bin_size=bin_sizes[3],
                proportions=True,
                zero_inflation_included=False
            )

        plotting_dict[key] = key_dict.copy()

    # Create dataframes for plotting
    plotting_df_counts_zi = create_count_reconstruction_dataframe(
        plotting_dict,
        likelihood_keys[0], # Either "NB" or ZINB
        True,
        True
    )
    plotting_df_counts_no_zi = create_count_reconstruction_dataframe(
        plotting_dict,
        likelihood_keys[0], # Either "NB" or ZINB
        False,
        True
    )

    # Later independent of likelihood
    plotting_df_expression_zi = create_count_reconstruction_dataframe(
        plotting_dict,
        likelihood_keys[-1], # "Gaussian"
        True,
        False
    )
    plotting_df_expression_no_zi = create_count_reconstruction_dataframe(
        plotting_dict,
        likelihood_keys[-1], # "Gaussian"
        False,
        False
    )

    # Evaluate embeddings and clustering and store them as evaluation dataframe
    #evaluation_df = evaluate_embedding_clustering(adata, tissue, likelihood_keys)

    # Create a figure with 2x3 subplots
    if cluster_eval_df is not None:
        #fig, axes = plt.subplots(4, 3, figsize=(18, 16), dpi=300)
        fig = plt.figure(figsize=(20, 16), dpi=300)
        gs = gridspec.GridSpec(4, 3, figure=fig)
    else:
        #fig, axes = plt.subplots(3, 3, figsize=(18, 16), dpi=300)
        fig = plt.figure(figsize=(18, 16), dpi=300)
        gs = gridspec.GridSpec(3, 3, figure=fig)
    #fig.suptitle(f"Gene Expression Data Distribution Analysis for {gene_name}", fontsize=12)

    # Gene counts bar chart
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')
    #ax00.set_title(f"Gene Counts Across Sorted Cells", fontsize=10)
    ax00.grid(axis='y', alpha=0.75, linestyle='--')
    ax00.bar(plotting_dict["counts"]["Cell indices"], plotting_dict["counts"]["Sorted gene counts"], color='navy')
    #ax00.set_xlabel("Cell Index (Sorted by Counts)", fontsize=8)
    ax00.set_xlabel("Cells")
    #ax00.set_ylabel("Gene Counts", fontsize=8)
    ax00.set_ylabel("GE counts")
    #ax00.tick_params(axis='both', labelsize=8)
    ax00.tick_params(axis='both')

    # Histogram of gene counts with zero-inflation
    ax01 = fig.add_subplot(gs[0, 1])
    sns.barplot(plotting_df_counts_zi, x='bin_labels', y='proportions', hue="data type", ax=ax01)
    #sns.barplot(x=plotting_dict["counts"]["Gene count bin labels ZI"], y=plotting_dict["counts"]["Gene count bins ZI"], color='blue', ax=axes[0, 1])
    """
    for index, value in enumerate(plotting_df_counts_zi["proportions"].to_numpy()):
        if value.item() > 0.001:
            axes[0, 1].text(index, value + 0.005, f'{value:.1%}', ha='center', va='bottom', fontsize=6)
    """
    ax01.set_title('b', loc='left', fontsize=20, fontweight='bold')
    #ax01.set_title(f'Distribution of Cells Across Counts of Gene', fontsize=10)
    #ax01.set_xlabel('Count Range', fontsize=8)
    ax01.set_xlabel('Count range')
    #ax01.set_ylabel('Proportion of Total Number of Cells', fontsize=8)
    ax01.set_ylabel('Proportion of cells')
    #ax01.tick_params(axis='x', rotation=45, labelsize=8)  # Rotate x-axis labels
    ax01.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    #ax01.tick_params(axis='y', labelsize=8)
    ax01.tick_params(axis='y')
    plt.setp(ax01.get_xticklabels(), ha='right')
    ax01.grid(axis='y', alpha=0.75, linestyle='--')
    ax01.set_ylim(0, np.max(plotting_df_counts_zi["proportions"].to_numpy()) * 1.15)
    ax01.get_legend().remove()

    # Histogram of gene counts without zero-inflation
    ax02 = fig.add_subplot(gs[0, 2])
    sns.barplot(plotting_df_counts_no_zi, x='bin_labels', y='proportions', hue="data type", ax=ax02)
    #sns.barplot(x=plotting_dict["counts"]["Gene count bin labels"], y=plotting_dict["counts"]["Gene count bins"], color='blue', ax=axes[0, 2])
    """
    for index, value in enumerate(plotting_df_counts_no_zi["proportions"].to_numpy()):
        if value.item() > 0.001:
            axes[0, 2].text(index, value + 0.005, f'{value:.1%}', ha='center', va='bottom', fontsize=6)
    """
    ax02.set_title('c', loc='left', fontsize=20, fontweight='bold')
    #ax02.set_title(f'Distribution of Cells Across Counts of Gene', fontsize=10)
    #ax02.set_xlabel('Count Range', fontsize=8)
    ax02.set_xlabel('GE count range')
    #ax02.set_ylabel('Proportion of Total Number of Cells', fontsize=8)
    ax02.set_ylabel('Proportion of cells')
    #ax02.tick_params(axis='x', rotation=45, labelsize=8)  # Rotate x-axis labels
    ax02.tick_params(axis='x', rotation=45)  # Rotate x-axis labels
    #ax02.tick_params(axis='y', labelsize=8)
    ax02.tick_params(axis='y')
    plt.setp(ax02.get_xticklabels(), ha='right')
    ax02.grid(axis='y', alpha=0.75, linestyle='--')
    ax02.set_ylim(0, np.max(plotting_df_counts_no_zi["proportions"].to_numpy()) * 1.15)
    # Move legend out of the plot
    #ax02.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    #ax02.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # In legend rename "Reconstructions" to "Recons"
    handles, labels = ax02.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label == "Reconstructions":
            new_labels.append("Recons")
        else:
            new_labels.append(label)
    ax02.legend(handles, new_labels, loc='upper left', bbox_to_anchor=(1, 1))

    # Gene expression bar chart
    ax10= fig.add_subplot(gs[1, 0])
    ax10.set_title('d', loc='left', fontsize=20, fontweight='bold')
    #ax10.set_title(f"Gene Expression Across Sorted Cells", fontsize=10)
    ax10.grid(axis='y', alpha=0.75, linestyle='--')
    ax10.bar(plotting_dict["expression"]["Cell indices"], plotting_dict["expression"]["Sorted gene expression"], color='cornflowerblue')
    #ax10.set_xlabel("Cell Index (Sorted by Expression)", fontsize=8)
    ax10.set_xlabel("Cells")
    #ax10.set_ylabel("Gene Expression (log(1+x)-transformed)", fontsize=8)
    ax10.set_ylabel("GE levels")
    ax10.tick_params(axis='both')

    # Histogram of gene expression with zero-inflation
    ax11 = fig.add_subplot(gs[1, 1])
    sns.barplot(plotting_df_expression_zi, x='bin_labels', y='proportions', hue="data type", ax=ax11)
    #sns.barplot(x=plotting_df_expression_zi["expression"]["Gene expression bin labels ZI"], y=plotting_df_expression_zi["expression"]["Gene expression bins ZI"], color='orange', ax=axes[1, 1])
    """
    for index, value in enumerate(plotting_df_expression_zi["proportions"].to_numpy()):
        if value.item() > 0.001:
            axes[1, 1].text(index, value + 0.005, f'{value:.1%}', ha='center', va='bottom', fontsize=6)
    """
    ax11.set_title('e', loc='left', fontsize=20, fontweight='bold')
    #ax11.set_title(f"Distribution of Cells Across Gene Expression (log(1+x)-transformed)", fontsize=10)
    #ax11.set_xlabel('Gene Expression (log(1+x)-transformed) Range', fontsize=8)
    ax11.set_xlabel('GE levels range')
    #ax11.set_ylabel('Proportion of Total Number of Cells', fontsize=8)
    ax11.set_ylabel('Proportion of cells')
    #ax11.tick_params(axis='x', rotation=45, labelsize=8)
    ax11.tick_params(axis='x', rotation=45)
    #ax11.tick_params(axis='y', labelsize=8)
    ax11.tick_params(axis='y')
    plt.setp(ax11.get_xticklabels(), ha='right')
    ax11.grid(axis='y', alpha=0.75, linestyle='--')
    ax11.set_ylim(0, np.max(plotting_df_expression_zi["proportions"].to_numpy()) * 1.15)
    ax11.get_legend().remove()

    # Histogram of gene expression
    ax12 = fig.add_subplot(gs[1, 2])
    sns.barplot(plotting_df_expression_no_zi, x='bin_labels', y='proportions', hue="data type", ax=ax12)
    #sns.barplot(x=plotting_dict["counts"]["Gene expression bin labels"], y=plotting_dict["counts"]["Gene expression bins"], color='orange', ax=axes[1, 2])
    """
    for index, value in enumerate(plotting_df_expression_no_zi["proportions"].to_numpy()):
        if value.item() > 0.001:
            axes[1, 2].text(index, value + 0.005, f'{value:.1%}', ha='center', va='bottom', fontsize=6)
    """
    ax12.set_title('f', loc='left', fontsize=20, fontweight='bold')
    #ax12.set_title(f"Distribution of Cells Across Gene Expression (log(1+x)-transformed)", fontsize=10)
    #ax12.set_xlabel('Gene Expression (log(1+x)-transformed) Range', fontsize=8)
    ax12.set_xlabel('GE levels range')
    #ax12.set_ylabel('Proportion of Total Number of Cells', fontsize=8)
    ax12.set_ylabel('Proportion of cells')
    #ax12.tick_params(axis='x', rotation=45, labelsize=8)
    ax12.tick_params(axis='x', rotation=45)
    #ax12.tick_params(axis='y', labelsize=8)
    ax12.tick_params(axis='y')
    plt.setp(ax12.get_xticklabels(), ha='right')
    ax12.grid(axis='y', alpha=0.75, linestyle='--')
    ax12.set_ylim(0, np.max(plotting_df_expression_no_zi["proportions"]) * 1.15)
    #ax12.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    #ax12.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # In legend rename "Reconstructions" to "Recons"
    handles, labels = ax12.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label == "Reconstructions":
            new_labels.append("Recons")
        else:
            new_labels.append(label)
    ax12.legend(handles, new_labels, loc='upper left', bbox_to_anchor=(1, 1))

    ax2022_full_row = fig.add_subplot(gs[2, :])
    image_path = "./figures/crecerelle_scGEVI.png"
    prob_model_img = mpimg.imread(image_path)
    ax2022_full_row.imshow(prob_model_img)
    ax2022_full_row.axis("off")
    ax2022_full_row.set_title('g', loc='left', fontsize=20, fontweight='bold')

    """
    # Load images to display in third row
    image_path_vae = "./figures/vae_generative_model_ge.jpg"
    image_path_eq_classic = "./figures/vae_generative_model_eq_classic.png"
    image_path_eq_datacentric = "./figures/vae_generative_model_eq_datacentric.png"
    prob_model_img = mpimg.imread(image_path_vae)
    eq_classic_img = mpimg.imread(image_path_eq_classic)
    eq_datacentric_img = mpimg.imread(image_path_eq_datacentric)

    # Display the equations of the classic view in the third column of the third row
    axes[2, 0].imshow(eq_datacentric_img)
    axes[2, 0].set_title("ZINB Likelihood Parameterisation", fontsize=10)
    axes[2, 0].axis('off')

    # Display the probabilistic model image in the second column of the third row
    axes[2, 1].imshow(prob_model_img)
    axes[2, 1].set_title("Probabilistic Model", fontsize=10)
    axes[2, 1].axis('off')

    # Display the equations of the classic view in the first column of the third row
    axes[2, 2].imshow(eq_classic_img)
    axes[2, 2].set_title("Gaussian Likelihood Parameterisation", fontsize=10)
    axes[2, 2].axis('off')
    """

    # Display the evaluation metrics as bar chart if predictions are provided
    if cluster_eval_df is not None:
        ax31 = fig.add_subplot(gs[3, 1])
        ax31.set_title('i', loc='left', fontsize=20, fontweight='bold')
        #ax31.set_title("Embedding and Clustering Evaluation", fontsize=10)
        sns.barplot(cluster_eval_df, x='Evaluation score', y='Score value', hue='Likelihood', ax=ax31)
        #ax31.set_xlabel("Evaluation score", fontsize=8)
        ax31.set_xlabel("Evaluation score")
        #ax31.set_ylabel("Score value", fontsize=8)
        ax31.set_ylabel("Value")
        #ax31.tick_params(axis='both', labelsize=8)
        ax31.tick_params(axis='both')
        ax31.grid(axis='y', alpha=0.75, linestyle='--')
        #ax31.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax31.legend(loc='lower right')
    else:
        ax31 = fig.add_subplot(gs[3, 1])
        ax31.axis('off')

    umap_axes = [fig.add_subplot(gs[3, col]) for col in [0, 2]]


    # Customize UMAPs with coordinate system
    x_bottom = 0.05
    y_bottom = 0.05
    length = 0.1

    if tissue is not None:
        for i, likelihood_key in enumerate(likelihood_keys):
            # Set the UMAP coordinates
            adata.obsm["X_umap"] = adata.obsm[likelihood_key + "_X_umap"].copy()
            sc.pl.umap(adata[adata.obs.tissue == tissue], color=cell_label_key, title=' ', ax=umap_axes[i],
                       show=False,
                       frameon=False)
            del adata.obsm["X_umap"]
            if i==0:
                # Remove legend from first UMAP
                umap_axes[i].get_legend().remove()
    else:
        for i, likelihood_key in enumerate(likelihood_keys):
            # Set the UMAP coordinates

            adata.obsm["X_umap"] = adata.obsm[likelihood_key + "_X_umap"].copy()
            sc.pl.umap(adata, color=cell_label_key, title=' ', ax=umap_axes[i], show=False, frameon=False)
            del adata.obsm["X_umap"]
            if i==0:
                # Remove legend from first UMAP
                umap_axes[i].get_legend().remove()

    plot_customized_UMAP_coordinates(
        axes=umap_axes, skip_axes=None, x_bottom=x_bottom, y_bottom=y_bottom, length=length
    )

    umap_axes[0].set_title('h', loc='left', fontsize=20, fontweight='bold')
    umap_axes[1].set_title('j', loc='left', fontsize=20, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.5, wspace=0.4)

    if save_fig:
        plt.savefig(f"./figures/tabulaMuris/gene_expression_data_distribution_analysis_{gene_name}.pdf", dpi=300, bbox_inches='tight')

    plt.show()

def create_count_reconstruction_dataframe(plotting_dict: Dict[str, Dict[str, np.ndarray]], likelihood_key: str,
                                          zero_inflation: bool = True, count_data: bool = True)-> pd.DataFrame:
    r"""
    Given a plotting dictionary with counts and reconstructions, create a dataframe for plotting. The dataframe contains
    the bin labels, proportions, and data type (counts or reconstructions). The bin labels are updated to ensure that the
    to ensure consistency between the last bin labels of counts and reconstructions. If zero_inflation is True, the
    bin labels and proportions are taken from the zero-inflated counts and reconstructions, otherwise from the non-zero
    inflated counts and reconstructions. The dataframe is then returned.

    :param plotting_dict: Dict[str, Dict[str, np.ndarray]], a dictionary containing the counts and reconstructions
    :param likelihood_key: str, the key for the likelihood (e.g., "Gaussian", "NB")
    :param zero_inflation: bool, whether to use zero-inflated counts and reconstructions
    :param count_data: bool, whether the data is count data (default is True)
    :return: pd.DataFrame, a dataframe containing the bin labels, proportions, and data type
    """

    if zero_inflation:
        # Given a plotting dictionary with counts and reconstructions, create a individual dataframes
        if count_data:
            data_df = pd.DataFrame({
                "bin_labels": plotting_dict["counts"]["Gene count bin labels ZI"],
                "proportions": plotting_dict["counts"]["Gene count bins ZI"],
                "data type": "Data"
            })
            recon_df = pd.DataFrame({
                "bin_labels": plotting_dict[likelihood_key + "_reconstructions"]["Gene count bin labels ZI"],
                "proportions": plotting_dict[likelihood_key + "_reconstructions"]["Gene count bins ZI"],
                "data type": "Reconstructions"
            })
        else:
            data_df = pd.DataFrame({
                "bin_labels": plotting_dict["expression"]["Gene expression bin labels ZI"],
                "proportions": plotting_dict["expression"]["Gene expression bins ZI"],
                "data type": "Data"
            })
            recon_df = pd.DataFrame({
                "bin_labels": plotting_dict[likelihood_key + "_reconstructions"]["Gene expression bin labels ZI"],
                "proportions": plotting_dict[likelihood_key + "_reconstructions"]["Gene expression bins ZI"],
                "data type": "Reconstructions"
            })
    else:
        # Given a plotting dictionary with counts and reconstructions, create a individual dataframes
        if count_data:
            data_df = pd.DataFrame({
                "bin_labels": plotting_dict["counts"]["Gene count bin labels"],
                "proportions": plotting_dict["counts"]["Gene count bins"],
                "data type": "Data"
            })
            recon_df = pd.DataFrame({
                "bin_labels": plotting_dict[likelihood_key + "_reconstructions"]["Gene count bin labels"],
                "proportions": plotting_dict[likelihood_key + "_reconstructions"]["Gene count bins"],
                "data type": "Reconstructions"
            })
        else:
            data_df = pd.DataFrame({
                "bin_labels": plotting_dict["expression"]["Gene expression bin labels"],
                "proportions": plotting_dict["expression"]["Gene expression bins"],
                "data type": "Data"
            })

            recon_df = pd.DataFrame({
                "bin_labels": plotting_dict[likelihood_key + "_reconstructions"]["Gene expression bin labels"],
                "proportions": plotting_dict[likelihood_key + "_reconstructions"]["Gene expression bins"],
                "data type": "Reconstructions"
            })

    # Merge the dataframes based on the bin labels
    if count_data:
        data_df_bin_labels = np.array([label.split("-") for label in data_df["bin_labels"].to_numpy()]).astype(int)
        recon_df_bin_labels = np.array([label.split("-") for label in recon_df["bin_labels"].to_numpy()]).astype(int)
    else:
        data_df_bin_labels = np.array([label.split("-") for label in data_df["bin_labels"].to_numpy()]).astype(float)
        recon_df_bin_labels = np.array([label.split("-") for label in recon_df["bin_labels"].to_numpy()]).astype(float)

    data_df_last_bin_label = data_df_bin_labels[-1]
    data_df_last_bin_label_upper = data_df_last_bin_label[-1]
    data_df_last_bin_label_lower = data_df_last_bin_label[0]

    recon_df_last_bin_label = recon_df_bin_labels[-1]
    recon_df_last_bin_label_upper = recon_df_last_bin_label[-1]
    recon_df_last_bin_label_lower = recon_df_last_bin_label[0]

    if recon_df_last_bin_label_upper > data_df_last_bin_label_upper:

        updated_bin_label_idx = np.argwhere(data_df_last_bin_label_lower == recon_df_bin_labels[:, 0])
        if len(updated_bin_label_idx.shape) > 1:
            updated_bin_label_idx = updated_bin_label_idx.flatten()[-1]
        updated_bin_label = str(data_df_last_bin_label_lower) + "-" + str(
            recon_df_bin_labels[updated_bin_label_idx, 1].item())
        data_df.loc[len(data_df) -1,  "bin_labels"] = updated_bin_label
    else:
        updated_bin_label_idx = np.argwhere(recon_df_last_bin_label_lower == data_df_bin_labels[:, 0])
        if len(updated_bin_label_idx.shape) > 1:
            updated_bin_label_idx = updated_bin_label_idx.flatten()[-1]
        updated_bin_label = str(recon_df_last_bin_label_lower) + "-" + str(
            data_df_bin_labels[updated_bin_label_idx, 1].item())
        recon_df.loc[len(recon_df) - 1, "bin_labels"] = updated_bin_label

    # Concatenate the dataframes
    plotting_df = pd.concat([data_df, recon_df])

    return plotting_df

def get_cell_type_colour(cell_type, adata, cell_label_key) -> str:
    r"""
    Extracts the colour for a given cell type from the AnnData object

    :param cell_type: str, the cell type for which the colour is to be extracted
    :param adata: AnnData object containing the cell type information
    :param cell_label_key: str, the key in adata.obs where the cell types are stored
    :return: str, the colour for the given cell type
    """
    # Get the ordered categories from the AnnData object's observation data
    categories = adata.obs[cell_label_key].cat.categories

    # Get the ordered color list from the AnnData object's unstructured data
    colors = adata.uns[f'{cell_label_key}_colors']

    # Create a dictionary mapping categories to colors
    color_map = dict(zip(categories, colors))

    # Return the color for the specified cell type
    return color_map.get(cell_type, 'black')

def plot_latent_space_evaluation_trvi(
        adata: Tuple[AnnData, AnnData],
        evaluation_df: pd.DataFrame,
        likelihoods: List[str] = ["ZINB", "ZIDM"],
        tissue: str = "Heart",
        dataset_name: str = "tabulaMuris",
        save_fig: bool = True
) -> None:

    # Check label_key for cell types
    if "cell_ontology_class" in adata[0].obs.keys():
        cell_label_key = "cell_ontology_class"
    elif "subclass_label" in adata[0].obs.keys():
        cell_label_key = "subclass_label"
    else:
        raise ValueError(
            "No cell type label found in adata.obs. Please provide a cell type label in adata.obs with key 'cell_ontology_class' or 'subclass_label'")

    # Unpack adata tuple
    adata_1, adata_2 = adata # gene expression (adata_1) and transcript usage (adata_2)

    if tissue is not None:
        adata_1 = adata_1[adata_1.obs.tissue == tissue].copy()
        adata_2 = adata_2[adata_2.obs.tissue == tissue].copy()

    # Create plot
    fig, axes = plt.subplots(4, 3, figsize=(16, 18), dpi=300)

    # Load images to display model
    image_path_trvi_shared_private = "./figures/crecerelle_trvi_shared_private_latent.png"
    shared_private_latent_img = mpimg.imread(image_path_trvi_shared_private)
    image_path_trvi_shared_mixing = "./figures/crecerelle_trvi_shared_latent_mixing.png"
    trvi_shared_mixing_img = mpimg.imread(image_path_trvi_shared_mixing)

    # Display the probabilistic model of TRVI
    axes[0, 0].imshow(shared_private_latent_img)
    axes[0, 0].axis('off')

    # Display the probabilistic model of TRVI
    axes[0, 1].imshow(trvi_shared_mixing_img)
    axes[0, 1].axis('off')

    # UMAP 1 (gene expression private)
    adata_1.obsm["X_umap"] = adata_1.obsm[likelihoods[0] + "_private_X_umap"].copy()
    sc.pl.umap(adata_1, color=cell_label_key, title=' ', legend_loc=None,
               ax=axes[1, 0], show=False, frameon=False)
    del adata_1.obsm["X_umap"]

    # UMAP 1 (gene expression shared)
    adata_1.obsm["X_umap"] = adata_1.obsm[likelihoods[0] + "_shared_X_umap"].copy()
    sc.pl.umap(adata_1, color=cell_label_key, title=' ', legend_loc=None,
               ax=axes[2, 0], show=False, frameon=False)
    del adata_1.obsm["X_umap"]

    # UMAP 3 (shared)
    adata_1.obsm["X_umap"] = adata_1.obsm["shared_X_umap"].copy()
    sc.pl.umap(adata_1, color=cell_label_key, title=' ', legend_loc=None,
               ax=axes[1, 1], show=False, frameon=False)
    del adata_1.obsm["X_umap"]

    # UMAP 2 (transcript usage private)
    adata_2.obsm["X_umap"] = adata_2.obsm[likelihoods[1] + "_private_X_umap"].copy()
    sc.pl.umap(adata_2, color=cell_label_key, title=' ', ax=axes[1, 2],
               show=False, frameon=False)
    del adata_2.obsm["X_umap"]

    # UMAP 2 (transcript usage shared)
    adata_2.obsm["X_umap"] = adata_2.obsm[likelihoods[1] + "_shared_X_umap"].copy()
    sc.pl.umap(adata_2, color=cell_label_key, title=' ', legend_loc=None, ax=axes[2, 1],
               show=False, frameon=False)
    del adata_2.obsm["X_umap"]

    # Clustering and embeddings evaluation
    axes[0, 2].set_title(" ") # Embedding and Clustering Evaluation
    sns.barplot(evaluation_df, x='Evaluation score', y='Score value', hue='Likelihood', ax=axes[0, 2])
    #axes[0, 2].set_xlabel("Evaluation score")
    #axes[0, 2].set_ylabel("Score value")
    axes[0, 2].set_xlabel(" ")
    axes[0, 2].set_ylabel(" ")
    axes[0, 2].tick_params(axis='both')
    axes[0, 2].grid(axis='y', alpha=0.75, linestyle='--')
    # Move legend out of the plot
    axes[0, 2].legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Distance matrix of embeddings
    embeddings_list = ["ZINB", "shared", "ZIDM"]

    cell_types = adata_1.obs[cell_label_key].to_numpy()
    cell_embeddings_1 = adata_1.obsm["ZINB_private_latent_mean"]
    cell_embeddings_2 = adata_2.obsm["ZIDM_private_latent_mean"]
    cell_embeddings_shared = adata_1.obsm["shared_latent_mean"]

    cell_embeddings = [cell_embeddings_1, cell_embeddings_shared, cell_embeddings_2]

    for embedding_idx in range(0, len(cell_embeddings)):
        dist_matrix, ordered_cell_types = distance_matrix(cell_embeddings[embedding_idx], cell_types)

        tick_positions = []
        tick_labels = []
        current_type = None

        for i, cell_type in enumerate(ordered_cell_types):
            if cell_type != current_type:
                # This is the start of a new cell type block
                tick_positions.append(i)
                tick_labels.append(cell_type)
                current_type = cell_type
        # Plot heatmap of distance matrix with color bar but without a color bar label
        sns.heatmap(dist_matrix, cmap='viridis',
                    ax=axes[3, embedding_idx]) #, cbar_kws={'label': 'Euclidean Distance'})
        # Hide ticks and tick labels of both axes
        axes[3, embedding_idx].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[3, embedding_idx].set_title(' ') # Euclidean Distance Matrix of Ordered Cell Embeddings
        axes[3, embedding_idx].set_xlabel('Ordered Cells')
        axes[3, embedding_idx].set_ylabel('Ordered Cells')

        # Set ticks and labels for both x and y axes
        # axes[1, embedding_idx+1].set_xticks(tick_positions, tick_labels, rotation=90, labelsize=4) # Rotate labels if they overlap
        # axes[1, embedding_idx+1].set_yticks(tick_positions, tick_labels, rotation=0, labelsize=4)

        # Add lines to visually separate the blocks (optional, but very helpful)
        for pos in tick_positions[1:]:  # Start from the second position to avoid a line at 0
            axes[3, embedding_idx].axvline(x=pos, color='red', linestyle='--', linewidth=1)
            axes[3, embedding_idx].axhline(y=pos, color='red', linestyle='--', linewidth=1)

    # TO DO add the modality weighting plot
    cell_types = adata_1.obs[cell_label_key].to_numpy()
    weightings_1 = adata_1.obs['weighting'].to_numpy()
    weightings_2 = adata_2.obs['weighting'].to_numpy()

    unique_cell_types = np.sort(np.unique(cell_types))
    num_cell_types = len(unique_cell_types)

    ordered_weightings_1 = []
    ordered_weightings_2 = []
    ordered_cell_types = []

    for cell_type in unique_cell_types:
        # Get the indices where the current cell_type matches
        indices = np.where(cell_types == cell_type)[0]
        # Append the embeddings for this cell type
        ordered_weightings_1.append(weightings_1[indices].reshape(-1, 1))
        ordered_weightings_2.append(weightings_2[indices].reshape(-1, 1))
        # Append the corresponding cell type labels
        ordered_cell_types.extend([cell_type] * len(indices))

    ordered_weightings_1 = np.vstack(ordered_weightings_1).squeeze()
    ordered_weightings_2 = np.vstack(ordered_weightings_2).squeeze()
    ordered_cell_types = np.array(ordered_cell_types)
    ordered_cell_types = np.concatenate((ordered_cell_types, ordered_cell_types))
    weighting = np.concatenate((ordered_weightings_1, ordered_weightings_2))
    # array of len(ordered_weightings_1) where each element contains string "gene expression"
    modality = np.array(
        ["GE"] * len(ordered_weightings_1) + ["TU"] * len(ordered_weightings_2))

    weighting_dict = {
        'cell type': ordered_cell_types,
        'weighting': weighting,
        'modality': modality
    }

    sns.barplot(data=pd.DataFrame(weighting_dict), x='cell type', y='weighting', hue='modality', ax=axes[2, 2])

    # Instead of plotting the x-tick labels plot just colored circles for each cell type (use the same colors as in the UMAPs)
    # Get the current x-tick labels
    xtick_labels = axes[2, 2].get_xticklabels()
    # Replace the x-tick labels with colored circles
    for i, label in enumerate(xtick_labels):
        cell_type = label.get_text()
        color = get_cell_type_colour(cell_type, adata_1, cell_label_key)
        label.set_text('●')  # Unicode character for a filled circle
        label.set_color(color)
        label.set_fontsize(14)  # Increase the size of the circle

    axes[2, 2].set_xticklabels(xtick_labels)

    axes[2, 2].set_title(' ') #Weighting of Modalities
    # Move legend out of the plot
    axes[2, 2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    #axes[1, 3].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=90)
    axes[2, 2].set_ylabel(' ')

    # Customize UMAPs with coordinate system
    x_bottom = 0.05
    y_bottom = 0.05
    length = 0.1
    skip_axes = [0, 1, 2, 8, 9, 10, 11]
    plot_customized_UMAP_coordinates(
        axes=axes, skip_axes=skip_axes, x_bottom=x_bottom, y_bottom=y_bottom, length=length, font_size=18
    )

    # Figure numbering
    axes[0, 0].set_title('a', loc='left', fontsize=24, fontweight='bold', pad=53)
    axes[0, 1].set_title('b', loc='left', fontsize=24, fontweight='bold')
    axes[0, 2].set_title('c', loc='left', fontsize=24, fontweight='bold')
    axes[1, 0].set_title('d', loc='left', fontsize=24, fontweight='bold')
    axes[1, 1].set_title('e', loc='left', fontsize=24, fontweight='bold')
    axes[1, 2].set_title('f', loc='left', fontsize=24, fontweight='bold')
    axes[2, 0].set_title('g', loc='left', fontsize=24, fontweight='bold')
    axes[2, 1].set_title('h', loc='left', fontsize=24, fontweight='bold')
    axes[2, 2].set_title('i', loc='left', fontsize=24, fontweight='bold')
    axes[3, 0].set_title('j', loc='left', fontsize=24, fontweight='bold')
    axes[3, 1].set_title('k', loc='left', fontsize=24, fontweight='bold')
    axes[3, 2].set_title('l', loc='left', fontsize=24, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    if save_fig:
        # plt.savefig(f"./figures/tabulaMuris/trvi_analysis_"+ tissue +".pdf", dpi=300, bbox_inches='tight')
        if tissue is None:
            plt.savefig(f"./figures/" + dataset_name + "/trvi_analysis.png", bbox_inches='tight')
        else:
            plt.savefig(f"./figures/" + dataset_name + "/trvi_analysis_" + tissue + ".png", bbox_inches='tight')

    plt.show()

def plot_cell_type_predictions_trvi_marker_genes(
        adata_full: AnnData,
        adata_eval: AnnData,
        tissue: str,
        cell_types: Tuple[str, str],
        pred_cell_types: Tuple[np.ndarray, np.ndarray],
        classifiers: Tuple[str, str],
        classification_dict: Dict[str, Dict[str, Dict[str, float]]],
        embedding_types: List[str],
        save_fig: bool = True,
) -> None:
    r"""
    """
    cell_type_1, cell_type_2 = cell_types
    classifier_1, classifier_2 = classifiers
    pred_cell_types_1, pred_cell_types_2 = pred_cell_types

    assert classifier_1 in ["private_1", "private_2","shared_uni_1", "shared_uni_2", "shared"], "Classifier 1 must use one of the following TRVI embeddings: 'private_1', 'private_2', 'shared_uni_1', 'shared_uni_2', 'shared'"
    assert classifier_2 in ["private_1", "private_2","shared_uni_1", "shared_uni_2", "shared"], "Classifier 2 must use one of the following TRVI embeddings: 'private_1', 'private_2', 'shared_uni_1', 'shared_uni_2', 'shared'"

    # Create dataframes (overall, classifer 1, classifer 2) for plotting
    classification_overall_df = cell_type_classification_trvi_dataframe(classification_dict, embedding_types)
    classification_cell_type_1_df = cell_classification_dataframe(cell_type_1, classifier_1, classification_dict)
    classification_cell_type_2_df = cell_classification_dataframe(cell_type_2, classifier_2, classification_dict)

    # Add gene marker classifiers to embedding_type
    embedding_types_for_prediction = embedding_types.copy()
    embedding_types_for_prediction.append("marker_gene_" + cell_type_1.replace(" ", "_"))
    embedding_types_for_prediction.append("marker_gene_" + cell_type_2.replace(" ", "_"))

    # Add cell type predictions of classifiers to AnnDatat object
    for emb_type in embedding_types_for_prediction:
        adata_full.obs[emb_type + "_pred_cell_ontology_class"] = "unpredicted"

        if emb_type == ("marker_gene_" + cell_type_1.replace(" ", "_")):
            filtered_preds = pred_cell_types_1  # From marker gene classifier
        elif emb_type == ("marker_gene_" + cell_type_2.replace(" ", "_")):
            filtered_preds = pred_cell_types_2  # From marker gene classifier
        else:
            preds_source_column = adata_eval.obs[emb_type + "_pred_cell_ontology_class"]
            # The predictions for the other embedding types are already in adata_full.obs
            filtered_preds = preds_source_column.to_numpy()

        mask = (adata_full.obs["data_partition"] == "test") & (adata_full.obs["tissue"] == tissue)
        adata_full.obs.loc[mask, emb_type + "_pred_cell_ontology_class"] = filtered_preds

        adata_full.obs[emb_type + "_pred_cell_ontology_class"] = adata_full.obs[
            emb_type + "_pred_cell_ontology_class"].astype('category')

    # Custom color palette for cell types
    all_ground_truth_classes = adata_full[adata_full.obs["tissue"] == tissue].obs[
        'cell_ontology_class'].cat.categories.tolist()
    all_categories_for_palette = sorted(list(set(all_ground_truth_classes)))
    all_categories_for_palette.append("unpredicted")

    num_cell_types = len(all_categories_for_palette) - 1

    if num_cell_types > 0:
        cmap = plt.cm.get_cmap('tab20', num_cell_types)
        # Generate colors for the actual cell types
        colors_for_types = [mcolors.rgb2hex(cmap(i)) for i in range(num_cell_types)]
    else:
        colors_for_types = []

    custom_palette = {}
    color_idx = 0
    for category in all_categories_for_palette:
        if category == 'unpredicted':
            custom_palette[category] = 'gainsboro'  # Assign grey to unpredicted
        else:
            # Assign colors from the generated list to cell types
            if color_idx < len(colors_for_types):
                custom_palette[category] = colors_for_types[color_idx]
                color_idx += 1
            else:
                # Fallback if somehow ran out of colors (shouldn't happen with robust cmap)
                custom_palette[category] = 'black'

    # Filter adata for the specified tissue
    adata_full.obsm["X_umap"] = adata_full.obsm["shared_X_umap"].copy()
    adata_tissue = adata_full[adata_full.obs.tissue == tissue].copy()

    adata_tissue.obs['plot_prediction_status'] = adata_tissue.obs['shared_pred_cell_ontology_class'].fillna(
        'unpredicted')
    adata_tissue.obs['plot_prediction_status'] = adata_tissue.obs['shared_pred_cell_ontology_class'].fillna(
        'unpredicted')

    # Plotting
    fig, axes = plt.subplots(3, 4, figsize=(18, 14), dpi=300)

    # Bar chart of classification metrics
    sns.barplot(classification_overall_df, x="Classification Metric", y="Value", hue="Embedding", ax=axes[0, 0])
    axes[0, 0].set_title('a', loc='left', fontsize=14, fontweight='bold')

    # UMAP train test
    sc.pl.umap(
        adata_tissue,
        color='data_partition',  # Our prepared column for predictions
        title=f" ", #f'Training and Test Data'
        ax=axes[0, 1],
        show=False,
        frameon=False,
    )
    axes[0, 1].set_title('b', loc='left', fontsize=14, fontweight='bold')

    # UMAP predictions
    sc.pl.umap(
        adata_tissue,
        color='plot_prediction_status',  # Our prepared column for predictions
        title=f" ", #'UMAP of {tissue} Cells (Shared Predictions)'
        palette=custom_palette,  # Use the consistent palette
        legend_loc=None,
        ax=axes[0, 2],
        show=False,
        frameon=False,
    )

    # UMAP groundtruth
    sc.pl.umap(
        adata_tissue,
        color='cell_ontology_class',  # Ground truth
        title=f" ", #'UMAP of {tissue} Cells (Ground Truth)',
        palette=custom_palette,  # Use the consistent palette
        ax=axes[0, 3],
        show=False,
        frameon=False
    )

    del adata_full.obsm["X_umap"]
    del adata_tissue.obs['plot_prediction_status']  # Clean up temporary column

    # Customize UMAPs with coordinate system
    x_bottom = 0.05
    y_bottom = 0.05
    length = 0.1
    skip_axes = [0, 4, 7, 8, 11]
    plot_customized_UMAP_coordinates(
        axes=axes, skip_axes=skip_axes, x_bottom=x_bottom, y_bottom=y_bottom, length=length
    )

    embedding_types.append("marker_genes")

    # ROC curves focus gene expression
    for embedding_type in embedding_types:
        fpr = classification_dict[embedding_type]["roc_auc"][cell_type_1]["fpr"]
        tpr = classification_dict[embedding_type]["roc_auc"][cell_type_1]["tpr"]
        auroc = classification_dict[embedding_type]["roc_auc"][cell_type_1]["auroc"]

        axes[1, 0].plot(fpr, tpr, lw=2, label=embedding_type + ' (area = %0.2f)' % auroc)
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].set_title(" ") #"ROC Curve for " + cell_type_1
    axes[1, 0].grid(True)

    # UMAP of predictions of cell type 1 with chosen embedding-cell type classifier on test set
    column_name = classifier_1 + "_pred_cell_ontology_class"
    cell_type_1_pred_plotting = adata_tissue.obs[column_name].to_numpy()
    mask = ((adata_tissue.obs[column_name] == cell_type_1) | (
                adata_tissue.obs[column_name] == "unpredicted")).to_numpy()
    cell_type_1_pred_plotting[~mask] = "other"
    adata_tissue.obs[column_name + "_plotting"] = cell_type_1_pred_plotting

    sc.pl.umap(
        adata_tissue,
        color=column_name + "_plotting",
        title=f" " ,#f'UMAP of {tissue} Cells',
        legend_loc=None,
        ax=axes[1, 1],
        show=False,
        frameon=False
    )

    del adata_tissue.obs[column_name + "_plotting"]

    # UMAP of predictions of cell type 1 with marker gene-cell type classifier on test set
    sc.pl.umap(
        adata_tissue,
        color="marker_gene_" + cell_type_1.replace(" ", "_") + "_pred_cell_ontology_class",
        title=f" ", #f'UMAP of {tissue} Cells',
        legend_loc=None,
        ax=axes[1, 2],
        show=False,
        frameon=False
    )

    # Cell specific classification scores
    sns.barplot(classification_cell_type_1_df, x="Classification Metric", y="Value", hue="Embedding", ax=axes[1, 3])
    #axes[1, 3].set_title("**h**")

    # ROC curves focus transcript usage
    for embedding_type in embedding_types:
        fpr = classification_dict[embedding_type]["roc_auc"][cell_type_2]["fpr"]
        tpr = classification_dict[embedding_type]["roc_auc"][cell_type_2]["tpr"]
        auroc = classification_dict[embedding_type]["roc_auc"][cell_type_2]["auroc"]

        axes[2, 0].plot(fpr, tpr, lw=2, label=embedding_type + ' (area = %0.2f)' % auroc)
    axes[2, 0].legend(loc="lower right")
    axes[2, 0].set_xlabel("False Positive Rate")
    axes[2, 0].set_ylabel("True Positive Rate")
    axes[2, 0].set_title(" ") #"ROC Curve for " + cell_type_2
    axes[2, 0].grid(True)

    # UMAP of predictions of cell type 2 with chosen embedding-cell type classifier on test set
    column_name = classifier_2 + "_pred_cell_ontology_class"
    cell_type_2_pred_plotting = adata_tissue.obs[column_name].to_numpy()
    mask = ((adata_tissue.obs[column_name] == cell_type_2) | (
                adata_tissue.obs[column_name] == "unpredicted")).to_numpy()
    cell_type_2_pred_plotting[~mask] = "other"
    adata_tissue.obs[column_name + "_plotting"] = cell_type_2_pred_plotting

    sc.pl.umap(
        adata_tissue,
        color=column_name + "_plotting",
        title=" ",#f'UMAP of {tissue} Cells',
        legend_loc=None,
        ax=axes[2, 1],
        show=False,
        frameon=False
    )

    del adata_tissue.obs[column_name + "_plotting"]

    # UMAP of predictions of cell type 2 with marker gene-cell type classifier on test set
    sc.pl.umap(
        adata_tissue,
        color="marker_gene_" + cell_type_2.replace(" ", "_") + "_pred_cell_ontology_class",  # Ground truth
        title=" ",#f'UMAP of {tissue} Cells',
        legend_loc=None,
        ax=axes[2, 2],
        show=False,
        frameon=False
    )

    # Cell specific classification scores
    sns.barplot(classification_cell_type_2_df, x="Classification Metric", y="Value", hue="Embedding", ax=axes[2, 3])
    axes[2, 3].set_title(" ")

    # Figure enumeration
    axes[0, 0].set_title('a', loc='left', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('b', loc='left', fontsize=14, fontweight='bold')
    axes[0, 2].set_title('c', loc='left', fontsize=14, fontweight='bold')
    axes[0, 3].set_title('d', loc='left', fontsize=14, fontweight='bold')
    axes[1, 0].set_title('e', loc='left', fontsize=14, fontweight='bold')
    axes[1, 1].set_title('f', loc='left', fontsize=14, fontweight='bold')
    axes[1, 2].set_title('g', loc='left', fontsize=14, fontweight='bold')
    axes[1, 3].set_title('h', loc='left', fontsize=14, fontweight='bold')
    axes[2, 0].set_title('i', loc='left', fontsize=14, fontweight='bold')
    axes[2, 1].set_title('j', loc='left', fontsize=14, fontweight='bold')
    axes[2, 2].set_title('k', loc='left', fontsize=14, fontweight='bold')
    axes[2, 3].set_title('l', loc='left', fontsize=14, fontweight='bold')

    # Adjust layout
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"./figures/tabulaMuris/trvi_cell_type_predictions_{tissue}_{cell_type_1}_{cell_type_2}.pdf", dpi=300, bbox_inches='tight')

    plt.show()

def plot_roc_auc_curve(
        classifier_evaluation_dict: Dict[str, Dict[str, np.ndarray]],
        cell_type: str,
        embedding_types: List[str]=["private_1", "private_2", "shared_uni_1", "shared_uni_2", "shared"],
        save_fig: bool = True,
) -> None:
    r"""
    Given an evaluation dictionary containing ROC AUC scores for different embedding types and a specific cell type,
    plot the ROC AUC curve for each embedding type.

    :param classifier_evaluation_dict: Dict[str, Dict[str, np.ndarray]], a dictionary containing the ROC AUC scores for different embedding types
    :param cell_type: str, the cell type for which the ROC AUC curve should be plotted
    :param embedding_types: List[str], a list of embedding types for which the ROC AUC curve should be plotted
    :param save_fig: bool, whether to save the figure
    """

    plt.figure()

    for embedding_type in embedding_types:
        fpr = classifier_evaluation_dict[embedding_type]["FPR"]
        tpr = classifier_evaluation_dict[embedding_type]["TPR"]
        roc_auc = classifier_evaluation_dict[embedding_type]["AUC"]

        plt.plot(fpr, tpr, lw=2, label=embedding_type + ' (area = %0.2f)' % roc_auc)
        plt.legend(loc="lower right")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for " + cell_type)
        plt.grid(True)

    if save_fig:
        plt.savefig(f"./figures/tabulaMuris/roc_auc_curve_{cell_type}.pdf", dpi=300, bbox_inches='tight')

    plt.show()



def evaluate_embedding_clustering(adata: AnnData, tissue: str, likelihood_keys: List[str]) -> pd.DataFrame:
    r"""
    Given an AnnData object containing latent mean embeddings (adata.obsm[likelihoood + "_latent_mean"]) of a generative
    model with different likelihood options given in likelihood_keys as well as a tissue type, the optimal resolution
    for clustering will be determined via scib. The optimal clustering is then evaluated via NMI, ARI, ASW, and Avg Bio
    scores. The results are returned as a dataframe.

    :param adata: AnnData object containing the latent mean embeddings
    :param tissue: str, the tissue type for which the clustering should be evaluated
    :param likelihood_keys: List[str], the likelihood keys for which the clustering should be evaluated
    :return: pd.DataFrame, a dataframe containing the evaluation results
    """
    if not isinstance(adata, AnnData):
        raise ValueError("Input must be an AnnData object")
    if tissue is not None:
        if tissue not in adata.obs["tissue"].to_numpy():
            raise ValueError("Provided tissue type is not contained in data")
    if not isinstance(likelihood_keys, list):
        raise ValueError("Likelihood keys must be a list of strings")

    # Check label_key for cell types
    if "cell_ontology_class" in adata.obs.keys():
        cell_label_key = "cell_ontology_class"
    elif "subclass_label" in adata.obs.keys():
        cell_label_key = "subclass_label"
    else:
        raise ValueError(
            "No cell type label found in adata.obs. Please provide a cell type label in adata.obs with key 'cell_ontology_class' or 'subclass_label'")

    # Assessment of clustering
    res_opt_list = []
    nmi_list = []
    ari_list = []
    asw_list = []
    avg_bio_list = []

    for idx, likelihood_key in enumerate(likelihood_keys):
        # Calculate optimal clustering achievable
        if tissue is not None:
            adata = adata[adata.obs["tissue"] == tissue]
        resolutions = np.round(np.linspace(0., 2.0, 21), 2)
        sc.pp.neighbors(adata, use_rep=likelihood_key + "_latent_mean")
        res_opt, nmi_opt = scib.me.cluster_optimal_resolution(
            adata,
            cluster_key="cluster",
            resolutions=resolutions,
            label_key=cell_label_key
        )

        # Report optimal resolution
        res_opt_list.append(res_opt.item())

        # Delete all cluster columns except the one with optimal resolution
        cluster_cols = [col for col in adata.obs.columns if col.startswith("cluster_")]
        for col in cluster_cols:
            if col == "cluster_membership":
                continue
            elif col == "cluster_label":
                continue
            elif col == "cluster_color":
                continue
            elif col != "cluster_" + str(res_opt):
                del adata.obs[col]
                del adata.uns[col]


        # Calculate NMI, ARI, ASW
        nmi_score = scib.me.nmi(adata, cluster_key="cluster_" + str(res_opt), label_key=cell_label_key)
        asw_score = scib.me.silhouette(adata, label_key=cell_label_key,
                                       embed=likelihood_key + "_latent_mean")
        ari_score = scib.me.ari(adata, cluster_key="cluster_" + str(res_opt), label_key=cell_label_key)
        avg_bio_score = (nmi_score + asw_score + ari_score) / 3

        # Delete the cluster column with optimal resolution
        del adata.obs["cluster_" + str(res_opt)]
        del adata.uns["cluster_" + str(res_opt)]
        del adata.obs["cluster"]

        nmi_list.append(nmi_score.item())
        asw_list.append(asw_score.item())
        ari_list.append(ari_score.item())
        avg_bio_list.append(avg_bio_score.item())

    # Create an evaluation dataframe
    score_values = np.concatenate((np.array(nmi_list), np.array(ari_list), np.array(asw_list), np.array(avg_bio_list)))
    if len(likelihood_keys) == 1:
        score_names = np.array(["NMI", "ARI", "ASW", "Avg Bio"])
    else:
        score_nmi_names = ["NMI"] * len(likelihood_keys)
        score_ari_names = ["ARI"] * len(likelihood_keys)
        score_asw_names = ["ASW"] * len(likelihood_keys)
        score_avg_bio_names = ["Avg Bio"] * len(likelihood_keys)

        score_names = np.array(score_nmi_names + score_ari_names + score_asw_names + score_avg_bio_names)
    likelihood_names = np.array((likelihood_keys * 4))

    evaluation_df = pd.DataFrame(
        {
            "Score value": score_values,
            "Evaluation score": score_names,
            "Likelihood": likelihood_names
        }
    )

    return evaluation_df

def compare_data_imputation(
        input_matrix_GE,
        input_matrix_TU,
        recon_matrix_GE,
        recon_matrix_TU,
        model_name: str,
        tissue=None,
        num_rows: int = 2,
        num_cols: int = 2,
        show_fig: bool = True,
        save_fig: bool = True
):
    r"""
    Given a gene expression matrix and a transcript usage matrix together with their according reconstructed
    versions, plot the matrices as heatmaps

    :param input_matrix_GE:
    :param input_matrix_TU:
    :param recon_matrix_GE:
    :param recon_matrix_TU:
    :param model_name:
    :param tissue:
    :param num_rows:
    :param num_cols:
    :param show_fig:
    :param save_fig:
    """

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))

    heatmap_input_GE = axes[0, 0].imshow(input_matrix_GE, cmap="viridis")
    heatmap_input_TU = axes[1, 0].imshow(input_matrix_TU, cmap="viridis")
    heatmap_recon_GE = axes[0, 1].imshow(recon_matrix_GE, cmap="viridis")
    heatmap_recon_TU = axes[1, 1].imshow(recon_matrix_TU, cmap="viridis")

    fig.colorbar(heatmap_input_GE, ax=axes[0, 0])
    fig.colorbar(heatmap_input_TU, ax=axes[1, 0])
    fig.colorbar(heatmap_recon_GE, ax=axes[0, 1])
    fig.colorbar(heatmap_recon_TU, ax=axes[1, 1])

    axes[0, 0].set_title("Gene Expression Matrix")
    axes[0, 1].set_title("Reconstructed Gene Expression Matrix")
    axes[1, 0].set_title("Transcript Usage Matrix")
    axes[1, 1].set_title("Reconstructed Transcript Usage Matrix")

    fig.suptitle(f"Data Imputation of Transcript Usage and Gene Expression")

    plt.tight_layout()
    plt.show()

    if save_fig:
        if tissue:
            plt.savefig("./figures/tabulaMuris/data_imputation_" + model_name + "_" + tissue + ".png", dpi=300,
                        bbox_inches='tight')
        else:
            plt.savefig("./figures/tabulaMuris/data_imputation_" + model_name + ".png", dpi=300,
                        bbox_inches='tight')

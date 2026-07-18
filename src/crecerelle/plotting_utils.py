#!/usr/bin/env python3
import pdb

import matplotlib
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from importlib import resources
import matplotlib.gridspec as gridspec
from matplotlib_venn import venn2 # Not installed in PyCharm atm
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

from .utils import filter_min_cells_per_feature, filter_min_cells_per_intron_group, cell_type_classification_scgetuvi_dataframe

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
        fig.savefig("./figures/" + dataset_name + "/random_seed_comparison_scTUVI_" + observation_model + ".pdf", dpi=300,
                    bbox_inches='tight')
    plt.show()

def plot_random_seed_comparison_scgetuvi(
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
        fig.savefig("./figures/" + dataset_name + "/random_seed_comparison_scGETUVI_" + observation_models[0] + "_" + observation_models[1] + ".pdf",
                    dpi=300,
                    bbox_inches='tight')
    plt.show()






def plot_cell_embeddings_umaps_scgetuvi(
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
    Given two AnnData objects containing the results of training a scGETUVI on gene expression and transcript usage data,
    plot UMAPs of the shared latent space of each modality and the joint latent space, as well as a bar plot of the
    importance weights for each modality. The UMAPs are colored by cell type and the bar plot shows the mean importance
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

        # 4. Importance Weights Bar Plot
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
        axs[i, 3].set_ylabel("Importance Weight")

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

        filename = f"./figures/{dataset_name}/{t_name}_{c_name}_scgetuvi_umaps.pdf"

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

def plot_comparison_sctuvi_zanidm_zidm(
        distance_matrices = Tuple[np.ndarray, np.ndarray],
        cluster_annotations = Tuple[np.ndarray, np.ndarray],
        computation_times = Dict[str, Dict[str, float]],
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given a tuple of two distance matrices where the first contains the distances of embeddings computed with
    scTUVI-ZANIDM, and the second one distances of embeddings computed with scTUVI-ZIDM, and a dictionary of computation
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
    axes[0,0].set_ylabel("Observation model of scTUVI")
    axes[0,0].tick_params(axis="both")
    axes[0,0].set_title('a', loc='left', fontsize=20, fontweight='bold')

    # Plot total training time
    total_training_times = np.array(
        [computation_times["ZANIDM"]["Total training time"], computation_times["ZIDM"]["Total training time"]])
    axes[0, 1].barh(np.array(["ZANIDM", "ZIDM"]), total_training_times)
    axes[0, 1].set_xlabel("Total training time (s)")
    axes[0, 1].set_ylabel("Observation model of scTUVI")
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
        fig_name = os.path.join(save_path, "comparison_sctuvi_zanidm_zidm.pdf")
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')

    plt.show()

def plot_umap_and_distance_matrix(
        adata_1: AnnData | None,
        adata_2: AnnData | None,
        cell_embeddings_keys: List[str],
        transcriptomic_facet_keys: List[str],
        distance_matrices: List[np.ndarray],
        cluster_annotations: np.ndarray,
        umap_color_display_key: str,
        model_type: str,
        save_fig: bool = True,
        **kwargs
):

    random_state = kwargs.get("random_state", 0)

    if kwargs.get("average_embeddings", True):
        heatmap_label = "Mean cell embeddings"
    else:
        heatmap_label = "Cell embeddings"

    vmin = min(np.min(m) for m in distance_matrices)
    vmax = max(np.max(m) for m in distance_matrices)

    num_rows = len(cell_embeddings_keys)
    alphabet_characters = ["a", "b", "c", "d", "e", "f"]

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    cluster_annotations = np.asarray(cluster_annotations)

    def get_color_from_legend_handle(handle):
        if hasattr(handle, "get_markerfacecolor"):
            return handle.get_markerfacecolor()

        if hasattr(handle, "get_facecolor"):
            color = handle.get_facecolor()
            if np.ndim(color) > 1 and len(color) > 0:
                return color[0]
            return color

        if hasattr(handle, "get_color"):
            return handle.get_color()

        return "black"

    def make_cluster_color_map(handles, labels):
        return {
            str(label): get_color_from_legend_handle(handle)
            for handle, label in zip(handles, labels)
        }

    def get_cluster_centers_labels_and_boundaries(cluster_annotations):
        boundary_positions = np.where(
            cluster_annotations[:-1] != cluster_annotations[1:]
        )[0] + 1

        starts = np.r_[0, boundary_positions]
        ends = np.r_[boundary_positions, len(cluster_annotations)]

        centers = (starts + ends) / 2
        cluster_labels = cluster_annotations[starts]

        return centers, cluster_labels, boundary_positions

    def plot_distance_heatmap(
            ax,
            distance_matrix,
            cluster_annotations,
            cluster_color_map
    ):
        sns.heatmap(
            distance_matrix,
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": "Distance"}
        )

        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError(
                f"Expected a square distance matrix, got shape {distance_matrix.shape}."
            )

        if distance_matrix.shape[0] != len(cluster_annotations):
            raise ValueError(
                f"Distance matrix has shape {distance_matrix.shape}, "
                f"but cluster_annotations has length {len(cluster_annotations)}."
            )

        centers, cluster_labels, boundary_positions = (
            get_cluster_centers_labels_and_boundaries(cluster_annotations)
        )

        for pos in boundary_positions:
            ax.axhline(pos, color="red", linestyle="--", linewidth=1.5)
            ax.axvline(pos, color="red", linestyle="--", linewidth=1.5)

        # Remove heatmap axis titles
        ax.set_xlabel("")
        ax.set_ylabel("")

        # One tick per cell type cluster
        ax.set_xticks(centers)
        ax.set_yticks(centers)

        # Colored bullet points as tick labels
        ax.set_xticklabels(["●"] * len(cluster_labels), rotation=0, fontsize=12)
        ax.set_yticklabels(["●"] * len(cluster_labels), rotation=0, fontsize=12)

        for tick_label, cluster_label in zip(ax.get_xticklabels(), cluster_labels):
            tick_label.set_color(cluster_color_map.get(str(cluster_label), "black"))

        for tick_label, cluster_label in zip(ax.get_yticklabels(), cluster_labels):
            tick_label.set_color(cluster_color_map.get(str(cluster_label), "black"))

        # Hide tick marks but keep bullet labels
        ax.tick_params(axis="both", length=0)

    handles, labels = None, None

    for i, transcriptomic_facet in enumerate(transcriptomic_facet_keys):

        if transcriptomic_facet in ["GE", "S-GE", "P-GE", "GE-TU"]:
            adata = adata_1
        elif transcriptomic_facet in ["TU", "S-TU", "P-TU"]:
            adata = adata_2
        else:
            raise ValueError(
                f"Invalid transcriptomic facet key: {transcriptomic_facet}. "
                "Expected one of 'GE', 'S-GE', 'P-GE', 'TU', 'S-TU', 'P-TU', or 'GE-TU'."
            )

        if adata is None:
            raise ValueError(
                f"AnnData object is None for transcriptomic facet {transcriptomic_facet}."
            )

        adata.obsm["X_umap"] = UMAP(
            n_components=2,
            random_state=random_state
        ).fit_transform(adata.obsm[cell_embeddings_keys[i]])

        sc.pl.umap(
            adata,
            color=umap_color_display_key,
            frameon=False,
            show=False,
            ax=axes[i, 0]
        )

        del adata.obsm["X_umap"]

        axes[i, 0].set_title(transcriptomic_facet)

        plot_customized_UMAP_coordinates(axes[i, 0], length=1.0)

        axes[i, 0].set_title(
            alphabet_characters[i],
            loc="left",
            fontsize=20,
            fontweight="bold"
        )

        handles, labels = axes[i, 0].get_legend_handles_labels()

        if axes[i, 0].get_legend():
            axes[i, 0].get_legend().remove()

        cluster_color_map = make_cluster_color_map(handles, labels)

        plot_distance_heatmap(
            axes[i, 1],
            distance_matrices[i],
            cluster_annotations,
            cluster_color_map
        )

    if handles is not None and labels is not None:
        n_cols = 5

        leg = fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.05, 0.88, 0.85, 0.05),
            ncol=n_cols,
            mode="expand",
            borderaxespad=0.,
            frameon=False,
            fontsize=12,
            title="Cell type",
            title_fontsize=12
        )

        leg._legend_box.align = "left"

    if save_fig:
        import os

        dataset_name = kwargs.get("dataset_name", "default")
        file_suffix = kwargs.get("file_suffix", "pdf")
        tax_level = kwargs.get("tax_level", "default")
        save_path = f"./figures/{dataset_name}/"
        os.makedirs(save_path, exist_ok=True)

        fig_name = os.path.join(
            save_path,
            model_type + "_cell_embeddings_umap_heatmaps_" + tax_level + "." + file_suffix
        )

        fig.savefig(fig_name, dpi=300, bbox_inches="tight")

    plt.show()



def plot_scVI_scTUVI_cell_embedding_comparison(
        adata_objects: Tuple[AnnData, AnnData],
        evaluation_df: pd.DataFrame,
        color_annotation: str,
        likelihood_1: str = "ZINB",
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given a tuple of two AnnData objects where the first contains the inferred gene expression cell embeddings from scVI
    (with a NB or ZINB observation model) and the second the inferred transcript usage cell embeddings from scTUVI (with
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

    # UMAP of atlas using scTUVI-DM
    ax01 = fig.add_subplot(gs[0, 1])
    adata_TU.obsm['X_umap'] = adata_TU.obsm['DM_X_umap'].copy()
    sc.pl.umap(adata_TU, color=color_annotation, frameon=False, show=False, ax=ax01)
    leg = ax01.get_legend()
    if leg:
        plt.setp(leg.get_texts(), fontsize=12)  # or a number like 8
        plt.setp(leg.get_title(), fontsize=12)
    del adata_TU.obsm['X_umap']
    ax01.set_title('scTUVI-DM')
    ax01.set_title('b', loc='left', fontsize=20, fontweight='bold')
    plot_customized_UMAP_coordinates(ax01, length=1.0)

    # Modelling splicing
    ax10 = fig.add_subplot(gs[1:3, :3])

    try:
        img_resource = resources.files('crecerelle.default_figures').joinpath('crecerelle_splicing_scTUVI.png')

        with resources.as_file(img_resource) as image_path:
            model_img = mpimg.imread(str(image_path))
            ax10.imshow(model_img)
    except (ImportError, FileNotFoundError):
        print("Warning: Splicing figure could not be loaded from package resources.")

    #image_path = "./default_figures/crecerelle_splicing_scTUVI.png"
    #model_img = mpimg.imread(image_path)
    #ax10.imshow(model_img)
    ax10.axis('off')
    # ax10.set_title('a', loc='left', fontsize=20, fontweight='bold')

    # UMAP of atlas using scTUVI-ZANIDM
    ax30 = fig.add_subplot(gs[3, 0])
    adata_TU.obsm['X_umap'] = adata_TU.obsm['ZANIDM_X_umap'].copy()
    sc.pl.umap(adata_TU, color=color_annotation, legend_loc=None, frameon=False, show=False, ax=ax30)
    del adata_TU.obsm['X_umap']
    ax30.set_title('scTUVI-ZANIDM')
    ax30.set_title('f', loc='left', fontsize=20, fontweight='bold')
    plot_customized_UMAP_coordinates(ax30, length=1.0)

    # UMAP of atlas using scTUVI-ZIDM
    ax31 = fig.add_subplot(gs[3, 1])
    adata_TU.obsm['X_umap'] = adata_TU.obsm['ZIDM_X_umap'].copy()
    sc.pl.umap(adata_TU, color=color_annotation, legend_loc=None, frameon=False, show=False, ax=ax31)
    del adata_TU.obsm['X_umap']
    ax31.set_title('scTUVI-ZIDM')
    ax31.set_title('g', loc='left', fontsize=20, fontweight='bold')
    plot_customized_UMAP_coordinates(ax31, length=1.0)

    # Bar chart of biological conservation metrics
    # optimal leiden resolution for scVI-ZINB: res = 0.6 (40 clusters /  cell types)
    # optimal leiden resolution for scTUVI-DM: res = 1.4 (32 clusters /  cell types)
    # optimal leiden resolution for scTUVI-ZANIDM: res = 0.9 ( 38 clusters /  cell types)
    # optimal leiden resolution for scTUVI-ZIDM: res =  0.9 ( 31 clusters /  cell types)

    ax32 = fig.add_subplot(gs[3, 2])
    sns.barplot(evaluation_df, x='Evaluation score', y='Score value', hue='Likelihood', ax=ax32)
    ax32.set_ylabel(" ")
    ax32.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax32.set_title('h', loc='left', fontsize=20, fontweight='bold')

    if save_fig:
        dataset_name = kwargs.get("dataset_name")
        file_suffix = kwargs.get("file_suffix", 'pdf')
        if dataset_name is not None:
            # plt.savefig("./figures/tabulaMuris/scTUVI_splicing_modelling.png", dpi=300, bbox_inches='tight')
            fig_name = "./figures/" + dataset_name + "/scTUVI_splicing_modelling." + file_suffix
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        else:
            fig_name = "./figures/scTUVI_splicing_modelling." + file_suffix
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
    for the cluster group specified. The DEG analysis should have been performed using the scVI or scGETUVI latent
    space. The chosen model and GE likelihood key (for scGETUVI also the embedding) should be represented in model_name
    (e.g. scVI_ZINB, scGETUVI_ZINB_ZIDM_shared, etc.). If the Anndata object was filtered to a cell taxonomy level
    (e.g. tissue (Heart, Brain_Non-Myeloid)) specify the cell taxonomy level name otherwise set taxonomy_level to None.
    The cluster_group specifies the Leiden cluster group for which the top 6 DEGs will be plotted. The figure is saved
    in the figures directory if save_fig is set to True.

    :param adata: AnnData object containing the results of the DEG analysis for a specific tissue and cluster group
    :param cluster_group: str, the Leiden cluster group for which the top 6 DEGs will be plotted
    :param model_name: str, the name of the model used for the DEG analysis (e.g. scVI_ZINB, scGETUVI_ZINB_ZIDM_shared, etc.)
    :param taxonomy_level: str, the name of the tissue for which the DEG analysis was performed, or None if the analysis was performed on all tissues
    :param save_fig: bool, whether to save the figure

    """
    # Assert that the key "leiden" is present in adata.obs
    assert "leiden" in adata.obs, "The key 'leiden' is not present in adata.obs. Please perform Leiden clustering and add the cluster labels to adata.obs['leiden'] before calling this function."

    # Assert that the DEG analysis has been performed and the key "rank_genes_groups" is present in adata.uns
    assert "rank_genes_groups" in adata.uns, "The key 'rank_genes_groups' is not present in adata.uns. Please perform DEG analysis using scVI or scGETUVI and add the results to adata.uns['rank_genes_groups'] before calling this function."


    fig = plt.figure(figsize=(16, 18), dpi=300)
    gs = gridspec.GridSpec(4, 3, figure=fig)

    ax00 = fig.add_subplot(gs[0, 0])
    sc.pl.umap(
        adata,
        color="leiden",
        legend_loc="on data",
        frameon=False,
        show=False,
        ax=ax00
    )
    # Delete title
    ax00.set_title(' ')
    # Set new title
    ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')

    ax02 = fig.add_subplot(gs[0, 1])
    sc.pl.umap(
        adata,
        color=kwargs.get("cell_type_key"),
        frameon=False,
        show=False,
        ax=ax02
    )

    # Delete title
    ax02.set_title(' ')
    # Set new title
    ax02.set_title('b', loc='left', fontsize=20, fontweight='bold')

    ax10_full_row = fig.add_subplot(gs[1, :])

    sc.pl.rank_genes_groups_dotplot(adata, groupby="leiden", standard_scale="var", n_genes=6, show=False,
                                    ax=ax10_full_row)
    # Set title for ax10
    ax10_full_row.set_title('c', loc='left', fontsize=20, fontweight='bold')

    dc_cluster_genes = sc.get.rank_genes_groups_df(adata, group=cluster_group).head(6)["names"]

    all_values = adata[:, dc_cluster_genes].X
    v_min, v_max = all_values.min(), all_values.max()

    ax20 = fig.add_subplot(gs[2, 0])

    sc.pl.umap(
        adata,
        color=dc_cluster_genes[0],
        legend_loc="on data",
        frameon=False,
        show=False,
        ax=ax20,
        vmin=v_min,
        vmax=v_max,
    )

    ax20.set_title('d', loc='left', fontsize=20, fontweight='bold')

    ax21 = fig.add_subplot(gs[2, 1])

    sc.pl.umap(
        adata,
        color=dc_cluster_genes[1],
        legend_loc="on data",
        frameon=False,
        show=False,
        ax=ax21,
        vmin=v_min,
        vmax=v_max,
    )

    ax22 = fig.add_subplot(gs[2, 2])

    sc.pl.umap(
        adata,
        color=dc_cluster_genes[2],
        legend_loc="on data",
        frameon=False,
        show=False,
        ax=ax22,
        vmin=v_min,
        vmax=v_max,
    )

    ax30 = fig.add_subplot(gs[3, 0])

    sc.pl.umap(
        adata,
        color=dc_cluster_genes[3],
        legend_loc="on data",
        frameon=False,
        show=False,
        ax=ax30,
        vmin=v_min,
        vmax=v_max,
    )

    ax31 = fig.add_subplot(gs[3, 1])

    sc.pl.umap(
        adata,
        color=dc_cluster_genes[4],
        legend_loc="on data",
        frameon=False,
        show=False,
        ax=ax31,
        vmin=v_min,
        vmax=v_max,
    )

    ax32 = fig.add_subplot(gs[3, 2])

    sc.pl.umap(
        adata,
        color=dc_cluster_genes[5],
        legend_loc="on data",
        frameon=False,
        show=False,
        ax=ax32,
        vmin=v_min,
        vmax=v_max,
    )

    plot_customized_UMAP_coordinates(ax00, length=1.0)
    plot_customized_UMAP_coordinates(ax02, length=1.0)

    plot_customized_UMAP_coordinates(ax20, length=1.0)
    plot_customized_UMAP_coordinates(ax21, length=1.0)
    plot_customized_UMAP_coordinates(ax22, length=1.0)
    plot_customized_UMAP_coordinates(ax30, length=1.0)
    plot_customized_UMAP_coordinates(ax31, length=1.0)
    plot_customized_UMAP_coordinates(ax32, length=1.0)

    plt.tight_layout()

    if save_fig:
        dataset_name = kwargs.get("dataset_name")
        file_suffix = kwargs.get("file_suffix", 'pdf')
        if taxonomy_level is not None:
            fig_name = "./figures/" + dataset_name + "/deg_analysis_" + model_name + "_" + taxonomy_level + "_clustergroup_" + cluster_group + "." + file_suffix
            fig.savefig(fig_name, dpi=300, bbox_inches="tight")
        else:
            fig_name = "./figures/" + dataset_name + "/deg_analysis_" + model_name + "_clustergroup_" + cluster_group + "." + file_suffix
            fig.savefig(fig_name, dpi=300, bbox_inches="tight")

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
    been performed using the scTUVI or scGETUVI latent space and the results should be present in
    adata.uns["rank_introns_groups"]. The cluster groups specified in cluster_groups should be present in
    adata.uns["rank_introns_groups"]["names"]. The raw PSI scores should be present in adata.layers["PSI_raw"].

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
    group specified. The DSG analysis should have been performed using the scTUVI or scGETUVI latent space. The chosen
    model and TU likelihood key should be represented in model_name (e.g. scTUVI_ZIDM, scGETUVI_ZINB_ZIDM, etc.). If the
    Anndata object was filtered to a taxonomy level (e.g. tissue) specify the taxonomy level name otherwise set taxonomy
    level to None. The cluster_group specifies the Leiden cluster group for which the top 3 DSG isoforms will be
    plotted. The isoforms are renamed numbered according to location if rename_isoforms is set to True.The figure is
    saved in the figures directory if save_fig is set to True.

    :param adata: AnnData object containing the results of the DSG analysis for a specific tissue and cluster group
    :param cluster_group: str, the Leiden cluster group for which the top 3 DSG isoforms will be plotted
    :param cluster_groups: List of str, the Leiden cluster groups to be included in the dotplot of DSGs across cluster groups
    :param model_name: str, the name of the model used for the DSG analysis (e.g. scTUVI_ZIDM, scGETUVI_ZINB_ZIDM, etc.)
    :param taxonomy_level: str or None, the taxonomy level to be plotted (e.g. tissue)
    :param rename_isoforms: bool, whether to rename the isoforms to "gene_name isoform_number" format, where isoform_number is assigned based on the order of appearance of the intron names for each gene
    :param save_fig: bool, whether to save the figure

    """

    # Assert that the key "leiden" is present in adata.obs
    assert "leiden" in adata.obs, "The key 'leiden' is not present in adata.obs. Please perform Leiden clustering and add the cluster labels to adata.obs['leiden'] before calling this function."

    # Assert that raw PSI scores have been added to adata.layers with the key "PSI_raw"
    assert "PSI_raw" in adata.layers, "The key 'PSI_raw' is not present in adata.layers. Please add the raw PSI scores to adata.layers['PSI_raw'] before calling this function."

    # Assert that the DSG analysis has been performed and the key "rank_introns_groups" is present in adata.uns
    assert "rank_introns_groups" in adata.uns, "The key 'rank_introns_groups' is not present in adata.uns. Please perform DSG analysis using scTUVI or scGETUVI and add the results to adata.uns['rank_introns_groups'] before calling this function."

    # This can be a dedicated plotting function
    fig = plt.figure(figsize=(15, 12), dpi=300)

    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1.25, 1]
    )

    # UMAP of clustering
    ax00 = fig.add_subplot(gs[0, 0])
    sc.pl.umap(
        adata,
        color="leiden",
        legend_loc="on data",
        frameon=False,
        show=False,
        ax=ax00
    )
    # Delete title
    ax00.set_title(' ')
    # Set new title
    ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')

    plot_customized_UMAP_coordinates(ax00, length=1.0)

    # UMAP of ground truth
    ax02 = fig.add_subplot(gs[0, 1])
    sc.pl.umap(
        adata,
        color=kwargs.get("cell_type_key"),
        frameon=False,
        show=False,
        ax=ax02
    )
    # Delete title
    ax02.set_title(' ')
    # Set new title
    ax02.set_title('b', loc='left', fontsize=20, fontweight='bold')

    plot_customized_UMAP_coordinates(ax02, length=1.0)


    # Number of DSG isoforms to be displayed has a default value of 3 if not specified otherwise by **kwargs
    num_intron_group_markers = kwargs.get("num_intron_group_markers", 3)

    start_idx = num_intron_group_markers * int(np.argwhere(np.array(cluster_groups) == cluster_group).squeeze())
    if start_idx == 0:
        end_idx = num_intron_group_markers - 1
    else:
        end_idx = start_idx + num_intron_group_markers - 1

    intron_names_indices = np.linspace(start_idx, end_idx, num_intron_group_markers).astype(int)


    # Helper function for DSG dotplot
    mean_psi_groups_introns, num_cells_valid_matrix, intron_names = dsg_dotplot_helper(
        adata=adata,
        cluster_groups=cluster_groups,
        num_intron_group_markers=num_intron_group_markers
    )

    # Rename isoforms to "gene_name isoform_number" format, where isoform_number is assigned based on the order of appearance of the intron names for each gene.
    if rename_isoforms:
        intron_names_plot = rename_isoform_helper(intron_names)
    else:
        intron_names_plot = intron_names

    # Actual dot plot
    # Control how tightly packed the rows are
    row_spacing = 0.75  # < 1.0 makes rows closer together

    # Create a grid of x, y coordinates
    X, Y = np.meshgrid(range(len(intron_names)), range(len(cluster_groups)))
    x_coords = X.flatten()
    y_coords = (Y * row_spacing).flatten()  # compress row spacing

    color_values = mean_psi_groups_introns.flatten()
    size_values = num_cells_valid_matrix.flatten()

    # Make the dot plot less wide
    ax10 = fig.add_subplot(gs[1, :2])

    psi_dotplot = ax10.scatter(
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
    ax10.set_xticks(range(len(intron_names_plot)))
    ax10.set_xticklabels(intron_names_plot, rotation=90, ha='right')

    # Y axis: ticks must match compressed row positions
    y_tick_positions = np.arange(len(cluster_groups)) * row_spacing
    ax10.set_yticks(y_tick_positions)
    ax10.set_yticklabels(cluster_groups)

    # Ensure first and last row dots are fully visible
    ax10.set_ylim(-0.5 * row_spacing, (len(cluster_groups) - 0.5) * row_spacing)

    # Optional: also give a little horizontal margin so edge dots are fully visible
    ax10.set_xlim(-0.5, len(intron_names) - 0.5)

    ax10.set_title('d', loc='left', fontsize=20, fontweight='bold')

    # -----------------------------
    # Dedicated legend column
    # -----------------------------
    size_legend_values = [50, 100, 250, 500]

    legend_ax = fig.add_subplot(gs[1, 2])

    legend_ax.set_xlim(
        -0.5,
        len(size_legend_values) - 0.5
    )
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

        # Small tick underneath each dot
        legend_ax.plot(
            [x, x],
            [0.47, 0.57],
            color="black",
            linewidth=1.3
        )

        # Size value
        legend_ax.text(
            x,
            0.35,
            str(s),
            ha="center",
            va="center",
            fontsize=9
        )

    # -----------------------------
    # Horizontal colorbar inside
    # the reserved legend column
    # -----------------------------
    cax = inset_axes(
        legend_ax,
        width="90%",
        height="12%",
        loc="lower center",
        borderpad=0
    )

    cbar = fig.colorbar(
        psi_dotplot,
        cax=cax,
        orientation="horizontal"
    )

    cbar.set_ticks([0.00, 0.50, 1.00])

    cbar.ax.set_title(
        "Mean PSI value\nin group",
        fontsize=10,
        pad=6
    )

    # UMAP of three DSGs

    color_0 = intron_names[intron_names_indices[0]]
    if color_0 != "0":
        ax20 = fig.add_subplot(gs[2, 0])
        sc.pl.umap(
            adata,
            color=color_0,
            layer="PSI_raw",
            frameon=False,
            show=False,
            ax=ax20
        )
        if rename_isoforms:
            ax20.set_title(intron_names_plot[intron_names_indices[0]], fontsize=12)

        plot_customized_UMAP_coordinates(ax20, length=1.0)
        ax20.set_title('d', loc='left', fontsize=20, fontweight='bold')

    color_1 = intron_names[intron_names_indices[1]]
    if color_1 != "0":
        ax21 = fig.add_subplot(gs[2, 1])
        sc.pl.umap(
            adata,
            color=color_1,
            layer="PSI_raw",
            frameon=False,
            show=False,
            ax=ax21
        )
        if rename_isoforms:
            ax21.set_title(intron_names_plot[intron_names_indices[1]], fontsize=12)

        plot_customized_UMAP_coordinates(ax21, length=1.0)

    color_2 = intron_names[intron_names_indices[2]]
    if color_2 != "0":

        ax22 = fig.add_subplot(gs[2, 2])
        sc.pl.umap(
            adata,
            color=color_2,
            layer="PSI_raw",
            frameon=False,
            show=False,
            ax=ax22
        )

        if rename_isoforms:
            ax22.set_title(intron_names_plot[intron_names_indices[2]], fontsize=12)

        plot_customized_UMAP_coordinates(ax22, length=1.0)

    plt.tight_layout()

    if save_fig:
        dataset_name = kwargs.get("dataset_name")
        file_suffix = kwargs.get("file_suffix", 'pdf')
        if dataset_name is not None:
            if taxonomy_level is not None:
                fig_name = "./figures/" + dataset_name + "/dsg_analysis_" + model_name + "_" + taxonomy_level+ "_clustergroup_" + cluster_group + "." + file_suffix
                fig.savefig(fig_name, dpi=300, pad_inches=0.3)
            else:
                fig_name = "./figures/" + dataset_name + "/dsg_analysis_" + model_name + "_clustergroup_" + cluster_group + "." + file_suffix
                fig.savefig(fig_name, dpi=300, pad_inches=0.3)
        else:
            if taxonomy_level is not None:
                fig_name = "./figures/dsg_analysis_" + model_name + "_" + taxonomy_level+ "_clustergroup_" + cluster_group + "." + file_suffix
                fig.savefig(fig_name, dpi=300, pad_inches=0.3)
            else:
                fig_name = "./figures/dsg_analysis_" + model_name + "_clustergroup_" + cluster_group + "." + file_suffix
                fig.savefig(fig_name, dpi=300, pad_inches=0.3)
    plt.show()

def plot_marker_analysis_scvi_sctuvi(
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
    on cell embeddings inferred by scVI and scTUVI respectively, analyse the top DEGs and DSGs for a specific cluster
    group and plot the UMAPs of the clusters, the dotplots of the DEGs and DSGs across cluster groups, and the top 3
    DEGs and DSGs for the cluster group specified. The DEG and DSG analyses should have been performed using the scVI or
    scTUVI latent space. The chosen model and GE likelihood key (for scTUVI also the embedding) should be represented in
    likelihoods (e.g. ["scVI_ZINB", "scTUVI_ZIDM"]). If the Anndata objects were filtered to a cell taxonomy level
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
    ax01.set_title('scTUVI-ZIDM')
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
    del adata_1.obsm['X_umap']
    del adata_2.obsm['X_umap']

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
        fig_name = "./figures/tabulaMuris/cell_annotation_scTUVI_" + taxonomy_level + "_clustergroup_ge_" + group_id_1 + "_clustergroup_tu_" + group_id_2 + "." + file_suffix
        fig.savefig(fig_name, dpi=300, bbox_inches="tight")

    plt.show()

def plot_differential_analysis_on_umap(
        adata: AnnData,
        cluster_groups: List[str],
        tax_level: str | None,
        model_name: str,
        modality: str,
        renamed_isoforms: List[str] | None = None,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given an Anndata object containing the results of a differential gene expression or differential splicing analysis
    for a specific tissue and cluster groups, plot for each cluster group the top 3 DEGs or DSGs on the UMAP. The DEG or
    DSG analysis should have been performed using the scVI, scTUVI, or scGETUVI latent space. The chosen model and GE/TU
    likelihood key should be represented in model_name (e.g. scVI_ZINB, scTUVI_ZIDM, scGETUVI_ZINB_ZIDM, etc.). If the
    Anndata object was filtered to a tissue specify the tissue name otherwise set tissue to None. The cluster_groups
    specifies the Leiden cluster groups for which the top 3 DEGs or DSGs will be plotted. If modality is set to "TU" and
    rename_isoforms is set to True, the DSGs will be renamed to "gene_name isoform_number" format, where isoform_number
    is assigned based on the order of appearance of the intron names for each gene. The figure is saved in the figures
    directory if save_fig is set to True.

    :param adata: AnnData object containing the results of a differential gene expression or differential splicing analysis for a specific tissue and cluster groups
    :param cluster_groups: List of str, the Leiden cluster groups for which the top 3 DEGs or DSGs will be plotted
    :param tax_level: str, the name of the tissue for which the analysis was performed, or None if the analysis was performed on all tissues
    :param model_name: str, the name of the model used for the analysis (e.g. scVI_ZINB, scTUVI_ZIDM, scGETUVI_ZINB_ZIDM, etc.)
    :param modality: str, the modality of the data (either "GE" for gene expression or "TU" for transcript usage)
    :param renamed_isoforms: List of str, the renamed isoform names in "gene_name isoform_number" format, where isoform_number is assigned based on the order of appearance of the intron names for each gene, to be used for plotting if modality is set to "TU" and rename_isoforms is set to True. The order of the names in the list should correspond to the order of the DSGs in de_entities. If modality is not "TU" or rename_isoforms is False, this parameter is ignored.
    :param save_fig: bool, whether to save the figure
    """
    # Assert that the key "leiden" is present in adata.obs
    assert "leiden" in adata.obs, "The key 'leiden' is not present in adata.obs. Please perform Leiden clustering and add the cluster labels to adata.obs['leiden'] before calling this function."

    # Assert that DEG or DSG analysis has been performed and the appropriate keys are present in adata.uns and adata.layers based on the modality
    if modality == "GE":
        # Assert that the DEG analysis has been performed and the key "rank_genes_groups" is present in adata.uns
        assert "rank_genes_groups" in adata.uns, "The key 'rank_genes_groups' is not present in adata.uns. Please perform DEG analysis using scVI or scGETUVI and add the results to adata.uns['rank_genes_groups'] before calling this function."
    elif modality == "TU":
        # Assert that raw PSI scores have been added to adata.layers with the key "PSI_raw"
        assert "PSI_raw" in adata.layers, "The key 'PSI_raw' is not present in adata.layers. Please add the raw PSI scores to adata.layers['PSI_raw'] before calling this function."
        # Assert that the DSG analysis has been performed and the key "rank_introns_groups" is present in adata.uns
        assert "rank_introns_groups" in adata.uns, "The key 'rank_introns_groups' is not present in adata.uns. Please perform DSG analysis using scTUVI or scGETUVI and add the results to adata.uns['rank_introns_groups'] before calling this function."
    else:
        raise ValueError("Invalid modality. Modality should be either 'GE' for gene expression or 'TU' for transcript usage.")

    # Asser that the X_umap coordinates have been computed and are present in adata.obsm
    assert "X_umap" in adata.obsm, "The key 'X_umap' is not present in adata.obsm. Please compute the UMAP coordinates and add them to adata.obsm['X_umap'] before calling this function."

    # Plot the top 3 DEGs or DSGs for each cluster group on the UMAP
    num_entities_per_group = kwargs.get("num_entities_per_group", 3)
    entity_names = np.zeros(num_entities_per_group * len(cluster_groups), dtype=object)
    for i, test_group in enumerate(cluster_groups):
        if modality == "GE":
            entity_names_group = adata.uns["rank_genes_groups"]["names"][test_group][:num_entities_per_group]
        elif modality == "TU":
            entity_names_group = adata.uns["rank_introns_groups"]["names"][test_group][:num_entities_per_group]
        else:
            raise ValueError("Invalid modality. Modality should be either 'GE' or 'TU'.")
        entity_names[i * num_entities_per_group: (i + 1) * num_entities_per_group] = entity_names_group

    fig, axs = plt.subplots(nrows=len(cluster_groups), ncols=num_entities_per_group, figsize=(num_entities_per_group * 4, len(cluster_groups) * 4))

    for i, test_group in enumerate(cluster_groups):
        # 1. Identify all entities for this specific test group
        group_entities = entity_names[i * num_entities_per_group: (i + 1) * num_entities_per_group]

        # 2. Calculate the global min and max for this group
        if modality == "GE":
            # Access data from adata.obs or adata.raw/X depending on where your entities are
            group_data = adata[:, group_entities].X
        elif modality == "TU":
            group_data = adata.layers["PSI_raw"][:, [adata.var_names.get_loc(e) for e in group_entities]]
        else:
            raise ValueError("Invalid modality. Modality should be either 'GE' or 'TU'.")

        # Flatten data to find true min/max across all cells and all entities in the group
        if modality == "GE":
            group_vmin = group_data.min()
            group_vmax = group_data.max()
        elif modality == "TU":
            group_vmin = 0.0
            group_vmax = 1.0
        else:
            raise ValueError("Invalid modality. Modality should be either 'GE' or 'TU'.")

        for j, entity_name in enumerate(group_entities):
            ax = axs[i, j]

            # 3. Pass vmin and vmax to ensure consistent colorbars
            if modality == "GE":
                sc.pl.umap(adata, color=entity_name, legend_loc=None, frameon=False,
                           show=False, ax=ax, vmin=group_vmin, vmax=group_vmax)
                ax.set_title(f"Cluster {test_group} - {entity_name}", fontsize=12)

            elif modality == "TU":
                sc.pl.umap(adata, color=entity_name, layer="PSI_raw", legend_loc=None,
                           frameon=False, show=False, ax=ax, vmin=group_vmin, vmax=group_vmax)

                if renamed_isoforms is not None:
                    ax.set_title(f"Cluster {test_group} - {renamed_isoforms[i * num_entities_per_group + j]}", fontsize=12)
                else:
                    ax.set_title(f"Cluster {test_group} - {entity_name}", fontsize=12)

            plot_customized_UMAP_coordinates(ax, length=1.0)

    plt.tight_layout()

    if save_fig:
        if modality == "GE":
            diff_analysis = "deg"
        elif modality == "TU":
            diff_analysis = "dsg"
        else:
            raise ValueError("Invalid modality. Modality should be either 'GE' for gene expression or 'TU' for transcript usage.")

        dataset_name = kwargs.get("dataset_name", "default")
        file_suffix = kwargs.get("file_suffix", "pdf")

        filename = f"./figures/{dataset_name}/{diff_analysis}_analysis_umaps_{model_name}_{tax_level}.{file_suffix}"
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {filename}")

    plt.show()

def plot_deg_and_dsg_analysis_scgetuvi(
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
    scGETUVI latent space. The chosen model, GE/TU likelihood key, and embedding should be represented in model_name
    (e.g. scGETUVI_ZINB_ZIDM_shared, etc.). If the Anndata objects were filtered to a tissue specify the tissue name
    otherwise set tissue to None. The embedding parameter specifies which cell embedding was used for the clustering
    and differential analysis (e.g. "shared") and is used for plotting the UMAPs. The figure is saved in the figures
    directory if save_fig is set to True.

    :param adata: Tuple of two AnnData objects, the first one containing the results of the DEG analysis and the second one containing the results of the DSG analysis for a specific tissue
    :param tax_level: str, the name of the tissue for which the analysis was performed,
    or None if the analysis was performed on all tissues
    :param model_name: str, the name of the model used for the analysis (e.g. scGETUVI_ZINB_ZIDM_shared, etc.)
    :param rename_isoforms: bool, whether to rename the DSG isoforms to "gene_name isoform_number" format, where isoform_number is assigned based on the order of appearance of the intron names for each gene
    :param save_fig: bool, whether to save the figure
    """

    # Assert that the key "leiden" is present in adata[0].obs and adata[1].obs
    assert "leiden" in adata[0].obs, "The key 'leiden' is not present in adata[0].obs. Please perform Leiden clustering and add the cluster labels to adata[0].obs['leiden'] before calling this function."
    assert "leiden" in adata[1].obs, "The key 'leiden' is not present in adata[1].obs. Please perform Leiden clustering and add the cluster labels to adata[1].obs['leiden'] before calling this function."

    # Assert that the DEG and DSG analyses have been performed and the appropriate keys are present in adata[0].uns, adata[1].uns, and adata[1].layers
    assert "rank_genes_groups" in adata[0].uns, "The key 'rank_genes_groups' is not present in adata[0].uns. Please perform DEG analysis using scGETUVI and add the results to adata[0].uns['rank_genes_groups'] before calling this function."
    assert "rank_introns_groups" in adata[1].uns, "The key 'rank_introns_groups' is not present in adata[1].uns. Please perform DSG analysis using scGETUVI and add the results to adata[1].uns['rank_introns_groups'] before calling this function."
    assert "PSI_raw" in adata[1].layers, "The key 'PSI_raw' is not present in adata[1].layers. Please add the raw PSI scores to adata[1].layers['PSI_raw'] before calling this function."

    # Assert that the UMAP coordinates have been computed and are present in adata[0].obsm and adata[1].obsm
    assert "X_umap" in adata[0].obsm, "The key 'X_umap' is not present in adata[0].obsm. Please compute the UMAP coordinates and add them to adata[0].obsm['X_umap'] before calling this function."
    assert "X_umap" in adata[1].obsm, "The key 'X_umap' is not present in adata[1].obsm. Please compute the UMAP coordinates and add them to adata[1].obsm['X_umap'] before calling this function."

    # Unpack cell type key from kwargs, otherwise set it to leiden as default
    cell_type_key = kwargs.get("cell_type_key", "leiden")

    # Unpack dataset name from kwargs, otherwise default
    dataset_name = kwargs.get("dataset_name", "default")

    fig = plt.figure(figsize=(16, 16), dpi=300)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    ax00 = fig.add_subplot(gs[0, 0])
    sc.pl.umap(
        adata[0],
        color="leiden",
        legend_loc="on data",
        frameon=False,
        show=False,
        ax=ax00
    )
    # Delete title
    ax00.set_title(' ')
    # Set new title
    ax00.set_title('a', loc='left', fontsize=20, fontweight='bold')
    plot_customized_UMAP_coordinates(ax00, length=1.0)

    ax02 = fig.add_subplot(gs[0, 1])
    sc.pl.umap(
        adata[0],
        color=cell_type_key,
        frameon=False,
        show=False,
        ax=ax02
    )

    # Delete title
    ax02.set_title(' ')
    # Set new title
    ax02.set_title('b', loc='left', fontsize=20, fontweight='bold')

    plot_customized_UMAP_coordinates(ax02, length=1.0)

    # Dot plot of top DEGs across cluster groups
    deg_cluster_groups = adata[0].obs["leiden"].unique().tolist()

    if tax_level == "Heart" and dataset_name == "tabulaMuris":
        deg_cluster_groups = deg_cluster_groups[:5]
    elif tax_level == "Brain_Non-Myeloid" and dataset_name == "tabulaMuris":
        deg_cluster_groups = deg_cluster_groups[:5]

    ax10_full_row = fig.add_subplot(gs[1, :])

    num_marker_genes = kwargs.get("num_marker_genes", 6)
    sc.pl.rank_genes_groups_dotplot(
        adata[0],
        groups=deg_cluster_groups,
        groupby="leiden",
        standard_scale="var",
        n_genes=num_marker_genes,
        show=False,
        dendrogram=False,
        ax=ax10_full_row
    )
    # Set title for ax10
    ax10_full_row.set_title('c', loc='left', fontsize=20, fontweight='bold')


    # Dot plot of top DSGs across cluster groups
    cluster_groups = adata[1].obs["leiden"].unique().tolist()

    if tax_level == "Heart" and dataset_name == "tabulaMuris":
        cluster_groups = cluster_groups[:5]
    elif tax_level == "Brain_Non-Myeloid" and dataset_name == "tabulaMuris":
        cluster_groups = cluster_groups[:4]

    num_intron_group_markers = kwargs.get("num_intron_group_markers", 3)

    mean_psi_groups_introns, num_cells_valid_matrix, intron_names = dsg_dotplot_helper(
        adata[1],
        cluster_groups,
        num_intron_group_markers
    )

    if rename_isoforms:
        intron_names_plot = rename_isoform_helper(intron_names)
    else:
        intron_names_plot = intron_names

    # Actual dot plot
    # Control how tightly packed the rows are
    row_spacing = 0.75  # < 1.0 makes rows closer together

    # Create a grid of x, y coordinates
    X, Y = np.meshgrid(range(len(intron_names)), range(len(cluster_groups)))
    x_coords = X.flatten()
    y_coords = (Y * row_spacing).flatten()  # compress row spacing

    color_values = mean_psi_groups_introns.flatten()
    size_values = num_cells_valid_matrix.flatten()

    # Make the dot plot less wide
    ax20 = fig.add_subplot(gs[2, :2])

    psi_dotplot = ax20.scatter(
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
    ax20.set_xticks(range(len(intron_names_plot)))
    ax20.set_xticklabels(intron_names_plot, rotation=90, ha='right')

    # Y axis: ticks must match compressed row positions
    y_tick_positions = np.arange(len(cluster_groups)) * row_spacing
    ax20.set_yticks(y_tick_positions)
    ax20.set_yticklabels(cluster_groups)

    # Ensure first and last row dots are fully visible
    ax20.set_ylim(-0.5 * row_spacing, (len(cluster_groups) - 0.5) * row_spacing)

    # Optional: also give a little horizontal margin so edge dots are fully visible
    ax20.set_xlim(-0.5, len(intron_names) - 0.5)

    ax20.set_title('d', loc='left', fontsize=20, fontweight='bold')

    # -----------------------------
    # Custom dot-size legend
    # -----------------------------
    size_legend_values = [50, 100, 250, 500]

    size_ax = inset_axes(
        ax20,
        width="100%",
        height="100%",
        loc="upper left",
        bbox_to_anchor=(1.03, 0.56, 0.38, 0.26),  # narrower legend box
        bbox_transform=ax20.transAxes,
        borderpad=0
    )

    size_ax.set_xlim(-0.5, len(size_legend_values) - 0.5)
    size_ax.set_ylim(0, 1)
    size_ax.axis("off")

    size_ax.set_title(
        "Number of cells\nin group",
        fontsize=10,
        pad=2
    )

    x_positions = np.arange(len(size_legend_values))

    for x, s in zip(x_positions, size_legend_values):
        # dot
        size_ax.scatter(
            x,
            0.62,
            s=s,
            color="gray",
            edgecolors="k",
            alpha=0.9
        )

        # tick below dot
        size_ax.plot(
            [x, x],
            [0.30, 0.42],
            color="black",
            linewidth=1.3
        )

        # number below tick
        size_ax.text(
            x,
            0.10,
            str(s),
            ha="center",
            va="center",
            fontsize=9
        )

    # -----------------------------
    # Horizontal colorbar
    # -----------------------------
    cax = inset_axes(
        ax20,
        width="100%",
        height="100%",
        loc="upper left",
        bbox_to_anchor=(1.03, 0.28, 0.38, 0.06),  # same narrower width
        bbox_transform=ax20.transAxes,
        borderpad=0
    )

    cbar = fig.colorbar(
        psi_dotplot,
        cax=cax,
        orientation="horizontal"
    )

    cbar.set_ticks([0.00, 0.50, 1.00])
    cbar.ax.set_title("Mean PSI value\nin group", fontsize=10, pad=6)

    plt.tight_layout()

    if save_fig:
        file_suffix = kwargs.get("file_suffix", "pdf")
        if tax_level is not None:
            fig_name = "./figures/" + dataset_name + "/deg_dsg_analysis_" + model_name + "_" + tax_level + "." + file_suffix
            fig.savefig(fig_name, dpi=300)
        else:
            fig.savefig("./figures/" + dataset_name + "/deg_dsg_analysis_" + model_name + "." + file_suffix, dpi=300)

def plot_latent_space_benchmarking_scgetuvi(
        adata_objects: Tuple[AnnData, AnnData],
        likelihood_keys: List[str],
        evaluation_df: pd.DataFrame,
        atlas_weight_dict: Dict[str, float],
        mean_weights_df: pd.DataFrame,
        importance_weight_likelihood_key: str,
        seed: int | None,
        random_seeds: List[int],
        umap_color_display_key: str,
        sort_cell_type_key: str,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given two AnnData objects (gene expression and transcript usage) containing the inferred cell embeddings, plot the
    UMAPs of S-GE, S-TU, and GE-TU cell embeddings as well as the bar plots of atlas and cell type specific importance
    weights.

    :param adata_objects: Tuple of two AnnData objects, the first one containing the inferred cell embeddings for gene expression and the second one containing the inferred cell embeddings for transcript usage
    :param likelihood_keys: List of str, the likelihood keys for gene expression and transcript usage
    :param evaluation_df: pd.DataFrame, the evaluation dataframe of the cell embeddings on the atlas level
    :param atlas_weight_dict: Dict of atlas weight dict
    :param mean_weights_df: pandas dataframe of mean importance weights per cell type averaged across random seeds
    :param importance_weight_likelihood_key: The key of the importance weight (either display gene expression or transcript usage importance weights)
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

    # Unpack the results from importance weight analysis
    total_mean_GE = atlas_weight_dict["total_mean_GE"]
    total_std_GE = atlas_weight_dict["total_std_GE"]
    total_mean_TU = atlas_weight_dict["total_mean_TU"]
    total_std_TU = atlas_weight_dict["total_std_TU"]

    # Reset index to make cell_type_key a column for plotting
    mean_weights_df = mean_weights_df.reset_index()

    likelihood_seeds_list = []

    for random_seed in random_seeds:
        likelihood_seeds_list.append(importance_weight_likelihood_key + "_" + str(random_seed))

    mean_weights_transcriptomic_facet = mean_weights_df[likelihood_seeds_list].mean(axis=1)
    std_weights_transcriptomic_facet = mean_weights_df[likelihood_seeds_list].std(axis=1)

    cell_types = mean_weights_df["index"]
    cell_organ_system = mean_weights_df[sort_cell_type_key]

    # Plotting function to be moved into plotting_utils
    ax00 = fig.add_subplot(gs[:2, :2])

    from importlib import resources
    try:
        img_resource = resources.files('crecerelle.default_figures').joinpath('crecerelle_scGETUVI.png')

        with resources.as_file(img_resource) as image_path:
            model_img = mpimg.imread(str(image_path))
            ax00.imshow(model_img)
    except (ImportError, FileNotFoundError):
        print("Warning: scGETUVI architecture figure could not be loaded from package resources.")

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
        fig_name = "./figures/" + dataset_name + "/scGETUVI_benchmarking." + file_suffix
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')


def plot_cell_embeddings_analysis_scgetuvi(
        adata_objects: Tuple[AnnData, AnnData],
        cluster_groups: List[str],
        likelihood_keys: List[str],
        cell_embeddings_selected: List[str],
        cell_embeddings_keys: List[str],
        seeds_selected: List[int],
        seed: int | None,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given two AnnData objects (gene expression and transcript usage) containing the inference results from scGETUVI, a
    specified group of Leiden clusters, likelihood keys, cell embeddings and their respective keys, as well as different
    random seeds, plot the latent spaces of the scGETUVI cell embeddings as UMAP, assess the robustness of the learnt
    importance weights of GE and TU per cell type present, and plot the results of DEG and DSG analysis per cluster
    showing sets of marker genes and sets of marker isoforms. The figure is saved in the figures directory if save_fig
    is set to True.

    :param adata_objects:
    :param cluster_groups:
    :param likelihood_keys:
    :param cell_embeddings_selected:
    :param cell_embeddings_keys:
    :param seeds_selected:
    :param seed:
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
    fig = plt.figure(figsize=(15, 22), dpi=300)
    gs = gridspec.GridSpec(
        6,
        4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.45],
        height_ratios=[1, 1, 1, 1, 1.25, 1.25]
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

    # Importance weights bar plot
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
    ax20.set_ylabel("Importance Weight")

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
    ax20.set_title('b', loc='left', fontsize=20, fontweight='bold')

    # UMAP displaying ZINB weighting
    ax21 = fig.add_subplot(gs[2, 1])
    sc.pl.umap(adata_1, color=f"{likelihood_key_1}_weighting", ax=ax21, show=False, frameon=False,
               title="GE weight", size=10, vmin=0.0, vmax=1.0, colorbar_loc=None)
    plot_customized_UMAP_coordinates(ax21, length=1.0)
    ax21.set_title('c', loc='left', fontsize=20, fontweight='bold')

    # UMAP displaying ZIDM weighting
    ax22 = fig.add_subplot(gs[2, 2])
    adata_2.obsm["X_umap"] = adata_1.obsm["X_umap"].copy()
    sc.pl.umap(adata_2, color=f"{likelihood_key_2}_weighting", ax=ax22, show=False, frameon=False,
               title="TU weight", size=10, vmin=0.0, vmax=1.0)

    plot_customized_UMAP_coordinates(ax22, length=1.0)

    # Boxplot
    ax30 = fig.add_subplot(gs[3, 0])
    sns.boxplot(
        data=df_combined,
        x=cell_type_key,
        y='weight',
        hue='Source',  # This creates the side-by-side comparison
        palette=modality_palette,
        ax=ax30,
        flierprops={
            'marker': '.',  # Use a point/dot instead of a circle
            'markersize': 4,  # Control the size
            'markerfacecolor': 'gray',
            'markeredgecolor': 'none'  # Remove the outline for a cleaner 'dot' look
        }
    )

    ax30.tick_params(axis='x', rotation=45)
    ax30.set_xticklabels([])
    ax30.set_xlabel("")
    ax30.set_ylabel("Importance weight")
    ax30.get_legend().remove()
    ax30.set_title('d', loc='left', fontsize=20, fontweight='bold')

    # Violinplot
    ax31 = fig.add_subplot(gs[3, 1:3])
    sns.violinplot(
        data=df_ratios_melted,
        x=cell_type_key,
        y='log_weight_ratio',
        palette=cell_type_palette,  # 'vlag' is great for centered data (diverging colors)
        inner='quartile',
        legend=False,
        ax=ax31
    )

    ax31.axhline(0, color='red', linestyle='--', alpha=0.6)
    ax31.yaxis.tick_right()
    ax31.yaxis.set_label_position("right")
    ax31.set_xticklabels([])
    ax31.set_xlabel("")
    ax31.set_ylabel(r'$\log\left(\frac{GE\ Weight}{TU\ Weight}\right)$')
    ax31.set_title('e', loc='left', fontsize=20, fontweight='bold')

    for ax in [ax20, ax30, ax31]:
        ax.scatter(x_positions, [-0.1] * len(categories), c=colors, s=80, transform=ax.get_xaxis_transform(),
                   clip_on=False, zorder=5)

    # Dotplot of DEGs
    deg_cluster_groups = adata_1.obs["leiden"].unique().tolist()

    if tax_level == "Heart" and dataset_name == "tabulaMuris":
        deg_cluster_groups = deg_cluster_groups[:5]
    elif tax_level == "Brain_Non-Myeloid" and dataset_name == "tabulaMuris":
        deg_cluster_groups = deg_cluster_groups[:5]

    ax40 = fig.add_subplot(gs[4, :3])
    sc.pl.rank_genes_groups_dotplot(
        adata_1,
        groups=deg_cluster_groups,
        groupby="leiden",
        standard_scale="var",
        n_genes=num_marker_genes,
        show=False,
        dendrogram=False,
        ax=ax40,
        return_fig=False
    )
    ax40.set_title('f', loc='left', fontsize=20, fontweight='bold')

    # Dotplot of DSGs
    ax50 = fig.add_subplot(gs[5, :2]) # prev [5, :3]
    row_spacing = 0.75

    y_coords = (Y * row_spacing).flatten()

    psi_dotplot = ax50.scatter(
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
    ax50.set_xticks(range(len(intron_names_plot)))
    ax50.set_xticklabels(intron_names_plot, rotation=90, ha='right', fontsize=12)

    # Y axis: ticks must match compressed row positions
    y_tick_positions = np.arange(len(cluster_groups)) * row_spacing
    ax50.set_yticks(y_tick_positions)
    ax50.set_yticklabels(cluster_groups)

    # Ensure first and last row dots are fully visible
    ax50.set_ylim(-0.5 * row_spacing, (len(cluster_groups) - 0.5) * row_spacing)

    # Optional: also give a little horizontal margin so edge dots are fully visible
    ax50.set_xlim(-0.5, len(intron_names) - 0.5)
    ax50.set_title('g', loc='left', fontsize=20, fontweight='bold')

    # Custom dot-size legend
    size_legend_values = [50, 100, 250, 500]

    legend_ax = fig.add_subplot(gs[5, 2]) # prev [5,3]
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
    cax = inset_axes(
        legend_ax,
        width="90%",
        height="12%",
        loc="lower center",
        borderpad=0
    )

    cbar = fig.colorbar(
        psi_dotplot,
        cax=cax,
        orientation="horizontal"
    )

    cbar.set_ticks([0.00, 0.50, 1.00])
    cbar.ax.set_title("Mean PSI value\nin group", fontsize=10, pad=6)

    plt.subplots_adjust(
        top=0.94,
        bottom=0.06,
        left=0.06,
        right=0.97,
        hspace=0.65,
        wspace=0.35
    )
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

    if save_fig:

        file_suffix = kwargs.get("file_suffix", 'pdf')
        fig_name = "./figures/" + dataset_name + "/clustering_scGETUVI_" + tax_level + "." + file_suffix
        fig.savefig(fig_name, dpi=300)

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

def plot_shared_and_unique_pathways_upsetplot(
        deg_pathways: int,
        dsg_pathways: int,
        shared_pathways: int,
        tissue: str | None,
        dataset_name: str,
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given the number of significantly enriched pathwats for DEGs, DSGs, and both, plot an upset plot to visualize the
    shared and unique significantly enriched pathways for DEGs and DSGs. The figure is saved in the figures directory
    if save_fig is set to True.

    :param deg_pathways: int, the number of significantly enriched pathways for DEGs
    :param dsg_pathways: int, the number of significantly enriched pathways for DSGs
    :param shared_pathways: int, the number of significantly enriched pathways for both DEGs and DSGs
    :param tissue: str, the name of the tissue for which the analysis was performed, or None if the analysis was performed on all tissues
    :param dataset_name: str, the name of the dataset (e.g. "tabulaMuris") for which the analysis was performed, used for saving the figure
    :param save_fig: bool, whether to save the figure
    """

    # Each list inside the first argument represents a set "membership"
    upset_data = upsetplot.from_memberships(
        [
            ['DEG BPs'],  # Only in DEG
            ['DSG BPs'],  # Only in DSG
            ['DEG BPs', 'DSG BPs']  # In both (Overlapping)
        ],
        data=[deg_pathways, dsg_pathways, shared_pathways]
    )

    # 3. Create and show the plot
    upset = upsetplot.UpSet(upset_data, subset_size='sum', show_counts=True)
    upset.plot()

    if save_fig:
        # Add unpacking of kwargs dataset_name
        if tissue is not None:
            plt.savefig(f"./figures/{dataset_name}/upset_plot_deg_dsg_pathways_{tissue}.png", bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f"./figures/{dataset_name}/upset_plot_deg_dsg_pathways.png", bbox_inches='tight', dpi=300)

    plt.show()

def plot_umap_latent_space_scgetuvi(
        adata_objects: Tuple[AnnData, AnnData],
        adata_1_latent_space_keys: List[str] | None,
        adata_2_latent_space_keys: List[str] | None,
        latent_space_display: List[str],
        color_key: List[str],
        save_fig: bool = True,
        **kwargs
):
    r"""
    Given a tuple of AnnData objects where the first contains the gene expression data and the second contains the
    transcript usage data, plot the UMAPs of the latent space of scGETUVI dependent on the keys given. The latent space
    keys must be present in the AnnData objects, and dependent on the specified keys, the according UMAPs will be
    displayed. The color key specifies what annotation should be used for coloring the UMAPs and must be present in the
    AnnData objects. The figure is saved in the figures directory if save_fig is set to True.

    :param adata_objects: Tuple[AnnData, AnnData]
    :param adata_1_latent_space_keys: List[str]
    :param adata_2_latent_space_keys: List[str]
    :param latent_space_display: List[str]
    :param color_key: str
    :param save_fig: bool, whether to save the figure

    """
    color_key_1, color_key_2 = color_key
    # Check if latent space keys are present in .obsm
    if adata_1_latent_space_keys is not None:
        assert all(key in adata_objects[0].obsm for key in adata_1_latent_space_keys), "Not all latent space keys are present in gene expression data adata_objects[0].obsm. Please ensure that the specified keys are present before calling this function."
    if adata_2_latent_space_keys is not None:
        assert all(key in adata_objects[1].obsm for key in adata_2_latent_space_keys), "Not all latent space keys are present in transcript usage data adata_objects[1].obsm. Please ensure that the specified keys are present before calling this function."

    # Check color_key is present in .obs
    assert color_key_1 in adata_objects[0].obs, f"The color key '{color_key_1}' is not present in gene expression data adata_objects[0].obs. Please ensure that the specified key is present before calling this function."
    assert color_key_2 in adata_objects[1].obs, f"The color key '{color_key_2}' is not present in transcript usage data adata_objects[1].obs. Please ensure that the specified key is present before calling this function."

    # Calculate UMAPs for GE AnnData object
    umap_keys_1 = []
    if adata_1_latent_space_keys is not None:
        for latent_space_key_1 in adata_1_latent_space_keys:
            # Keep part of latent_space_key_1 without "_latent_mean"

            umap_key_1 = latent_space_key_1[:-12] + "_X_umap"

            seed = kwargs.get("seed", None)
            if seed is not None:
                adata_objects[0].obsm[umap_key_1] = UMAP(n_components=2, random_state=seed).fit_transform(adata_objects[0].obsm[latent_space_key_1])
            else:
                adata_objects[0].obsm[umap_key_1] = UMAP(n_components=2).fit_transform(adata_objects[0].obsm[latent_space_key_1])

            umap_keys_1.append(umap_key_1)

    # Calculate UMAPs for TU AnnData object
    umap_keys_2 = []
    if adata_2_latent_space_keys is not None:
        for latent_space_key_2 in adata_2_latent_space_keys:
            umap_key = latent_space_key_2[:-12] + "_X_umap"
            seed = kwargs.get("seed", None)
            if seed is not None:
                adata_objects[1].obsm[umap_key] = UMAP(n_components=2, random_state=seed).fit_transform(adata_objects[1].obsm[latent_space_key_2])
            else:
                adata_objects[1].obsm[umap_key] = UMAP(n_components=2).fit_transform(adata_objects[1].obsm[latent_space_key_2])

            umap_keys_2.append(umap_key)

    seed = kwargs.get("seed", None)

    num_umaps = len(adata_1_latent_space_keys) + len(adata_2_latent_space_keys)

    fig = plt.figure(figsize=(20, 25), dpi=300)
    gs = gridspec.GridSpec(4, 2, figure=fig)


    # Joint GE-TU
    if "GE-TU" in latent_space_display:
        ax00 = fig.add_subplot(gs[:2, :2])

        if seed is None:
            pattern = re.compile(r"^(ZINB|NB)_\d+_(ZIDM|DM|ZANIDM)_\d+_shared_X_umap$")
        else:
            pattern = re.compile(r"^(ZINB|NB)_" + str(seed) + "_(ZIDM|DM|ZANIDM)_" + str(seed) + "_shared_X_umap$")


        matches = [key for key in umap_keys_1 if pattern.match(key)]

        x_umap_key = matches[0]  # We only expect one match

        adata_objects[0].obsm["X_umap"] = adata_objects[0].obsm[x_umap_key].copy()
        if "weighting" in color_key_1:
            sc.pl.umap(adata_objects[0], color=color_key_1, ax=ax00, show=False, frameon=False, vmin=0.0, vmax=1.0)
        else:
            sc.pl.umap(adata_objects[0], color=color_key_1, ax=ax00, show=False, frameon=False)
        plot_customized_UMAP_coordinates(ax00, length=1.0)
        ax00.set_title("GE-TU", loc='center', fontsize=16)
        del adata_objects[0].obsm["X_umap"]

    # S-GE UMAP
    if "S-GE" in latent_space_display:
        ax20 = fig.add_subplot(gs[2, 0])
        if seed is None:
            pattern = re.compile(r"^(ZINB|NB)_\d+_shared_X_umap$")
        else:
            pattern = re.compile(r"^(ZINB|NB)_" + str(seed) +"_shared_X_umap$")

        matches = [key for key in umap_keys_1 if pattern.match(key)]

        x_umap_key = matches[0] # We only expect one match

        adata_objects[0].obsm["X_umap"] = adata_objects[0].obsm[x_umap_key].copy()
        if "weighting" in color_key_1:
            sc.pl.umap(adata_objects[0], color=color_key_1, ax=ax20, show=False, frameon=False, vmin=0.0, vmax=1.0)
        else:
            sc.pl.umap(adata_objects[0], color=color_key_1, ax=ax20, show=False, frameon=False)
        plot_customized_UMAP_coordinates(ax20, length=1.0)
        ax20.set_title("S-GE", loc='center', fontsize=16)
        del adata_objects[0].obsm["X_umap"]

    # S-TU UMAP
    if "S-TU" in latent_space_display:
        ax21 = fig.add_subplot(gs[2, 1])
        if seed is None:
            pattern = re.compile(r"^(ZIDM|DM|ZANIDM)_\d+_shared_X_umap$")
        else:
            pattern = re.compile(r"^(ZIDM|DM|ZANIDM)_" + str(seed) + "_shared_X_umap$")

        matches = [key for key in umap_keys_2 if pattern.match(key)]

        x_umap_key = matches[0]  # We only expect one match

        adata_objects[1].obsm["X_umap"] = adata_objects[1].obsm[x_umap_key].copy()
        if "weighting" in color_key_2:
            sc.pl.umap(adata_objects[1], color=color_key_2, ax=ax21, show=False, frameon=False, vmin=0.0, vmax=1.0)
        else:
            sc.pl.umap(adata_objects[1], color=color_key_2, ax=ax21, show=False, frameon=False)
        plot_customized_UMAP_coordinates(ax21, length=1.0)
        ax21.set_title("S-TU", loc='center', fontsize=16)
        del adata_objects[1].obsm["X_umap"]

    # P-GE UMAP
    if "P-GE" in latent_space_display:
        ax30 = fig.add_subplot(gs[3, 0])
        if seed is None:
            pattern = re.compile(r"^(ZINB|NB)_\d+_private_X_umap$")
        else:
            pattern = re.compile(r"^(ZINB|NB)_" + str(seed) + "_private_X_umap$")

        matches = [key for key in umap_keys_1 if pattern.match(key)]

        x_umap_key = matches[0]  # We only expect one match

        adata_objects[0].obsm["X_umap"] = adata_objects[0].obsm[x_umap_key].copy()
        if "weighting" in color_key_1:
            sc.pl.umap(adata_objects[0], color=color_key_1, ax=ax30, show=False, frameon=False, vmin=0.0, vmax=1.0)
        else:
            sc.pl.umap(adata_objects[0], color=color_key_1, ax=ax30, show=False, frameon=False)
        plot_customized_UMAP_coordinates(ax30, length=1.0)
        ax30.set_title("P-GE", loc='center', fontsize=16)
        del adata_objects[0].obsm["X_umap"]


    # P-TU UMAP
    if "P-TU" in latent_space_display:
        ax31 = fig.add_subplot(gs[3, 1])
        if seed is None:
            pattern = re.compile(r"^(ZIDM|DM|ZANIDM)_\d+_private_X_umap$")
        else:
            pattern = re.compile(r"^(ZIDM|DM|ZANIDM)_" + str(seed) + "_private_X_umap$")

        matches = [key for key in umap_keys_2 if pattern.match(key)]

        x_umap_key = matches[0]  # We only expect one match

        adata_objects[1].obsm["X_umap"] = adata_objects[1].obsm[x_umap_key].copy()
        if "weighting" in color_key_1:
            sc.pl.umap(adata_objects[1], color=color_key_2, ax=ax31, show=False, frameon=False, vmin=0.0, vmax=1.0)
        else:
            sc.pl.umap(adata_objects[1], color=color_key_2, ax=ax31, show=False, frameon=False)
        plot_customized_UMAP_coordinates(ax31, length=1.0)
        ax31.set_title("P-TU", loc='center', fontsize=16)
        del adata_objects[1].obsm["X_umap"]

    if save_fig:
        dataset_name = kwargs.get("dataset_name", "default")
        tax_level = kwargs.get("tax_level", "atlas_level")
        file_suffix = kwargs.get("file_suffix", "pdf")

        fig_name = "./figures/" + dataset_name + "/umap_latent_space_scgetuvi_" + tax_level + "." + file_suffix
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')


    plt.show()



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
        image_path = "./figures/crecerelle_scTUVI.png"
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
    :param predictions: bool, if the model contains predictions (reconstructions, latent means, and UMAP)
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

def plot_latent_space_evaluation_scgetuvi(
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
    image_path_scgetuvi_shared_private = "./figures/crecerelle_scGETUVI_shared_private_latent.png"
    shared_private_latent_img = mpimg.imread(image_path_scgetuvi_shared_private)
    image_path_scgetuvi_shared_mixing = "./figures/crecerelle_scGETUVI_shared_latent_mixing.png"
    scgetuvi_shared_mixing_img = mpimg.imread(image_path_scgetuvi_shared_mixing)

    # Display the probabilistic model of scGETUVI
    axes[0, 0].imshow(shared_private_latent_img)
    axes[0, 0].axis('off')

    # Display the probabilistic model of scGETUVI
    axes[0, 1].imshow(scgetuvi_shared_mixing_img)
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
        # plt.savefig(f"./figures/tabulaMuris/scgetuvi_analysis_"+ tissue +".pdf", dpi=300, bbox_inches='tight')
        if tissue is None:
            plt.savefig(f"./figures/" + dataset_name + "/scgetuvi_analysis.png", bbox_inches='tight')
        else:
            plt.savefig(f"./figures/" + dataset_name + "/scgetuvi_analysis_" + tissue + ".png", bbox_inches='tight')

    plt.show()

def plot_cell_type_predictions_scgetuvi_marker_genes(
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

    assert classifier_1 in ["private_1", "private_2","shared_uni_1", "shared_uni_2", "shared"], "Classifier 1 must use one of the following scGETUVI embeddings: 'private_1', 'private_2', 'shared_uni_1', 'shared_uni_2', 'shared'"
    assert classifier_2 in ["private_1", "private_2","shared_uni_1", "shared_uni_2", "shared"], "Classifier 2 must use one of the following scGETUVI embeddings: 'private_1', 'private_2', 'shared_uni_1', 'shared_uni_2', 'shared'"

    # Create dataframes (overall, classifer 1, classifer 2) for plotting
    classification_overall_df = cell_type_classification_scgetuvi_dataframe(classification_dict, embedding_types)
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
        plt.savefig(f"./figures/tabulaMuris/scgetuvi_cell_type_predictions_{tissue}_{cell_type_1}_{cell_type_2}.pdf", dpi=300, bbox_inches='tight')

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

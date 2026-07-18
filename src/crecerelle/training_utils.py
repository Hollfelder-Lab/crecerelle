#!/usr/bin/env python3
import pdb
import time

import numpy as np
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn

from anndata import AnnData

import os

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import device

from .models import BetaVAE, SCGETUVI, LogisticRegressionClassifier
import matplotlib.pyplot as plt

def plot_loss(
        epoch_train_loss_list: list,
        epoch_val_loss_list: list,
        loss_type: str,
        dataset_name: str,
        model_name: str,
        save_fig: bool = False
) -> None:
    r"""
    Given a list of epoch losses, the evolution of the loss is plotted over the epochs

    :param epoch_train_loss_list: list, comprising the loss of each epoch
    :param loss_type: str, the name of the applied loss function e.g. log evidence, Variational ELBO
    :param dataset_name: str, name of dataset
    :param model_name: str, name of model
    :param save_fig: boolean variable, if True it saves the plot as PDF file
    :return: None
    """
    epoch_list = np.arange(1, len(epoch_train_loss_list) + 1)
    epoch_train_loss_list = np.array(epoch_train_loss_list)
    epoch_val_loss_list = np.array(epoch_val_loss_list)

    title = loss_type
    x_label = "Number of epochs"
    y_label = "Loss"
    fig, ax = plt.subplots()
    ax.set_title(title)
    # ax.set_xticks(np.arange(1, len(epoch_train_loss_list) + 1))
    plt.plot(epoch_list, epoch_train_loss_list)
    plt.plot(epoch_list, epoch_val_loss_list)
    # ax.legend(loc="lower right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()

    if save_fig:
        plt.savefig("./figures/tabulaMuris/" + dataset_name + "_" + model_name + "_" + loss_type + ".png", bbox_inches='tight')


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


def random_dataset_split(
        dataset: Dataset,
        dataset_ratio: np.ndarray = np.array([0.8, 0.1, 0.1])
) -> Tuple[Dataset, Dataset, Dataset, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""
    Given a PyTorch dataset and a dataset ratio of (% training, % validation, % test), a set of random indices is
    created to then split the dataset into a training, validation, and test set

    :param dataset:
    :param dataset_ratio:
    """

    # Convert ratio into sample sizes
    dataset_size = len(dataset)
    val_test_sizes = np.ceil(dataset_ratio[1:] * dataset_size).astype(int)
    train_size = dataset_size - np.sum(val_test_sizes)
    val_size = val_test_sizes[0]

    # Generate random indices
    random_indices = torch.randperm(dataset_size)
    train_indices = torch.sort(random_indices[:train_size])[0]
    val_indices = torch.sort(random_indices[train_size:train_size + val_size])[0]
    test_indices = torch.sort(random_indices[train_size + val_size:])[0]

    # Original indices (training, validation, test)
    original_indices = (train_indices.numpy(), val_indices.numpy(), test_indices.numpy())

    # Create training, validation, and test dataset
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset, original_indices


def random_multiple_dataset_split(
        datasets: Tuple[Dataset, Dataset],
        num_datasets: int = 2,
        dataset_ratio: np.ndarray = np.array([0.8, 0.1, 0.1])
) -> Tuple[Dict[str, Dict[str, Dataset]], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""
    Given a tuple of PyTorch datasets and a dataset ratio of (% training, % validation, % test), a set of random indices is
    created to then split each dataset with the same random indices into a training, validation, and test set

    :param datasets:
    :param num_datasets:
    :param dataset_ratio:
    :return:
    """
    # Check all datasets have same number of data points
    dataset_size = len(datasets[0])

    for dataset_idx in range(0, num_datasets):
        assert dataset_size == len(datasets[dataset_idx]), "All datasets must have the same number of data points"

    # Convert ratio into sample sizes
    val_test_sizes = np.ceil(dataset_ratio[1:] * dataset_size).astype(int)
    train_size = dataset_size - np.sum(val_test_sizes)
    val_size = val_test_sizes[0]

    # Generate random indices
    random_indices = torch.randperm(dataset_size)
    train_indices = torch.sort(random_indices[:train_size])[0]
    val_indices = torch.sort(random_indices[train_size:train_size + val_size])[0]
    test_indices = torch.sort(random_indices[train_size + val_size:])[0]

    # Original indices (training, validation, test)
    original_indices = (train_indices.numpy(), val_indices.numpy(), test_indices.numpy())

    # Create a dictionary for all datasets
    dataset_dict = {}
    for dataset_idx in range(0, num_datasets):
        dataset_key = "Dataset " + str(dataset_idx + 1)
        dataset_dict[dataset_key] = None

    # Create for each dictionary entry training, validation, and test dataset
    for dataset_key in dataset_dict.keys():
        print(int(dataset_key.split()[-1]) - 1)
        dataset_dict[dataset_key] = {
            "Training": torch.utils.data.Subset(datasets[int(dataset_key.split()[-1]) - 1], train_indices),
            "Validation": torch.utils.data.Subset(datasets[int(dataset_key.split()[-1]) - 1], val_indices),
            "Test": torch.utils.data.Subset(datasets[int(dataset_key.split()[-1]) - 1], test_indices)
        }

    return dataset_dict, original_indices


def train_VAE_one_epoch_OLD(
        model: BetaVAE,
        optimizer: Adam,
        train_loader: DataLoader,
        count_data_included: bool = True
) -> Tuple[float, float, float]:
    r"""
    Given the training data loader, train the VAE one epoch and report the batch and the loss during each iteration.
    The last loss is returned.

    :param model:
    :param optimizer:
    :param train_loader:
    :param count_data_included:
    :return:
    """

    # Set model to training mode
    model.train()

    avg_loss = 0
    avg_nll = 0
    avg_kld = 0

    running_loss = 0
    running_nll = 0
    running_kld = 0
    num_trained_batches = 0

    for i, batch in enumerate(train_loader):

        # Zero gradients for every batch
        optimizer.zero_grad()

        # To do later: include noise sampling here instead of inside the model
        eps = auxiliary_noise((batch[1].shape[0], model.latent_dim))

        if count_data_included:
            input_batch_counts, input_batch_levels, _, _ = batch
            input_batch = (input_batch_counts.to(model.device), input_batch_levels.to(model.device))
            eps = eps.to(model.device)
        else:
            _, input_batch, _, _ = batch

            input_batch = input_batch.to(model.device)
            eps = eps.to(model.device)

        # Compute the ELBO and its gradients
        elbo, kld, nll = model.forward(input_batch, eps)
        loss = elbo.mean()
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        running_nll += nll.mean().item()
        running_kld += kld.mean().item()

        avg_loss += loss.item()
        avg_nll += nll.mean().item()
        avg_kld += kld.mean().item()
        num_trained_batches += 1

        if i % 10 == 9:
            last_loss = running_loss / num_trained_batches  # average loss per data point (previously running_loss / 10)
            print(f"   batch {i + 1} loss: {last_loss}")
            print(f"        NLL: {running_nll / num_trained_batches}, KLD: {running_kld / num_trained_batches}")
            running_loss = 0
            running_nll = 0
            running_kld = 0

            num_trained_batches = 0

    # avg_loss = avg_loss / len(train_loader.dataset)
    avg_loss = avg_loss / len(train_loader)
    avg_nll = avg_nll / len(train_loader)
    avg_kld = avg_kld / len(train_loader)

    return avg_loss, avg_nll, avg_kld

def train_VAE_one_epoch(
        model: BetaVAE,
        optimizer: Adam,
        train_loader: DataLoader,
        count_data_included: bool = True
) -> Tuple[float, float, float]:
    r"""
    Given the training data loader, train the VAE one epoch and report the batch and the loss during each iteration.
    The last loss is returned.

    :param model:
    :param optimizer:
    :param train_loader:
    :param count_data_included:
    :return:
    """

    model.train()

    total_loss = 0.0
    total_nll = 0.0
    total_kld = 0.0
    total_samples = 0

    running_loss = 0.0
    running_nll = 0.0
    running_kld = 0.0
    running_samples = 0

    for i, batch in enumerate(train_loader):

        optimizer.zero_grad()

        if count_data_included:
            input_batch_counts, input_batch_levels, _, _ = batch
            batch_size = input_batch_counts.shape[0]

            input_batch = (
                input_batch_counts.to(model.device),
                input_batch_levels.to(model.device)
            )
        else:
            _, input_batch, _, _ = batch
            batch_size = input_batch.shape[0]

            input_batch = input_batch.to(model.device)

        eps = auxiliary_noise((batch_size, model.latent_dim)).to(model.device)

        # model.forward is assumed to return per-sample tensors
        # or tensors whose mean gives the batch objective.
        elbo, kld, nll = model.forward(input_batch, eps)

        # If `elbo` is actually the mathematical ELBO, use:
        # loss = -elbo.mean()
        loss = elbo.mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_loss = loss.item()
        batch_nll = nll.mean().item()
        batch_kld = kld.mean().item()

        # Weight batch means by batch size so the epoch average is per-sample.
        total_loss += batch_loss * batch_size
        total_nll += batch_nll * batch_size
        total_kld += batch_kld * batch_size
        total_samples += batch_size

        running_loss += batch_loss * batch_size
        running_nll += batch_nll * batch_size
        running_kld += batch_kld * batch_size
        running_samples += batch_size

        if i % 10 == 9:
            print(f"   batch {i + 1} loss: {running_loss / running_samples}")
            print(
                f"        NLL: {running_nll / running_samples}, "
                f"KLD: {running_kld / running_samples}"
            )

            running_loss = 0.0
            running_nll = 0.0
            running_kld = 0.0
            running_samples = 0

    avg_loss = total_loss / total_samples
    avg_nll = total_nll / total_samples
    avg_kld = total_kld / total_samples

    return avg_loss, avg_nll, avg_kld

def train_MMVAEplus_one_epoch(
        model: SCGETUVI,
        optimizer: Adam,
        train_loader: DataLoader,
        beta_kl_warmup_epoch: Optional[List[float]] = None,
        temp_epoch: float = 0.1,
        num_samples: int = 1,
        device: str = "cuda"
) -> tuple[float, float, float, float, float, float, float]:
    r"""
    Given a dataloader containing the training dataset, the optimiser, and a model, train SCGETUVI one epoch.

    :param model: SCGETUVI
    :param optimizer: torch.optim.Adam
    :param train_loader: torch.utils.data.DataLoader
    :param beta_kl_warmup_epoch: list of float, the beta values for KL warm-up for the current epoch for both modalities
    :param temp_epoch: float, the temperature value for modality weight annealing for the current epoch
    :param num_samples: int, number of noise samples to draw for Monte Carlo estimation
    :param device: selected torch.device either "cpu" or "cuda"
    :return: float, the average loss of the epoch

    Returns per-sample averages for:
        - total loss / negative ELBO
        - modality 1 ELBO/loss
        - modality 2 ELBO/loss
        - modality 1 NLL
        - modality 2 NLL
        - modality 1 KLD
        - modality 2 KLD
    """

    model.train()

    if beta_kl_warmup_epoch is None:
        beta_kl_warmup_epoch = [1.0, 1.0]

    total_loss = 0.0
    total_elbo_1 = 0.0
    total_elbo_2 = 0.0
    total_nll_1 = 0.0
    total_nll_2 = 0.0
    total_kld_1 = 0.0
    total_kld_2 = 0.0
    total_samples = 0

    running_loss = 0.0
    running_elbo_1 = 0.0
    running_elbo_2 = 0.0
    running_nll_1 = 0.0
    running_nll_2 = 0.0
    running_kld_1 = 0.0
    running_kld_2 = 0.0
    running_samples = 0

    for i, batch in enumerate(train_loader):
        # 1 is gene expression data, 2 is transcript usage data
        (
            input_batch_1_counts,
            input_batch_1_levels,
            input_batch_2_counts,
            input_batch_2_levels,
            _,
            _
        ) = batch

        batch_size = input_batch_1_counts.shape[0]

        optimizer.zero_grad(set_to_none=True)

        eps_1 = auxiliary_noise(
            (batch_size, model.latent_dim[2] + model.latent_dim[0]),
            num_samples
        ).to(device)

        eps_2 = auxiliary_noise(
            (batch_size, model.latent_dim[2] + model.latent_dim[1]),
            num_samples
        ).to(device)

        eps_3 = auxiliary_noise(
            (batch_size, model.latent_dim[0]),
            num_samples
        ).to(device)

        eps_4 = auxiliary_noise(
            (batch_size, model.latent_dim[1]),
            num_samples
        ).to(device)

        input_batch_1_counts = input_batch_1_counts.to(device)
        input_batch_1_levels = input_batch_1_levels.to(device)
        input_batch_2_counts = input_batch_2_counts.to(device)
        input_batch_2_levels = input_batch_2_levels.to(device)

        input_batch = (
            input_batch_1_counts,
            input_batch_1_levels,
            input_batch_2_counts,
            input_batch_2_levels
        )

        eps = (eps_1, eps_2, eps_3, eps_4)

        elbo, elbo_1, elbo_2, nll_1, nll_2, kld_1, kld_2 = model(
            input_batch,
            eps,
            beta_kl_warmup_epoch,
            temp_epoch
        )

        # This "elbo" is already the negative ELBO for minimization
        loss = elbo.mean()

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss detected: loss={loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_loss = loss.item()
        batch_elbo_1 = elbo_1.mean().item()
        batch_elbo_2 = elbo_2.mean().item()
        batch_nll_1 = nll_1.mean().item()
        batch_nll_2 = nll_2.mean().item()
        batch_kld_1 = kld_1.mean().item()
        batch_kld_2 = kld_2.mean().item()

        # Weight each batch mean by batch size so the epoch average is per-sample.
        total_loss += batch_loss * batch_size
        total_elbo_1 += batch_elbo_1 * batch_size
        total_elbo_2 += batch_elbo_2 * batch_size
        total_nll_1 += batch_nll_1 * batch_size
        total_nll_2 += batch_nll_2 * batch_size
        total_kld_1 += batch_kld_1 * batch_size
        total_kld_2 += batch_kld_2 * batch_size
        total_samples += batch_size

        running_loss += batch_loss * batch_size
        running_elbo_1 += batch_elbo_1 * batch_size
        running_elbo_2 += batch_elbo_2 * batch_size
        running_nll_1 += batch_nll_1 * batch_size
        running_nll_2 += batch_nll_2 * batch_size
        running_kld_1 += batch_kld_1 * batch_size
        running_kld_2 += batch_kld_2 * batch_size
        running_samples += batch_size

        if i % 10 == 9:
            print(f"   batch {i + 1} loss: {running_loss / running_samples}")
            print(
                f"        ELBO 1: {running_elbo_1 / running_samples}, "
                f"ELBO 2: {running_elbo_2 / running_samples}"
            )
            print(
                f"        NLL 1: {running_nll_1 / running_samples}, "
                f"NLL 2: {running_nll_2 / running_samples}"
            )
            print(
                f"        KLD 1: {running_kld_1 / running_samples}, "
                f"KLD 2: {running_kld_2 / running_samples}"
            )

            running_loss = 0.0
            running_elbo_1 = 0.0
            running_elbo_2 = 0.0
            running_nll_1 = 0.0
            running_nll_2 = 0.0
            running_kld_1 = 0.0
            running_kld_2 = 0.0
            running_samples = 0

    avg_loss = total_loss / total_samples
    avg_elbo_1 = total_elbo_1 / total_samples
    avg_elbo_2 = total_elbo_2 / total_samples
    avg_nll_1 = total_nll_1 / total_samples
    avg_nll_2 = total_nll_2 / total_samples
    avg_kld_1 = total_kld_1 / total_samples
    avg_kld_2 = total_kld_2 / total_samples

    return (
        avg_loss,
        avg_elbo_1,
        avg_elbo_2,
        avg_nll_1,
        avg_nll_2,
        avg_kld_1,
        avg_kld_2
    )


def train_MMVAEplus_one_epoch_OLD(
        model: SCGETUVI,
        optimizer: Adam,
        train_loader: DataLoader,
        beta_kl_warmup_epoch: List[float] = [1.0, 1.0],
        temp_epoch: float = 0.1, # ADDED FOR SCGETUVI_EXP
        num_samples: int = 1,
        device: device = "cuda"
) -> tuple[float, float, float, float, float, float, float]:
    r"""
    Given a dataloader containing the training dataset, the optimiser, and a model, train the MMVAE+ one epoch.

    :param model: GeneExpressionTranscriptUsageMMVAEplus
    :param optimizer: torch.optim.Adam
    :param train_loader: torch.utils.data.DataLoader
    :param beta_kl_warmup_epoch: list of float, the beta values for KL warm-up for the current epoch for both modalities
    :param temp_epoch: float, the temperature value for modality weight annealing for the current epoch
    :param num_samples: int, number of noise samples to draw for Monte Carlo estimation
    :param device: selected torch.device either "cpu" or "cuda"
    :return: float, the average loss of the epoch
    """
    model.train()

    avg_loss = 0
    avg_elbo_1 = 0
    avg_elbo_2 = 0
    avg_nll_1 = 0
    avg_nll_2 = 0
    avg_kld_1 = 0
    avg_kld_2 = 0

    running_loss = 0
    running_elbo_1 = 0
    running_elbo_2 = 0
    running_nll_1 = 0
    running_nll_2 = 0
    running_kld_1 = 0
    running_kld_2 = 0
    num_trained_batches = 0

    for i, batch in enumerate(train_loader):
        # 1 is for gene expression data and 2 for transcript usage data
        input_batch_1_counts, input_batch_1_levels, input_batch_2_counts, input_batch_2_levels, _, _ = batch
        batch_size = input_batch_1_counts.shape[0]
        # Zero gradients for every batch
        optimizer.zero_grad()

        # To do later: include noise sampling here instead of inside the model
        eps_1 = auxiliary_noise((batch_size, model.latent_dim[2] + model.latent_dim[0]), num_samples) # noise for [z_1, w_1]
        eps_2 = auxiliary_noise((batch_size, model.latent_dim[2] + model.latent_dim[1]), num_samples) # noise for [z_2, w_2]
        eps_3 = auxiliary_noise((batch_size, model.latent_dim[0]), num_samples) # noise for w_aux_1
        eps_4 = auxiliary_noise((batch_size, model.latent_dim[1]), num_samples) # noise for w_aux_2

        # Push data to device ("cuda" or "cpu")
        input_batch_1_counts = input_batch_1_counts.to(device)
        input_batch_1_levels = input_batch_1_levels.to(device)
        input_batch_2_counts= input_batch_2_counts.to(device)
        input_batch_2_levels = input_batch_2_levels.to(device)
        eps_1 = eps_1.to(device)
        eps_2 = eps_2.to(device)
        eps_3 = eps_3.to(device)
        eps_4 = eps_4.to(device)

        input_batch = (input_batch_1_counts, input_batch_1_levels, input_batch_2_counts, input_batch_2_levels)
        eps = (eps_1, eps_2, eps_3, eps_4)

        # Compute the ELBO and its gradients
        elbo, elbo_1, elbo_2, nll_1, nll_2, kld_1, kld_2 = model.forward(input_batch, eps, beta_kl_warmup_epoch, temp_epoch) # ADDED temp_epoch for weight regularization annealing
        loss = elbo.mean()
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        running_elbo_1 += elbo_1.mean().item()
        running_elbo_2 += elbo_2.mean().item()
        running_nll_1 += nll_1.mean().item()
        running_nll_2 += nll_2.mean().item()
        running_kld_1 += kld_1.mean().item()
        running_kld_2 += kld_2.mean().item()

        avg_loss += loss.item()
        avg_elbo_1 += elbo_1.mean().item()
        avg_elbo_2 += elbo_2.mean().item()
        avg_nll_1 += nll_1.mean().item()
        avg_nll_2 += nll_2.mean().item()
        avg_kld_1 += kld_1.mean().item()
        avg_kld_2 += kld_2.mean().item()

        num_trained_batches += 1

        if i % 10 == 9:
            last_loss = running_loss / num_trained_batches  # average loss per batch
            print(f"   batch {i + 1} loss: {last_loss}")
            print(f"        ELBO 1: {running_elbo_1 / num_trained_batches}, ELBO 2: {running_elbo_2 / num_trained_batches}")
            print(f"        NLL 1: {running_nll_1 / num_trained_batches}, NLL 2: {running_nll_2 / num_trained_batches}")
            print(f"        KLD 1: {running_kld_1 / num_trained_batches}, KLD 2: {running_kld_2 / num_trained_batches}")
            running_loss = 0
            running_elbo_1 = 0
            running_elbo_2 = 0
            running_nll_1 = 0
            running_nll_2 = 0
            running_kld_1 = 0
            running_kld_2 = 0
            num_trained_batches = 0

    # avg_loss = avg_loss / len(train_loader.dataset)
    avg_loss = avg_loss / len(train_loader)
    avg_elbo_1 = avg_elbo_1 / len(train_loader)
    avg_elbo_2 = avg_elbo_2 / len(train_loader)
    avg_nll_1 = avg_nll_1 / len(train_loader)
    avg_nll_2 = avg_nll_2 / len(train_loader)
    avg_kld_1 = avg_kld_1 / len(train_loader)
    avg_kld_2 = avg_kld_2 / len(train_loader)

    return avg_loss, avg_elbo_1, avg_elbo_2, avg_nll_1, avg_nll_2, avg_kld_1, avg_kld_2

def train_vae_embedding_cell_type_classifier_one_epoch(
        model,
        optimizer: Adam,
        train_loader: DataLoader,
        loss_fn: nn.CrossEntropyLoss
) -> float:
    r"""

    """
    # Set model to training mode
    model.train()

    avg_loss = 0
    running_loss = 0
    num_trained_batches = 0

    for i, batch in enumerate(train_loader):

        # Zero gradients for every batch
        optimizer.zero_grad()

        input_batch, target_batch, _ = batch

        # Push input_batch and target_batch to device
        input_batch = input_batch.to(model.device)
        target_batch = target_batch.to(model.device)

        logits = model.forward(input_batch)

        loss = loss_fn(logits, target_batch) # not necessary to take .mean()
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        avg_loss += loss.item()
        num_trained_batches += 1

        if i % 10 == 9:
            last_loss = running_loss / num_trained_batches  # average loss per data point (previously running_loss / 10)
            print(f"   batch {i + 1} loss: {last_loss}")
            running_loss = 0
            num_trained_batches = 0

    # avg_loss = avg_loss / len(train_loader.dataset)
    avg_loss = avg_loss / len(train_loader)

    return avg_loss

def train_classifier_one_epoch(
        model,
        optimizer: Adam,
        train_loader: DataLoader,
        embedding_indices: Tuple[int],
        loss_fn: nn.CrossEntropyLoss
) -> float:
    r"""

    """
    # Set model to training mode
    model.train()

    avg_loss = 0
    running_loss = 0
    num_trained_batches = 0

    for i, batch in enumerate(train_loader):

        # Zero gradients for every batch
        optimizer.zero_grad()

        input_private_emb_1, input_private_emb_2, shared_unimodal_emb_1, shared_unimodal_emb_2, shared_emb, target_batch, _ = batch

        input_batch_embs = (input_private_emb_1, input_private_emb_2, shared_unimodal_emb_1, shared_unimodal_emb_2, shared_emb)

        if len(embedding_indices) == 1:
            input_batch = [input_batch_embs[i].to(model.device) for i in embedding_indices][0]
        else:
            input_batch = torch.cat([input_batch_embs[i].to(model.device) for i in embedding_indices])

        # Push target_batch to device
        target_batch = target_batch.to(model.device)

        logits = model.forward(input_batch)

        loss = loss_fn(logits, target_batch) # not necessary to take .mean()
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        avg_loss += loss.item()
        num_trained_batches += 1

        if i % 10 == 9:
            last_loss = running_loss / num_trained_batches  # average loss per data point (previously running_loss / 10)
            print(f"   batch {i + 1} loss: {last_loss}")
            running_loss = 0
            num_trained_batches = 0

    # avg_loss = avg_loss / len(train_loader.dataset)
    avg_loss = avg_loss / len(train_loader)

    return avg_loss

def train_marker_gene_classifier_one_epoch(
        model,
        optimizer: Adam,
        train_loader: DataLoader,
        loss_fn: nn.BCEWithLogitsLoss
) -> float:
    # Set model to training mode
    model.train()

    avg_loss = 0
    running_loss = 0
    num_trained_batches = 0

    for i, batch in enumerate(train_loader):
        # Zero gradients for every batch
        optimizer.zero_grad()

        input_batch, target_batch = batch

        # Push input_batch and target_batch to device
        input_batch = input_batch.to(model.device)
        target_batch = target_batch.to(model.device).reshape(-1, 1)  # Reshape target_batch to be of shape (batch_size, 1)

        logits = model.forward(input_batch)

        loss = loss_fn(logits, target_batch)

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        avg_loss += loss.item()
        num_trained_batches += 1

        if i % 10 == 9:
            last_loss = running_loss / num_trained_batches  # average loss per data point (previously running_loss / 10)
            print(f"   batch {i + 1} loss: {last_loss}")
            running_loss = 0
            num_trained_batches = 0

        # avg_loss = avg_loss / len(train_loader.dataset)
    avg_loss = avg_loss / len(train_loader)

    return avg_loss



def train_VAE_OLD(
        model: BetaVAE,
        optimizer: Adam,
        dataloaders: Tuple[DataLoader, DataLoader],
        lr_scheduler: torch.optim.lr_scheduler,
        num_epochs: int,
        model_name: str,
        dataset_name: str,
        loss_type: str,
        count_data_included: bool = True,
        patience: int = 10
) -> BetaVAE:
    r"""
    Given a GeneExpressionVAE or TranscriptUsageVAE, an optimiser, a tuple of training and validation dataloader,
    the model is trained for num_epochs epochs. The trained model is returned. Also, the loss is plotted and saved as
    a figure.

    :param model:
    :param optimizer:
    :param dataloaders:
    :param lr_scheduler:
    :param num_epochs:
    :param model_name:
    :param dataset_name:
    :param loss_type:
    :param count_data_included:
    :param patience:
    :return:
    """
    train_loader, val_loader = dataloaders
    best_val_loss = 100_000_000.

    epoch_train_loss_list = []
    epoch_train_nll_list = []
    epoch_train_kld_list = []

    epoch_val_loss_list = []
    epoch_val_nll_list = []
    epoch_val_kld_list = []

    num_epochs_wo_improvement = 0

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}:")

        # Ensure model is set to training mode
        model.train(True)

        time_point_0 = time.time()
        train_loss, train_nll, train_kld = train_VAE_one_epoch(model, optimizer, train_loader, count_data_included)

        # Ensure model is set to evaluation mode
        model.eval()
        running_val_loss = 0.0
        running_val_nll = 0.0
        running_val_kld = 0.0
        best_epoch = ""

        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):

                eps = auxiliary_noise((val_batch[1].shape[0], model.latent_dim))

                if count_data_included:
                    input_val_batch_counts, input_val_batch_levels, _, _ = val_batch

                    input_val_batch_counts = input_val_batch_counts.to(model.device)
                    input_val_batch_levels = input_val_batch_levels.to(model.device)
                    input_val_batch = (input_val_batch_counts, input_val_batch_levels)
                    eps = eps.to(model.device)
                else:
                    _, input_val_batch, _, _ = val_batch

                    input_val_batch = input_val_batch.to(model.device)
                    eps = eps.to(model.device)

                # Compute model output and loss
                elbo, kld, nll = model(input_val_batch, eps)
                # val_loss = elbo.sum(dim=0)
                val_loss = elbo.mean()
                running_val_loss += val_loss
                running_val_nll += nll.mean()
                running_val_kld += kld.mean()

        # avg_val_loss = running_val_loss / len(val_loader.dataset) # len(val_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_nll = running_val_nll / len(val_loader)
        avg_val_kld = running_val_kld / len(val_loader)
        print(f"LOSS training {train_loss} validation {avg_val_loss}")
        print(f"       NLL: training {train_nll} validation {avg_val_nll}, KLD: training {train_kld} validation {avg_val_kld}")
        print(f" TIME needed '{time.time() - time_point_0} seconds")

        epoch_train_loss_list.append(train_loss)
        epoch_train_nll_list.append(train_nll)
        epoch_train_kld_list.append(train_kld)

        epoch_val_loss_list.append(avg_val_loss.item())
        epoch_val_nll_list.append(avg_val_nll.item())
        epoch_val_kld_list.append(avg_val_kld.item())

        extra_metrics_to_save = {
            # Training Metrics
            'train_nll': epoch_train_nll_list,
            'train_kld': epoch_train_kld_list,
            # Validation Metrics
            'val_nll': epoch_val_nll_list,
            'val_kld': epoch_val_kld_list
        }

        # Learning rate scheduler
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(avg_val_loss)
        elif lr_scheduler is not None:
            lr_scheduler.step()

        # Track best performance, and save the model and training checkpoint
        if avg_val_loss < best_val_loss:
            # Delete previous best model to save disk space
            if best_epoch != "":
                import os
                os.remove("./models/" + dataset_name + "_" + model_name + "_epochs_" + best_epoch + '_checkpoint.pth')

            # Save current best model
            best_epoch = str(epoch + 1)
            best_val_loss = avg_val_loss
            num_epochs_wo_improvement = 0

            save_model_checkpoint(
                model_name,
                dataset_name,
                best_epoch,
                model,
                optimizer,
                epoch_train_loss_list,
                epoch_val_loss_list,
                **extra_metrics_to_save
            )
        else:
            num_epochs_wo_improvement += 1
            if num_epochs_wo_improvement == patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    plot_loss(
        epoch_train_loss_list,
        epoch_val_loss_list,
        loss_type,
        dataset_name,
        model_name,
        save_fig=True
    )

    return model

def train_VAE(
        model: BetaVAE,
        optimizer: Adam,
        dataloaders: Tuple[DataLoader, DataLoader],
        lr_scheduler: torch.optim.lr_scheduler,
        num_epochs: int,
        model_name: str,
        dataset_name: str,
        loss_type: str,
        count_data_included: bool = True,
        patience: int = 10
) -> BetaVAE:
    r"""
    Train a BetaVAE using train/validation dataloaders.

    Fixes:
        - validation loss is averaged per sample, not per batch
        - ReduceLROnPlateau receives a Python float
        - best_val_loss is kept as a Python float
        - best_epoch is not reset every epoch
        - previous best checkpoint can actually be removed
    """

    train_loader, val_loader = dataloaders

    best_val_loss = float("inf")
    best_epoch = ""

    epoch_train_loss_list = []
    epoch_train_nll_list = []
    epoch_train_kld_list = []

    epoch_val_loss_list = []
    epoch_val_nll_list = []
    epoch_val_kld_list = []

    num_epochs_wo_improvement = 0

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}:")

        model.train(True)

        time_point_0 = time.time()

        train_loss, train_nll, train_kld = train_VAE_one_epoch(
            model,
            optimizer,
            train_loader,
            count_data_included
        )

        model.eval()

        total_val_loss = 0.0
        total_val_nll = 0.0
        total_val_kld = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for val_batch in val_loader:

                if count_data_included:
                    input_val_batch_counts, input_val_batch_levels, _, _ = val_batch
                    batch_size = input_val_batch_counts.shape[0]

                    input_val_batch = (
                        input_val_batch_counts.to(model.device),
                        input_val_batch_levels.to(model.device)
                    )
                else:
                    _, input_val_batch, _, _ = val_batch
                    batch_size = input_val_batch.shape[0]

                    input_val_batch = input_val_batch.to(model.device)

                eps = auxiliary_noise((batch_size, model.latent_dim)).to(model.device)

                elbo, kld, nll = model(input_val_batch, eps)

                # If `elbo` is actually the mathematical ELBO, use:
                # val_loss = -elbo.mean()
                val_loss = elbo.mean()

                total_val_loss += val_loss.item() * batch_size
                total_val_nll += nll.mean().item() * batch_size
                total_val_kld += kld.mean().item() * batch_size
                total_val_samples += batch_size

        avg_val_loss = total_val_loss / total_val_samples
        avg_val_nll = total_val_nll / total_val_samples
        avg_val_kld = total_val_kld / total_val_samples

        print(f"LOSS training {train_loss} validation {avg_val_loss}")
        print(
            f"       NLL: training {train_nll} validation {avg_val_nll}, "
            f"KLD: training {train_kld} validation {avg_val_kld}"
        )
        print(f" TIME needed '{time.time() - time_point_0} seconds")

        epoch_train_loss_list.append(train_loss)
        epoch_train_nll_list.append(train_nll)
        epoch_train_kld_list.append(train_kld)

        epoch_val_loss_list.append(avg_val_loss)
        epoch_val_nll_list.append(avg_val_nll)
        epoch_val_kld_list.append(avg_val_kld)

        extra_metrics_to_save = {
            "train_nll": epoch_train_nll_list,
            "train_kld": epoch_train_kld_list,
            "val_nll": epoch_val_nll_list,
            "val_kld": epoch_val_kld_list,
        }

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(avg_val_loss)
        elif lr_scheduler is not None:
            lr_scheduler.step()

        if avg_val_loss < best_val_loss:
            if best_epoch != "":
                previous_checkpoint = (
                    "./models/"
                    + dataset_name
                    + "_"
                    + model_name
                    + "_epochs_"
                    + best_epoch
                    + "_checkpoint.pth"
                )

                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            best_epoch = str(epoch + 1)
            best_val_loss = avg_val_loss
            num_epochs_wo_improvement = 0

            save_model_checkpoint(
                model_name,
                dataset_name,
                best_epoch,
                model,
                optimizer,
                epoch_train_loss_list,
                epoch_val_loss_list,
                **extra_metrics_to_save
            )
        else:
            num_epochs_wo_improvement += 1

            if num_epochs_wo_improvement == patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    plot_loss(
        epoch_train_loss_list,
        epoch_val_loss_list,
        loss_type,
        dataset_name,
        model_name,
        save_fig=True
    )

    return model

def train_MMVAEplus(
        model: SCGETUVI,
        optimizer: Adam,
        dataloaders: Tuple[DataLoader, DataLoader],
        lr_scheduler: torch.optim.lr_scheduler,
        num_epochs: int,
        num_epochs_kl_warmup: int,
        num_epochs_temp_annealing: int,
        model_name: str,
        dataset_name: str,
        loss_type: str,
        patience: int = 10,
        num_samples: int = 1,
        device: str = "cuda",
        min_delta: float = 0.0
) -> SCGETUVI:
    r"""
    Given SCGETUVI, an optimizer, a tuple of training and validation dataloader,
    a learning rate scheduler, the model is trained for num_epochs epochs. The trained model is returned. Also,
    the loss is plotted and saved as a figure. The training stops if the validation loss does not improve for a number
    of epochs defined by patience.

    :param model: GeneExpressionTranscriptUsageMMVAEplus
    :param optimizer: torch.optim.Adam
    :param dataloaders: torch.utils.data.DataLoader
    :param lr_scheduler: torch.optim.lr_scheduler
    :param num_epochs: int
    :param num_epochs_kl_warmup: int
    :param num_epochs_temp_annealing: int
    :param model_name: str
    :param dataset_name: str
    :param loss_type: str
    :param patience: int, number of epochs without improvement before stopping
    :param num_samples: int
    :param device: either "cuda" or "cpu"
    :param min_delta: float, the minimum change in the monitored quantity to qualify as an improvement
    :return: SCGETUVI

    The training helper `train_MMVAEplus_one_epoch` is assumed to return
    per-sample averages.

    Validation is also computed as a per-sample average, not as an
    unweighted average over batches.

    Assumes that the model returns a negative ELBO / loss-like quantity
    called `elbo` that should be minimized.
    """

    train_loader, val_loader = dataloaders

    best_val_loss = float("inf")
    best_epoch = ""
    num_epochs_wo_improvement = 0

    epoch_train_loss_list = []
    epoch_train_elbo_1_list = []
    epoch_train_elbo_2_list = []
    epoch_train_nll_1_list = []
    epoch_train_nll_2_list = []
    epoch_train_kld_1_list = []
    epoch_train_kld_2_list = []

    epoch_val_loss_list = []
    epoch_val_elbo_1_list = []
    epoch_val_elbo_2_list = []
    epoch_val_nll_1_list = []
    epoch_val_nll_2_list = []
    epoch_val_kld_1_list = []
    epoch_val_kld_2_list = []

    # Convert target values to Python floats.
    beta_1_target = float(model.vae_1.beta)
    beta_2_target = float(model.vae_2.beta)
    temp_target = float(model.temp)

    # KL warm-up schedules.
    if num_epochs_kl_warmup > 0:
        beta_1_values_kl_warmup = torch.linspace(
            0.0,
            beta_1_target,
            steps=num_epochs_kl_warmup
        ).tolist()

        beta_2_values_kl_warmup = torch.linspace(
            0.0,
            beta_2_target,
            steps=num_epochs_kl_warmup
        ).tolist()
    else:
        beta_1_values_kl_warmup = []
        beta_2_values_kl_warmup = []

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}:")

        model.train(True)

        # Freeze weighting encoder during KL warm-up.
        weighting_encoder_trainable = epoch >= num_epochs_kl_warmup

        for param in model.weighting_encoder.parameters():
            param.requires_grad = weighting_encoder_trainable

        # KL beta for this epoch.
        if epoch < num_epochs_kl_warmup:
            beta_1_epoch = beta_1_values_kl_warmup[epoch]
            beta_2_epoch = beta_2_values_kl_warmup[epoch]
        else:
            beta_1_epoch = beta_1_target
            beta_2_epoch = beta_2_target

        beta_kl_warmup_epoch = [beta_1_epoch, beta_2_epoch]

        # Temperature annealing starts after KL warm-up.
        if epoch < num_epochs_kl_warmup:
            temp_epoch = 0.0
        elif num_epochs_temp_annealing > 0 and epoch < num_epochs_kl_warmup + num_epochs_temp_annealing:
            temp_progress = epoch - num_epochs_kl_warmup + 1
            temp_epoch = temp_target * (temp_progress / num_epochs_temp_annealing)
        else:
            temp_epoch = temp_target

        train_loss, train_elbo_1, train_elbo_2, train_nll_1, train_nll_2, train_kld_1, train_kld_2 = train_MMVAEplus_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            beta_kl_warmup_epoch=beta_kl_warmup_epoch,
            temp_epoch=temp_epoch,
            num_samples=num_samples,
            device=device
        )

        model.eval()

        total_val_loss = 0.0
        total_val_elbo_1 = 0.0
        total_val_elbo_2 = 0.0
        total_val_nll_1 = 0.0
        total_val_nll_2 = 0.0
        total_val_kld_1 = 0.0
        total_val_kld_2 = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for val_batch in val_loader:
                (
                    input_val_batch_1_counts,
                    input_val_batch_1_levels,
                    input_val_batch_2_counts,
                    input_val_batch_2_levels,
                    _,
                    _
                ) = val_batch

                batch_size = input_val_batch_1_counts.shape[0]

                input_val_batch_1_counts = input_val_batch_1_counts.to(device)
                input_val_batch_1_levels = input_val_batch_1_levels.to(device)
                input_val_batch_2_counts = input_val_batch_2_counts.to(device)
                input_val_batch_2_levels = input_val_batch_2_levels.to(device)

                eps_1 = auxiliary_noise(
                    (batch_size, model.latent_dim[2] + model.latent_dim[0]),
                    num_samples
                ).to(device)

                eps_2 = auxiliary_noise(
                    (batch_size, model.latent_dim[2] + model.latent_dim[1]),
                    num_samples
                ).to(device)

                eps_3 = auxiliary_noise(
                    (batch_size, model.latent_dim[0]),
                    num_samples
                ).to(device)

                eps_4 = auxiliary_noise(
                    (batch_size, model.latent_dim[1]),
                    num_samples
                ).to(device)

                input_val_batch = (
                    input_val_batch_1_counts,
                    input_val_batch_1_levels,
                    input_val_batch_2_counts,
                    input_val_batch_2_levels
                )

                eps = (eps_1, eps_2, eps_3, eps_4)

                elbo, elbo_1, elbo_2, nll_1, nll_2, kld_1, kld_2 = model(
                    input_val_batch,
                    eps,
                    beta_kl_warmup_epoch,
                    temp_epoch
                )

                # This "elbo" is assumed to be the negative ELBO / loss.
                val_loss = elbo.mean()

                batch_val_loss = val_loss.item()
                batch_val_elbo_1 = elbo_1.mean().item()
                batch_val_elbo_2 = elbo_2.mean().item()
                batch_val_nll_1 = nll_1.mean().item()
                batch_val_nll_2 = nll_2.mean().item()
                batch_val_kld_1 = kld_1.mean().item()
                batch_val_kld_2 = kld_2.mean().item()

                total_val_loss += batch_val_loss * batch_size
                total_val_elbo_1 += batch_val_elbo_1 * batch_size
                total_val_elbo_2 += batch_val_elbo_2 * batch_size
                total_val_nll_1 += batch_val_nll_1 * batch_size
                total_val_nll_2 += batch_val_nll_2 * batch_size
                total_val_kld_1 += batch_val_kld_1 * batch_size
                total_val_kld_2 += batch_val_kld_2 * batch_size
                total_val_samples += batch_size

        if total_val_samples == 0:
            raise ValueError("val_loader is empty.")

        avg_val_loss = total_val_loss / total_val_samples
        avg_val_elbo_1 = total_val_elbo_1 / total_val_samples
        avg_val_elbo_2 = total_val_elbo_2 / total_val_samples
        avg_val_nll_1 = total_val_nll_1 / total_val_samples
        avg_val_nll_2 = total_val_nll_2 / total_val_samples
        avg_val_kld_1 = total_val_kld_1 / total_val_samples
        avg_val_kld_2 = total_val_kld_2 / total_val_samples

        print(f"LOSS training {train_loss} validation {avg_val_loss}")
        print(
            f"       ELBO 1: training {train_elbo_1} validation {avg_val_elbo_1}, "
            f"ELBO 2: training {train_elbo_2} validation {avg_val_elbo_2}"
        )
        print(
            f"       NLL 1: training {train_nll_1} validation {avg_val_nll_1}, "
            f"NLL 2: training {train_nll_2} validation {avg_val_nll_2}"
        )
        print(
            f"       KLD 1: training {train_kld_1} validation {avg_val_kld_1}, "
            f"KLD 2: training {train_kld_2} validation {avg_val_kld_2}"
        )
        print(
            f"       beta_1: {beta_1_epoch}, beta_2: {beta_2_epoch}, "
            f"temp: {temp_epoch}"
        )

        epoch_train_loss_list.append(train_loss)
        epoch_train_elbo_1_list.append(train_elbo_1)
        epoch_train_elbo_2_list.append(train_elbo_2)
        epoch_train_nll_1_list.append(train_nll_1)
        epoch_train_nll_2_list.append(train_nll_2)
        epoch_train_kld_1_list.append(train_kld_1)
        epoch_train_kld_2_list.append(train_kld_2)

        epoch_val_loss_list.append(avg_val_loss)
        epoch_val_elbo_1_list.append(avg_val_elbo_1)
        epoch_val_elbo_2_list.append(avg_val_elbo_2)
        epoch_val_nll_1_list.append(avg_val_nll_1)
        epoch_val_nll_2_list.append(avg_val_nll_2)
        epoch_val_kld_1_list.append(avg_val_kld_1)
        epoch_val_kld_2_list.append(avg_val_kld_2)

        extra_metrics_to_save = {
            "train_elbo_1": epoch_train_elbo_1_list,
            "train_elbo_2": epoch_train_elbo_2_list,
            "train_nll_1": epoch_train_nll_1_list,
            "train_nll_2": epoch_train_nll_2_list,
            "train_kld_1": epoch_train_kld_1_list,
            "train_kld_2": epoch_train_kld_2_list,

            "val_elbo_1": epoch_val_elbo_1_list,
            "val_elbo_2": epoch_val_elbo_2_list,
            "val_nll_1": epoch_val_nll_1_list,
            "val_nll_2": epoch_val_nll_2_list,
            "val_kld_1": epoch_val_kld_1_list,
            "val_kld_2": epoch_val_kld_2_list,

            "beta_1_epoch": beta_1_epoch,
            "beta_2_epoch": beta_2_epoch,
            "temp_epoch": temp_epoch,
        }

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(avg_val_loss)
        elif lr_scheduler is not None:
            lr_scheduler.step()

        if avg_val_loss < best_val_loss - min_delta:
            if best_epoch != "":
                previous_checkpoint = (
                        "./models/"
                        + dataset_name
                        + "_"
                        + model_name
                        + "_epochs_"
                        + best_epoch
                        + "_checkpoint.pth"
                )

                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            best_epoch = str(epoch + 1)
            best_val_loss = avg_val_loss
            num_epochs_wo_improvement = 0

            save_model_checkpoint(
                model_name,
                dataset_name,
                best_epoch,
                model,
                optimizer,
                epoch_train_loss_list,
                epoch_val_loss_list,
                **extra_metrics_to_save
            )
        else:
            num_epochs_wo_improvement += 1

            if num_epochs_wo_improvement == patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    plot_loss(
        epoch_train_loss_list,
        epoch_val_loss_list,
        loss_type,
        dataset_name,
        model_name,
        save_fig=True
    )

    return model


def train_MMVAEplus_OLD(
        model: SCGETUVI,
        optimizer: Adam,
        dataloaders: Tuple[DataLoader, DataLoader],
        lr_scheduler: torch.optim.lr_scheduler,
        num_epochs: int,
        num_epochs_kl_warmup: int,
        num_epochs_temp_annealing: int,
        model_name: str,
        dataset_name: str,
        loss_type: str,
        patience: int = 10,
        num_samples: int = 1,
        device: str = "cuda"
):
    r"""
    Given SCGETUVI, an optimizer, a tuple of training and validation dataloader,
    a learning rate scheduler, the model is trained for num_epochs epochs. The trained model is returned. Also,
    the loss is plotted and saved as a figure. The training stops if the validation loss does not improve for a number
    of epochs defined by patience.

    :param model: GeneExpressionTranscriptUsageMMVAEplus
    :param optimizer: torch.optim.Adam
    :param dataloaders: torch.utils.data.DataLoader
    :param lr_scheduler: torch.optim.lr_scheduler
    :param num_epochs: int
    :param num_epochs_kl_warmup: int
    :param num_epochs_temp_annealing: int
    :param model_name: str
    :param dataset_name: str
    :param loss_type: str
    :param patience: int, number of epochs without improvement before stopping
    :param num_samples: int
    :param device: either "cuda" or "cpu"
    """
    train_loader, val_loader = dataloaders
    best_val_loss = 100_000_000.
    epoch_train_loss_list = []
    epoch_train_elbo_1_list = []
    epoch_train_elbo_2_list = []
    epoch_train_nll_1_list = []
    epoch_train_nll_2_list = []
    epoch_train_kld_1_list = []
    epoch_train_kld_2_list = []

    epoch_val_loss_list = []
    epoch_val_elbo_1_list = []
    epoch_val_elbo_2_list = []
    epoch_val_nll_1_list = []
    epoch_val_nll_2_list = []
    epoch_val_kld_1_list = []
    epoch_val_kld_2_list = []

    num_epochs_wo_improvement = 0

    beta_1_values_kl_warmup = torch.linspace(0.0, model.vae_1.beta, num_epochs_kl_warmup)
    beta_2_values_kl_warmup = torch.linspace(0.0, model.vae_2.beta, num_epochs_kl_warmup)

    # Annealing of modality weight regularisation starts after KL warm-up
    temp_init = 0
    temp_values_kl_warmup = torch.ones(num_epochs_kl_warmup) * temp_init
    temp_values_annealing = torch.linspace(temp_init, model.temp, num_epochs_temp_annealing)
    temp_values = torch.cat((temp_values_kl_warmup, temp_values_annealing))

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}:")

        # Ensure model is set to training mode
        model.train(True)

        # Disable training of weighting_encoder during KL warm-up
        weighting_encoder_params = model.weighting_encoder.parameters()
        if epoch < num_epochs_kl_warmup:
            for param in weighting_encoder_params:
                param.requires_grad = False
        else:
            for param in weighting_encoder_params:
                param.requires_grad = True

        # TO DO: KL WARM UP here
        beta_1_epoch = beta_1_values_kl_warmup[epoch] if epoch < num_epochs_kl_warmup else model.vae_1.beta
        beta_2_epoch = beta_2_values_kl_warmup[epoch] if epoch < num_epochs_kl_warmup else model.vae_2.beta
        beta_kl_warmup_epoch = [beta_1_epoch, beta_2_epoch]
        temp_epoch = temp_values[epoch] if epoch < len(temp_values) else model.temp

        train_loss, train_elbo_1, train_elbo_2, train_nll_1, train_nll_2, train_kld_1, train_kld_2 = train_MMVAEplus_one_epoch(
            model,
            optimizer,
            train_loader,
            beta_kl_warmup_epoch,
            temp_epoch,
            num_samples,
            device
        )

        # Ensure model is set to evaluation mode
        model.eval()
        running_val_loss = 0.0
        running_elbo_1 = 0.0
        running_elbo_2 = 0.0
        running_val_nll_1 = 0.0
        running_val_nll_2 = 0.0
        running_val_kld_1 = 0.0
        running_val_kld_2 = 0.0

        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):
                # 1 is for gene expression and 2 for transcript usage
                input_val_batch_1_counts, input_val_batch_1_levels, input_val_batch_2_counts, input_val_batch_2_levels, _, _ = val_batch
                batch_size = input_val_batch_1_counts.shape[0]

                # Zero gradients for every batch
                optimizer.zero_grad()

                # To do later: include noise sampling here instead of inside the model
                eps_1 = auxiliary_noise((batch_size, model.latent_dim[2] + model.latent_dim[0]))  # noise for [z_1, w_1]
                eps_2 = auxiliary_noise((batch_size, model.latent_dim[2] + model.latent_dim[1]))  # noise for [z_2, w_2]
                eps_3 = auxiliary_noise((batch_size, model.latent_dim[0]))  # noise for w_aux_1
                eps_4 = auxiliary_noise((batch_size, model.latent_dim[1]))  # noise for w_aux_2

                # Push data to device ("cpu" or "cuda") if available
                input_val_batch_1_counts = input_val_batch_1_counts.to(device)
                input_val_batch_1_levels = input_val_batch_1_levels.to(device)
                input_val_batch_2_counts = input_val_batch_2_counts.to(device)
                input_val_batch_2_levels = input_val_batch_2_levels.to(device)
                eps_1 = eps_1.to(device)
                eps_2 = eps_2.to(device)
                eps_3 = eps_3.to(device)
                eps_4 = eps_4.to(device)

                input_val_batch = (input_val_batch_1_counts, input_val_batch_1_levels, input_val_batch_2_counts, input_val_batch_2_levels)
                eps = (eps_1, eps_2, eps_3, eps_4)

                # Compute model output and loss
                elbo, elbo_1, elbo_2, nll_1, nll_2, kld_1, kld_2 = model(input_val_batch, eps, beta_kl_warmup_epoch, temp_epoch)
                # val_loss = elbo.sum(dim=0)
                val_loss = elbo.mean()

                running_val_loss += val_loss
                running_elbo_1 += elbo_1.mean()
                running_elbo_2 += elbo_2.mean()
                running_val_nll_1 += nll_1.mean()
                running_val_nll_2 += nll_2.mean()
                running_val_kld_1 += kld_1.mean()
                running_val_kld_2 += kld_2.mean()

        # avg_val_loss = running_val_loss / len(val_loader.dataset) # len(val_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_elbo_1 = running_elbo_1 / len(val_loader)
        avg_val_elbo_2 = running_elbo_2 / len(val_loader)
        avg_val_nll_1 = running_val_nll_1 / len(val_loader)
        avg_val_nll_2 = running_val_nll_2 / len(val_loader)
        avg_val_kld_1 = running_val_kld_1 / len(val_loader)
        avg_val_kld_2 = running_val_kld_2 / len(val_loader)

        print(f"LOSS training {train_loss} validation {avg_val_loss}")
        print(f"       ELBO 1: {running_elbo_1 / len(val_loader)}, ELBO 2: {running_elbo_2 / len(val_loader)}")
        print(f"       NLL 1: {running_val_nll_1 / len(val_loader)}, NLL 2: {running_val_nll_2 / len(val_loader)}")
        print(f"       KLD 1: {running_val_kld_1 / len(val_loader)}, KLD 2: {running_val_kld_2 / len(val_loader)}")

        epoch_train_loss_list.append(train_loss)
        epoch_train_elbo_1_list.append(train_elbo_1)
        epoch_train_elbo_2_list.append(train_elbo_2)
        epoch_train_nll_1_list.append(train_nll_1)
        epoch_train_nll_2_list.append(train_nll_2)
        epoch_train_kld_1_list.append(train_kld_1)
        epoch_train_kld_2_list.append(train_kld_2)

        epoch_val_loss_list.append(avg_val_loss.item())
        epoch_val_elbo_1_list.append(avg_val_elbo_1.item())
        epoch_val_elbo_2_list.append(avg_val_elbo_2.item())
        epoch_val_nll_1_list.append(avg_val_nll_1.item())
        epoch_val_nll_2_list.append(avg_val_nll_2.item())
        epoch_val_kld_1_list.append(avg_val_kld_1.item())
        epoch_val_kld_2_list.append(avg_val_kld_2.item())

        extra_metrics_to_save = {
            # Training Metrics
            'train_elbo_1': epoch_train_elbo_1_list,
            'train_elbo_2': epoch_train_elbo_2_list,
            'train_nll_1': epoch_train_nll_1_list,
            'train_nll_2': epoch_train_nll_2_list,
            'train_kld_1': epoch_train_kld_1_list,
            'train_kld_2': epoch_train_kld_2_list,

            # Validation Metrics
            'val_elbo_1': epoch_val_elbo_1_list,
            'val_elbo_2': epoch_val_elbo_2_list,
            'val_nll_1': epoch_val_nll_1_list,
            'val_nll_2': epoch_val_nll_2_list,
            'val_kld_1': epoch_val_kld_1_list,
            'val_kld_2': epoch_val_kld_2_list,
        }

        # Learning rate scheduler
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(avg_val_loss)
        elif lr_scheduler is not None:
            lr_scheduler.step()

        # Track best performance, and save the model and training checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            num_epochs_wo_improvement = 0
            save_model_checkpoint(
                model_name,
                dataset_name,
                str(epoch + 1),
                model,
                optimizer,
                epoch_train_loss_list,
                epoch_val_loss_list,
                **extra_metrics_to_save
            )
        else:
            num_epochs_wo_improvement += 1
            if num_epochs_wo_improvement == patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    plot_loss(
        epoch_train_loss_list,
        epoch_val_loss_list,
        loss_type, dataset_name,
        model_name,
        save_fig=True
    )

def train_vae_embedding_cell_type_classifier(
        model: LogisticRegressionClassifier,
        optimizer: Adam,
        dataloaders: Tuple[DataLoader, DataLoader],
        num_epochs: int,
        model_name: str,
        dataset_name: str,
        loss_type: str = "Cross Entropy",
        patience: int = 10,
) -> None:

    train_loader, val_loader = dataloaders
    best_val_loss = 100_000_000.
    epoch_train_loss_list = []
    epoch_val_loss_list = []
    num_epochs_wo_improvement = 0

    if loss_type == "Cross Entropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError("Given loss function is not supported")


    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}:")

        # Train model one epoch
        train_loss = train_vae_embedding_cell_type_classifier_one_epoch(model, optimizer, train_loader, loss_fn)

        # Ensure model is set to evaluation mode
        model.eval()
        running_val_loss = 0.0

        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):

                input_batch, target_batch, _ = val_batch

                # Push input_batch and target_batch to device
                input_batch = input_batch.to(model.device)
                target_batch = target_batch.to(model.device)

                logits = model.forward(input_batch)
                val_loss = loss_fn(logits, target_batch) # not necessary to take .mean()

                running_val_loss += val_loss

        # avg_val_loss = running_val_loss / len(val_loader.dataset) # len(val_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"LOSS training {train_loss} validation {avg_val_loss}")
        epoch_train_loss_list.append(train_loss)
        epoch_val_loss_list.append(avg_val_loss.item())

        # Track best performance, and save the model and training checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            num_epochs_wo_improvement = 0
            save_model_checkpoint(
                model_name,
                dataset_name,
                str(epoch + 1),
                model,
                optimizer,
                epoch_train_loss_list,
                epoch_val_loss_list
            )
        else:
            num_epochs_wo_improvement += 1
            if num_epochs_wo_improvement == patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    plot_loss(
        epoch_train_loss_list,
        epoch_val_loss_list,
        loss_type,
        dataset_name,
        model_name,
        save_fig=True
    )

def train_classifier(
        model: LogisticRegressionClassifier,
        optimizer: Adam,
        dataloaders: Tuple[DataLoader, DataLoader],
        embedding_indices: List[int],
        num_epochs: int,
        model_name: str,
        dataset_name: str,
        loss_type: str = "Cross Entropy",
        patience: int = 10,
) -> None:

    train_loader, val_loader = dataloaders
    best_val_loss = 100_000_000.
    epoch_train_loss_list = []
    epoch_val_loss_list = []
    num_epochs_wo_improvement = 0

    if loss_type == "Cross Entropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError("Given loss function is not supported")


    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}:")

        # Train model one epoch
        train_loss = train_classifier_one_epoch(model, optimizer, train_loader, embedding_indices, loss_fn)

        # Ensure model is set to evaluation mode
        model.eval()
        running_val_loss = 0.0

        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):

                input_private_emb_1, input_private_emb_2, shared_unimodal_emb_1, shared_unimodal_emb_2, shared_emb, target_batch, _ = val_batch

                input_batch_embs = (input_private_emb_1, input_private_emb_2, shared_unimodal_emb_1,
                                    shared_unimodal_emb_2, shared_emb)

                if len(embedding_indices) == 1:
                    input_batch = [input_batch_embs[i].to(model.device) for i in embedding_indices][0]
                else:
                    input_batch = torch.cat([input_batch_embs[i].to(model.device) for i in embedding_indices])

                # Push input_batch and target_batch to device
                target_batch = target_batch.to(model.device)

                logits = model.forward(input_batch)
                val_loss = loss_fn(logits, target_batch) # not necessary to take .mean()

                running_val_loss += val_loss

        # avg_val_loss = running_val_loss / len(val_loader.dataset) # len(val_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"LOSS training {train_loss} validation {avg_val_loss}")
        epoch_train_loss_list.append(train_loss)
        epoch_val_loss_list.append(avg_val_loss.item())

        # Track best performance, and save the model and training checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            num_epochs_wo_improvement = 0
            save_model_checkpoint(
                model_name,
                dataset_name,
                str(epoch + 1),
                model,
                optimizer,
                epoch_train_loss_list,
                epoch_val_loss_list
            )
        else:
            num_epochs_wo_improvement += 1
            if num_epochs_wo_improvement == patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    plot_loss(
        epoch_train_loss_list,
        epoch_val_loss_list,
        loss_type,
        dataset_name,
        model_name,
        save_fig=True
    )

def train_marker_gene_classifier(
        model: LogisticRegressionClassifier,
        optimizer: Adam,
        dataloaders: Tuple[DataLoader, DataLoader],
        num_epochs: int,
        model_name: str,
        dataset_name: str,
        loss_type: str = "Binary Cross Entropy",
        patience: int = 10
) -> None:

    train_loader, val_loader = dataloaders
    best_val_loss = 100_000_000.
    epoch_train_loss_list = []
    epoch_val_loss_list = []
    num_epochs_wo_improvement = 0

    if loss_type == "Binary Cross Entropy":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Given loss function is not supported")

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}:")

        # Train model one epoch
        train_loss = train_marker_gene_classifier_one_epoch(model, optimizer, train_loader, loss_fn)

        # Ensure model is set to evaluation mode
        model.eval()
        running_val_loss = 0.0

        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for i, val_batch in enumerate(val_loader):
                input_batch, target_batch = val_batch

                # Push input_batch and target_batch to device
                input_batch = input_batch.to(model.device)
                target_batch = target_batch.to(model.device).reshape(-1, 1)  # Reshape target_batch to be of shape (batch_size, 1)

                logits = model.forward(input_batch)

                val_loss = loss_fn(logits, target_batch)

                running_val_loss += val_loss

        # avg_val_loss = running_val_loss / len(val_loader.dataset) # len(val_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"LOSS training {train_loss} validation {avg_val_loss}")
        epoch_train_loss_list.append(train_loss)
        epoch_val_loss_list.append(avg_val_loss.item())

        # Track best performance, and save the model and training checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            num_epochs_wo_improvement = 0
            save_model_checkpoint(
                model_name,
                dataset_name,
                str(epoch + 1),
                model,
                optimizer,
                epoch_train_loss_list,
                epoch_val_loss_list
            )
        else:
            num_epochs_wo_improvement += 1
            if num_epochs_wo_improvement == patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    plot_loss(
        epoch_train_loss_list,
        epoch_val_loss_list,
        loss_type,
        dataset_name,
        model_name,
        save_fig=True
    )


def save_model_checkpoint(
        model_name: str,
        dataset_name: str,
        epoch: str,
        model,
        optimizer: Adam,
        epoch_train_loss_list: list,
        epoch_val_loss_list: list,
        **kwargs
) -> None:
    r"""
    Creates checkpoint / saves a model

    :param model_name: str, name of the model
    :param dataset_name: str
    :param epoch: int, last epoch of training
    :param model: TranscriptUsageVAE, any kind of trained model
    :param optimizer: Adam, optimizer used for training
    :param epoch_train_loss_list: list, the training losses of the model
    :param epoch_val_loss_list: list, the validation losses of the model
    :return: None
    """

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_train_loss_list,
        'val_loss': epoch_val_loss_list,
        **kwargs
    }
    torch.save(checkpoint, "./models/" + dataset_name + "_" + model_name + "_epochs_" + epoch + '_checkpoint.pth')

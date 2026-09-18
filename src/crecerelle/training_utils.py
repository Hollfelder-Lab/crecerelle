#!/usr/bin/env python3
import time

import numpy as np
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn

import os

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LRScheduler

from .models import BetaVAE, TRVI, LogisticRegressionClassifier
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

    :param epoch_train_loss_list: list, comprising the training loss of each epoch
    :param epoch_val_loss_list: list, comprising the validation loss of each epoch
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
    Sample independent standard Gaussian noise, $\epsilon\sim\mathcal{N}(0,1)$, for reparameterization.

    :param latent_shape: (Tuple[int, int]) Base shape (batch size, latent dimension).
    :param num_samples: (int) A leading sample dimension is added only when this value is greater than 1.
    :return: (torch.Tensor) Noise of shape latent_shape or (num_samples, *latent_shape). Move it to the model device
        before use.
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
    Randomly split observations into training, validation, and test subsets.
    Validation and test sizes are rounded up; training receives the remainder. Ratios must leave a nonnegative
    training size. Indices are sorted within each partition. Reproducibility follows the PyTorch random state.

    :param dataset: (Dataset) Dataset to partition.
    :param dataset_ratio: (np.ndarray) Three fractions in training/validation/test order. Validation and test fractions
        determine the rounded sizes; the first fraction is not directly used. Default is [0.8, 0.1, 0.1].
    :return: (tuple) Training Subset, validation Subset, test Subset, and a tuple of their original NumPy index arrays.
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
    Randomly split observations into training, validation, and test subsets.
    Validation and test sizes are rounded up; training receives the remainder. Ratios must leave a nonnegative
    training size. Indices are sorted within each partition. Reproducibility follows the PyTorch random state.
    The same indices are applied to every selected dataset, preserving alignment across modalities.

    :param datasets: (Tuple[Dataset, ...]) Aligned datasets with identical observation counts.
    :param num_datasets: (int) Number of leading datasets to split; must not exceed the number supplied.
    :param dataset_ratio: (np.ndarray) Three fractions in training/validation/test order. Validation and test fractions determine
        the rounded sizes; the first fraction is not directly used. Default is [0.8, 0.1, 0.1].
    :return: (tuple) Dataset dictionary and original training/validation/test index arrays. The dictionary
        uses "Dataset 1", "Dataset 2", etc., each containing "Training", "Validation", and "Test" Subsets.
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

def train_VAE_one_epoch(
        model: BetaVAE,
        optimizer: Adam,
        train_loader: DataLoader,
        count_data_included: bool = True
) -> Tuple[float, float, float]:
    r"""
    Train a unimodal VAE for one epoch by minimizing its returned negative ELBO. Batches contain counts, levels,
    ontology, and tissue. Fresh Gaussian noise is sampled for each batch, and gradient norms are clipped to 1.0. Epoch
    metrics are batch-size-weighted averages over observations.

    :param model: (torch.nn.Module) Model updated in place. Its parameters must already be on the training device.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param train_loader: (DataLoader) Nonempty loader yielding the batch layout described above.
    :param count_data_included: (bool) If True, pass (counts, levels) to the model; otherwise pass only levels.
        Each loader batch must still contain counts, levels, ontology, and tissue, in that order.
    :return: (Tuple[float, float, float]) Mean negative ELBO, negative log-likelihood, and KL divergence, in that order.
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
        model: TRVI,
        optimizer: Adam,
        train_loader: DataLoader,
        beta_kl_warmup_epoch: Optional[List[float]] = None,
        temp_epoch: float = 0.1,
        num_samples: int = 1,
        device: str = "cuda"
) -> tuple[float, float, float, float, float, float, float]:
    r"""
    Train TRVI for one epoch with paired gene expression and transcript usage data. Batches contain GE counts, GE
    levels, TU counts, TU levels, ontology, and tissue. Four Gaussian noise tensors are drawn for the unimodal
    posteriors and auxiliary private distributions. The returned negative objective is minimized, gradient norms are
    clipped to 1.0, and a non-finite loss raises FloatingPointError.

    :param model: (torch.nn.Module) Model updated in place. Its parameters must already be on the training device.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param train_loader: (DataLoader) Nonempty loader yielding the batch layout described above.
    :param beta_kl_warmup_epoch: (List[float] | None) KL weights for gene expression and transcript usage. None uses
        [1.0, 1.0].
    :param temp_epoch: (float) Value passed as temp to TRVI.forward; the current model uses it for modality-weight
        regularization.
    :param num_samples: (int) Number of Gaussian noise samples. Values greater than 1 add a leading sample dimension;
        supported shapes depend on the selected model and decoder.
    :param device: (str) Device used for data and noise tensors; must match the model device.
    :return: (tuple[float, ...]) Observation-weighted means of the combined negative objective, GE negative objective,
        TU negative objective, GE NLL, TU NLL, GE pseudo-KL, and TU pseudo-KL, in that order.
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

def train_vae_embedding_cell_type_classifier_one_epoch(
        model,
        optimizer: Adam,
        train_loader: DataLoader,
        loss_fn: nn.CrossEntropyLoss
) -> float:
    r"""
    Train a classifier for one epoch, updating the model and optimizer in place. Batches contain an embedding matrix,
    class targets, and tissue labels. The epoch loss is an unweighted mean of batch losses, including any smaller final
    batch.

    :param model: (torch.nn.Module) Model updated in place. Its parameters must already be on the training device.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param train_loader: (DataLoader) Nonempty loader yielding the batch layout described above.
    :param loss_fn: (torch.nn.Module) Scalar loss accepting model logits and the corresponding targets.
    :return: (float) Mean scalar loss across training batches.
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
    Train a classifier for one epoch, updating the model and optimizer in place. Batches contain five embedding
    matrices, class targets, and tissue labels, as returned by EmbeddingCellTypeDataset. The epoch loss is an unweighted
    mean of batch losses, including any smaller final batch.

    :param model: (torch.nn.Module) Model updated in place. Its parameters must already be on the training device.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param train_loader: (DataLoader) Nonempty loader yielding the batch layout described above.
    :param embedding_indices: (Sequence[int]) Select from private GE (0), private TU (1), shared GE (2), shared TU (3),
        and joint shared (4). Use one index for the normal classification path: multiple selections are concatenated
        along the batch dimension by the current implementation, without repeating targets.
    :param loss_fn: (torch.nn.Module) Scalar loss accepting model logits and the corresponding targets.
    :return: (float) Mean scalar loss across training batches.
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
    r"""
    Train a classifier for one epoch, updating the model and optimizer in place. Batches contain marker-expression
    vectors and binary float targets. Targets are reshaped to (batch size, 1), so the model must produce one logit per
    cell. The epoch loss is an unweighted mean of batch losses, including any smaller final batch.

    :param model: (torch.nn.Module) Model updated in place. Its parameters must already be on the training device.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param train_loader: (DataLoader) Nonempty loader yielding the batch layout described above.
    :param loss_fn: (torch.nn.Module) Scalar loss accepting model logits and the corresponding targets.
    :return: (float) Mean scalar loss across training batches.
    """
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

def train_VAE(
        model: BetaVAE,
        optimizer: Adam,
        dataloaders: Tuple[DataLoader, DataLoader],
        lr_scheduler: LRScheduler,
        num_epochs: int,
        model_name: str,
        dataset_name: str,
        loss_type: str,
        count_data_included: bool = True,
        patience: int = 10
) -> BetaVAE:
    r"""
    Train and validate a unimodal VAE with early stopping and optional learning-rate scheduling. Training and validation
    metrics are weighted by batch size. Each new best checkpoint replaces the previous best checkpoint created by this
    run. Checkpoints are written beneath ./models/ and a PNG loss plot beneath ./figures/tabulaMuris/. These directories
    must exist. The model ends in evaluation mode; best checkpoint weights are not reloaded.

    :param model: (torch.nn.Module) Model updated in place. Its parameters must already be on the training device.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param dataloaders: (Tuple[DataLoader, DataLoader]) Nonempty training and validation loaders, in that order.
    :param lr_scheduler: (LRScheduler | None) Optional scheduler. ReduceLROnPlateau receives the validation loss;
        other schedulers are stepped once per epoch without an argument.
    :param num_epochs: (int) Maximum number of training epochs.
    :param model_name: (str) Model identifier used in checkpoint and loss-figure filenames.
    :param dataset_name: (str) Dataset identifier used in checkpoint and loss-figure filenames.
    :param loss_type: (str) Label used in the loss plot; VAE objectives are supplied by the model.
    :param count_data_included: (bool) If True, pass (counts, levels) to the model; otherwise pass only levels.
        Each loader batch must still contain counts, levels, ontology, and tissue, in that order.
    :param patience: (int) Number of consecutive epochs without validation improvement before stopping.
    :return: (BetaVAE) The supplied model with weights from the last executed epoch, not necessarily the best epoch.
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
        model: TRVI,
        optimizer: Adam,
        dataloaders: Tuple[DataLoader, DataLoader],
        lr_scheduler: LRScheduler,
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
) -> TRVI:
    r"""
    Train and validate TRVI with KL warm-up, modality-weight scheduling, and early stopping. The model must expose a
    weighting_encoder, which is frozen during KL warm-up. Both training and validation metrics are weighted by batch
    size. Each new best checkpoint replaces this run's previous best checkpoint. Checkpoints are written beneath
    ./models/ and a PNG loss plot beneath ./figures/tabulaMuris/. These directories must exist. The model ends in
    evaluation mode; best checkpoint weights are not reloaded.

    :param model: (torch.nn.Module) Model updated in place. Its parameters must already be on the training device.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param dataloaders: (Tuple[DataLoader, DataLoader]) Nonempty training and validation loaders, in that order.
    :param lr_scheduler: (LRScheduler | None) Optional scheduler. ReduceLROnPlateau receives the validation loss;
        other schedulers are stepped once per epoch without an argument.
    :param num_epochs: (int) Maximum number of training epochs.
    :param num_epochs_kl_warmup: (int) Number of epochs in the linear schedule from zero to the initial unimodal beta
        values. The weighting encoder is frozen during this interval; zero disables the warm-up schedule.
    :param num_epochs_temp_annealing: (int) Number of epochs after KL warm-up over which temp increases to model.temp.
        Zero applies the target value immediately after warm-up.
    :param model_name: (str) Model identifier used in checkpoint and loss-figure filenames.
    :param dataset_name: (str) Dataset identifier used in checkpoint and loss-figure filenames.
    :param loss_type: (str) Label used in the loss plot; VAE objectives are supplied by the model.
    :param patience: (int) Number of consecutive epochs without validation improvement before stopping.
    :param num_samples: (int) Number of Gaussian noise samples. Values greater than 1 add a leading sample dimension;
        supported shapes depend on the selected model and decoder.
    :param device: (str) Device used for data and noise tensors; must match the model device.
    :param min_delta: (float) Required reduction below the best validation loss to count as an improvement.
    :return: (TRVI) The supplied model with weights and schedule settings from the last executed epoch.
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
    r"""
    Train a classifier with validation-based early stopping. Batches contain embeddings, class targets, and tissue. Loss
    histories use unweighted means of batch losses. Every validation improvement saves a checkpoint; previous
    checkpoints from this run are retained. Checkpoints are written beneath ./models/ and a PNG loss plot beneath
    ./figures/tabulaMuris/. These directories must exist. The model ends in evaluation mode; best checkpoint weights are
    not reloaded.

    :param model: (torch.nn.Module) Model updated in place. Its parameters must already be on the training device.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param dataloaders: (Tuple[DataLoader, DataLoader]) Nonempty training and validation loaders, in that order.
    :param num_epochs: (int) Maximum number of training epochs.
    :param model_name: (str) Model identifier used in checkpoint and loss-figure filenames.
    :param dataset_name: (str) Dataset identifier used in checkpoint and loss-figure filenames.
    :param loss_type: (str) Only "Cross Entropy" is supported.
    :param patience: (int) Number of consecutive epochs without validation improvement before stopping.
    :return: None. The supplied model and optimizer are updated in place.
    """

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

                running_val_loss += val_loss.item()

        # avg_val_loss = running_val_loss / len(val_loader.dataset) # len(val_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"LOSS training {train_loss} validation {avg_val_loss}")
        epoch_train_loss_list.append(train_loss)
        epoch_val_loss_list.append(avg_val_loss)

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
    r"""
    Train a classifier with validation-based early stopping. Batches contain five embeddings, class targets, and tissue.
    Loss histories use unweighted means of batch losses. Every validation improvement saves a checkpoint; previous
    checkpoints from this run are retained. Checkpoints are written beneath ./models/ and a PNG loss plot beneath
    ./figures/tabulaMuris/. These directories must exist. The model ends in evaluation mode; best checkpoint weights are
    not reloaded.

    :param model: (torch.nn.Module) Model updated in place. Its parameters must already be on the training device.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param dataloaders: (Tuple[DataLoader, DataLoader]) Nonempty training and validation loaders, in that order.
    :param embedding_indices: (Sequence[int]) Select from private GE (0), private TU (1), shared GE (2), shared TU (3),
        and joint shared (4). Use one index for the normal classification path: multiple selections are concatenated
        along the batch dimension by the current implementation, without repeating targets.
    :param num_epochs: (int) Maximum number of training epochs.
    :param model_name: (str) Model identifier used in checkpoint and loss-figure filenames.
    :param dataset_name: (str) Dataset identifier used in checkpoint and loss-figure filenames.
    :param loss_type: (str) Only "Cross Entropy" is supported.
    :param patience: (int) Number of consecutive epochs without validation improvement before stopping.
    :return: None. The supplied model and optimizer are updated in place.
    """

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

                running_val_loss += val_loss.item()

        # avg_val_loss = running_val_loss / len(val_loader.dataset) # len(val_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"LOSS training {train_loss} validation {avg_val_loss}")
        epoch_train_loss_list.append(train_loss)
        epoch_val_loss_list.append(avg_val_loss)

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
    r"""
    Train a classifier with validation-based early stopping. Batches contain marker-expression vectors and binary
    targets; the model must output one logit per cell. Loss histories use unweighted means of batch losses. Every
    validation improvement saves a checkpoint; previous checkpoints from this run are retained. Checkpoints are written
    beneath ./models/ and a PNG loss plot beneath ./figures/tabulaMuris/. These directories must exist. The model ends
    in evaluation mode; best checkpoint weights are not reloaded.

    :param model: (torch.nn.Module) Model updated in place. Its parameters must already be on the training device.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param dataloaders: (Tuple[DataLoader, DataLoader]) Nonempty training and validation loaders, in that order.
    :param num_epochs: (int) Maximum number of training epochs.
    :param model_name: (str) Model identifier used in checkpoint and loss-figure filenames.
    :param dataset_name: (str) Dataset identifier used in checkpoint and loss-figure filenames.
    :param loss_type: (str) Only "Binary Cross Entropy" is supported.
    :param patience: (int) Number of consecutive epochs without validation improvement before stopping.
    :return: None. The supplied model and optimizer are updated in place.
    """
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

                running_val_loss += val_loss.item()

        # avg_val_loss = running_val_loss / len(val_loader.dataset) # len(val_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"LOSS training {train_loss} validation {avg_val_loss}")
        epoch_train_loss_list.append(train_loss)
        epoch_val_loss_list.append(avg_val_loss)

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
    Save model and optimizer state plus loss histories to a PyTorch checkpoint.
    The path is ./models/{dataset_name}_{model_name}_epochs_{epoch}_checkpoint.pth. The directory must exist;
    a file with the same name is overwritten.

    :param model_name: (str) Model identifier used in checkpoint and loss-figure filenames.
    :param dataset_name: (str) Dataset identifier used in checkpoint and loss-figure filenames.
    :param epoch: (str) Epoch identifier as a string, used in the filename and stored in the checkpoint.
    :param model: (torch.nn.Module) Model whose state_dict is saved.
    :param optimizer: (Adam) Optimizer bound to the model parameters; its state is updated during training.
    :param epoch_train_loss_list: (list) Training-loss history to store under train_loss.
    :param epoch_val_loss_list: (list) Validation-loss history to store under val_loss.
    :param kwargs: (dict) Extra checkpoint entries, such as NLL and KL histories. Duplicate keys override standard entries.
    :return: None. Writes the checkpoint file.
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

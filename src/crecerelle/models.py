#!/usr/bin/env python3
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from typing import Tuple, Dict, Iterable, List

import ast # DOUBLE CHECK

import math


def create_set_zero_subsets(d: int) -> List[Tuple[int, ...]]:
    r"""
    Given a tensor of dimensionality $d$, create a set

    $$
        \tilde{\mathcal{K}} = \{\mathcal{K} \subseteq \{1, \dots, d \}; 1 \leq |\mathcal{K}| \leq d - 2 \}
    $$

    which are all subsets $\mathcal{K}$ of $\{1, \dots, d \}$ with cardinality
    $|\mathcal{K}|$ between $1$ and $d-2$. A subset $\mathcal{K}$ gathers categories with counts of zero
    excluding the cases where exactly $d$ and $d-1$ categories are zero-inflated.

    :param d: (int) The dimension of the count vector of an intron group (intron group size)
    :return set_zero_subsets: (List[Tuple[int, ...]]) A list of tuples where each tuple is a subset of indices
    """

    if d < 3:
        set_zero_subsets = []
        return set_zero_subsets

    # Set of indices
    index_set = range(0, d)  # indices are adjusted for Python indexing
    min_set_size = 1
    max_set_size = d - 2
    valid_set_sizes = range(min_set_size, max_set_size + 1)

    # Generate all subsets and collect them
    set_zero_subsets = []
    for set_size in valid_set_sizes:
        for subset in itertools.combinations(index_set, set_size):
            set_zero_subsets.append(subset)

    return set_zero_subsets


class FCLayers(nn.Module):
    r"""
        Fully connected neural network used by the encoders and decoders.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_cat_list: list = None,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            non_linearity: str = "ReLU",
            bias: bool = True
    ) -> None:
        r"""
        This class provides a fully connected neural network with a specified number of hidden layers and units.
        The network can also include batch normalization and dropout regularization. The non-linearity (ReLU by default)
        can be specified as well. Each block of the network consists of a linear layer, followed by optional batch
        normalization, a non-linear activation function, and optional dropout. This class is used as a building block
        for the encoder and decoder networks in the BetaVAE model and its child classes. The class is based on the
        scvi.nn.FCLayers class from the scVI package
        (https://docs.scvi-tools.org/en/stable/api_reference/scvi.nn.html#scvi.nn.FCLayers).

        :param input_dim: (int) The dimension of the input features.
        :param output_dim: (int) The dimension of the output features.
        :param num_cat_list: (list) A list of integers representing the number of categories for each categorical variable.
        :param num_hidden_layers: (int) The number of hidden layers in the network.
        :param num_hidden_units: (int) The number of units in each hidden layer.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param use_batch_norm: (bool) Whether to use batch normalization after each linear layer. Default is True.
        :param non_linearity: (str) The non-linear activation function to use. Default is "ReLU".
        :param bias: (bool) Whether to include a bias term in the linear layers. Default is True.
        """

        super(FCLayers, self).__init__()
        layers_dim = [input_dim] + (num_hidden_layers - 1) * [num_hidden_units] + [output_dim]

        if num_cat_list is not None:
            self.num_cat_list = [num_cat if num_cat > 1 else 0 for num_cat in num_cat_list]
        else:
            self.num_cat_list = []

        self.fc_layers = nn.ModuleList()

        for i, (input_dim, output_dim) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            layer = nn.Sequential()
            layer.add_module("linear", nn.Linear(input_dim + sum(self.num_cat_list), output_dim, bias=bias))
            if use_batch_norm:
                layer.add_module("batchnorm", nn.BatchNorm1d(output_dim, momentum=0.01, eps=0.001))
            if non_linearity == "ReLU":
                layer.add_module("activation", nn.ReLU())
            if dropout_rate > 0:
                layer.add_module("dropout", nn.Dropout(p=dropout_rate))
            self.fc_layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass through the fully connected layers. The input tensor is passed through each layer in the network,
        and the output of the final layer is returned.

        :param x: (torch.Tensor) The input tensor of shape (batch_size, input_dim).
        :return: (torch.Tensor) The output tensor of shape (batch_size, output_dim).
        """
        for layers in self.fc_layers:
            for layer in layers:
                x = layer(x)

        return x


class Encoder(nn.Module):
    r"""
        Encoder for a Gaussian variational posterior returning latent means and diagonal variances
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_cat_list: list = None,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
            architecture: str = "Fully Connected",
            variational_posterior: str = "Gaussian"
    ) -> None:
        r"""
        This class provides an encoder network for the BetaVAE model and its child classes. The encoder takes the input
        data and encodes it into a latent representation. The encoder can be configured to use different architectures
        (currently only "Fully Connected" is implemented) and variational posterior distributions (currently "Gaussian"
        is implemented). The encoder consists of a series of fully connected layers followed by two linear layers that
        output the mean and variance of the latent representation.

        :param input_dim: (int) The dimension of the input features.
        :param output_dim: (int) The dimension of the latent representation.
        :param num_cat_list: (list) A list of integers representing the number of categories for each categorical variable.
        :param num_hidden_layers: (int) The number of hidden layers in the encoder network.
        :param num_hidden_units: (int) The number of units in each hidden layer.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param architecture: (str) The architecture of the encoder network. Currently only "Fully Connected" is implemented.
        :param variational_posterior: (str) The variational posterior distribution to use. Currently only "Gaussian" is implemented.
        """
        super(Encoder, self).__init__()

        self.variational_posterior = variational_posterior
        if architecture == "Fully Connected":
            self.encoder = FCLayers(
                input_dim=input_dim,
                output_dim=num_hidden_units,
                num_cat_list=num_cat_list,
                num_hidden_layers=num_hidden_layers,
                num_hidden_units=num_hidden_units,
                dropout_rate=dropout_rate
            )
        else:
            raise TypeError("Chosen architecture is not implemented")

        self.mean_encoder = nn.Linear(num_hidden_units, output_dim)
        self.var_encoder = nn.Linear(num_hidden_units, output_dim)

        if variational_posterior == "Gaussian":
            pass
        else:
            raise TypeError("Variational posterior distribution is not implemented")

    def forward(
            self,
            x: torch.Tensor,
            *cat_list: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass through the encoder network. The input tensor is passed through the fully connected layers, and the
        mean and variance of the latent representation are computed and returned.

        :param x: (torch.Tensor) The input tensor of shape (batch_size, input_dim).
        :param *cat_list: (int) Categorical variables to be concatenated to the input of the encoder.
        :return: Tuple[torch.Tensor, torch.Tensor] The mean and variance of the latent representation, each of shape (batch_size, output_dim).
        """

        if not torch.isfinite(x).all():
            raise FloatingPointError("Encoder input contains NaN or Inf.")

        #encoding = self.encoder(x, *cat_list)
        encoding = self.encoder(x)

        if not torch.isfinite(encoding).all():
            raise FloatingPointError("Encoder hidden representation contains NaN or Inf.")

        latent_mean = self.mean_encoder(encoding)

        # Stable variance parameterisation
        #latent_var = torch.exp(self.var_encoder(encoding))  # + 1e-4 in scVI
        raw_log_var = self.var_encoder(encoding)
        raw_log_var = torch.clamp(raw_log_var, min=-8.0, max=8.0)
        latent_var = torch.exp(raw_log_var) + 1e-4

        if not torch.isfinite(latent_mean).all():
            raise FloatingPointError("Latent mean contains NaN or Inf.")

        if not torch.isfinite(latent_var).all():
            raise FloatingPointError("Latent variance contains NaN or Inf.")

        return latent_mean, latent_var

class ScaleEncoder(nn.Module):
    r"""
    Encoder returning positive scaling factors or unconstrained modality-relevance weight logits.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 1, # Default output_dim for scaling factors
            num_cat_list: list = None,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
            architecture: str = "Fully Connected",
            variational_posterior: str = "Gaussian",
            learn_modality_weighting: bool = False # Scenario to encode the mixing coefficient of the latent distribution
    ) -> None:
        r"""
        This class provides an encoder network for learning scaling factors in the BetaVAE model and its child classes.
        The scale encoder takes the input data and encodes it into a scaling factor representation. The encoder can be
        configured to use different architectures (currently only "Fully Connected" is implemented) and variational
        posterior distributions (currently "Gaussian" is implemented). The encoder consists of a series of fully
        connected layers followed by a linear layer that outputs the scaling factor. In TRVI it is used to learn the
        scaling factor of a cell or the modality-relevance weights of the shared variational posterior of a latent
        mixture model.

        :param input_dim: (int) The dimension of the input features.
        :param output_dim: (int) The dimension of the scaling factor representation. Default is 1.
        :param num_cat_list: (list) A list of integers representing the number of categories for each categorical variable.
        :param num_hidden_layers: (int) The number of hidden layers in the scale encoder network.
        :param num_hidden_units: (int) The number of units in each hidden layer.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param architecture: (str) The architecture of the scale encoder network. Currently only "Fully Connected" is implemented.
        :param variational_posterior: (str) The variational posterior distribution to use. Currently only "Gaussian" is implemented.
        :param learn_modality_weighting: (bool) Whether to learn the mixing coefficient of a latent mixture model. Default is False.
        """
        super(ScaleEncoder, self).__init__()
        self.variational_posterior = variational_posterior

        if architecture == "Fully Connected":
            self.encoder = FCLayers(
                input_dim=input_dim,
                output_dim=num_hidden_units,
                num_cat_list=num_cat_list,
                num_hidden_layers=num_hidden_layers,
                num_hidden_units=num_hidden_units,
                dropout_rate=dropout_rate
            )
        else:
            raise TypeError("Chosen architecture is not implemented")


        if learn_modality_weighting:
            # Scenario of learning the mixing coefficient of latent mixture model
            self.scale_encoder = nn.Linear(num_hidden_units, output_dim)
        else:
            # Standard scenario of learning the scale of a cell
            self.scale_encoder = nn.Sequential(
                nn.Linear(num_hidden_units, output_dim),
                nn.Softplus()  # Ensure the scale is greater than 0
            )

        if variational_posterior == "Gaussian":
            pass
        else:
            raise TypeError("Variational posterior distribution is not implemented")

    def forward(
            self,
            x: torch.Tensor,
            *cat_list: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass through the scale encoder network. The input tensor is passed through the fully connected layers,
        and the scaling factor is computed and returned.

        :param x: (torch.Tensor) The input tensor of shape (batch_size, input_dim).
        :param *cat_list: (int) Categorical variables to be concatenated to the input of the scale encoder.
        :return: (torch.Tensor) The scaling factor of shape (batch_size, output_dim).
        """

        #encoding = self.encoder(x, *cat_list)
        encoding = self.encoder(x)
        scale = self.scale_encoder(encoding)

        return scale


class Decoder(nn.Module):
    r"""
    Decoder for a generative model with a Gaussian observaton model returning reconstructed meand and diagonal variances
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_cat_list: Iterable[int] = None,
            dropout_rate: float = 0.1,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128
    ) -> None:
        r"""
        Default class for decoders used in class BetaVAE. As likelihood, the decoder assumes a multivariate Gaussian
        distribution with diagonal covariance matrix. The decoder is a neural network that takes the latent representation
        as input and computes the mean and variance vectors of the likelihood back in the data space.

        :param input_dim: (int) The dimension of the latent representation.
        :param output_dim: (int) The dimension of the output features (data space).
        :param num_cat_list: (list) A list of integers representing the number of categories for each categorical variable.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param num_hidden_layers: (int) The number of hidden layers in the decoder network.
        :param num_hidden_units: (int) The number of units in each hidden layer.
        """
        super(Decoder, self).__init__()

        self.decoding_layers = FCLayers(
            input_dim=input_dim,
            output_dim=num_hidden_units,
            num_cat_list=num_cat_list,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )

        self.mean_decoder = nn.Linear(num_hidden_units, output_dim)
        self.var_decoder = nn.Linear(num_hidden_units, output_dim)

    def forward(
            self,
            x: torch.Tensor,
            *cat_list: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass of the decoder. The decoding neural network takes the latent representation as input and
        decodes it into mean and variance vectors of the likelihood back in the data space.

        :param x: (torch.Tensor) Latent representation of the data.
        :param *cat_list: (int) Categorical variables to be concatenated to the input of the decoder.
        :return: Tuple[torch.Tensor, torch.Tensor] Reconstructed mean and variance vectors of the likelihood.
        """
        # Apply the decoding layers to the latent variable
        #x_decoded = self.decoding_layers(x, *cat_list)
        x_decoded = self.decoding_layers(x)

        pred_mean = self.mean_decoder(x_decoded)
        pred_var = torch.exp(self.var_decoder(x_decoded))

        return pred_mean, pred_var

class NBGeneExpressionDecoder(nn.Module):
    r"""
    Decoder for a generative model of gene expression data with a negative binomial (NB) observation model returning
    positive means.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_cat_list: Iterable[int] = None,
            dropout_rate: float = 0.1,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128
    ) -> None:
        r"""
        Decoder for gene expression data if a negative binomial likelihood is used. The decoder is a neural network that
        takes the latent representation as input and decodes it into a mean reconstruction that parameterize
        the negative-binomial likelihood together with an inverse dispersion which is a simple learnable parameter.

        :param input_dim: (int) The dimension of the latent representation.
        :param output_dim: (int) The dimension of the output features (data space).
        :param num_cat_list: (list) A list of integers representing the number of categories for each categorical variable.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param num_hidden_layers: (int) The number of hidden layers in the decoder network.
        :param num_hidden_units: (int) The number of units in each hidden layer.
        """
        super(NBGeneExpressionDecoder, self).__init__()

        self.decoding_layers = FCLayers(
            input_dim=input_dim,
            output_dim=num_hidden_units,
            num_cat_list=num_cat_list,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )

        self.mean_decoder = nn.Sequential(
            nn.Linear(num_hidden_units, output_dim),
            nn.Softplus()  # Ensure the mean is greater than 0
        )

    def forward(self, x: torch.Tensor, *cat_list: int, eps: float = 1e-8) -> torch.Tensor:
        r"""
        Forward pass of the decoder. The decoding neural network takes the latent representation as input and
        decodes it into a mean reconstruction which is returned.

        :param x: (torch.Tensor) Latent representation of the data.
        :param *cat_list: (int) Categorical variables to be concatenated to the input of the decoder.
        :param eps: (float) Small epsilon value for numerical stability. Default is 1e-8.
        :return: (torch.Tensor) Reconstructed mean vector of the likelihood.
        """
        # Apply the decoding layers to the latent variable
        #x_decoded = self.decoding_layers(x, *cat_list)
        x_decoded = self.decoding_layers(x)

        # Decode the mean
        mean_reconstruction = self.mean_decoder(x_decoded) + eps  # Add small epsilon for numerical stability

        return mean_reconstruction

class ZINBGeneExpressionDecoder(nn.Module):
    r"""
    Decoder for a generative model of gene expression data with a zero-inflated negative binomial (ZINB) observation
    model returning positive means, and zero-inflation logits.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_cat_list: Iterable[int] = None,
            dropout_rate: float = 0.1,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128
    ) -> None:
        r"""
        Decoder for gene expression data if a zero-inflated negative binomial (ZINB) likelihood is used. The decoder is a
        neural network that takes the latent representation as input and decodes it into a mean reconstruction and logit
        zero-inflation probability that parameterize the ZINB likelihood together with an inverse dispersion which is a
        simple learnable parameter.

        :param input_dim: (int) The dimension of the latent representation.
        :param output_dim: (int) The dimension of the output features (data space).
        :param num_cat_list: (list) A list of integers representing the number of categories for each categorical variable.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param num_hidden_layers: (int) The number of hidden layers in the decoder network.
        :param num_hidden_units: (int) The number of units in each hidden layer.
        """
        super(ZINBGeneExpressionDecoder, self).__init__()

        self.decoding_layers = FCLayers(
            input_dim=input_dim,
            output_dim=num_hidden_units,
            num_cat_list=num_cat_list,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )

        # Mean decoder for the ZINB likelihood
        self.mean_decoder = nn.Sequential(
            nn.Linear(num_hidden_units, output_dim),
            nn.Softplus()  # Ensure the mean is greater than 0
        )

        # Logit zero-inflation decoder for the ZINB likelihood
        self.logit_zero_inflation_decoder = nn.Linear(num_hidden_units, output_dim)

    def forward(self, x: torch.Tensor, *cat_list: int, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass of the decoder. The decoding neural network takes the latent representation as input and
        decodes it into a mean reconstruction and logit zero-inflation probability which are returned as a tuple.

        :param x: (torch.Tensor) Latent representation of the data.
        :param cat_list: (int) Categorical variables to be concatenated to the input of the decoder.
        :param eps: (float) Small epsilon value for numerical stability. Default is 1e-8.
        :return: Tuple[torch.Tensor, torch.Tensor] Reconstructed mean vector and logit zero-inflation probability.
        """
        # Apply the decoding layers to the latent variable
        #x_decoded = self.decoding_layers(x, *cat_list)
        x_decoded = self.decoding_layers(x)

        # Decode the mean
        mean_reconstruction = torch.clamp(self.mean_decoder(x_decoded), min=eps)  # Add small epsilon for numerical stability

        # Check if mean_reconstruction contains any NaN or Inf values
        if torch.isnan(mean_reconstruction).any() or torch.isinf(mean_reconstruction).any():
            pdb.set_trace()

        # Decode the logit zero-inflation probability
        logit_zero_inflation = self.logit_zero_inflation_decoder(x_decoded)

        return mean_reconstruction, logit_zero_inflation




class GaussianGeneExpressionDecoder(nn.Module):
    r"""
    Decoder for a generative model of gene expression data (i.e. expression levels) with a Gaussian observation model
    returning means and variances.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_cat_list: Iterable[int] = None,
            dropout_rate: float = 0.1,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128
    ) -> None:
        r"""
        Decoder for gene expression data if a Gaussian likelihood is used. The decoder is a neural network that takes the
        latent representation as input and decodes it into a mean reconstruction and variance reconstruction.

        :param input_dim: (int) The dimension of the latent representation.
        :param output_dim: (int) The dimension of the output features (data space).
        :param num_cat_list: (list) A list of integers representing the number of categories for each categorical variable.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param num_hidden_layers: (int) The number of hidden layers in the decoder network.
        :param num_hidden_units: (int) The number of units in each hidden layer.
        """
        super(GaussianGeneExpressionDecoder, self).__init__()

        self.decoding_layers = FCLayers(
            input_dim=input_dim,
            output_dim=num_hidden_units,
            num_cat_list=num_cat_list,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )

        self.mean_decoder = nn.Linear(num_hidden_units, output_dim)
        self.var_decoder = nn.Linear(num_hidden_units, output_dim)

    def forward(
            self,
            x: torch.Tensor,
            *cat_list: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass of the decoder. The decoding neural network takes the latent representation as input and
        decodes it into a mean reconstruction and variance reconstruction which are returned as a tuple.

        :param x: (torch.Tensor) Latent representation of the data.
        :param cat_list: (int) Categorical variables to be concatenated to the input of the decoder.
        :return: Tuple[torch.Tensor, torch.Tensor] Reconstructed mean vector and variance vector of the likelihood.
        """
        # Apply the decoding layers to the latent variable
        #x_decoded = self.decoding_layers(x, *cat_list)
        x_decoded = self.decoding_layers(x)

        # Decode the mean and variance
        mean_reconstruction = self.mean_decoder(x_decoded)
        var_reconstruction = torch.exp(self.var_decoder(x_decoded)) # consider if it better to return log variance

        return mean_reconstruction, var_reconstruction


class IntronsDecoder(nn.Module):
    r"""
    Decoder for a generative model of alternative-splicing induced transcript usage data with a Dirichlet-Multinomial
    (DM) observation model returning exon junction (intron) proportions normalized within each isoform (intron) group.
    """
    def __init__(
            self,
            intron_groups: torch.Tensor,
            intron_group_summation: torch.sparse_coo_tensor,
            input_dim: int,
            output_dim: int,
            num_cat_list: list = None,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
    ) -> None:
        r"""
        This class is the decoder for the transcript usage Variational Inference (tuVI) variational autoencoder with a
        Dirichlet-Multinomial observation model (DMTranscriptUsageVAE). It takes the latent representation as input and
        decodes it into a vector of reconstructed intron proportions (psi scores) which are returned. The decoder is a
        neural network that consists of a series of fully connected layers followed by a linear layer that outputs the
        reconstructed intron proportions. The decoder also takes into account the intron groups and their summation to
        ensure that the reconstructed intron proportions are valid.

        :param intron_groups: (torch.Tensor) A tensor containing the indices of the first intron of each intron group.
        :param intron_group_summation: (torch.sparse_coo_tensor) A sparse tensor representing the summation of intron groups.
        :param input_dim: (int) The dimension of the latent representation.
        :param output_dim: (int) The dimension of the output features (data space).
        :param num_cat_list: (list) A list of integers representing the number of categories for each categorical variable.
        :param num_hidden_layers: (int) The number of hidden layers in the decoder network.
        :param num_hidden_units: (int) The number of units in each hidden layer.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        """
        super(IntronsDecoder, self).__init__()
        #self.intron_groups = intron_groups
        #self.intron_group_summation = intron_group_summation

        self.register_buffer("intron_groups", intron_groups.clone())
        self.register_buffer("intron_group_summation", intron_group_summation.clone())

        self.non_linear_layers = FCLayers(
            input_dim=input_dim,
            output_dim=num_hidden_units,
            num_cat_list=num_cat_list,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate
        )
        self.linear_layer = nn.Linear(num_hidden_units, output_dim)

    def forward(self, latent_variable: torch.Tensor, intron_group_idx_start: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the decoder. The decoding neural network takes the latent representation as input and decodes it
        into a vector of reconstructed intron proportions (psi scores) which are returned.
        """
        potentials = self.linear_layer(self.non_linear_layers(latent_variable)) # batch_size x num_introns
        potentials[:, intron_group_idx_start] = 0.0

        potentials = torch.clamp(potentials, min=-20.0, max=20.0)

        p_u = torch.exp(potentials)
        intron_group_sums = torch.sparse.mm(self.intron_group_summation, p_u.T).T
        norm_factor = intron_group_sums[:, self.intron_groups]

        # For future p (reconstructed intron proportion / psi score) must be in range [0, 1]
        p = p_u / (norm_factor + 1e-8)
        p = torch.clamp(p, min=1e-8, max=1.0 - 1e-8)

        assert (torch.any(p < 0) or torch.any(p > 1)) == False, "Reconstructed intron proportions are out of range [0, 1]"

        return p

class ZIDMTranscriptUsageDecoder(nn.Module):
    r"""
    Decoder for a generative model of alternative-splicing induced transcript usage data with the heuristic
    Zero-Inflated Dirichlet-Multinomial (ZIDM) surrogate of the ZANDIM log-likelihood. It returns exon junction (intron)
    proportions normalized within each isoform (intron) group and logits of the excess-of-zeros probabilities.
    """
    def __init__(
            self,
            intron_groups: torch.Tensor,
            intron_group_summation: torch.sparse_coo_tensor,
            input_dim: int,
            output_dim: int,
            num_cat_list: list = None,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
    ) -> None:
        r"""
        This class is the decoder for the transcript usage Variational Inference (tuVI) variational autoencoder with the
        heuristic ZIDM as surrogate for the ZANIDM log-likelihood (ZIDMTranscriptUsageVAE). It takes the latent
        representation as input and decodes it into a vector of reconstructed intron proportions (psi scores) and logits
        of the excess-of-zeros parameter which are returned. The decoder is a neural network that consists of a series
        of fully connected layers followed by two linear layers that output the reconstructed intron proportions and
        logits of the excess-of-zeros parameter. The decoder also takes into account the intron groups and their
        summation to ensure that the reconstructed intron proportions are valid.

        :param intron_groups: (torch.Tensor) A tensor containing the indices of the first intron of each intron group.
        :param intron_group_summation: (torch.sparse_coo_tensor) A sparse tensor representing the summation of intron groups.
        :param input_dim: (int) The dimension of the latent representation.
        :param output_dim: (int) The dimension of the output features (data space).
        :param num_cat_list: (list) A list of integers representing the number of categories for each categorical variable.
        :param num_hidden_layers: (int) The number of hidden layers in the decoder network
        :param num_hidden_units: (int) The number of units in each hidden layer.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        """
        super(ZIDMTranscriptUsageDecoder, self).__init__()
        #self.intron_groups = intron_groups
        #self.intron_group_summation = intron_group_summation
        self.register_buffer("intron_groups", intron_groups.clone())
        self.register_buffer("intron_group_summation", intron_group_summation.clone())

        self.decoding_layers = FCLayers(
            input_dim=input_dim,
            output_dim=num_hidden_units,
            num_cat_list=num_cat_list,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )

        # Decoder for the intron proportions (psi scores)
        self.intron_proportion_decoder = nn.Linear(num_hidden_units, output_dim)

        # Logit zero-inflation decoder for the ZIDM likelihood
        self.logit_zero_inflation_decoder = nn.Linear(num_hidden_units, output_dim)


    def forward(self, x: torch.Tensor, intron_group_idx_start: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass of the decoder. The decoding neural network takes the latent representation as input and decodes it
        into an intron proportion reconstruction and logit excess-of-zero parameter which are returned as a tuple.

        :param x: (torch.Tensor) Latent representation of the data.
        :param intron_group_idx_start: (torch.Tensor) Indices of the first intron of each intron group.
        :returns: Tuple[torch.Tensor, torch.Tensor] The reconstructed intron proportions and the logit excess-of-zero parameters.
        """
        # Apply the decoding layers to the latent variable
        x_decoded = self.decoding_layers(x)

        # Decode the logit zero-inflation probability
        logit_zero_inflation = self.logit_zero_inflation_decoder(x_decoded)

        # Decode the intron proportions (psi scores)
        potentials = self.intron_proportion_decoder(x_decoded)  # batch_size x num_introns
        potentials[:, intron_group_idx_start] = 0.0
        potentials = torch.clamp(potentials, min=-20.0, max=20.0)

        p_u = torch.exp(potentials)
        intron_group_sums = torch.sparse.mm(self.intron_group_summation, p_u.T).T
        norm_factor = intron_group_sums[:, self.intron_groups]

        # For future intron_proportion (reconstructed intron proportion / psi score) must be in range [0, 1]
        intron_proportion = p_u / (norm_factor + 1e-8)
        intron_proportion = torch.clamp(intron_proportion, min=1e-8, max=1.0 - 1e-8)

        assert (torch.any(intron_proportion < 0) or torch.any(intron_proportion > 1)) == False, "Reconstructed intron proportions are out of range [0, 1]"

        return intron_proportion, logit_zero_inflation

class ZANIDMTranscriptUsageDecoder(nn.Module):
    r"""
    Decoder for a generative model of alternative-splicing induced transcript usage data with the zero-and-N-inflated
    Dirichlet-Multinomial (ZANIDM) observation model. It returns exon junction (intron) proportions normalized within
    each isoform (intron) group and the excess-of-zeros probabilities.
    """
    def __init__(
            self,
            intron_groups: torch.Tensor,
            intron_group_summation: torch.sparse_coo_tensor,
            input_dim: int,
            output_dim: int,
            num_cat_list: list = None,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
    ) -> None:
        r"""
        This class is the decoder for the transcript usage Variational Inference (tuVI) variational autoencoder with the
        ZANIDM observation model (ZANIDMTranscriptUsageVAE). It takes the latent representation as input and decodes it
        into a vector of reconstructed intron proportions (psi scores) and excess-of-zero parameters which are returned.
        The decoder is a neural network that consists of a series of fully connected layers followed by two linear
        layers that output the reconstructed intron proportions and excess-of-zero parameters. The decoder also takes
        into account the intron groups and their summation to ensure that the reconstructed intron proportions are
        valid. Also, the excess-of-zero parameters for each exon-junction (intron) are scaled to be in the range [0, 1]
        using a sigmoid activation function.

        :param intron_groups: (torch.Tensor) A tensor containing the indices of the first intron of each intron group.
        :param intron_group_summation: (torch.sparse_coo_tensor) A sparse tensor representing the summation of intron groups.
        :param input_dim: (int) The dimension of the latent representation.
        :param output_dim: (int) The dimension of the output features (data space).
        :param num_cat_list: (list) A list of integers representing the number of categories for each categorical variable.
        :param num_hidden_layers: (int) (int) The number of hidden layers in the decoder network.
        :param num_hidden_units: (int) The number of units in each hidden layer.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        """
        super(ZANIDMTranscriptUsageDecoder, self).__init__()
        #self.intron_groups = intron_groups
        #self.intron_group_summation = intron_group_summation
        self.register_buffer("intron_groups", intron_groups.clone())
        self.register_buffer("intron_group_summation", intron_group_summation.clone())

        self.decoding_layers = FCLayers(
            input_dim=input_dim,
            output_dim=num_hidden_units,
            num_cat_list=num_cat_list,
            num_hidden_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )

        # Decoder for the intron proportions (psi scores)
        self.intron_proportion_decoder = nn.Linear(num_hidden_units, output_dim)

        # Decoder for the excess-of-zero parameter of the ZANIDM likelihood
        self.excess_of_zero_decoder = nn.Linear(num_hidden_units, output_dim)

    def forward(self, x: torch.Tensor, intron_group_idx_start: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass of the decoder. The decoding neural network takes the latent representation as input and decodes it
        into an intron proportion which will become the concentration parameter of the ZANIDM likelihood and an
        excess-of-zero parameter of the ZANIDM likelihood.

        :param x: (torch.Tensor) Latent representation of the data.
        :param intron_group_idx_start: (torch.Tensor) Indices of the first intron of each intron group.
        :returns: Tuple[torch.Tensor, torch.Tensor] The reconstructed intron proportions and the excess-of-zero parameters.
        """
        # Apply the decoding layers to the latent variable
        x_decoded = self.decoding_layers(x)

        # Decode the excess-of-zero parameter and scale it to be in range [0, 1]
        excess_of_zero = torch.sigmoid(self.excess_of_zero_decoder(x_decoded))

        # Decode the intron proportions (psi scores)
        potentials = self.intron_proportion_decoder(x_decoded)  # batch_size x num_introns
        potentials[:, intron_group_idx_start] = 0.0
        potentials = torch.clamp(potentials, min=-20.0, max=20.0)

        p_u = torch.exp(potentials)
        intron_group_sums = torch.sparse.mm(self.intron_group_summation, p_u.T).T
        norm_factor = intron_group_sums[:, self.intron_groups]

        # Intron_proportion (reconstructed intron proportion / psi score) must be in range [0, 1]
        intron_proportion = p_u / (norm_factor + 1e-8)
        intron_proportion = torch.clamp(intron_proportion, min=1e-8, max=1.0 - 1e-8)

        assert (torch.any(intron_proportion < 0) or torch.any(intron_proportion > 1)) == False, "Reconstructed intron proportions are out of range [0, 1]"
        assert (torch.any(excess_of_zero < 0) or torch.any(excess_of_zero > 1)) == False, "Excess-of-zero parameters are out of range [0, 1]"

        return intron_proportion, excess_of_zero

class BetaVAE(nn.Module):
    r"""
    Base beta variational autoencoders with a Gaussian observation model and optional learned scaling.
    """
    def __init__(
            self,
            input_dim: int,
            latent_dim: int = 10,
            beta: float = 1.0,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
            device: str = "cuda",
            scaling_factor: bool = True,
    ):
        r"""
        This is the base class for the Beta Variational Autoencoder (BetaVAE) and its child classes (tuVI and used
        within TRVI). The BetaVAE is a deep generative model that learns a latent representation of the input data by
        maximizing the evidence lower bound (ELBO) with a Kullback-Leibler divergence term weighted by a factor beta.
        The model consists of an encoder network that maps the input data to a latent space, and a decoder network that
        reconstructs the input data from the latent representation. The BetaVAE can be configured to learn scaling
        factors for the latent representation, which can be useful for certain applications. When training with
        KL-warm-up, the beta parameter can be gradually increased from 0 to the desired value over a specified number
        of epochs.

        :param input_dim: (int) The dimension of the input features.
        :param latent_dim: (int) The dimension of the latent representation. Default is 10.
        :param beta: (float) The weight of the Kullback-Leibler divergence term in the ELBO. Default is 1.0.
        :param num_hidden_layers: (int) The number of hidden layers in the encoder and decoder networks. Default is 1.
        :param num_hidden_units: (int) The number of units in each hidden layer. Default is 128.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param device: (str) The device to run the model on. Default is "cuda".
        :param scaling_factor: (bool) Whether to learn scaling factors for the latent representation. Default is True.
        """
        super().__init__()

        self.beta = beta
        self.latent_dim = latent_dim
        self.device = device
        self.scaling_factor = scaling_factor

        self.encoder = Encoder(
            input_dim=input_dim,
            output_dim=latent_dim,
            num_cat_list=None,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate,
            architecture="Fully Connected",
            variational_posterior="Gaussian"
        )
        if self.scaling_factor:
            self.scale_encoder = ScaleEncoder(
                input_dim=input_dim,
                output_dim=1,  # Default output_dim for scaling factors
                num_cat_list=None,
                num_hidden_layers=num_hidden_layers,
                num_hidden_units=num_hidden_units,
                dropout_rate=dropout_rate,
                architecture="Fully Connected",
                variational_posterior="Gaussian"
            )

        self.decoder = Decoder(
            input_dim=latent_dim,
            output_dim=input_dim,
            num_cat_list=None,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate,
        )

    def kullback_leibler_divergence(self, latent_mean: torch.Tensor, latent_var: torch.Tensor) -> torch.Tensor:
        r"""
        Given the latent mean and the latent variance of the variational posterior,  the negative Kullback-Leibler
        divergence (since we minimize the loss)
        $-\mathrm{D_{KL}} (q_{\boldsymbol{\phi}_{TU}}(\mathbf{z}_n \mid \mathbf{x}_n^{(TU)}) \middle\| p(\mathbf{z}_n))$
        between the variational posterior
        $q_{\boldsymbol{\phi}_{TU}}(\mathbf{z}_n \mid \mathbf{x}_n^{(TU)}) = \mathcal{N} (\mathbf{z}_n \rvert \boldsymbol{\mu}_{\boldsymbol{\phi}_{TU}}( \mathbf{x}^{(TU)}_n ), \mathrm{diag}(\boldsymbol{\sigma}_{\boldsymbol{\phi}_{TU}}^2 (\mathbf{x}^{(TU)}_n ) ))$
        and the prior $p(\mathbf{z}_n) = \mathcal{N}(\mathbf{0}, \mathbf{I}_L)$ is computed. Here, both distributions are
        considered as Gaussian as in Kingma et al. 2013.

        $$
            - \mathrm{D_{KL}} (q_{\boldsymbol{\phi}_{TU}}(\mathbf{z}_n \mid \mathbf{x}_n^{(TU)}) \middle\| p(\mathbf{z}_n)) = \frac{1}{2} \sum_{l=1}^L ( 1 + 2 \log (\sigma_{n,l}) - \mu^2_{n,l} - \sigma^2_{n,l} ).
        $$

        with $\mu_{n,l} \coloneqq (\boldsymbol{\mu}_{\boldsymbol{\phi}_{TU}}( \mathbf{x}^{(TU)}_n ))_l$ and
        $\sigma_{n,l} \coloneqq  (\boldsymbol{\sigma}_{\boldsymbol{\phi}_{TU}} (\mathbf{x}^{(TU)}_n ))_l$ being computed
        by the decoder.

        :param latent_mean: (torch.Tensor) The mean vector of shape (batch size, latent_dim) of the variational posterior
        :param latent_var: (torch.Tensor) The variance vector of shape (batch size, latent_dim) of the variaitonal posterior
        :return: kld (torch.Tensor)
        """
        # Check the support of the parameters
        assert torch.all(latent_var > 0).item(), "The variance of the likelihood must be greater than 0"

        kld = -0.5 * (1 + torch.log(latent_var) - latent_mean ** 2 - latent_var).sum(dim=1)
        return kld

    def log_likelihood(
            self, x: torch.Tensor,
            generative_model: Dict[str, torch.Tensor],
            min_variance: float=1e-6
    ) -> torch.Tensor:
        r"""
        The standard Beta VAE assumes a generative model with a Gaussian likelihood. Thus, the implemented
        log-likelihood function is derived from the multivariate Gaussian distribution $\mathcal{N}(\mathbf{x}
        \rvert \boldsymbol{\mu}, \boldsymbol{\sigma}^2 \odot \mathbf{I})$ with a mean vector $\boldsymbol{\mu}
        \in \mathbb{R}^D$ and diagonal covariance matrix $\boldsymbol{\sigma}^2 \odot \mathbf{I} \in
        \mathbb{R}_+^{N \times N}$. The log-likelihood function is evaluated using the given data point x (torch.Tensor)
         as well as the torch.Tensors corresponding to the keys "Mean" and the "Variance" of the dictionary
         generative_model. The Gaussian log-likelihood is then computed according to

         $$
            \mathrm{log} \, \mathcal{N}(\mathbf{x} \rvert \boldsymbol{\mu}, \boldsymbol{\sigma}^2 \odot \mathbf{I}) =
            -\frac{D}{2} \mathrm{log}(2 \pi) -\frac{1}{2}\sum_{d=1}^D \mathrm{log}(\sigma_d^2)
            - \frac{1}{2} \sum_{d=1}^D \frac{(x_d - \mu_d)^2}{\sigma_d^2}
        $$

        :param x: (torch.Tensor) The data point(s). Shape is be (batch size, D) with D being the feature dimension.
        :param generative_model: (Dict[str, torch.Tensor]) Dictionary of the predicted parameters of the generative model.
        :param min_variance: (float) minimum variance for numerical stability
        :return: log_likelihood (torch.Tensor) The sum of log-likelihoods over the last dimension (D) for each sample.
            The output shape will be of (batch_size, ).
        """

        # Minimum variance to avoid numerical stability
        variance = torch.clamp(generative_model["Variance"], min_variance)

        # Check the support of the distribution
        assert torch.all(variance > 0).item(), "The variance of the likelihood must be greater than 0"

        # Constant term
        constant = -0.5 * x.shape[-1] * math.log(2 * math.pi)

        # Log-determinant
        log_determinant = -0.5 * torch.log(variance).sum(dim=-1)

        # Quadratic term
        if self.scaling_factor:
            quadratic_term = -0.5 * (1 / variance * (x - generative_model["Scaled mean"]) ** 2).sum(dim=-1)
        else:
            quadratic_term = -0.5 * (1 / variance * (x - generative_model["Mean"]) ** 2).sum(dim=-1)

        # Final log-likelihood
        log_likelihood = constant + log_determinant + quadratic_term

        return log_likelihood


    def negative_log_likelihood(self, x: torch.Tensor, generative_model: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""
        The negative log-likelihood is computed as the negative of the log-likelihood function. It is used as a part of
        the loss function to be minimized during training. The negative log-likelihood is computed using the
        log_likelihood function defined above.

        :param x: (torch.Tensor) The data point(s). Shape is be (batch size, D) with D being the feature dimension.
        :param generative_model: (Dict[str, torch.Tensor]) Dictionary of the predicted parameters of the generative model.
        :return: negative_log_likelihood (torch.Tensor) The sum of negative log-likelihoods over the last dimension (D)
            for each sample. The output shape will be of (batch_size, ).
        """
        return -self.log_likelihood(x, generative_model)

    def variational_posterior(self, x: torch.Tensor, eps: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        r"""
        The variational posterior computes the mean and variance of the latent variable given the input data. It uses
        the encoder network to compute the mean and variance of the latent variable. The variational posterior is
        assumed to be a Gaussian distribution with a diagonal covariance matrix. The mean and variance are computed
        using the encoder network and all computed entities necessary to define the variational posterior are returned
        in a variational posterior dictionary. If eps is provided, a sample from the variational posterior is computed
        using the reparameterization trick.

        :param x: (torch.Tensor) The input data point(s). Shape is be (batch size, D) with D being the feature dimension.
        :param eps: (torch.Tensor | None) A tensor of random noise for the reparameterization trick. If None, no sample is computed. Shape is (batch size, latent_dim) or (num_samples, batch size, latent_dim) if multiple samples are desired.
        :return: variational_posterior_dict (Dict[str, torch.Tensor]) Dictionary containing the computed parameters of the variational posterior, including "Latent mean", "Latent variance", and optionally "Latent variable" if eps is provided, and "Scale factor" if scaling_factor is True.
        """

        # Check input data are positive real numbers
        assert torch.all(x >= 0).item(), "Input data must be positive real numbers including 0"

        # Minimum variance for numerical stability
        minimum_variance = 1e-3

        # Create dictionary which stores the computed parameters of the variational posterior
        variational_posterior_dict = {}

        # Compute the latent mean and the latent variance of the variational posterior using the encoder
        variational_posterior_dict["Latent mean"], latent_var = self.encoder(x)
        variational_posterior_dict["Latent variance"] = torch.clamp(latent_var, min=minimum_variance)

        # Check if Latent mean or Latent variance contain any NaN or Inf values
        if torch.isnan(variational_posterior_dict["Latent mean"]).any() or torch.isinf(variational_posterior_dict["Latent mean"]).any():
            pdb.set_trace()
        if torch.isnan(variational_posterior_dict["Latent variance"]).any() or torch.isinf(variational_posterior_dict["Latent variance"]).any():
            pdb.set_trace()

        # Compute a sample from the variational posterior through the reparameterization trick
        if eps is not None:

            # If num_samples is greater than 1, eps has shape (num_samples, batch_size, latent_dim)
            variational_posterior_dict["Latent variable"] = (
                    variational_posterior_dict["Latent mean"] + eps * variational_posterior_dict["Latent variance"].sqrt()
            )


            # Check if Latent mean or Latent variance contain any NaN or Inf values
            if torch.isnan(variational_posterior_dict["Latent variable"]).any() or torch.isinf(variational_posterior_dict["Latent variable"]).any():
                pdb.set_trace()

        if self.scaling_factor:
            # Compute the scale factor using the scale encoder
            variational_posterior_dict["Scale factor"] = self.scale_encoder(x)

        return variational_posterior_dict

    def generative_model(self, latent_variable: torch.Tensor, scale: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        r"""
        The generative model computes the mean and variance of the Gaussian likelihood given the latent variable. It
        uses the decoder network to compute the mean and variance of the Gaussian likelihood. The generative model is
        assumed to be a Gaussian distribution with a diagonal covariance matrix. The mean and variance are computed
        using the decoder network and all computed entities necessary to define the generative model are returned in a
        generative model dictionary. If scale is provided, the mean is scaled by the scale factor.

        :param latent_variable: (torch.Tensor) The latent variable(s) sampled from the variational posterior. Shape is (batch size, latent_dim) or (num_samples, batch size, latent_dim) if multiple samples are desired.
        :param scale: (torch.Tensor | None) A tensor of scale factors for the latent variable. If None, no scaling is applied. Shape is (batch size, 1) or (num_samples, batch size, 1) if multiple samples are desired.
        :return: generative_model_dict (Dict[str, torch.Tensor]) Dictionary containing the computed parameters of the generative model, including "Mean", "Variance", and optionally "Scaled mean" if scale is provided.
        """

        # Create dictionary which stores the computed parameters of the generative model
        generative_model_dict = {}

        # Compute the mean reconstruction and the variance using the decoder
        pred_mean, pred_var = self.decoder(latent_variable)

        # Since gene expression data are non-negative, we ensure the mean is strictly positive
        generative_model_dict["Mean"] = torch.relu(pred_mean) # potentially torch.clamp(pred_mean, min=1e-6) or torch.exp(pred_mean)
        generative_model_dict["Variance"] = pred_var

        if self.scaling_factor:
            # If scaling factor is used, compute the scale factor
            generative_model_dict["Scaled mean"] = scale * generative_model_dict["Mean"]

        return generative_model_dict

    def forward(self, x: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        The forward pass of the BetaVAE computes the negative evidence lower bound (ELBO) given the input data and a
        sample from the variational posterior. It computes the variational posterior using the encoder, the generative
        model using the decoder, the Kullback-Leibler divergence, and the negative log-likelihood. The negative ELBO is
        then computed as the sum of the negative log-likelihood and the Kullback-Leibler divergence weighted by the beta
        parameter. The function returns the ELBO, Kullback-Leibler divergence, and negative log-likelihood.

        :param x: (torch.Tensor) The input data point(s). Shape is be (batch size, D) with D being the feature dimension.
        :param eps: (torch.Tensor) A tensor of random noise for the reparameterization trick. Shape is (batch size, latent_dim) or (num_samples, batch size, latent_dim) if multiple samples are desired.
        :return: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] The negative ELBO, Kullback-Leibler divergence, and negative log-likelihood. Each tensor has shape (batch size, ) or (num_samples, batch size) if multiple samples are desired.
        """
        # Compute variational posterior using the encoder
        variational_posterior_dict = self.variational_posterior(x, eps)

        # Compute generative model using the decoder
        if self.scaling_factor:
            generative_model_dict = self.generative_model(variational_posterior_dict["Latent variable"],
                                                          variational_posterior_dict["Scale factor"])
        else:
            generative_model_dict = self.generative_model(variational_posterior_dict["Latent variable"], None)

        # Kullback-Leibler divergence
        kld = self.kullback_leibler_divergence(
            variational_posterior_dict["Latent mean"],
            variational_posterior_dict["Latent variance"]
        )

        # Negative log-likelihood
        nll = self.negative_log_likelihood(x, generative_model_dict)

        # Negative ELBO with beta implementation i.e. -mathcal{L} = \beta D_{KL} + NLL since we minimise
        elbo = nll + self.beta * kld

        return elbo, kld, nll

class NBGeneExpressionVAE(BetaVAE):
    r"""
    Beta variational autoencoder for gene expression data with a negative binomial (NB) observation model. This
    variational autoencoder is applied within TRVI if the NB is specified as observation model for gene expression data.
    """
    def __init__(
            self,
            input_dim: int,
            latent_dim: int = 10,
            beta: float = 1.0,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
            device: str = "cuda",
            scaling_factor: bool = True
    ) -> None:
        r"""
        This is the Beta Variational Autoencoder (BetaVAE) for gene expression data with a negative binomial likelihood
        and is used within TRVI if a negative binomial distribution is used as observation model for gene expression
        data. It inherits from the BetaVAE class and overrides the log_likelihood and generative_model methods to
        implement the negative binomial likelihood. The decoder is also replaced with a NBGeneExpressionDecoder that
        outputs the mean of the negative binomial distribution. The inverse dispersion parameter of the negative
        binomial distribution is a learnable parameter that is initialized randomly and optimized during training.
        The forward method is also overridden to compute the negative ELBO, Kullback-Leibler divergence, and negative
        log-likelihood using the negative binomial likelihood. The model can be configured to learn scaling factors for
        the latent representation, which can be useful for certain applications. When training with KL-warm-up, the beta
        parameter can be gradually increased from 0 to the desired value over a specified number of epochs.

        :param input_dim: (int) The dimension of the input features.
        :param latent_dim: (int) The dimension of the latent representation. Default is 10.
        :param beta: (float) The weight of the Kullback-Leibler divergence term in the ELBO. Default is 1.0.
        :param num_hidden_layers: (int) The number of hidden layers in the encoder and decoder networks. Default is 1.
        :param num_hidden_units: (int) The number of units in each hidden layer. Default is 128.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param device: (str) The device to run the model on. Default is "cuda".
        :param scaling_factor: (bool) Whether to learn scaling factors for the latent representation. Default is True.
        """
        super(NBGeneExpressionVAE, self).__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=beta,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate,
            device=device,
            scaling_factor=scaling_factor
        )

        # Inverse dispersion of negative binomial likelihood as learnable parameter
        self.inverse_dispersion = torch.nn.Parameter(torch.randn(input_dim))

        self.decoder = NBGeneExpressionDecoder(
            input_dim=latent_dim,
            output_dim=input_dim,
            num_cat_list=None,
            dropout_rate=dropout_rate,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units
        )

    def log_likelihood(
            self, x: torch.Tensor,
            generative_model: Dict[str, torch.Tensor],
            eps: float=1e-8
    ) -> torch.Tensor:
        r"""
        This implementation of a Beta-VAE for gene expression data assumes a generative model with a negative binomial
        likelihood. Thus, the log-likelihood function is derived from the negative binomial distribution $
        \mathrm{NB}(x \rvert \mu, \theta)$ with mean $\mu > 0$ and inverse dispersion $\theta > 0$. The
         log-likelihood function is evaluated using the given data point x (torch.Tensor) as well as the torch.Tensors
         corresponding to the keys "Mean" and "Inverse dispersion" of the dictionary generative_model. The negative
         binomial log-likelihood is then computed according to

         $$
            \begin{split}
                \mathrm{log} \, \mathrm{NB}(x \rvert \mu, \theta) &= \mathrm{log} \, \Gamma(x + \theta)
                - \mathrm{log} \, \Gamma(x + 1) - \mathrm{log} \, \Gamma(\theta) \\
                &\quad + \theta(\mathrm{log}(\theta + \epsilon) - \mathrm{log} \,(\theta + \mu + \epsilon)) \\
                &\quad + x (\mathrm{log} (\mu + \epsilon) - \mathrm{log} \,(\theta + \mu + \epsilon))
            \end{split}
        $$

        where $\epsilon = 1e-8$ is included for the purpose of numerical stability. The function returns the
        log-likelihood.

        :param x: (torch.Tensor) The data point(s). Shape is be (batch size, D) with D being the feature dimension.
        :param generative_model: (Dict[str, torch.Tensor) Dictionary of the predicted parameters of the generative model.
        :param eps: (float) numerical stability constant
        :returns: log_likelihood (torch.Tensor) The sum of log-likelihoods over the last dimension (D) for each sample.
            The output shape will be of (batch_size, )
        """

        # Check the data point x is non-negative count data
        assert torch.all((x >= 0) & (x == x.floor())).item(), "The data must be non-negative count data"

        # Extract mean and inverse dispersion from dictionary of generative model
        if self.scaling_factor:
            mu = generative_model["Scaled mean"]
        else:
            mu = generative_model["Mean"]
        theta = generative_model["Inverse dispersion"]

        # Check the support of the parameters of the distribution
        assert torch.all(mu > 0).item(), "The mean must be greater or equal 0"
        assert torch.all(theta > 0).item(), "The inverse dispersion must be greater 0"

        log_gamma_terms = torch.lgamma(x + theta) - torch.lgamma(x + 1) - torch.lgamma(theta)
        theta_log_term = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        x_log_term = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))

        # Log likelihood
        log_likelihood = (log_gamma_terms + theta_log_term + x_log_term).sum(dim=-1)

        return log_likelihood

    def generative_model(self, latent_variable: torch.Tensor, scale: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        r"""
        The generative model computes the mean and inverse dispersion of the negative binomial likelihood given the
        latent variable. It uses the decoder network to compute the mean of the negative binomial likelihood. The
        inverse dispersion parameter is a learnable parameter that is initialized randomly and optimized during
        training. The mean and inverse dispersion are computed using the decoder network and all computed entities
        necessary to define the generative model are returned in a generative model dictionary. If scale is provided,
        the mean is scaled by the scale factor.

        :param latent_variable: (torch.Tensor) The latent variable(s) sampled from the variational posterior. Shape is (batch size, latent_dim) or (num_samples, batch size, latent_dim) if multiple samples are desired.
        :param scale: (torch.Tensor | None) A tensor of scale factors for the latent variable. If None, no scaling is applied. Shape is (batch size, 1) or (num_samples, batch size, 1) if multiple samples are desired.
        :return: generative_model_dict (Dict[str, torch.Tensor]) Dictionary containing the computed parameters of the generative model, including "Mean", "Inverse dispersion", and optionally "Scaled mean" if scale is provided
        """

        # Create dictionary which stores the computed parameters of the generative model
        generative_model_dict = {}

        # Compute the mean using the decoder
        generative_model_dict["Mean"] = self.decoder(latent_variable)

        if self.scaling_factor:
            # If scaling factor is used, compute the scaled mean
            generative_model_dict["Scaled mean"] = scale * generative_model_dict["Mean"]

        # Inverse dispersion
        generative_model_dict["Inverse dispersion"] = torch.exp(self.inverse_dispersion)

        return generative_model_dict

    def forward(
            self,
            x: Tuple[torch.Tensor, torch.Tensor],
            eps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        The forward pass of the BetaVAE computes the negative evidence lower bound (ELBO) given the input data and a
        sample from the variational posterior. It computes the variational posterior using the encoder, the generative
        model using the decoder, the Kullback-Leibler divergence, and the negative log-likelihood. The negative ELBO is
        then computed as the sum of the negative log-likelihood and the Kullback-Leibler divergence weighted by the beta
        parameter. The function returns the ELBO, Kullback-Leibler divergence, and negative log-likelihood.

        :param x: (Tuple[torch.Tensor, torch.Tensor]) The input data point(s) and gene expression levels. The first tensor is the gene expression counts of shape (batch size, D) with D being the feature dimension. The second tensor is the gene expression levels of shape (batch size, D).
        :param eps: (torch.Tensor) A tensor of random noise for the reparameterization trick. Shape is (batch size, latent_dim) or (num_samples, batch size, latent_dim) if multiple samples are desired.
        :return: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] The negative ELBO, Kullback-Leibler divergence, and negative log-likelihood. Each tensor has shape (batch size, ) or (num_samples, batch size) if multiple samples are desired.
        """

        # Extract gene expression counts and levels
        x_counts, x_levels = x

        # Compute variational posterior using encoder and gene expression levels
        variational_posterior_dict = self.variational_posterior(x_levels, eps)

        # Compute generative model using the decoder
        if self.scaling_factor:
            generative_model_dict = self.generative_model(variational_posterior_dict["Latent variable"],
                                                          variational_posterior_dict["Scale factor"])
        else:
            generative_model_dict = self.generative_model(variational_posterior_dict["Latent variable"], None)

        # Kullback-Leibler divergence
        kld = self.kullback_leibler_divergence(
            variational_posterior_dict["Latent mean"],
            variational_posterior_dict["Latent variance"]
        )

        # Negative log-likelihood is evaluated on actual gene expression counts
        nll = self.negative_log_likelihood(x_counts, generative_model_dict)

        # Negative ELBO with beta implementation i.e. -mathcal{L} = \beta D_{KL} + NLL since we minimise
        elbo = nll + self.beta * kld

        return elbo, kld, nll

class ZINBGeneExpressionVAE(BetaVAE):
    r"""
    Beta variational autoencoder for gene expression data with a zero-inflated negative binomial (ZINB) observation
    model. This variational autoencoder is applied within TRVI if the ZINB is specified as observation model for gene
    expression data.
    """
    def __init__(
            self,
            input_dim: int,
            latent_dim: int = 10,
            beta: float = 1.0,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
            device: str = "cuda",
            scaling_factor: bool = True
    ) -> None:
        r"""
        This is the Beta Variational Autoencoder (BetaVAE) for gene expression data with a zero-inflated negative
        binomial likelihood (ZINB) and is used within TRVI if a zero-inflated negative binomial distribution is used as
        observation model for gene expression data. It inherits from the BetaVAE class and overrides the log_likelihood
        and generative_model methods to implement the ZINB likelihood. The decoder is also replaced with a
        ZINBGeneExpressionDecoder that outputs the mean and the logits of the zero-inflation probability of the ZINB
        distribution. The inverse dispersion parameter of the ZINB distribution is a learnable parameter that is
        initialized randomly and optimized during training. The forward method is also overridden to compute the
        negative ELBO, Kullback-Leibler divergence, and negative log-likelihood using the ZINB likelihood. The model can
        be configured to learn scaling factors for the latent representation, which can be useful for certain
        applications. When training with KL-warm-up, the beta parameter can be gradually increased from 0 to the desired
        value over a specified number of epochs.

        :param input_dim: (int) The dimension of the input features.
        :param latent_dim: (int) The dimension of the latent representation. Default is 10.
        :param beta: (float) The weight of the Kullback-Leibler divergence term in the ELBO. Default is 1.0.
        :param num_hidden_layers: (int) The number of hidden layers in the encoder and decoder networks. Default is 1.
        :param num_hidden_units: (int) The number of units in each hidden layer. Default is 128.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param device: (str) The device to run the model on. Default is "cuda".
        :param scaling_factor: (bool) Whether to learn scaling factors for the latent representation. Default is True.
        """
        super(ZINBGeneExpressionVAE, self).__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=beta,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate,
            device=device,
            scaling_factor=scaling_factor
        )

        # Inverse dispersion of negative binomial likelihood as learnable parameter
        self.inverse_dispersion = torch.nn.Parameter(torch.randn(input_dim))

        self.decoder = ZINBGeneExpressionDecoder(
            input_dim=latent_dim,
            output_dim=input_dim,
            num_cat_list=None,
            dropout_rate=dropout_rate,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units
        )

    def log_likelihood(
            self, x: torch.Tensor,
            generative_model: Dict[str, torch.Tensor],
            eps: float=1e-8
    ) -> torch.Tensor:
        r"""
        This implementation of a Beta-VAE for gene expression data assumes a generative model with a zero-inflated
        negative binomial (ZINB) likelihood. Thus, the log-likelihood function is derived from the ZINB distribution
        $\mathrm{ZINB}(x \rvert \mu, \theta, \pi_0)$ with mean `$\mu > 0$, inverse dispersion
        $\theta > 0$, and zero-inflation probability $\pi_0 \in [0, 1]$. Instead of the zero-inflation
        probability $\pi_0$, the log-likelihood uses the logits of the zero-inflation probability
        $\rho = \mathrm{logit} \, \pi_0$  which are the direct output of the decoder. The log-likelihood function
        is evaluated using the given data point x (torch.Tensor) as well as the torch.Tensors corresponding to the keys
        "Mean" or "Scaled mean", "Inverse dispersion", and "Logit of zero-inflation probability" of the dictionary
        generative_model. The ZINB log-likelihood is then computed according to

        $$
            \mathrm{log} \, \mathrm{ZINB}(x \rvert \mu, \theta, \rho) = \begin{cases}
                \mathrm{softplus}(-\rho + \mathrm{log} \, \mathrm{NB}(x \rvert \mu, \theta)) - \mathrm{softplus}(-\rho)
                & \text{if } x = 0 \\
                -\rho - \mathrm{softplus}(-\rho) + \mathrm{log} \,\mathrm{NB}(x \rvert \mu, \theta)
                & \text{if } x \neq 0
            \end{cases}
        $$

        where $$\mathrm{NB}(x \rvert \mu, \theta)$$ is the negative binomial distribution. The function returns the
        log-likelihood.

        :param x: (torch.Tensor) The data point(s). Shape is be (batch size, D) with D being the feature dimension.
        :param generative_model: (Dict[str, torch.Tensor) Dictionary of the predicted parameters of the generative model.
        :param eps: (float) constant for numerical stability
        :returns: log_likelihood (torch.Tensor) The sum of log-likelihoods over the last dimension (D) for each sample.

        """
        # Check the data point x is non-negative count data
        assert torch.all((x >= 0) & (x == x.floor())).item(), "The data must be non-negative count data"

        # Extract the (scaled) mean, inverse dispersion, and zero-inflation probability from dictionary of generative model
        if self.scaling_factor:
            mu = generative_model["Scaled mean"]
        else:
            mu = generative_model["Mean"]
        theta = generative_model["Inverse dispersion"]
        logit_pi_0 = generative_model["Logit zero-inflation probability"] # this is rho

        # Check the support of the parameters of the distribution
        if torch.all(mu > 0).item() == False:
            pdb.set_trace()

        if torch.all(theta > 0).item() == False:
            pdb.set_trace()

        assert torch.all(mu > 0).item(), "The mean must be greater 0"
        assert torch.all(theta > 0).item(), "The inverse dispersion must be greater 0"
        # The zero-inflation probability is modeled as a logit transformation, thus it can take any real value.

        # Log-likelihood implementation according to scVI
        softplus_pi = F.softplus(-logit_pi_0)
        log_theta_eps = torch.log(theta + eps)
        log_theta_mu_eps = torch.log(theta + mu + eps)
        pi_theta_log = -logit_pi_0 + theta * (log_theta_eps - log_theta_mu_eps)

        # Case x = 0
        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

        # Case x > 0
        case_non_zero = (
                -softplus_pi
                + pi_theta_log
                + x * (torch.log(mu + eps) - log_theta_mu_eps)
                + torch.lgamma(x + theta)
                - torch.lgamma(theta)
                - torch.lgamma(x + 1)
        )
        mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

        log_likelihood = (mul_case_zero + mul_case_non_zero).sum(dim=-1)

        return log_likelihood

    def generative_model(self, latent_variable: torch.Tensor, scale: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        r"""
        The generative model computes the mean, inverse dispersion, and zero-inflation probability of the ZINB
        observation model given the latent variable. It uses the decoder network to compute the mean and the logits of
        the zero-inflation probability of the ZINB likelihood. The inverse dispersion parameter is a learnable parameter
        that is initialized randomly and optimized during training. The mean, inverse dispersion, and zero-inflation
        probability are computed using the decoder network and all computed entities necessary to define the generative
        model are returned in a generative model dictionary. If scale is provided, the mean is scaled by the scale
        factor.

        :param latent_variable: (torch.Tensor) The latent variable(s) sampled from the variational posterior. Shape is (batch size, latent_dim) or (num_samples, batch size, latent_dim) if multiple samples are desired.
        :param scale: (torch.Tensor | None) A tensor of scale factors for the latent variable. If None, no scaling is applied. Shape is (batch size, 1) or (num_samples, batch size, 1) if multiple samples are desired.
        :return: generative_model_dict (Dict[str, torch.Tensor]) Dictionary containing the computed parameters of the
            generative model, including "Mean", "Inverse dispersion", "Logit zero-inflation probability", and optionally "Scaled mean" if scale is provided.
        """
        # Create dictionary which stores the computed parameters of the generative model
        generative_model_dict = {}

        # Check if latent variable is [num_samples, batch_size, latent_dim]
        if len(latent_variable.shape) == 3:
            num_mc_samples, batch_size, _ = latent_variable.shape
            latent_variable = latent_variable.view(num_mc_samples * batch_size, -1)
        else:
            num_mc_samples = 1
            batch_size = latent_variable.shape[0]

        # Compute the mean and the logit of zero-inflation probability using the decoder
        generative_mean, generative_logit_zi = self.decoder(latent_variable)
        generative_zi_mean = generative_mean * (1 - F.sigmoid(generative_logit_zi))

        if num_mc_samples > 1:
            generative_model_dict["Mean"] = generative_mean.view(num_mc_samples, batch_size, -1)
            generative_model_dict["Logit zero-inflation probability"] = generative_logit_zi.view(num_mc_samples, batch_size, -1)
            generative_model_dict["ZI mean"] = generative_zi_mean.view(num_mc_samples, batch_size, -1)
        else:
            generative_model_dict["Mean"] = generative_mean
            generative_model_dict["Logit zero-inflation probability"] = generative_logit_zi
            generative_model_dict["ZI mean"] = generative_zi_mean

        if self.scaling_factor:
            # NOT adjusted for MC samples yet
            # If scaling factor is used, compute the scaled mean
            generative_model_dict["Scaled mean"] = scale * generative_model_dict["Mean"]
            generative_model_dict["ZI scaled mean"] = generative_model_dict["Scaled mean"] * (1 - F.sigmoid(generative_model_dict["Logit zero-inflation probability"]))

        # Inverse dispersion
        generative_model_dict["Inverse dispersion"] = torch.exp(self.inverse_dispersion)

        return generative_model_dict

    def forward(
            self,
            x: Tuple[torch.Tensor, torch.Tensor],
            eps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        The forward pass of the BetaVAE computes the negative evidence lower bound (ELBO) given the input data and a
        sample from the variational posterior. It computes the variational posterior using the encoder, the generative
        model using the decoder, the Kullback-Leibler divergence, and the negative log-likelihood. The negative ELBO is
        then computed as the sum of the negative log-likelihood and the Kullback-Leibler divergence weighted by the beta
        parameter. The function returns the ELBO, Kullback-Leibler divergence, and negative log-likelihood.

        :param x: (Tuple[torch.Tensor, torch.Tensor]) The input data point(s) and gene expression levels. The first tensor is the gene expression counts of shape (batch size, D) with D being the feature dimension. The second tensor is the gene expression levels of shape (batch size, D).
        :param eps: (torch.Tensor) A sample from the variational posterior.
        :return: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] The negative ELBO, Kullback-Leibler divergence, and negative log-likelihood. Each tensor has shape (batch size, ) or (num_samples, batch size) if multiple samples are desired.
        """

        # Extract gene expression counts and levels
        x_counts, x_levels = x

        # Compute variational posterior using encoder and gene expression levels
        variational_posterior_dict = self.variational_posterior(x_levels, eps)

        # Compute generative model using the decoder
        if self.scaling_factor:
            generative_model_dict = self.generative_model(variational_posterior_dict["Latent variable"],
                                                          variational_posterior_dict["Scale factor"])
        else:
            generative_model_dict = self.generative_model(variational_posterior_dict["Latent variable"], None)

        # Kullback-Leibler divergence
        kld = self.kullback_leibler_divergence(
            variational_posterior_dict["Latent mean"],
            variational_posterior_dict["Latent variance"]
        )

        # Negative log-likelihood is evaluated on actual gene expression counts
        nll = self.negative_log_likelihood(x_counts, generative_model_dict)

        # Negative ELBO with beta implementation i.e. -mathcal{L} = \beta D_{KL} + NLL since we minimise
        elbo = nll + self.beta * kld

        return elbo, kld, nll

class DMTranscriptUsageVAE(BetaVAE):
    r"""
    transcript usage Variational Inference (tuVI) variational autoencoder for alternative-splicing induced transcript
    usage data when a Dirichlet-Multinomial (DM) observation model is applied (tuVI-DM). It is also used within TRVI if its
    observation model is specified as DM.
    """
    def __init__(
            self,
            input_dim: int, # number of introns
            num_intron_groups: int,
            intron_groups: np.ndarray,
            latent_dim: int = 10,
            beta: float = 1.0,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
            device: str = "cuda",
            scaling_factor: bool = False
    ) -> None:
        r"""
        This is the class for the transcript usage Variational Inference (tuVI) variational autoencoder when a
        Dirichlet-Multinomial (DM) observation model is used. It inherits from the BetaVAE class and overrides the
        log_likelihood and generative_model methods to implement the DM likelihood. The decoder is
        also replaced with a IntronsDecoder that outputs the exon junction (intron) proportion vector used for the
        concentration parameter of the DM distribution. The feature precision parameter of the DM distribution is a
        learnable parameter that is initialized randomly and optimized during training. The forward method is also
        overridden to compute the negative ELBO, Kullback-Leibler divergence, and negative log-likelihood using the DM
        likelihood. The model can be configured to learn scaling factors for the latent representation, which can be
        useful for certain applications. When training with KL-warm-up, the beta parameter can be gradually increased
        from 0 to the desired value over a specified number of epochs.

        :param input_dim: (int) The dimension of the input features (number of introns).
        :param num_intron_groups: (int) The number of intron groups.
        :param intron_groups: (np.ndarray) An array of shape (input_dim,) containing the group index for each intron.
        :param latent_dim: (int) The dimension of the latent representation. Default is 10.
        :param beta: (float) The weight of the Kullback-Leibler divergence term in the ELBO. Default is 1.0.
        :param num_hidden_layers: (int) The number of hidden layers in the encoder and decoder networks. Default is 1.
        :param num_hidden_units: (int) The number of units in each hidden layer. Default is 128.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param device: (str) The device to run the model on. Default is "cuda".
        :param scaling_factor: (bool) Whether to learn scaling factors for the latent representation. Default is False.
        """
        super(DMTranscriptUsageVAE, self).__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=beta,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate,
            device=device,
            scaling_factor=scaling_factor
        )

        # Learnable feature precision parameter for concentration of Dirichlet-Multinomial distribution
        # self.feature_precision = nn.Parameter(torch.randn(num_intron_groups))
        # self.feature_precision = torch.zeros(num_intron_groups, dtype=torch.float32, device=device)

        self.register_buffer(
            "feature_precision",
            torch.zeros(num_intron_groups, dtype=torch.float32)
        )

        unique_intron_groups, intron_group_idx_start = np.unique(intron_groups, return_index=True)

        self.register_buffer(
            "intron_group_idx_nostart",
            torch.tensor(np.delete(np.arange(input_dim), intron_group_idx_start), dtype=torch.long)
        )
        self.register_buffer(
            "intron_group_idx_start",
            torch.tensor(intron_group_idx_start, dtype=torch.long)
        )
        self.register_buffer(
            "intron_groups",
            torch.tensor(intron_groups, dtype=torch.long)
        )

        self.register_buffer(
            "intron_group_indices",
            torch.stack(
                (
                    torch.tensor(intron_groups, dtype=torch.long),
                    torch.arange(0, input_dim, dtype=torch.long),
                ),
                dim=0,
            ),
        )
        """
        intron_group_summation = torch.sparse_coo_tensor(
            self.intron_group_indices,
            torch.ones(input_dim, dtype=torch.float),
            torch.Size([num_intron_groups, input_dim]),
        )
        """
        self.register_buffer(
            "intron_group_summation",
            torch.sparse_coo_tensor(
                self.intron_group_indices,
                torch.ones(input_dim, dtype=torch.float),
                torch.Size([num_intron_groups, input_dim]),
            )
        )

        self.decoder = IntronsDecoder(
            intron_groups=self.intron_groups,
            intron_group_summation=self.intron_group_summation,
            input_dim=latent_dim,
            output_dim=input_dim,
            num_cat_list=None,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate
        )

    def log_likelihood(
            self,
            x: torch.Tensor,
            generative_model: Dict[str, torch.Tensor],
            **kwargs
    ) -> torch.Tensor:
        r"""
        Given the concentration in the the generative model dictionary of the log likelihood of the
        Dirichlet-Multinomial distribution for a count data point x (torch.Tensor) is computed via

        $$
            \mathrm{log} \, \mathrm{DirMult}(\mathbf{x} \rvert n, \boldsymbol{\alpha}) = \mathrm{log} \,
            \Gamma(\alpha_0) + \mathrm{log} \, \Gamma(n+1) - \mathrm{log} \, \Gamma(n + \alpha_0) + \sum_{k=1}^K
            \mathrm{log} \, \Gamma(x_k + \alpha_k) - \mathrm{log} \,\Gamma(\alpha_k) - \mathrm{log} \, \Gamma(\alpha_k) -
            \mathrm{log} \, \Gamma(x_k + 1)
        $$

        The function returns the log-likelihood as torch.Tensor.

        :param x: (torch.Tensor) The data point(s). Shape is be (batch size, D) with D being the feature dimension.
        :param generative_model: (Dict[str, torch.Tensor) Dictionary of the predicted parameters of the generative model.
        :return: log_likelihood (torch.Tensor) The sum of log-likelihoods over the last dimension (D) for each sample. The output shape will be of (batch_size, )
        """
        # Extract alpha from the generative model
        alpha = generative_model["Concentration"]

        # Determine alpha_0 and number of trials
        alpha_0 = torch.sparse.mm(self.intron_group_summation, alpha.T).T  # equivalent to alpha_sum in scQuint
        num_trials = torch.sparse.mm(self.intron_group_summation, x.T).T  # equivalent to x_sum

        # Check the support of the distribution
        assert torch.all(x >= 0).item(), "The data must be non-negative integers"
        assert torch.all(x == x.floor()).item(), "The data must be integers"
        assert torch.all(alpha > 0).item(), "The concentration parameter must be greater or equal 0"
        assert torch.all(num_trials >= 0).item(), "The number of trials must be non-negative integers"
        assert torch.all(num_trials == num_trials.floor()).item(), "The number of trials must be integers"

        log_beta_t1 = (torch.lgamma(alpha_0) + torch.lgamma(num_trials + 1) - torch.lgamma(num_trials + alpha_0)).sum(
            dim=1)
        log_beta_t2 = (torch.lgamma(x + alpha) - torch.lgamma(alpha) - torch.lgamma(x + 1)).sum(dim=1)

        log_likelihood = log_beta_t1 + log_beta_t2

        return log_likelihood

    def generative_model(self, latent_variable: torch.Tensor, x: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        r"""
        The generative model computes the concentration parameter of the Dirichlet-Multinomial distribution given the
        latent variable. It uses the decoder network to compute the intron proportion vector used for the concentration
        parameter of the Dirichlet-Multinomial distribution. The feature precision parameter is a learnable parameter
        that is initialized randomly and optimized during training. The concentration parameter is computed using the
        decoder network and all computed entities necessary to define the generative model are returned in a generative
        model dictionary. If x is provided, the number of trials is computed as the sum of counts for each intron group
        and added to the generative model dictionary.

        :param latent_variable: (torch.Tensor) The latent variable(s) sampled from the variational posterior. Shape is (batch size, latent_dim) or (num_samples, batch size, latent_dim) if multiple samples are desired.
        :param x: (torch.Tensor | None) A tensor of counts for each intron group. If None, the number of trials is not computed. Shape is (batch size, D) with D being the feature dimension or (num_samples, batch size, D) if multiple samples are desired.
        :return: generative_model_dict (Dict[str, torch.Tensor]) Dictionary containing the computed parameters of the generative model, including "Feature precision", "Intron proportion", "Concentration", "Sum of concentration", and optionally "Number of trials" and "Mean" if x is provided.
        """
        # Create dictionary which stores the computed parameters of the generative model
        generative_model_dict = {}

        # Compute the reconstructed intron proportion using the decoder and extract the feature precision
        generative_model_dict["Feature precision"] = torch.exp(self.feature_precision)[self.intron_groups] + 1e-6 # add small constant for numerical stability
        generative_model_dict["Intron proportion"] = self.decoder(latent_variable, self.intron_group_idx_start)

        # Compute the concentration parameter alpha
        generative_model_dict["Concentration"] = generative_model_dict["Intron proportion"] * generative_model_dict["Feature precision"]
        assert torch.all(generative_model_dict["Concentration"] > 0).item(), "The concentration must be greater than 0"

        # Compute the sum of concentration parameter
        generative_model_dict["Sum of concentration"] = torch.sparse.mm(
            self.intron_group_summation,
            generative_model_dict["Concentration"].T
        ).T  # equivalent to alpha_sum in scQuint

        # If x is None cross-modal generation of proportions is carried out (necessary for TRVI)
        if x is not None:
            generative_model_dict["Number of trials"] = torch.sparse.mm(self.intron_group_summation, x.T).T
            assert torch.all(generative_model_dict["Number of trials"] >= 0).item(), "The number of trials must be non-negative integers"
            assert torch.all(generative_model_dict["Number of trials"] == generative_model_dict["Number of trials"].floor()).item(), "The number of trials must be integers"

            # Calculate the mean of the Dirichlet-Multinomial distribution
            num_trials_per_intron = generative_model_dict["Number of trials"][:, self.intron_groups]
            sum_concentration_per_intron = generative_model_dict["Sum of concentration"][:, self.intron_groups]
            generative_model_dict["Mean"] = num_trials_per_intron / sum_concentration_per_intron * generative_model_dict["Concentration"]

        return generative_model_dict

    def forward(
            self,
            x: Tuple[torch.Tensor, torch.Tensor],
            eps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        The forward pass of the BetaVAE computes the negative evidence lower bound (ELBO) given the input data and a
        sample from the variational posterior. It computes the variational posterior using the encoder, the generative
        model using the decoder, the Kullback-Leibler divergence, and the negative log-likelihood. The negative ELBO is
        then computed as the sum of the negative log-likelihood and the Kullback-Leibler divergence weighted by the beta
        parameter. The function returns the ELBO, Kullback-Leibler divergence, and negative log-likelihood.

        :param x:
        :param eps:
        """

        # Extract intron counts and intron levels (i.e. log(1+x)-normalised counts or PSI-scores)
        x_counts, x_levels = x

        # Compute variational posterior using the encoder
        variational_posterior_dict = self.variational_posterior(x_levels, eps)

        # Compute generative model using the decoder
        generative_model_dict = self.generative_model(variational_posterior_dict["Latent variable"], x_counts)

        # Kullback-Leibler divergence
        kld = self.kullback_leibler_divergence(
            variational_posterior_dict["Latent mean"],
            variational_posterior_dict["Latent variance"]
        )

        # Negative log-likelihood is evaluated on actual intron counts
        nll = self.negative_log_likelihood(x_counts, generative_model_dict)

        # Negative ELBO with beta implementation i.e. -mathcal{L} = \beta D_{KL} + NLL since we minimise
        elbo = nll + self.beta * kld

        return elbo, kld, nll

class ZIDMTranscriptUsageVAE(BetaVAE):
    r"""
    transcript usage Variational Inference (tuVI) variational autoencoder for alternative-splicing induced transcript
    usage data when a heuristic Zero-Inflated Dirichlet-Multinomial (ZIDM) surrogate of the ZANIDM log-likelihood is
    applied (tuVI-ZIDM). It is also used within TRVI if its observation model is specified as ZIDM. Note that the ELBO
    here is strictly speaking a training objective since ZIDM is a surrogate and not a true log-likelihood.
    """
    def __init__(
            self,
            input_dim: int,  # number of introns
            num_intron_groups: int,
            intron_groups: np.ndarray,
            latent_dim: int = 10,
            beta: float = 1.0,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
            device: str = "cuda",
            scaling_factor: bool = False
    ) -> None:
        r"""
        This is the class for the transcript usage Variational Inference (tuVI) variational autoencoder when the
        heuristic Zero-Inflated Dirichlet-Multinomial (ZIDM) surrogate for the ZANIDM log-likelihood is used. It
        inherits from the BetaVAE class and overrides the log_likelihood and generative_model methods to implement the
        heuristic ZIDM surrogate for the ZANIDM log-likelihood. The decoder is also replaced with a
        ZIDMTranscriptUsageDecoder that outputs the exon junction (intron) proportion vector used for the concentration
        parameter of the heuristic ZIDM surrogate as well as the logits of the zero-inflation probability. The feature
        precision parameter of the heurstic ZIDM surrogate is a learnable parameter that is initialized randomly and
        optimized during training. The forward method is also overridden to compute the negative ELBO, Kullback-Leibler
        divergence, and negative log-likelihood using the ZIDM likelihood. The model can be configured to learn scaling
        factors for the latent representation, which can be useful for certain applications. When training with
        KL-warm-up, the beta parameter can be gradually increased from 0 to the desired value over a specified number
        of epochs.

        :param input_dim: (int) The dimension of the input features (number of introns).
        :param num_intron_groups: (int) The number of intron groups.
        :param intron_groups: (np.ndarray) An array of shape (input_dim,) containing the group index for each intron.
        :param latent_dim: (int) The dimension of the latent representation. Default is 10.
        :param beta: (float) The weight of the Kullback-Leibler divergence term in the ELBO. Default is 1.0.
        :param num_hidden_layers: (int) The number of hidden layers in the encoder and decoder networks. Default is 1.
        :param num_hidden_units: (int) The number of units in each hidden layer. Default is 128.
        :param dropout_rate: (float) The dropout rate for regularization. Default is 0.1.
        :param device: (str) The device to run the model on. Default is "cuda".
        :param scaling_factor: (bool) Whether to learn scaling factors for the latent representation. Default is False.
        """
        super(ZIDMTranscriptUsageVAE, self).__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=beta,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate,
            device=device,
            scaling_factor=scaling_factor
        )

        # Learnable feature precision parameter for concentration of Dirichlet-Multinomial distribution
        #self.feature_precision = nn.Parameter(torch.randn(num_intron_groups))
        #self.feature_precision = torch.zeros(num_intron_groups, dtype=torch.float32, device=device)

        self.register_buffer(
            "feature_precision",
            torch.zeros(num_intron_groups, dtype=torch.float32)
        )

        unique_intron_groups, intron_group_idx_start = np.unique(intron_groups, return_index=True)

        self.register_buffer(
            "intron_group_idx_nostart",
            torch.tensor(np.delete(np.arange(input_dim), intron_group_idx_start), dtype=torch.long)
        )
        self.register_buffer(
            "intron_group_idx_start",
            torch.tensor(intron_group_idx_start, dtype=torch.long)
        )
        self.register_buffer(
            "intron_groups",
            torch.tensor(intron_groups, dtype=torch.long)
        )

        self.register_buffer(
            "intron_group_indices",
            torch.stack(
                (
                    torch.tensor(intron_groups, dtype=torch.long),
                    torch.arange(0, input_dim, dtype=torch.long),
                ),
                dim=0,
            ),
        )
        """
        intron_group_summation = torch.sparse_coo_tensor(
            self.intron_group_indices,
            torch.ones(input_dim, dtype=torch.float),
            torch.Size([num_intron_groups, input_dim]),
        )
        """
        self.register_buffer(
            "intron_group_summation",
            torch.sparse_coo_tensor(
                self.intron_group_indices,
                torch.ones(input_dim, dtype=torch.float),
                torch.Size([num_intron_groups, input_dim]),
            )
        )

        self.decoder = ZIDMTranscriptUsageDecoder(
            intron_groups=self.intron_groups,
            intron_group_summation=self.intron_group_summation,
            input_dim=latent_dim,
            output_dim=input_dim,
            num_cat_list=None,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate
        )

    def log_likelihood(
            self,
            x: torch.Tensor,
            generative_model: Dict[str, torch.Tensor],
            **kwargs
    ) -> torch.Tensor:
        r"""
        Given a count data point x (torch.Tensor) and the generative model dictionary generative_model containing the
        concentration parameter alpha (torch.Tensor), and the logit of the zero-inflation probability rho
        (torch.Tensor), this implementation computes the heuristic ZIDM surrogate of the ZANIDM log likelihood. Instead
        of the zero-inflation probability $\pi_0$, the log-likelihood uses the logits of
        the zero-inflation probability $\mathrm{logit} \, \pi_0$ which are the direct output of the
        decoder. The log-likelihood function is computed according to

        $$
            H^{\mathrm{ZIDM}}(\mathbf{x}^{(TU)} \rvert \boldsymbol{\alpha}, \boldsymbol{\rho}, N, \overline{N}^{(\mathcal{B})}) =  \begin{cases}
                \sum_{d=1}^{D} (\mathcal{S}(-\rho_{d} + \log  (h^{\mathrm{DM}}(\boldsymbol{\alpha}, \overline{N}^{(\mathcal{B})}))) - \mathcal{S}(-\rho_{d})) & \text{if } N = 0 \\
                \sum_{d=1}^{D} (-\rho_{d} - \mathcal{S}(-\rho_{d}) + \log  \mathrm{DM} \left(\mathbf{x}^{(TU)} \rvert \boldsymbol{\alpha}, N \right)) & \text{if } N > 0
        \end{cases}
        $$

        with

        $$
            h^{\mathrm{DM}}(\boldsymbol{\alpha}, \overline{N}^{(\mathcal{B})}) = \mathrm{sigmoid}\left(\log \frac{\Gamma(\alpha_{0}) \Gamma(\overline{N}^{(\mathcal{B})} + 1)}{\Gamma(\overline{N}^{(\mathcal{B})} + \alpha_{0})} \right) \in [0, 1]
        $$

        where $\mathrm{DM} \left(\mathbf{x}^{(TU)} \rvert \boldsymbol{\alpha}, N \right)$ is the Dirichlet-Multinomial
        distribution. The number of trials $N$ is computed as the sum of the counts in each intron group for
        each batch. For the case $\mathbf{x} = \mathbf{0}$, the number of trials is set to

        $$
            \overline{N}^{(\mathcal{B})} = \ceil*{\frac{1}{M}{\sum_{n \in \mathcal{B}} (N_{n} + 1)}}
        $$
        which is the average of number trials for each intron group in the batch where the minimum is set to 1.
        This ensures the zero-inflation model is still a valid probability mass function. The function returns the
        log-likelihood as a torch.Tensor.

        :param x: (torch.Tensor) The count data point(s). Shape is (batch size, D) with D being the feature dimension.
        :param generative_model: (Dict[str, torch.Tensor]) Dictionary of the predicted parameters of the generative model.
        :return: log_likelihood (torch.Tensor) The sum of log-likelihoods over the last dimension (D)
        """
        # Check the data point x is non-negative count data
        assert torch.all((x >= 0) & (x == x.floor())).item(), "The data must be non-negative count data"

        # Extract alpha and logit of zero-inflation probability from the generative model
        alpha = generative_model["Concentration"]
        logit_pi_0 = generative_model["Logit zero-inflation probability"]  # this is rho

        # Extract the number of trials, the sum of concentration, and the mean number of trials
        num_trials = generative_model["Number of trials"]
        alpha_0 = generative_model["Sum of concentration"]
        mean_num_trials = generative_model["Mean number of trials"]

        # Check the support of the distribution
        # Done in generative_model
        # The zero-inflation probability is modeled as a logit transformation, thus it can take any real value.

        # Log-likelihood implementation
        softplus_pi = F.softplus(-logit_pi_0)

        # Determine the a mask where False is count absent for any intron in an intron group and True if present
        mask = torch.zeros_like(x, dtype=torch.bool)

        for idx in range(0, x.shape[0]):
            non_zeros = torch.argwhere(x[idx] > 0).flatten().tolist()
            non_zero_intron_groups = torch.unique(self.intron_group_indices[:, non_zeros][0])
            mask[idx, :] = torch.isin(self.intron_group_indices[0], non_zero_intron_groups) # True is count present, False is count absent


        # Case x = 0
        #log_dm_case_zero = torch.lgamma(alpha_0) + torch.lgamma(mean_num_trials + 1) - torch.lgamma(mean_num_trials + alpha_0) # THIS IS ORIGINAL
        # TRIAL OF Valid probability

        log_dm_case_zero = F.logsigmoid(torch.lgamma(alpha_0) + torch.lgamma(mean_num_trials + 1) - torch.lgamma(mean_num_trials + alpha_0)) # NEW TRIAL
        log_dm_case_zero = log_dm_case_zero[..., self.intron_groups] # expand case_zero from intron groups to all introns

        case_zero = F.softplus(-logit_pi_0 + log_dm_case_zero) - softplus_pi

        # TROUBLESHOOTING: [..., ] might cause issues in no MC sample case
        case_zero[..., mask == True] = 0 # only apply case_zero to intron groups where x is 0

        # Case x > 0
        log_beta_t1 = (torch.lgamma(alpha_0) + torch.lgamma(num_trials + 1) - torch.lgamma(num_trials + alpha_0))

        log_beta_t1 = log_beta_t1[..., self.intron_groups]  # expand case_zero from intron groups to all introns
        log_beta_t2 = (torch.lgamma(x + alpha) - torch.lgamma(alpha) - torch.lgamma(x + 1))
        log_dm_case_non_zero = log_beta_t1 + log_beta_t2

        case_non_zero = -logit_pi_0 - softplus_pi + log_dm_case_non_zero

        # TROUBLESHOOTING: [..., ] might cause issues in no MC sample case
        case_non_zero[..., mask == False] = 0

        # Combine both cases
        log_likelihood = (case_zero + case_non_zero).sum(dim=-1)

        return log_likelihood



    def generative_model(self, latent_variable: torch.Tensor, x: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        r"""
        The generative model computes the concentration parameter, and the logit of the zero-inflation probability given
        the latent variable for the heuristic ZIDM surrogate. It uses the decoder network to compute the intron
        proportion vector used for the concentration parameter of the heuristic ZIDM surrogate as well as the logits of
        the zero-inflation probability. The feature precision parameter is a learnable parameter that is initialized
        randomly and optimized during training. The concentration parameter is computed using the decoder network and
        all computed entities necessary to define the generative model are returned in a generative model dictionary.
        If x is provided, the number of trials is computed as the sum of counts for each intron group and added to the
        generative model dictionary.

        :param latent_variable: (torch.Tensor) The latent variable(s) sampled from the variational posterior. Shape is (batch size, latent_dim) or (num_samples, batch size, latent_dim) if multiple samples are desired.
        :param x: (torch.Tensor | None) A tensor of counts for each intron group. If None, the number of trials is not computed. Shape is (batch size, D) with D being the feature dimension or (num_samples, batch size, D) if multiple samples are desired.
        :return: generative_model_dict (Dict[str, torch.Tensor]) Dictionary containing the computed parameters of the generative model, including "Feature precision", "Intron proportion", "Logit zero-inflation probability", "Concentration", "Sum of concentration", and optionally "Number of trials" and "Mean"
        """
        # Create dictionary which stores the computed parameters of the generative model
        generative_model_dict = {}

        # Check if latent variable is [num_samples, batch_size, latent_dim]
        if len(latent_variable.shape) == 3:
            num_mc_samples, batch_size, _ = latent_variable.shape
            latent_variable = latent_variable.view(num_mc_samples * batch_size, -1)
        else:
            num_mc_samples = 1
            batch_size = latent_variable.shape[0]

        # Compute the reconstructed intron proportion and the logit of the zero-inflation probability using the decoder
        generative_intron_prop, generative_logit_zi = self.decoder(latent_variable, self.intron_group_idx_start)

        # Extract the feature precision
        generative_model_dict["Feature precision"] = torch.exp(self.feature_precision)[
                                                         self.intron_groups] + 1e-6  # add small constant for numerical stability

        # Compute the concentration parameter alpha
        generative_concentration = (generative_intron_prop * generative_model_dict["Feature precision"]).clamp(min=1e-8) # clamp NEW
        # DEBUG numerical instability
        if torch.all(generative_concentration < 1e-8).item():
            pdb.set_trace()
        #assert torch.all(generative_concentration > 0).item(), "The concentration must be greater than 0"

        generative_concentration_sum = torch.sparse.mm(
            self.intron_group_summation,
            generative_concentration.T
        ).T  # equivalent to alpha_sum in scQuint

        # If x is None a cross-modal generation of the intron proportion is carried out (necessary in TRVI)
        if x is not None:
            generative_model_dict["Number of trials"] = torch.sparse.mm(self.intron_group_summation, x.T).T
            assert torch.all(generative_model_dict["Number of trials"] >= 0).item(), "The number of trials must be non-negative integers"
            assert torch.all(generative_model_dict["Number of trials"] == generative_model_dict["Number of trials"].floor()).item(), "The number of trials must be integers"

            # Calculate the pseudo number of trials for zero-inflation
            pseudo_trials = torch.ones(generative_model_dict["Number of trials"].shape, dtype=torch.float32).to(self.device)
            generative_model_dict["Mean number of trials"] = (generative_model_dict["Number of trials"] + pseudo_trials).mean(dim=0).ceil()

            assert torch.all(generative_model_dict["Mean number of trials"] >= 0).item(), "The mean number of trials (pseudo trials) must be non-negative integers"
            assert torch.all(generative_model_dict["Mean number of trials"] == generative_model_dict["Mean number of trials"].floor()).item(), "The mean number of trials (pseudo trials) must be integers"

            # Calculate the mean of the Dirichlet-Multinomial distribution
            if num_mc_samples > 1:
                num_trials_per_intron_unexpanded = generative_model_dict["Number of trials"][:, self.intron_groups]
                # Expand to match num_mc_samples
                num_trials_per_intron_expanded = num_trials_per_intron_unexpanded.unsqueeze(0).expand(10, -1, -1)
                num_trials_per_intron = num_trials_per_intron_expanded.contiguous().view(num_mc_samples * batch_size, -1)
            else:
                num_trials_per_intron = generative_model_dict["Number of trials"][:, self.intron_groups]

            sum_concentration_per_intron = generative_concentration_sum[:, self.intron_groups]
            generative_mean = num_trials_per_intron / sum_concentration_per_intron * generative_concentration
            generative_zi_mean = generative_mean * (1 - F.sigmoid(generative_logit_zi))

            if num_mc_samples > 1:
                generative_model_dict["Mean"] = generative_mean.view(num_mc_samples, batch_size, -1)
                generative_model_dict["ZI mean"] = generative_zi_mean.view(num_mc_samples, batch_size, -1)
            else:
                generative_model_dict["Mean"] = generative_mean
                generative_model_dict["ZI mean"] = generative_zi_mean

        if num_mc_samples > 1:
            generative_model_dict["Intron proportion"] = generative_intron_prop.view(num_mc_samples, batch_size, -1)
            generative_model_dict["Logit zero-inflation probability"] = generative_logit_zi.view(num_mc_samples, batch_size, -1)
            generative_model_dict["Concentration"] = generative_concentration.view(num_mc_samples, batch_size, -1)
            generative_model_dict["Sum of concentration"] = generative_concentration_sum.view(num_mc_samples, batch_size, -1)
        else:
            generative_model_dict["Intron proportion"] = generative_intron_prop
            generative_model_dict["Logit zero-inflation probability"] = generative_logit_zi
            generative_model_dict["Concentration"] = generative_concentration
            generative_model_dict["Sum of concentration"] = generative_concentration_sum

        """
        #OLD FROM START HERE

        # Compute the reconstructed intron proportion and the logit of the zero-inflation probability using the decoder
        generative_model_dict["Intron proportion"], generative_model_dict["Logit zero-inflation probability"] = self.decoder(latent_variable, self.intron_group_idx_start)

        # Extract the feature precision
        generative_model_dict["Feature precision"] = torch.exp(self.feature_precision)[self.intron_groups] + 1e-6  # add small constant for numerical stability

        # Compute the concentration parameter alpha
        generative_model_dict["Concentration"] = (generative_model_dict["Intron proportion"] * generative_model_dict[
            "Feature precision"]).clamp(min=1e-8) # clamp NEW
        # DEBUG numerical instability
        if torch.all(generative_model_dict["Concentration"] < 1e-8).item():
            pdb.set_trace()
        #assert torch.all(generative_model_dict["Concentration"] > 0).item(), "The concentration must be greater than 0"

        # Compute the sum of concentration parameter
        generative_model_dict["Sum of concentration"] = torch.sparse.mm(
            self.intron_group_summation,
            generative_model_dict["Concentration"].T
        ).T  # equivalent to alpha_sum in scQuint

        # If x is None a cross-modal generation of the intron proportion is carried out (necessary in TRVI)
        if x is not None:
            generative_model_dict["Number of trials"] = torch.sparse.mm(self.intron_group_summation, x.T).T
            assert torch.all(generative_model_dict["Number of trials"] >= 0).item(), "The number of trials must be non-negative integers"
            assert torch.all(generative_model_dict["Number of trials"] == generative_model_dict["Number of trials"].floor()).item(), "The number of trials must be integers"

            # Calculate the pseudo number of trials for zero-inflation
            pseudo_trials = torch.ones(generative_model_dict["Number of trials"].shape, dtype=torch.float32).to(self.device)
            generative_model_dict["Mean number of trials"] = (generative_model_dict["Number of trials"] + pseudo_trials).mean(dim=0).ceil()

            assert torch.all(generative_model_dict["Mean number of trials"] >= 0).item(), "The mean number of trials (pseudo trials) must be non-negative integers"
            assert torch.all(generative_model_dict["Mean number of trials"] == generative_model_dict["Mean number of trials"].floor()).item(), "The mean number of trials (pseudo trials) must be integers"

            # Calculate the mean of the Dirichlet-Multinomial distribution
            num_trials_per_intron = generative_model_dict["Number of trials"][:, self.intron_groups]
            sum_concentration_per_intron = generative_model_dict["Sum of concentration"][:, self.intron_groups]
            generative_model_dict["Mean"] = num_trials_per_intron / sum_concentration_per_intron * generative_model_dict["Concentration"]

            generative_model_dict["ZI mean"] = generative_model_dict["Mean"] * (1 - F.sigmoid(generative_model_dict["Logit zero-inflation probability"]))
        """

        return generative_model_dict

    def forward(
            self,
            x: torch.Tensor,
            eps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        The forward pass of the BetaVAE computes the negative evidence lower bound (ELBO) given the input data and a
        sample from the variational posterior. It computes the variational posterior using the encoder, the generative
        model using the decoder, the Kullback-Leibler divergence, and the negative log-likelihood. The negative ELBO
        (objective since the heuristic ZIDM surrogate is not a true log-likelihood) is then computed as the sum of the
        negative heuristic ZIDM surrogate and the Kullback-Leibler divergence weighted by the beta parameter. The
        function returns the negative "ELBO", Kullback-Leibler divergence, and negative heuristic ZIDM surrogate.

        :param x: (torch.Tensor) The input data point(s) are log(1 + x) transformed intron group count vectors. Shape is (batch size, D) with D being the feature dimension.
        :param eps: (torch.Tensor) A sample from the variational posterior. Shape is (batch size, latent_dim) or (num_samples, batch size, latent_dim) if multiple samples are desired.
        """

        # Extract intron counts and intron levels (i.e. log(1+x)-normalised counts or PSI-scores)
        x_counts, x_levels = x

        # Compute variational posterior using the encoder
        variational_posterior_dict = self.variational_posterior(x_levels, eps)

        # Compute generative model using the decoder
        generative_model_dict = self.generative_model(variational_posterior_dict["Latent variable"], x_counts)

        # Kullback-Leibler divergence
        kld = self.kullback_leibler_divergence(
            variational_posterior_dict["Latent mean"],
            variational_posterior_dict["Latent variance"]
        )

        # Negative log-likelihood is evaluated on actual intron counts
        nll = self.negative_log_likelihood(x_counts, generative_model_dict)

        # Negative ELBO with beta implementation i.e. -mathcal{L} = \beta D_{KL} + NLL since we minimise
        elbo = nll + self.beta * kld

        return elbo, kld, nll

class ZANIDMTranscriptUsageVAE(BetaVAE):
    r"""
    transcript usage Variational Inference (tuVI) variational autoencoder for alternative-splicing induced transcript
    usage data when a zero-and-N-inflated Dirichlet-Multinomial (ZANIDM) observation model is applied (tuVI-ZANIDM). It
    is also used within TRVI if its observation model is specified as ZANIDM.
    """
    def __init__(
            self,
            input_dim: int,  # number of introns
            num_intron_groups: int,
            intron_groups: np.ndarray,
            latent_dim: int = 10,
            beta: float = 1.0,
            num_hidden_layers: int = 1,
            num_hidden_units: int = 128,
            dropout_rate: float = 0.1,
            device: str = "cuda",
            scaling_factor: bool = False
    ) -> None:
        super(ZANIDMTranscriptUsageVAE, self).__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            beta=beta,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate,
            device=device,
            scaling_factor=scaling_factor
        )

        # Learnable feature precision parameter for concentration of Dirichlet-Multinomial distribution
        # self.feature_precision = nn.Parameter(torch.randn(num_intron_groups))
        # self.feature_precision = torch.zeros(num_intron_groups, dtype=torch.float32, device=device)

        self.register_buffer(
            "feature_precision",
            torch.zeros(num_intron_groups, dtype=torch.float32)
        )

        unique_intron_groups, intron_group_idx_start = np.unique(intron_groups, return_index=True)

        self.register_buffer(
            "intron_group_idx_nostart",
            torch.tensor(np.delete(np.arange(input_dim), intron_group_idx_start), dtype=torch.long)
        )
        self.register_buffer(
            "intron_group_idx_start",
            torch.tensor(intron_group_idx_start, dtype=torch.long)
        )
        self.register_buffer(
            "intron_groups",
            torch.tensor(intron_groups, dtype=torch.long)
        )

        self.register_buffer(
            "intron_group_indices",
            torch.stack(
                (
                    torch.tensor(intron_groups, dtype=torch.long),
                    torch.arange(0, input_dim, dtype=torch.long),
                ),
                dim=0,
            ),
        )
        """
        intron_group_summation = torch.sparse_coo_tensor(
            self.intron_group_indices,
            torch.ones(input_dim, dtype=torch.float),
            torch.Size([num_intron_groups, input_dim]),
        )
        """
        self.register_buffer(
            "intron_group_summation",
            torch.sparse_coo_tensor(
                self.intron_group_indices,
                torch.ones(input_dim, dtype=torch.float),
                torch.Size([num_intron_groups, input_dim]),
            )
        )

        # Create a dictionary mapping the intron groups to potential set of zero-inflated subsets
        #self.intron_group_set_zero_subsets = {}
        #unique_intron_groups, intron_group_sizes = torch.unique(self.intron_groups, return_counts=True)
        #for intron_group in range(0, len(unique_intron_groups)):
        #    if intron_group_sizes[intron_group] > 2:
        #        self.intron_group_set_zero_subsets[str(intron_group)] = create_set_zero_subsets(intron_group_sizes[intron_group])

        self.decoder = ZANIDMTranscriptUsageDecoder(
            intron_groups=self.intron_groups,
            intron_group_summation=self.intron_group_summation,
            input_dim=latent_dim,
            output_dim=input_dim,
            num_cat_list=None,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            dropout_rate=dropout_rate
        )

    def log_likelihood(self, x: torch.Tensor, generative_model: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        r"""
        Given a count data point x (torch.Tensor) and the generative model dictionary generative_model containing the
        concentration parameter alpha (torch.Tensor), and the excess-of-zeros probability (torch.Tensor), and the
        corresponding mixing weights, the log-likelihood of the zero-and-one-inflated Dirichlet-Multinomial is computed
        for counts $\mathbf{x}^{(TU)} = [x_1, \dots, x_D]^T$ of one isoform group count vector. Dependent on the
        observed isoform group count vector $\mathbf{x}^{(TU)}$, the ZANIDM distribution can be broken down into the
        four cases and their respective log-likelihood is computed. For case 1
        ($N > 0 \, \land \forall d \, x^{(TU)}_d > 0$),  the log-likelihood is obtained by

        $$
            \begin{split}
                \log \mathrm{ZANIDM}(\mathbf{x}^{(TU)} \rvert \boldsymbol{\alpha}, \boldsymbol{\zeta}, N) &= \log \eta^{(D)} + \log \Gamma(\alpha_0) + \log \Gamma(N + 1) - \log \Gamma(N + \alpha_0) \\
                &\quad+ \sum_{d=1}^D \log  \Gamma(x^{(TU)}_d + \alpha_d) - \log \Gamma(\alpha_d) - \log \Gamma(x^{(TU)}_d + 1).
            \end{split}
        $$

        For case 2 ($N = 0 \land \mathbf{x}^{(TU)} = \mathbf{0}$), the  log-likelihood is simply obtained form the mixture weight as

        $$
            \log  \mathrm{ZANIDM}(\mathbf{x}^{(TU)} \rvert \boldsymbol{\alpha}, \boldsymbol{\zeta}, N) = \log \eta^{(0)}.
        $$

        For case 3 ($N > 0 $ and $D-1$ zero-inflated intron counts), the set of zero-inflated subsets $\tilde{\mathcal{K}} = \{ \}$ is empty by definition. The log-likelihood is of case 3 then results in

        $$
            \begin{split}
                \log \mathrm{ZANIDM}(\mathbf{x}^{(TU)} \rvert \boldsymbol{\alpha}, \boldsymbol{\zeta}, \hat{N}) &= \log (\eta^{(D)} \frac{\Gamma(\alpha_{0}) \Gamma(N  + 1)}{\Gamma(N  + \alpha_{0})} \prod_{d=1}^{D} \frac{\Gamma \left(x_{d}^{(TU)} + \alpha_{d} \right)}{\Gamma(\alpha_{d})\Gamma \left( x_{d}^{(TU)} + 1 \right)} \\
                &+ \sum_{d=1}^{D} \eta^{(N)}_{d} \left(\mathbf{1}_0 \left(\sum_{k:k \neq d} x_{k}^{(TU)} \right) \right)) \end{split}.
        $$

        For case 4 ($N > 0$ and at most $ D - 2$ zero-inflated intron counts), the  log-likelihood is then given by

        $$
            \begin{split}
                \log \mathrm{ZANIDM}(\mathbf{x}^{(g,i)} \rvert \boldsymbol{\alpha}, \boldsymbol{\zeta}, \hat{N}) &= \log (\eta^{(D)} \frac{\Gamma(\alpha_{0}) \Gamma(N  + 1)}{\Gamma(N  + \alpha_{0})} \prod_{d=1}^{D} \frac{\Gamma \left(x_{d}^{(TU)} + \alpha_{d} \right)}{\Gamma(\alpha_{d})\Gamma \left( x_{d}^{(TU)} + 1 \right)} \\
                &+ \sum_{\mathcal{K} \in \tilde{\mathcal{K}}} \eta^{(\tilde{\mathcal{K}})}_{\mathcal{K}} \left( \mathbf{1}_0 \left( \sum_{j \in \mathcal{K}} x_{j}^{(TU)} \right) \right) \\ & \quad \quad \frac{\Gamma(\alpha_{\mathcal{K}}) \Gamma( N+ 1)}{\Gamma \left(N + \alpha_{\mathcal{K}} \right)} \prod_{d \notin \mathcal{K}} \frac{\Gamma \left(x_{d}^{(TU)} + \alpha_{d} \right)}{\Gamma (\alpha_{d}) \Gamma \left( x_{d}^{(TU)} + 1 \right)})
            \end{split}.
        $$


        The function returns the log-likelihood as a torch.Tensor.

        :param x: (torch.Tensor) The count data point(s). Shape is (batch size, D) with D being the feature dimension.
        :param generative_model: (Dict[str, torch.Tensor]) Dictionary of the predicted parameters of the generative model.
        :return: log_likelihood (torch.Tensor) The sum of log-likelihoods over the last dimension (D)
        """
        # Numerical stability constant
        eps = 1e-10

        # Extract alpha from the generative model
        alpha = generative_model["Concentration"] # shape (batch size, D)
        alpha_0 = generative_model["Sum of concentration"] # shape (batch size, num intron groups)
        num_trials = generative_model["Number of trials"] # shape (batch size, num intron groups)
        mix_weight_d = generative_model["Mixing weight d"] # shape (batch size, num intron groups)
        mix_weight_0 = generative_model["Mixing weight 0"] # shape (batch size, num intron groups)
        mix_weight_N = generative_model["Mixing weight N"] # shape (batch size, D)
        subset_alpha_mixing_params_batch_dict = generative_model["Subset concentration and mixing parameters"]

        # Check the data point x is non-negative count data
        assert torch.all((x >= 0) & (x == x.floor())).item(), "The data must be non-negative count data"
        assert torch.all(alpha > 0).item(), "The concentration parameter must be greater or equal 0"
        assert torch.all(num_trials >= 0).item(), "The number of trials must be non-negative integers"
        assert torch.all(num_trials == num_trials.floor()).item(), "The number of trials must be integers"

        # General calculation of Dirichlet-Multinomial term
        log_beta_t2_feature = torch.lgamma(x + alpha) - torch.lgamma(alpha) - torch.lgamma(x + 1)  # shape (batch size, D)
        log_beta_t2_group = torch.sparse.mm(self.intron_group_summation, log_beta_t2_feature.T).T  # shape (batch size, num intron groups)
        log_beta_t1_group = torch.lgamma(alpha_0) + torch.lgamma(num_trials + 1) - torch.lgamma(
            num_trials + alpha_0)  # shape (batch size, num intron groups)
        log_dm_group = log_beta_t1_group + log_beta_t2_group  # shape (batch size, num intron groups)
        log_likelihood_dm_term = torch.log(mix_weight_d.clamp(min=eps)) + log_dm_group  # shape (batch size, num intron groups)


        # Case 1: all counts in intron group are greater than zero i.e. for all j: x_j > 0 and N > 0
        mask_non_zero_trials = (num_trials > 0)
        group_indices = self.intron_group_indices[0].unsqueeze(0).expand(x.shape[0], -1)

        x_is_zero = (x == 0).float()
        num_zeros_group = torch.zeros(x.shape[0], num_trials.shape[-1], dtype=x.dtype, device=x.device)
        num_zeros_group.scatter_add_(dim=1, index=group_indices.long(), src=x_is_zero)
        mask_min_count_positive = (num_zeros_group == 0)
        mask_case_1 = mask_non_zero_trials & mask_min_count_positive

        #case_non_zeros = log_likelihood_dm_term * mask_case_1.float()
        case_non_zeros = torch.where(mask_case_1, log_likelihood_dm_term, torch.zeros_like(log_likelihood_dm_term))

        num_case_non_zeros = mask_case_1.sum()

        # Case 2: all counts in intron group are zero i.e. x = 0 and N = 0
        mask_all_zero = (num_trials == 0)  # shape (batch size, num intron groups)
        log_likelihood_case_zeros_all = torch.zeros_like(mix_weight_0) + torch.log(mix_weight_0.clamp(eps))  # shape (batch size, num intron groups)
        #case_zeros = log_likelihood_case_zeros_all * mask_all_zero.float()

        case_zeros = torch.where(mask_all_zero, log_likelihood_case_zeros_all, torch.zeros_like(log_likelihood_case_zeros_all))

        num_case_zeros = mask_all_zero.sum()

        # Case 3: d - 1 categories are zero and N > 0
        mask_d_1 = torch.full((x.shape[0], num_trials.shape[-1]),fill_value=False,dtype=torch.bool).to(self.device)

        for intron_group in range(0, num_trials.shape[-1]):
            ig_mask = (self.intron_group_indices[0] == intron_group)
            intron_indices_mask = self.intron_group_indices[1, ig_mask]
            ig_counts = x[:, intron_indices_mask]
            num_trials_ig = num_trials[:, intron_group]
            mask_d_1[:, intron_group] = (num_trials_ig > 0) & ( (ig_counts == 0).sum(dim=-1) == (ig_counts.shape[-1] - 1) )

        batch_indices, group_indices = torch.where(mask_d_1)
        num_active_cases = batch_indices.shape[0]

        if num_active_cases > 0:
            log_dm_active = log_dm_group[batch_indices, group_indices]
            mix_weight_d_active = mix_weight_d[batch_indices, group_indices]

            inflated_N_term_values = torch.zeros(num_active_cases, device=x.device)

            for ig in torch.unique(group_indices):
                group_mask = (group_indices == ig)
                active_k_indices = torch.where(group_mask)[0]

                ig_mask = (self.intron_group_indices[0] == ig)
                intron_indices_mask = self.intron_group_indices[1, ig_mask]

                current_x_rows = batch_indices[group_mask]
                ig_counts = x[current_x_rows[:, None], intron_indices_mask]

                mask_positive = (ig_counts > 0)

                idx_in_group = torch.argwhere(mask_positive)[:, 1]
                global_intron_indices = intron_indices_mask[idx_in_group]

                eta_N_values = mix_weight_N[current_x_rows, global_intron_indices]

                inflated_N_term_values[active_k_indices] = eta_N_values

            #pmf_active = mix_weight_d_active * torch.exp(log_dm_active) + inflated_N_term_values
            #log_likelihood_d_1_active = torch.log(pmf_active)
            # Numerically stable log sum exp implementation
            log_term_a = torch.log(mix_weight_d_active.clamp(eps)) + log_dm_active
            log_term_b = torch.log(inflated_N_term_values.clamp(eps))
            log_likelihood_d_1_active = torch.logaddexp(log_term_a, log_term_b)

            case_d_1 = torch.zeros_like(mix_weight_d)
            case_d_1[batch_indices, group_indices] = log_likelihood_d_1_active

            num_case_d_1 = (case_d_1 != 0).sum().item()
        else:

            case_d_1 = torch.zeros_like(mix_weight_d)
            num_case_d_1 = 0



        """
        # Original case_d_1
        case_d_1 = torch.zeros_like(mix_weight_d)  # shape (batch size, num intron groups)
        for x_idx in range(0, x.shape[0]):
            d1_intron_groups = torch.where(mask_d_1[x_idx, :] == True)[0]
            for ig in d1_intron_groups:
                ig_mask = (self.intron_group_indices[0] == ig)
                intron_indices_mask = self.intron_group_indices[1, ig_mask]
                ig_counts = x[x_idx, intron_indices_mask]

                num_trials_ig = num_trials[x_idx, ig]
                alpha_ig = alpha[x_idx, intron_indices_mask]
                alpha_0_ig = alpha_0[x_idx, ig]

                log_beta_t1 = torch.lgamma(alpha_0_ig) + torch.lgamma(num_trials_ig + 1) - torch.lgamma(num_trials_ig + alpha_0_ig)
                log_beta_t2 = torch.lgamma(ig_counts + alpha_ig) - torch.lgamma(alpha_ig) - torch.lgamma(ig_counts + 1)
                log_dm_ig = log_beta_t1 + log_beta_t2.sum()
                dm_ig = torch.exp(log_dm_ig)

                mix_weight_N_ig = mix_weight_N[x_idx, intron_indices_mask]
                inflated_num_trials_term = 0
                for j in range(0, len(ig_counts)):
                    sum_not_j = ig_counts.sum() - ig_counts[j]
                    inflated_num_trials_term = inflated_num_trials_term + mix_weight_N_ig[j] * (sum_not_j == 0).float()

                case_d_1[x_idx, ig] = case_d_1[x_idx, ig] + torch.log(mix_weight_d[x_idx, ig] * dm_ig + inflated_num_trials_term)
        """

        # Case 4: d - 2 categories are zero and N > 0
        mask_d_2 = torch.full((x.shape[0], num_trials.shape[-1]),fill_value=False,dtype=torch.bool).to(self.device)

        for intron_group in range(0, num_trials.shape[-1]):
            ig_mask = (self.intron_group_indices[0] == intron_group)
            intron_indices_mask = self.intron_group_indices[1, ig_mask]
            ig_counts = x[:, intron_indices_mask]
            num_trials_ig = num_trials[:, intron_group]
            cond_1 = (num_trials_ig > 0)
            cond_2 = 1 <= (ig_counts == 0).sum(dim=-1)
            cond_3 = (ig_counts == 0).sum(dim=-1) <= (ig_counts.shape[-1] - 2)

            mask_d_2[:, intron_group] = cond_1 & cond_2 & cond_3

        case_d_2 = torch.zeros_like(mix_weight_d) # shape (batch size, num intron groups)

        for x_idx in range(0, x.shape[0]):
            d2_intron_groups = torch.where(mask_d_2[x_idx, :] == True)[0].tolist()
            for ig in d2_intron_groups:
                ig_mask = (self.intron_group_indices[0] == ig)
                intron_indices_mask = self.intron_group_indices[1, ig_mask]
                ig_counts = x[x_idx, intron_indices_mask]

                num_trials_ig = num_trials[x_idx, ig]
                alpha_ig = alpha[x_idx, intron_indices_mask]
                alpha_0_ig = alpha_0[x_idx, ig]

                log_beta_t1 = torch.lgamma(alpha_0_ig) + torch.lgamma(num_trials_ig + 1) - torch.lgamma(num_trials_ig + alpha_0_ig)
                log_beta_t2 = torch.lgamma(ig_counts + alpha_ig) - torch.lgamma(alpha_ig) - torch.lgamma(ig_counts + 1)
                log_dm_ig = log_beta_t1 + log_beta_t2.sum()
                #dm_ig = torch.exp(log_dm_ig)


                log_term_DM_d = torch.log(mix_weight_d[x_idx, ig].clamp(eps)) + log_dm_ig # Numerically stable
                case_d_2_ig_log_pmf = log_term_DM_d

                #case_d_2_pmf[x_idx, ig] = case_d_2_pmf[x_idx, ig] + mix_weight_d[x_idx, ig] * dm_ig
                #case_d_2_ig_pmf = (mix_weight_d[x_idx, ig] * dm_ig).reshape(-1, 1)

                assert str(x_idx) in subset_alpha_mixing_params_batch_dict.keys(), "Data point index not found in subset_alpha_mixing_params_batch_dict"
                assert str(ig) in subset_alpha_mixing_params_batch_dict[str(x_idx)].keys(), "Intron group index not found in subset_alpha_mixing_params_batch_dict for given data point"

                zero_inflated_subsets_ig = [ast.literal_eval(key) for key in subset_alpha_mixing_params_batch_dict[str(x_idx)][str(ig)].keys()]

                # Problems in backpropagation might arise here
                for subset in zero_inflated_subsets_ig:
                    alpha_0_red_ig = subset_alpha_mixing_params_batch_dict[str(x_idx)][str(ig)][str(subset)]["Sum of reduced concentration"]
                    mix_weight_subset_K_ig = subset_alpha_mixing_params_batch_dict[str(x_idx)][str(ig)][str(subset)]["Mixing weight subset K"]

                    log_beta_t1_red = torch.lgamma(alpha_0_red_ig) + torch.lgamma(num_trials_ig + 1) - torch.lgamma(num_trials_ig + alpha_0_red_ig)
                    log_beta_t2_red_inc_j = torch.lgamma(ig_counts + alpha_ig) - torch.lgamma(alpha_ig) - torch.lgamma(ig_counts + 1)

                    log_beta_t2_red = log_beta_t2_red_inc_j[ [i for i in range(len(ig_counts)) if i not in subset] ]
                    log_dm_red_ig = log_beta_t1_red + log_beta_t2_red.sum()

                    log_term_subset_K_ig = torch.log(mix_weight_subset_K_ig.clamp(eps)) + log_dm_red_ig # numerically stable
                    case_d_2_ig_log_pmf = torch.logaddexp(case_d_2_ig_log_pmf, log_term_subset_K_ig) #numerically stable

                    #case_d_2_ig_pmf = torch.cat((case_d_2_ig_pmf, (mix_weight_subset_K_ig * torch.exp(log_dm_red_ig)).reshape(-1,1)))

                # Calculate log-likelihood of case_d_2_pmf
                #case_d_2[x_idx, ig] = torch.log(case_d_2_ig_pmf.sum())
                case_d_2[x_idx, ig] = case_d_2_ig_log_pmf.sum() # numerically stable

        num_case_d_2 = (case_d_2 != 0).sum().item()
        num_cases_all = num_case_non_zeros + num_case_zeros + num_case_d_1 + num_case_d_2

        assert num_cases_all == len(num_trials.flatten()), "Number of cases do not sum up to total number of data points times intron groups"

        # Full log-likelihood
        log_likelihood = (case_non_zeros + case_zeros + case_d_1 + case_d_2).sum(dim=-1)

        return log_likelihood


    def generative_model(self, latent_variable: torch.Tensor, x: torch.Tensor | None) -> Dict[str, torch.Tensor]:
        r"""
        Given the latent variable (torch.Tensor) and the count data point (torch.Tensor), compute the parameters of the
        generative model using the ZANIDM observation model. The parameters of the ZANIDM include the concentration
        computed from the exon-junction (intron) proportion, and the mixture weights computed from the excess-of-zeros
        probabilities. The mixture weights
        $\boldsymbol{\eta} = [\eta^{(D)}, \eta^{(0)}, \eta^{(N)}_{1}, \dots, \eta^{(N)}_{D}, \boldsymbol{\eta}^{(\tilde{\mathcal{K}})} ]^T$
        of ZANIDM are functions of the excess-of-zero parameters. A mixture weight quantifies the relative importance of
        a specific event scenario. Four different event scenarios can be defined for the ZANIDM. In case 1, all $D$ exon
        junctions have counts greater than zero ($N > 0 \, \land \forall d \, x^{(TU)}_{d} > 0$) which is covered by
        the mixture weight

        $$
            \eta^{(D)} = \prod_{d=1}^{D} (1 - \zeta_{d}).
        $$

        The opposite scenario is the event the event in which no exon-junction counts are observed
        ($N = 0$ and $\mathbf{x}^{(TU)} = \mathbf{0}$) constituting case 2 which has the mixture weight

        $$
            \eta^{(0)}= \prod_{d=1}^{D} \zeta_{d}.
        $$

        If $D - 1$ exon junctions are zero-inflated and $N > 0$, one exon junction will be $N$-inflated (case 3). This
        event scenario is weighted by

        $$
            \eta^{(N)}_{d} = (1 - \zeta_{d}) \prod_{k:k\neq d} \zeta_{k}.
        $$

        All other events are instances of the case where $N > 0$ and at most $D - 2$ exon junctions are zero-inflated
        (case 4). Therefore,
        $\tilde{\mathcal{K}} = \{\mathcal{K} \subseteq \{1, \dots, D \}; 1 \leq |\mathcal{K}| \leq D - 2 \}$ is the set
        of all subsets $\mathcal{K}$ of $\{1, \dots, D \}$ with cardinality $|\mathcal{K}| \in \{1, \dots, D - 2 \}$
        where a subset $\mathcal{K}$ gathers all exon junctions with zero counts excluding the cases where exactly $D$
        and $D - 1$ exon junctions are zero-inflated. The mixture weight for a subset
        $\mathcal{K} \in \tilde{\mathcal{K}}$ is then obtained by

        $$
            \eta^{(\tilde{\mathcal{K}})}_{\mathcal{K}} = \prod_{k \in \mathcal{K}} \zeta_{k} \prod_{d \notin \mathcal{K}} (1 - \zeta_{d})
        $$

        such that all mixture weights of these subsets are given by the vector
        $\boldsymbol{\eta}^{(\tilde{\mathcal{K}})}  = \{\eta^{(\tilde{\mathcal{K}})}_{\mathcal{K}}; \mathcal{K} \in \tilde{\mathcal{K}} \}$.
        As additional constraint for each subset $\mathcal{K}$, a truncated sum of concentration parameters is defined
        by $\alpha_{\mathcal{K}} = \sum_{d \notin \mathcal{K}} \alpha_{d}$.

        A dictionary containing all entities necessary to compute the generative model is returned.

        :param latent_variable: (torch.Tensor) The latent variable. Shape is (batch size, latent dim).
        :param x: (torch.Tensor) The count data point(s). Shape is (batch size, D) with D being the feature dimension.
        :return: generative_model_dict: (Dict[str, torch.Tensor]) A dictionary containing the parameters of the generative model

        x are the counts
        """
        # Create dictionary which stores the computed parameters of the generative model
        generative_model_dict = {}

        # Compute the reconstructed intron proportion and the excess of zeros probability using the decoder
        generative_model_dict["Intron proportion"], generative_model_dict["Excess-of-zeros"] = self.decoder(latent_variable, self.intron_group_idx_start)

        # Extract the feature precision
        generative_model_dict["Feature precision"] = torch.exp(self.feature_precision)[self.intron_groups] + 1e-6  # add small constant for numerical stability

        # Compute the concentration parameter alpha
        generative_model_dict["Concentration"] = generative_model_dict["Intron proportion"] * generative_model_dict[
            "Feature precision"]
        assert torch.all(generative_model_dict["Concentration"] > 0).item(), "The concentration must be greater than 0"

        # Compute the number of trials
        generative_model_dict["Number of trials"] = torch.sparse.mm(self.intron_group_summation, x.T).T
        assert torch.all(generative_model_dict["Number of trials"] >= 0).item(), "The number of trials must be non-negative integers"
        assert torch.all(generative_model_dict["Number of trials"] == generative_model_dict["Number of trials"].floor()).item(), "The number of trials must be integers"

        unique_intron_groups, intron_group_sizes = torch.unique(self.intron_groups, return_counts=True)
        num_intron_groups = len(unique_intron_groups)

        # Compute the mixture weights from the excess-of-zeros parameter
        excess_of_zeros_complement = (1 - generative_model_dict["Excess-of-zeros"])

        mix_weight_d_list = []
        mix_weight_0_list = []

        for ig in range(0, num_intron_groups):
            ig_mask = (self.intron_group_indices[0] == ig)

            excess_of_zeros_complement_ig = excess_of_zeros_complement[:, ig_mask]
            excess_of_zeros_ig = generative_model_dict["Excess-of-zeros"][:, ig_mask]

            prod_excess_of_zeros_complement_ig = torch.prod(excess_of_zeros_complement_ig, dim=1)
            prod_excess_of_zeros_ig = torch.prod(excess_of_zeros_ig, dim=1)

            mix_weight_d_list.append(prod_excess_of_zeros_complement_ig.unsqueeze(1))
            mix_weight_0_list.append(prod_excess_of_zeros_ig.unsqueeze(1))

        # Final out-of-place assembly of mix_weight_d and mix_weight_0
        mix_weight_d = torch.cat(mix_weight_d_list, dim=1)
        mix_weight_0 = torch.cat(mix_weight_0_list, dim=1)

        # New
        mix_weight_N_group_list = []

        for ig in range(0, num_intron_groups):
            ig_mask = (self.intron_group_indices[0] == ig)

            excess_of_zeros_complement_ig = excess_of_zeros_complement[:, ig_mask]  # (1 - zeta_j)
            excess_of_zeros_ig = generative_model_dict["Excess-of-zeros"][:, ig_mask]  # zeta_j

            log_excess_of_zeros_ig = torch.log(excess_of_zeros_ig + 1e-9)

            sum_log_excess_of_zeros = log_excess_of_zeros_ig.sum(dim=1, keepdim=True)

            log_prod_others = sum_log_excess_of_zeros - log_excess_of_zeros_ig

            mix_weight_N_ig = excess_of_zeros_complement_ig * torch.exp(log_prod_others)

            mix_weight_N_group_list.append(mix_weight_N_ig)

        mix_weight_N = torch.cat(mix_weight_N_group_list, dim=1)

        # Add mixture weights to generative model dict
        generative_model_dict["Mixing weight d"] = mix_weight_d
        generative_model_dict["Mixing weight 0"] = mix_weight_0
        generative_model_dict["Mixing weight N"] = mix_weight_N

        # Compute the sum of concentration parameter
        generative_model_dict["Sum of concentration"] = torch.sparse.mm(
            self.intron_group_summation,
            generative_model_dict["Concentration"].T
        ).T  # CHECK: is the same across data points, why?

        # Compute the sum of reduced concentration parameter for each intron group and each data point
        # TO DO: DO this as in log-likelihood via masking etc
        mask_d_2 = torch.full((x.shape[0], num_intron_groups),fill_value=False,dtype=torch.bool).to(self.device)

        for intron_group in range(0, num_intron_groups):
            ig_mask = (self.intron_group_indices[0] == intron_group)
            intron_indices_mask = self.intron_group_indices[1, ig_mask]
            ig_counts = x[:, intron_indices_mask]
            num_trials_ig = generative_model_dict["Number of trials"][:, intron_group]
            cond_1 = (num_trials_ig > 0)
            cond_2 = 1 <= (ig_counts == 0).sum(dim=-1)
            cond_3 = (ig_counts == 0).sum(dim=-1) <= (ig_counts.shape[-1] - 2)

            mask_d_2[:, intron_group] = cond_1 & cond_2 & cond_3

        subset_concentration_mixing_params_batch_dict = {}

        for x_idx in range(0, x.shape[0]):
            d2_intron_groups = torch.where(mask_d_2[x_idx, :] == True)[0]
            for ig in d2_intron_groups:
                ig_mask = (self.intron_group_indices[0] == ig)
                intron_indices_mask = self.intron_group_indices[1, ig_mask]
                ig_counts = x[x_idx, intron_indices_mask]

                # Compute the sets of zero subsets
                zero_subset_index_set = torch.argwhere(ig_counts == 0).flatten().tolist()
                min_set_size = 1
                max_set_size = int(intron_group_sizes[ig] - 2)
                valid_set_sizes = range(min_set_size, max_set_size + 1) # +1 since range is exclusive at the end

                set_zero_subsets = []
                for set_size in valid_set_sizes:
                    for subset in itertools.combinations(zero_subset_index_set, set_size):
                        set_zero_subsets.append(subset)

                subset_concentration_mixing_params_dict = {}
                # For each zero subset compute the sum of reduced concentration parameter

                for zero_subset in set_zero_subsets:
                    # Sum the concentration parameters of the introns of intron_indices_mask that are not in the zero_subset
                    non_zero_indices = np.arange(0, len(intron_indices_mask)).tolist()
                    non_zero_subset = [idx for idx in non_zero_indices if idx not in zero_subset]
                    zero_subset_list = [idx for idx in non_zero_indices if idx in zero_subset]
                    non_zero_intron_indices = [intron_indices_mask[idx].item() for idx in non_zero_subset]
                    zero_intron_indices = [intron_indices_mask[idx].item() for idx in zero_subset_list]

                    excess_of_zeros_k = generative_model_dict["Excess-of-zeros"][x_idx, zero_intron_indices]
                    excess_of_zeros_complement_j = excess_of_zeros_complement[
                        x_idx, non_zero_intron_indices]

                    prod_excess_of_zeros_k = torch.prod(excess_of_zeros_k)
                    prod_excess_of_zeros_complement_j = torch.prod(excess_of_zeros_complement_j)

                    mix_weight_subset_K = prod_excess_of_zeros_k * prod_excess_of_zeros_complement_j

                    sum_reduced_concentration = generative_model_dict["Concentration"][
                        x_idx, non_zero_intron_indices].sum()

                    subset_concentration_mixing_params_dict[str(zero_subset_list)] = {
                        "Sum of reduced concentration": sum_reduced_concentration,
                        "Mixing weight subset K": mix_weight_subset_K
                    }

                subset_concentration_mixing_params_batch_dict.setdefault(str(x_idx), {})[
                    str(ig.item())] = subset_concentration_mixing_params_dict.copy()

        # Add dictionary of subset concentration and mixing parameters to generative model dict
        generative_model_dict["Subset concentration and mixing parameters"] = subset_concentration_mixing_params_batch_dict

        # Calculate the mean of the Dirichlet-Multinomial distribution component (NOTE: THIS IS NOT THE CORRECT MEAN FOR THE ZANIDM)
        num_trials_per_intron = generative_model_dict["Number of trials"][:, self.intron_groups]
        sum_concentration_per_intron = generative_model_dict["Sum of concentration"][:, self.intron_groups]
        generative_model_dict["Mean"] = num_trials_per_intron / sum_concentration_per_intron * generative_model_dict[
            "Concentration"]

        return generative_model_dict

    def forward(
            self,
            x: torch.Tensor,
            eps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward pass of the VAE. Given the vector x (log transformed intron counts), and eps (Gaussian noise for
        reparameterization), the ELBO, KL divergence and negative log-likelihood are computed.

        :param x: (torch.Tensor) The input data point(s). Shape is (batch size, D) with D being the feature dimension.
        :param eps: (torch.Tensor) Gaussian noise for reparameterisation. Shape is (batch size, latent dim).
        :return: (torch.Tensor) The ELBO, KL divergence and negative log-likelihood.
        """


        # Extract intron counts and intron levels (i.e. log(1+x)-normalised counts or PSI-scores)
        x_counts, x_levels = x

        # Compute variational posterior using the encoder
        variational_posterior_dict = self.variational_posterior(x_levels, eps)

        # Compute generative model using the decoder
        generative_model_dict = self.generative_model(variational_posterior_dict["Latent variable"], x_counts)

        # Kullback-Leibler divergence
        kld = self.kullback_leibler_divergence(
            variational_posterior_dict["Latent mean"],
            variational_posterior_dict["Latent variance"]
        )

        # Negative log-likelihood is evaluated on actual intron counts
        nll = self.negative_log_likelihood(x_counts, generative_model_dict)

        # Negative ELBO with beta implementation i.e. -mathcal{L} = \beta D_{KL} + NLL since we minimise
        elbo = nll + self.beta * kld

        return elbo, kld, nll

class TRVI(nn.Module):
    r"""
    Transcriptomic Regulation Variational Inference (TRVI) consolidates gene expression and alternative splicing-induced
    transcript usage data by learning modality-specific (P-GE, P-TU), and joint cell embeddings (GE-TU with S-GE and
    S-TU). The joint cell embeddings are modelled via a modality-relevance weighted mixture-of-experts variational
    posterior where the modality-relevance weight (for gene expression and transcript usage) are learned from the
    single-cell profiles and indicate the modality's contribution to the joint representation.
    """
    def __init__(
            self,
            input_dim: List[int], # input_dim[0] for gene expression, input_dim[1] for transcript usage
            num_intron_groups: int, # number of intron groups for transcript usage # TO DO ELIMINATE THIS PARAMETER LATER
            intron_groups: np.ndarray, # intron groups for transcript usage # TO DO ELIMINATE THIS PARAMETER LATER
            latent_dim: List[int], # latent_dim[0] for gene expression, latent_dim[1] for transcript usage, latent_dim[2] for shared latent
            beta: List[float], # beta[0] for gene expression, beta[1] for transcript usage
            num_hidden_layers: List[int], # num_hidden_layers[0] for gene expression, num_hidden_layers[1] for transcript usage
            num_hidden_units: List[int], # num_hidden_units[0] for gene expression, num_hidden_units[1] for transcript usage
            likelihoods: List[str], # likelihoods[0] for gene expression, likelihoods[1] for transcript usage
            scaling_factor: List[bool], # scaling_factor[0] for gene expression, scaling_factor[1] for transcript usage TO DO LATER ADJUST SCALING FACTOR
            temp: float = 0.1,
            learn_modality_weighting: bool = True, # Learn the mixing coefficients of the two shared latent distributions of the mixture model
            dropout_rate: float = 0.1,
            device: str = "cuda",
    ) -> None:
        r"""
        This is the class of Transcriptomic Regulation Variational Inference (TRVI). TRVI is a multimodal variational
        autoencoder that jointly learns cell embeddings from single-cell gene expression (GE) and alternative
        splicing-induced transcript usage (TU) profiles. TRVI combines shared and modality-specific latent
        representations with a cell-specific, modality-relevance-weighted mixture-of-experts
        variational posterior.

        :param input_dim: (List[int]) Input dimensions in the order [number of genes, number of introns].
        :param num_intron_groups: (int) Number of transcript usage intron groups.
        :param intron_groups: (np.ndarray) Zero-based intron-group index for each transcript usage feature.
        :param latent_dim: (List[int]) Latent dimensions in the order [gene expression private, transcript usage private,
            shared]. Each unimodal encoder outputs the shared coordinates followed by its private coordinates.
        :param beta: (List[float]) KL weights in the order [gene expression, transcript usage].
        :param num_hidden_layers: (List[int]) Numbers of hidden layers for gene expression and transcript usage.
        :param num_hidden_units: (List[int]) Hidden layer widths for gene expression and transcript usage.
        :param likelihoods: (List[str]) Observation models in modality order. Gene expression supports "Gaussian", "NB",
            or "ZINB"; transcript usage supports "DM", "ZIDM", or "ZANIDM".
        :param scaling_factor: (List[bool]) Scaling options passed to the two unimodal VAEs. The joint forward path
            currently passes no scale factors to the decoders; use [False, False] for that path.
        :param temp: (float) Stored temperature setting. Default is 0.1. The current mixture weights use ordinary softmax.
        :param learn_modality_weighting: (bool) Whether to construct the modality-weighting encoder. Default is True.
        :param dropout_rate: (float) Dropout probability for the networks. Default is 0.1.
        :param device: (str) Device setting passed to the unimodal models. Default is "cuda".
            Move the module and input tensors to the same device before use.
        """
        super(TRVI, self).__init__()

        self.latent_dim = latent_dim
        self.learn_modality_weighting = learn_modality_weighting
        self.pre_auxiliary_latent_var_1 = nn.Parameter(torch.zeros(latent_dim[0]), requires_grad=True) # DOUBLE CHECK THIS it should be rather batch_size x latent_dim[0]
        self.pre_auxiliary_latent_var_2 = nn.Parameter(torch.zeros(latent_dim[1]),requires_grad=True)  # DOUBLE CHECK THIS it should be rather batch_size x latent_dim[0]
        self.temp = temp
        self.scale_log_likelihood = True
        self.running_scale_unimodal = 82.23
        self.running_scale_crossmodal = 84.70
        self.modality_weight_balancing_coeff = 500 # previously 200
        self.modality_weight_entropy_coeff = 500 # previously 200
        self.device = device

        # VAE for gene expression (modality 1)
        if likelihoods[0] == "Gaussian":
            self.vae_1 = BetaVAE(
                input_dim=input_dim[0],
                latent_dim=latent_dim[2] + latent_dim[0], # Output is considered as vector [z_1, w_1]
                beta=beta[0],
                num_hidden_layers=num_hidden_layers[0],
                num_hidden_units=num_hidden_units[0],
                dropout_rate=dropout_rate,
                device=device,
                scaling_factor=scaling_factor[0]
            )
        elif likelihoods[0] == "NB":
            self.vae_1 = NBGeneExpressionVAE(
                input_dim=input_dim[0],
                latent_dim=latent_dim[2] + latent_dim[0],  # Output is considered as vector [z_1, w_1]
                beta=beta[0],
                num_hidden_layers=num_hidden_layers[0],
                num_hidden_units=num_hidden_units[0],
                dropout_rate=dropout_rate,
                device=device,
                scaling_factor=scaling_factor[0]
            )
        elif likelihoods[0] == "ZINB":
            self.vae_1 = ZINBGeneExpressionVAE(
                input_dim=input_dim[0],
                latent_dim=latent_dim[2] + latent_dim[0],  # Output is considered as vector [z_1, w_1]
                beta=beta[0],
                num_hidden_layers=num_hidden_layers[0],
                num_hidden_units=num_hidden_units[0],
                dropout_rate=dropout_rate,
                device=device,
                scaling_factor=scaling_factor[0]
            )
        else:
            raise ValueError(f"Likelihood {likelihoods[0]} not supported for gene expression modality")

        # VAE for transcript usage (modality 2)
        if likelihoods[1] == "DM":
            self.vae_2 = DMTranscriptUsageVAE(
                input_dim=input_dim[1],
                num_intron_groups=num_intron_groups,
                intron_groups=intron_groups,
                latent_dim=latent_dim[2] + latent_dim[1],  # Output is considered as vector [z_2, w_2]
                beta=beta[1],
                num_hidden_layers=num_hidden_layers[1],
                num_hidden_units=num_hidden_units[1],
                dropout_rate=dropout_rate,
                device=device,
                scaling_factor=scaling_factor[1]
            )
        elif likelihoods[1] == "ZIDM":
            self.vae_2 = ZIDMTranscriptUsageVAE(
                input_dim=input_dim[1],
                num_intron_groups=num_intron_groups,
                intron_groups=intron_groups,
                latent_dim=latent_dim[2] + latent_dim[1],  # Output is considered as vector [z_2, w_2]
                beta=beta[1],
                num_hidden_layers=num_hidden_layers[1],
                num_hidden_units=num_hidden_units[1],
                dropout_rate=dropout_rate,
                device=device,
                scaling_factor=scaling_factor[1]
            )
        elif likelihoods[1] == "ZANIDM":
            self.vae_2 = ZANIDMTranscriptUsageVAE(
                input_dim=input_dim[1],
                num_intron_groups=num_intron_groups,
                intron_groups=intron_groups,
                latent_dim=latent_dim[2] + latent_dim[1],  # Output is considered as vector [z_2, w_2]
                beta=beta[1],
                num_hidden_layers=num_hidden_layers[1],
                num_hidden_units=num_hidden_units[1],
                dropout_rate=dropout_rate,
                device=device,
                scaling_factor=scaling_factor[1]
            )
        else:
            raise ValueError(f"Likelihood {likelihoods[1]} not supported for transcript usage modality")

        if self.learn_modality_weighting:
            self.weighting_encoder = ScaleEncoder(
                input_dim= 2 * latent_dim[2],
                output_dim=2,  # Default output_dim for two modalities
                num_cat_list=None,
                num_hidden_layers=num_hidden_layers[0],
                num_hidden_units=latent_dim[2],
                dropout_rate=dropout_rate,
                architecture="Fully Connected",
                variational_posterior="Gaussian",
                learn_modality_weighting=self.learn_modality_weighting
            )

    def gaussian_log_likelihood(
            self, x: torch.Tensor,
            parameter_dict: Dict[str, torch.Tensor],
            min_variance: float=1e-6
    ) -> torch.Tensor:
        r"""
        The implemented Gaussian log-likelihood function is derived from the multivariate Gaussian distribution
        $\mathcal{N}(\mathbf{x} \rvert \boldsymbol{\mu}, \boldsymbol{\sigma}^2 \odot \mathbf{I})$ with a mean
        vector $\boldsymbol{\mu} \in \mathbb{R}^D$ and diagonal covariance matrix $\boldsymbol{\sigma}^2
        \odot \mathbf{I} \in \mathbb{R}_+^{N \times N}$. The log-likelihood function is evaluated using the given data
        point x (torch.Tensor) as well as the torch.Tensors corresponding to the keys "Mean" and the "Variance" of the
        dictionary parameter_model. The Gaussian log-likelihood is then computed according to

        $$
            \mathrm{log} \, \mathcal{N}(\mathbf{x} \rvert \boldsymbol{\mu}, \boldsymbol{\sigma}^2 \odot \mathbf{I}) =
            -\frac{D}{2} \mathrm{log}(2 \pi) -\frac{1}{2}\sum_{d=1}^D \mathrm{log}(\sigma_d^2)
            - \frac{1}{2} \sum_{d=1}^D \frac{(x_d - \mu_d)^2}{\sigma_d^2}
        $$

        :param x: (torch.Tensor) The data point(s). Shape is be (batch size, D) with D being the feature dimension.
        :param parameter_dict: (Dict[str, torch.Tensor) Dictionary of the predicted parameters of the variational posterior.
        :param min_variance: (float) minimum variance for numerical stability
        :returns: log_likelihood (torch.Tensor) The sum of log-likelihoods over the last dimension (D) for each sample.
            The output shape will be of (batch_size, ).
        """

        # Minimum variance to avoid numerical stability
        variance = torch.clamp(parameter_dict["Variance"], min_variance)

        # Check the support of the distribution
        assert torch.all(variance > 0).item(), "The variance of the likelihood must be greater than 0"

        # Constant term
        constant = -0.5 * x.shape[-1] * math.log(2 * math.pi)

        # Log-determinant
        log_determinant = -0.5 * torch.log(variance).sum(dim=-1)

        # Quadratic term
        quadratic_term = -0.5 * (1 / variance * (x - parameter_dict["Mean"]) ** 2).sum(dim=-1)

        # Final log-likelihood
        log_likelihood = constant + log_determinant + quadratic_term

        return log_likelihood

    def variational_posterior(
            self,
            x: Tuple[torch.Tensor | None, torch.Tensor | None], # x[0] for gene expression levels, x[1] for transcript usage levels
            eps: Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None, # eps[0] gene levels, eps[1] transcript usage levels, eps[2] for auxiliary prior of gene levels, eps[3] for auxiliary prior of transcript usage levels
            temp: float = 0.5
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        r"""
        The full variational posterior of TRVI is computed from the gene expression and transcript usage levels. If a
        noise variable (eps) is supplied, the reparameterization trick during training is applied. For a data point
        $\mathbf{x}_n = \{\mathbf{x}_n^{(GE)},\mathbf{x}_n^{(TU)} \}$ the full variational posterior is obtained by

        $$
            q_{\boldsymbol{\Phi}} \left( \mathbf{z}_n, \{\mathbf{w}_n^{(m)}\}_{m\in\mathcal M} \mid\mathbf{x}_n \right) &= \left( \sum_{m\in\mathcal M} \pi_{\boldsymbol{\phi}^{\pi}}^{(m)}(\mathbf{x}_n)\, q_{\boldsymbol{\phi}^{\mathbf z}_m} \left( \mathbf{z}_n\mid\mathbf{x}_n^{(m)} \right) \right) \\
            &\quad \prod_{m\in\mathcal M} q_{\boldsymbol{\phi}^{\mathbf w}_m} \left( \mathbf{w}_n^{(m)} \mid\mathbf{x}_n^{(m)} \right)
        $$

        where
        $\boldsymbol{\Phi} = \{\boldsymbol{\phi}^{\pi}, {\boldsymbol{\phi}^{\mathbf z}_m, \boldsymbol{\phi}^{\mathbf w}_m} \mid {m\in\mathcal{M}}\}$
        denotes the set of all variational parameters.

        All entities necessary to compute the variational posterior are returned as dictionary
        variational_posterior_dict.

        :param x: (Tuple[torch.Tensor | None, torch.Tensor | None]) Gene expression levels and transcript usage levels,
            each with shape (batch size, modality input dim). A missing modality can be supplied as None.
        :param eps: (Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]) Four noise entries for the
            gene expression posterior, transcript usage posterior, gene expression auxiliary distribution, and transcript
            usage auxiliary distribution. Posterior noise has final dimension shared dim plus private dim; auxiliary
            noise has the corresponding private dimension. An optional leading sample dimension is supported here.
            Individual entries may be None to omit sampling, but the tuple itself is required by the implementation.
        :param temp: (float) Retained temperature argument, currently unused because weights use ordinary softmax.
        :return: variational_posterior_dict (Dict[str, Dict[str, torch.Tensor]]) Nested dictionaries under "Modality 1",
            "Modality 2", and "Mixture of experts". Modality entries contain available shared/private moments, samples,
            auxiliary samples, and optional scale factors. Mixture moments and weights are included only when both
            modalities are present; equal weights are stored as scalar floats.
        """

        # Extract gene expression levels and transcript usage levels
        x_1_levels, x_2_levels = x

        # Create dictionary to store the variational posterior
        variational_posterior_dict = {"Modality 1": {}, "Modality 2": {}, "Mixture of experts": {}}

        # Compute the variational posterior for gene expression data using the first VAE
        if x_1_levels is not None:
            variational_posterior_1 = self.vae_1.variational_posterior(x_1_levels, eps[0])

            variational_posterior_dict["Modality 1"]["Shared latent mean"] = variational_posterior_1["Latent mean"][:, :self.latent_dim[2]]
            variational_posterior_dict["Modality 1"]["Shared latent variance"] = variational_posterior_1["Latent variance"][:, :self.latent_dim[2]]
            variational_posterior_dict["Modality 1"]["Private latent mean"] = variational_posterior_1["Latent mean"][:, self.latent_dim[2]:]
            variational_posterior_dict["Modality 1"]["Private latent variance"] = variational_posterior_1["Latent variance"][:, self.latent_dim[2]:]

            if self.vae_1.scaling_factor:
                variational_posterior_dict["Modality 1"]["Scale factor"] = variational_posterior_1["Scale factor"]

            if eps[0] is not None:
                # Use of ellipsis [..., ]to ensure correct slicing in case of num_samples > 1
                variational_posterior_dict["Modality 1"]["Shared latent variable"] = variational_posterior_1["Latent variable"][..., :self.latent_dim[2]]
                variational_posterior_dict["Modality 1"]["Private latent variable"] = variational_posterior_1["Latent variable"][..., self.latent_dim[2]:]

        if eps[2] is not None:
            # Necessary for cross-modal generation
            auxiliary_latent_var_1 = F.softplus(self.pre_auxiliary_latent_var_1) + 1e-20
            variational_posterior_dict["Modality 1"]["Auxiliary latent variable"] = eps[2] * auxiliary_latent_var_1.sqrt() # zero mean auxiliary prior assumed
            # Check if variational_posterior_dict["Modality 1"]["Auxiliary latent variable"] contains inf or nan
            if torch.isinf(variational_posterior_dict["Modality 1"]["Auxiliary latent variable"]).any().item():
                pdb.set_trace()


        # Compute the variational posterior for transcript usage data using the second VAE
        if x_2_levels is not None:
            variational_posterior_2 = self.vae_2.variational_posterior(x_2_levels, eps[1])
            variational_posterior_dict["Modality 2"]["Shared latent mean"] = variational_posterior_2["Latent mean"][:, :self.latent_dim[2]]
            variational_posterior_dict["Modality 2"]["Shared latent variance"] = variational_posterior_2["Latent variance"][:, :self.latent_dim[2]]
            variational_posterior_dict["Modality 2"]["Private latent mean"] = variational_posterior_2["Latent mean"][:, self.latent_dim[2]:]
            variational_posterior_dict["Modality 2"]["Private latent variance"] = variational_posterior_2["Latent variance"][:, self.latent_dim[2]:]

            if self.vae_2.scaling_factor:
                variational_posterior_dict["Modality 2"]["Scale factor"] = variational_posterior_2["Scale factor"]

            if eps[1] is not None:
                # Use of ellipsis [..., ]to ensure correct slicing in case of num_samples > 1
                variational_posterior_dict["Modality 2"]["Shared latent variable"] = variational_posterior_2["Latent variable"][..., :self.latent_dim[2]]
                variational_posterior_dict["Modality 2"]["Private latent variable"] = variational_posterior_2["Latent variable"][..., self.latent_dim[2]:]

        if eps[3] is not None:
            # Necessary for cross-modal generation
            auxiliary_latent_var_2 = F.softplus(self.pre_auxiliary_latent_var_2) + 1e-20
            variational_posterior_dict["Modality 2"]["Auxiliary latent variable"] = eps[3] * auxiliary_latent_var_2.sqrt()
            # Check if variational_posterior_dict["Modality 2"]["Auxiliary latent variable"] contains inf or nan
            if torch.isinf(variational_posterior_dict["Modality 2"]["Auxiliary latent variable"]).any().item():
                pdb.set_trace()


        # Compute the mean and the variance of the mixture of experts variational posterior distribution
        if (x_1_levels is not None) and (x_2_levels is not None):
            if self.learn_modality_weighting:
                # Scenario of learnt weighting of two modalities

                cat_shared_latent_means = torch.cat((
                    variational_posterior_dict["Modality 1"]["Shared latent mean"],
                    variational_posterior_dict["Modality 2"]["Shared latent mean"]
                ), dim=-1)

                logits = self.weighting_encoder(cat_shared_latent_means)

                # Gumbel-Softmax sampling for differentiable weights during training and softmax during evaluation
                #if self.training:
                #    weightings = F.gumbel_softmax(logits, tau=temp, hard=False, dim=-1)
                #else:
                #    weightings = F.softmax(logits, dim=-1)

                weightings = F.softmax(logits, dim=-1)

                weight_1 = weightings[:, 0].reshape(-1,1)
                weight_2 = weightings[:, 1].reshape(-1,1)

            else:
                # Scenario of equal weighting of two modalities
                weight_1 = 0.5
                weight_2 = 0.5

            variational_posterior_dict["Mixture of experts"]["Latent mean"] =  (
                    weight_1 * variational_posterior_dict["Modality 1"]["Shared latent mean"]
                    + weight_2 * variational_posterior_dict["Modality 2"]["Shared latent mean"]
            )

            variational_posterior_dict["Mixture of experts"]["Latent variance"] = (
                    weight_1 * variational_posterior_dict["Modality 1"]["Shared latent variance"]
                    + weight_2 * variational_posterior_dict["Modality 2"]["Shared latent variance"]
                    + (
                            weight_1 * variational_posterior_dict["Modality 1"]["Shared latent mean"]**2
                            + weight_2 * variational_posterior_dict["Modality 2"]["Shared latent mean"]**2
                            - (
                                    weight_1 * variational_posterior_dict["Modality 1"]["Shared latent mean"]
                                    + weight_2 * variational_posterior_dict["Modality 2"]["Shared latent mean"]
                            )**2
                    )
            )
            if self.learn_modality_weighting:
                variational_posterior_dict["Mixture of experts"]["Weighting modality 1"] = weight_1.squeeze()
                variational_posterior_dict["Mixture of experts"]["Weighting modality 2"] = weight_2.squeeze()
            else:
                variational_posterior_dict["Mixture of experts"]["Weighting modality 1"] = weight_1
                variational_posterior_dict["Mixture of experts"]["Weighting modality 2"] = weight_2

        return variational_posterior_dict

    def generative_model(
            self,
            shared_latent_variables: Tuple[torch.Tensor | None, torch.Tensor | None], # z_1, z_2
            private_latent_variables: Tuple[torch.Tensor | None, torch.Tensor | None], # w_1, w_2
            auxiliary_latent_variables: Tuple[torch.Tensor | None, torch.Tensor | None], # w_aux_1, w_aux_2
            scales: Tuple[torch.Tensor | None, torch.Tensor | None], # Scaling factors
            x_2_counts: torch.Tensor | None
    ) -> Dict[str, torch.Tensor]:
        r"""
        Given the share latent cell embeddings (GE-TU with S-GE and S-TU), modality-specific cell embeddings (P-GE,
        P-TU), auxiliary latent variables $\tilde{\mathbf{w}}_n^{(m')} \sim r_{m'} (\cdot)$, scales, and isoform group
        count vector $\mathbf{x}^{TU}$, generative model of TRVI is computed by

        $$
            p_{\boldsymbol{\Theta}} \left(\mathbf{X}, \mathbf{Z}, \{\mathbf{W}^{(m)}\}_{m\in\mathcal M} \right) = \prod_{n=1}^{N} p(\mathbf{z}_n) \prod_{m\in\mathcal M} p_{\boldsymbol{\theta}_m} \left( \mathbf{x}_n^{(m)} \mid \mathbf{z}_n,\mathbf{w}_n^{(m)} \right) p(\mathbf{w}_n^{(m)}).
        $$

        where $p(\mathbf{z}_n) = \mathcal{N}(\mathbf{z}_n \mid \mathbf{0}, \mathbf{I}_L)$, $p(\mathbf{w}_n^{(m)}) = \mathcal{N}(\mathbf{w}_n \mid \mathbf{0}, \mathbf{I}_L)$
        are the respective priors on the latent cell embeddings and
        $p_{\boldsymbol{\theta}_m} \left( \mathbf{x}_n^{(m)} \mid \mathbf{z}_n,\mathbf{w}_n^{(m)} \right)$  the data
        likelihood for modality $m$ with $\boldsymbol{\theta}_m$ being the parameters of the deep neural network of the
        respective modality decoder. The set of all parameters of the generative model is denoted by
        $\boldsymbol{\Theta} = \{\boldsymbol{\theta}_{GE}, \boldsymbol{\theta}_{TU} \}$ For gene expression $m= GE$, the
         data likelihood factorises across $G_{\mathrm{GE}}$ genes as

        $$
            p_{\boldsymbol{\theta}_{GE}} \left( \mathbf{x}_n^{(GE)} \mid \mathbf{z}_n, \mathbf{w}_n^{(GE)} \right) = \prod_{g = 1}^{G_{GE}} p_{\boldsymbol{\theta}_{GE}}(x^{(GE)}_{n,g} \mid \mathbf{z}_n, \mathbf{w}_n^{(GE)}).
        $$

        Here, $p_{\boldsymbol{\theta}_{GE}}(x^{(GE)}_{n,g} \mid \mathbf{z}_n, \mathbf{w}_n^{(GE)})$ is the observation
        model of a gene expression $x^{(GE)}_{n,g}$ in cell $n$ similar to scVI. TRVI supports Gaussian,
        negative-binomial (NB), and zero-inflated negative-binomial (ZINB) observation models. All TRVI analyses
        reported here used the ZINB observation model. For transcript usage data $m= TU$, the data likelihood across
        spliced genes and their isoform groups

        $$
            p_{\boldsymbol{\theta}_{TU}} \left( \mathbf{x}^{(TU)}_n \rvert \mathbf{z}_n, \mathbf{w}_n^{(TU)}\right) = \prod_{g=1}^{G_{TU}} \prod_{i=1}^{I_g} p_{\boldsymbol{\theta}_{TU}} \left(\mathbf{x}_{n,g,i}^{(TU)} \rvert \mathbf{z}_n, \mathbf{w}_n^{(TU)}\right)
        $$

        where
        $p_{\boldsymbol{\theta}_{TU}} \left(\mathbf{x}_{n,g,i}^{(TU)} \rvert \mathbf{z}_n, \mathbf{w}_n^{(TU)}\right)$
        is the observation model of an isoform group vector $\mathbf{x}_{n,g,i}^{(TU)}$. TRVI supports the same
        transcript usage observation models as tuVI (DM, ZANIDM) and the heuristic ZIDM. All TRVI analyses reported used
        the heuristic ZIDM objective.

        All entities necessary to compute the generative model are stored in the dictionary generative_model_dict.
        Unimodal generation concatenates a modality's shared and private samples. Cross-modal generation combines the
        other modality's shared sample with an auxiliary private sample for the target modality.

        :param shared_latent_variables: (Tuple[torch.Tensor | None, torch.Tensor | None]) Shared samples from the gene
            expression and transcript usage encoders, respectively.
        :param private_latent_variables: (Tuple[torch.Tensor | None, torch.Tensor | None]) Private samples from the two
            encoders in the same modality order. A unimodal branch requires both its shared and private samples.
        :param auxiliary_latent_variables: (Tuple[torch.Tensor | None, torch.Tensor | None]) Auxiliary private samples
            for gene expression and transcript usage. Each non-None entry requires the other modality's shared sample.
        :param scales: (Tuple[torch.Tensor | None, torch.Tensor | None]) Optional modality scales. Only the first entry
            is passed to unimodal gene expression generation; cross-modal gene expression receives None, and the second
            entry is unused. Scaling must be disabled for gene expression cross-modal generation.
        :param x_2_counts: (torch.Tensor | None) Observed intron counts with shape (batch size, number of introns),
            required by transcript usage generation to compute group totals and count-dependent parameters.
        :return: generative_model_dict (dict) Nested parameter dictionaries for the available branches, keyed by
            "Modality 1 unimodal", "Modality 2 unimodal", "Modality 1 crossmodal", and "Modality 2 crossmodal".
            Individual parameter dictionaries follow the selected unimodal VAE observation model.
        """

        # Create dictionary to store the generative model
        generative_model_dict = {}

        if (shared_latent_variables[0] is not None) and (private_latent_variables[0] is not None):
            # Compute the uni-modal generative model for gene expression data (1) using the first VAE

            latent_variable_1_unimodal = torch.cat((shared_latent_variables[0], private_latent_variables[0]), dim=-1)
            generative_model_1_unimodal = self.vae_1.generative_model(latent_variable_1_unimodal, scales[0])
            generative_model_dict["Modality 1 unimodal"] = generative_model_1_unimodal

        if (shared_latent_variables[1] is not None) and (private_latent_variables[1] is not None):
            # Compute the uni-modal generative model for transcript usage data (2) using the second VAE

            latent_variable_2_unimodal = torch.cat((shared_latent_variables[1], private_latent_variables[1]), dim=-1)
            generative_model_2_unimodal = self.vae_2.generative_model(latent_variable_2_unimodal, x_2_counts) # Currently only supports DM or ZIDM for vae_2 ISSUE ARE THE NUMBER OF TRIALS being counts from x_2_counts
            generative_model_dict["Modality 2 unimodal"] = generative_model_2_unimodal

        if auxiliary_latent_variables[0] is not None:
            # Compute the cross-modal generative model for gene expression data (1) using the first VAE

            latent_variable_2_crossmodal = torch.cat((shared_latent_variables[1], auxiliary_latent_variables[0]), dim=-1)
            generative_model_1_crossmodal = self.vae_1.generative_model(latent_variable_2_crossmodal, None) # CURRENTLY no scale supported
            generative_model_dict["Modality 1 crossmodal"] = generative_model_1_crossmodal

        if auxiliary_latent_variables[1] is not None:
            # Compute the cross-modal generative model for transcript usaeg data (2) using the second VAE

            latent_variable_1_crossmodal = torch.cat((shared_latent_variables[0], auxiliary_latent_variables[1]), dim=-1)
            generative_model_2_crossmodal = self.vae_2.generative_model(latent_variable_1_crossmodal, x_2_counts) # Currently only supports DM or ZIDM for vae_2 ISSUE ARE THE NUMBER OF TRIALS being counts from x_2_counts
            generative_model_dict["Modality 2 crossmodal"] = generative_model_2_crossmodal

        return generative_model_dict

    def pseudo_kullback_leibler_divergence(
            self,
            variational_posterior_dict: Dict[str, Dict[str, torch.Tensor]],
            modality: int
    ) -> torch.Tensor:
        r"""
        Given the mixture of experts variational posterior, and the modality of the sampled unimodal shared latent
        variable, compute the "pseudo" Kullback-Leibler divergence (KLD) as

        ..math::
            \mathbb{E}_{\begin{matrix} \mathbf{z} \sim q_{\boldsymbol{\phi}^{\mathbf{z}}_m}(\mathbf{z} \rvert
            \mathbf{x}_m) \end{matrix}} \left[\log \frac{p(\mathbf{z})}{\sum_{k=1}^2 \pi_k(\mathbf{x}_k)
            q_{\boldsymbol{\phi}_k^{\mathbf{z}}}(\mathbf{z} \rvert \mathbf{x}_k)} \right]

        """
        # Check the modality is either 1 or 2
        if modality not in [1, 2]:
            raise ValueError("Modality must be either 1 or 2")

        # Extract the shared latent variable (MC sample) according to the modality
        shared_latent_variable = variational_posterior_dict["Modality " + str(modality)]["Shared latent variable"]

        # Extract the parameters of the mixture of experts variational posterior
        shared_latent_mean_1 = variational_posterior_dict["Modality 1"]["Shared latent mean"]
        shared_latent_variance_1 = variational_posterior_dict["Modality 1"]["Shared latent variance"]
        shared_latent_mean_2 = variational_posterior_dict["Modality 2"]["Shared latent mean"]
        shared_latent_variance_2 = variational_posterior_dict["Modality 2"]["Shared latent variance"]

        if self.learn_modality_weighting:
            # Scenario of learnt weighting of two modalities
            weight_1 = variational_posterior_dict["Mixture of experts"]["Weighting modality 1"]
            weight_2 = variational_posterior_dict["Mixture of experts"]["Weighting modality 2"]
        else:
            # Scenario of equal weighting of two modalities
            weight_1 = 0.5
            weight_2 = 0.5

        # log-likelihood of the prior on the shared latent variable
        prior_dict = {
            "Mean": torch.zeros_like(shared_latent_variable),
            "Variance": torch.ones_like(shared_latent_variable)
        }
        shared_variational_posterior_1_dict= {
            "Mean": shared_latent_mean_1,
            "Variance": shared_latent_variance_1
        }
        shared_variational_posterior_2_dict = {
            "Mean": shared_latent_mean_2,
            "Variance": shared_latent_variance_2
        }

        # Compute the log-likelihood of the prior on the shared latent variable (does not matter if vae_1 or vae_2)
        prior_log_likelihood = self.gaussian_log_likelihood(shared_latent_variable, prior_dict)

        # Calculate the uni-modal likelihoods for the shared latent variable (does not matter if vae_1 or vae_2)
        shared_1_likelihood = self.gaussian_log_likelihood(shared_latent_variable, shared_variational_posterior_1_dict).exp()
        shared_2_likelihood = self.gaussian_log_likelihood(shared_latent_variable, shared_variational_posterior_2_dict).exp()

        # Calculate the likelihood of the mixture of experts variational posterior
        mixture_of_experts_log_likelihood = (weight_1 * shared_1_likelihood + weight_2 * shared_2_likelihood).clamp(min=1e-10).log()

        # Calculate the pseudo KLD (note there is a minus sign in front of the log-likelihoods to mimic the KLD but this will be )
        pseudo_kld = mixture_of_experts_log_likelihood - prior_log_likelihood

        # Return the pseudo KLD
        return pseudo_kld

    def moe_log_likelihood(self,
            variational_posterior_dict: Dict[str, Dict[str, torch.Tensor]],
            modality: int
    ) -> torch.Tensor:
        r"""
        Give the variational_posterior_dict, the modality-relevance weighted mixture-of-experts variational posterior is
        computed. TRVI uses a cell-specific modality-relevance-weighted mixture-of-experts variational posterior for the
        joint GE--TU cell embeddings. TRVI infers the relative contribution of each modality separately for each cell
        through the modality-relevance weight $\pi_{\boldsymbol{\phi}^{\pi}}^{(m)}(\mathbf{x}_n)$. The log-likelihood of
        the modality-relevance-weighted variational posterior of a GE--TU cell embedding is then obtained by

        $$
            \log q_{\boldsymbol{\Phi}} \left( \mathbf{z}_n,  \mid\mathbf{x}_n \right) = \log \sum_{m\in\mathcal M} \pi_{\boldsymbol{\phi}^{\pi}}^{(m)}(\mathbf{x}_n)\, q_{\boldsymbol{\phi}^{\mathbf z}_m} \left( \mathbf{z}_n\mid\mathbf{x}_n^{(m)} \right).
        $$

        For $\mathcal{M}={\mathrm{GE},\mathrm{TU}})$,  defines a two-component Gaussian mixture. The GE-derived and
        TU-derived components constitute the shared gene expression (S--GE) and shared transcript usage (S--TU)
        posterior components
        $q_{\boldsymbol{\phi}^{\mathbf z}_m} ( \mathbf{z}_n\mid\mathbf{x}_n^{(m)} ) = \mathcal{N}(\mathbf{z}_n \mid \boldsymbol{\mu}_{\boldsymbol{\phi}^{\mathbf z}_m}(\mathbf{x}_n^{(m)}), \mathrm{diag}(\boldsymbol{\sigma^2}_{\boldsymbol{\phi}^{\mathbf z}_m})(\mathbf{x}_n^{(m)})))$,
        respectively. The modality-relevance weight is constrained by
        $\pi_{\boldsymbol{\phi}^{\pi}}^{(m)}(\mathbf{x}_n) \geq 0$ and
        $\sum_{m \in \mathcal{M}} \pi_{\boldsymbol{\phi}^{\pi}}^{(m)}(\mathbf{x}_n) = 1$.

        The function returns the log-likelihood of the shared mixture-of-experts variational posterior at samples from
        the selected modality. Both Gaussian component densities are evaluated at the same samples and combined with
        their mixture weights using log-sum-exp for numerical stability. Equal weights are used when learned modality
        weighting is disabled.

        :param variational_posterior_dict: (Dict[str, Dict[str, torch.Tensor]]) Posterior dictionary containing both
            modalities' shared means and variances, the selected shared sample, and any learned mixture weights.
        :param modality: (int) Source of the shared sample: 1 for gene expression or 2 for transcript usage.
        :return: mixture_of_experts_log_likelihood (torch.Tensor) Mixture log density summed over shared latent
            coordinates, with shape (batch size,) or (number of samples, batch size).
        """

        # Check the modality is either 1 or 2
        if modality not in [1, 2]:
            raise ValueError("Modality must be either 1 or 2")

        # Extract the shared latent variable (MC sample) according to the modality
        shared_latent_variable = variational_posterior_dict["Modality " + str(modality)]["Shared latent variable"]

        # Extract the parameters of the mixture of experts variational posterior
        shared_latent_mean_1 = variational_posterior_dict["Modality 1"]["Shared latent mean"]
        shared_latent_variance_1 = variational_posterior_dict["Modality 1"]["Shared latent variance"]
        shared_latent_mean_2 = variational_posterior_dict["Modality 2"]["Shared latent mean"]
        shared_latent_variance_2 = variational_posterior_dict["Modality 2"]["Shared latent variance"]

        shared_variational_posterior_1_dict = {
            "Mean": shared_latent_mean_1,
            "Variance": shared_latent_variance_1
        }
        shared_variational_posterior_2_dict = {
            "Mean": shared_latent_mean_2,
            "Variance": shared_latent_variance_2
        }

        if self.learn_modality_weighting:
            # Scenario of learnt weighting of two modalities
            weight_1 = variational_posterior_dict["Mixture of experts"]["Weighting modality 1"]
            weight_2 = variational_posterior_dict["Mixture of experts"]["Weighting modality 2"]
        else:
            # Scenario of equal weighting of two modalities
            weight_1 = torch.tensor(
                0.5,
                device=shared_latent_variable.device,
                dtype=shared_latent_variable.dtype,
            )
            weight_2 = torch.tensor(
                0.5,
                device=shared_latent_variable.device,
                dtype=shared_latent_variable.dtype,
            )
        """
        OLD without Logsumexp
        # Calculate the uni-modal likelihoods for the shared latent variable (does not matter if vae_1 or vae_2)
        shared_1_likelihood = self.gaussian_log_likelihood(shared_latent_variable,
                                                               shared_variational_posterior_1_dict).exp()
        shared_2_likelihood = self.gaussian_log_likelihood(shared_latent_variable,
                                                               shared_variational_posterior_2_dict).exp()

        # Calculate the likelihood of the mixture of experts variational posterior
        mixture_of_experts_log_likelihood = (weight_1 * shared_1_likelihood + weight_2 * shared_2_likelihood).clamp(min=1e-10).log()
        """

        shared_1_ll = self.gaussian_log_likelihood(
            shared_latent_variable,
            shared_variational_posterior_1_dict,
        )
        shared_2_ll = self.gaussian_log_likelihood(
            shared_latent_variable,
            shared_variational_posterior_2_dict,
        )

        log_weight_1 = torch.log(weight_1 + 1e-10)
        log_weight_2 = torch.log(weight_2 + 1e-10)

        mixture_of_experts_log_likelihood = torch.logsumexp(
            torch.stack(
                [
                    log_weight_1 + shared_1_ll,
                    log_weight_2 + shared_2_ll
                ],
                dim=0
            ),
            dim=0
        )

        return mixture_of_experts_log_likelihood

    def forward(
            self,
            x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            eps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            beta_kl_warmup: List[float] = [1.0, 1.0],
            temp: float = 500
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Given a batch of inputs consisting of two torch.Tensors of gene expression data x_1_counts and x_1_levels as
        well as two torch.Tensors of transcript usage data x_2_counts and x_2_levels, the forward pass computes the
        evidence lower bound (ELBO) for TRVI. The full ELBO is an average of the ELBO for the first data modality (i.e.
        gene expressions) and the second data modality (i.e.transcript usage). Since optimization takes place here
        through minimization, the negative ELBO will be computed. The negative ELBO for a data modality is then built
        up by the negative log-likelihoods (NLLs) for uni-modal and cross-modal generation and the Kullback-Leibler
        divergences (KLDs) between the mixture of experts variational posterior on the shared latent variable and the
        prior on the shared latent variable as well as the private variational posterior on the private latent variable
        and the prior on the private latent variable. The negative ELBO for an input data point is then yielded by

        $$
            \mathcal{L}(\boldsymbol{\Phi}, \boldsymbol{\Theta}; \mathbf{x}) &\geq \sum_{m=1}^2 \pi_m(\mathbf{x}_m) (\mathbb{E}_{\begin{matrix}
            \mathbf{z} \sim q_{\boldsymbol{\phi^{\mathbf{z}}_m}}(\mathbf{z} \rvert \mathbf{x}_m) \\
            \mathbf{w}_m \sim q_{\boldsymbol{\phi}^{\mathbf{w}}_m}(\mathbf{w}_m \rvert \mathbf{x}_m)
            \end{matrix}} \left[ \log p_{\boldsymbol{\theta}_m}(\mathbf{x}_m \rvert \mathbf{z}, \mathbf{w}_m) \right] \\
            &\quad+ \mathbb{E}_{\begin{matrix}
            \mathbf{z} \sim q_{\boldsymbol{\phi^{\mathbf{z}}_m}}(\mathbf{z} \rvert \mathbf{x}_m) \\
            \tilde{\mathbf{w}}_n \sim r_n(\mathbf{w}_n)
            \end{matrix}} \left[\log p_{\boldsymbol{\theta}_{n\neq m}}(\mathbf{x}_{n \neq m} \rvert \mathbf{z}, \tilde{\mathbf{w}}_n) \right] \\
            &\quad-  \mathrm{D}_{\mathrm{KL}}(q_{\boldsymbol{\phi}^{\mathbf{w}}_m}(\mathbf{w}_m \rvert \mathbf{x}_m) \rvert \rvert p(\mathbf{w}_m)) \\
            &\quad+ \mathbb{E}_{\begin{matrix}
            \mathbf{z} \sim q_{\boldsymbol{\phi}^{\mathbf{z}}_m}(\mathbf{z} \rvert \mathbf{x}_m)
            \end{matrix}} \left[\log \frac{p(\mathbf{z})}{\sum_{k=1}^2 \pi_k(\mathbf{x}_k) q_{\boldsymbol{\phi}_k^{\mathbf{z}}}(\mathbf{z} \rvert \mathbf{x}_k)} \right] )
        $$

        where $\mathbf{x} = \{\mathbf{x}_1, \mathbf{x}_2\}$ is the input data consisting of the gene expression
        data $\mathbf{x}_1$ and the transcript usage data $\mathbf{x}_2$, $\mathbf{z}$ is the shared
        latent variable, $\mathbf{w}_1$ is the private latent variable for the gene expression modality, and
        $\mathbf{w}_2$ is the private latent variable for the transcript usage modality. The priors on the latent
        variables are assumed to be standard Gaussian distributions, i.e. $p(\mathbf{z}) = \mathcal{N}(0, I)$ and
        $p(\mathbf{w}_1) = \mathcal{N}(0, I)$ and $p(\mathbf{w}_2) = \mathcal{N}(0, I)$.
        The function returns the negative ELBO, the KLDs, and the NLLs for both modalities. The current implementation
        combines Monte Carlo importance weights with log-sum-exp over the leading sample dimension and adds
        modality-weight regularization when enabled.

        :param x: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) A tuple containing the gene expression
            counts (x_1_counts), gene expression levels (x_1_levels), transcript usage counts (x_2_counts), and
            transcript usage levels (x_2_levels).
        :param eps: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) Noise for the gene expression
            posterior, transcript usage posterior, gene expression auxiliary distribution, and transcript usage auxiliary
            distribution, in that order. Use shape (number of samples, batch size, shared dim + private dim) for posterior
            noise and (number of samples, batch size, private dim) for the corresponding auxiliary noise. Decoder support
            for a leading sample dimension depends on the selected observation model.
        :param beta_kl_warmup: (List[float]) KL weights assigned to the two VAEs. If either is below 1, learned weighting
            and its regularization are disabled. Otherwise they are enabled, requiring a constructed weighting encoder.
            Passing None leaves the current settings unchanged.
        :param temp: (float) Coefficient assigned to both modality-weight regularizers after KL warm-up. Default is 500.
        :return: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])
            Combined negative objective, gene expression negative objective, transcript usage negative objective,
            gene expression NLL, transcript usage NLL, gene expression pseudo-KL, and transcript usage pseudo-KL,
            in that order. Objectives and pseudo-KL terms reduce the leading sample dimension; NLL terms retain it.
        """
        if beta_kl_warmup is not None:
            self.vae_1.beta = beta_kl_warmup[0]
            self.vae_2.beta = beta_kl_warmup[1]
            if beta_kl_warmup[0] < 1.0 or beta_kl_warmup[1] < 1.0:
                self.learn_modality_weighting = False
                self.modality_weight_balancing_coeff = 0.0
                self.modality_weight_entropy_coeff = 0.0
            else:
                self.learn_modality_weighting = True
                self.modality_weight_balancing_coeff = temp
                self.modality_weight_entropy_coeff = temp

        # Extract the inputs for gene expression (1) and transcript usage (2)
        x_1_counts, x_1_levels, x_2_counts, x_2_levels = x

        # Compute the full variational posterior using the encoders
        variational_posterior_dict = self.variational_posterior((x_1_levels, x_2_levels), eps, temp)

        # Compute the full generative model using the decoders
        generative_model_dict = self.generative_model(
            (variational_posterior_dict["Modality 1"]["Shared latent variable"], variational_posterior_dict["Modality 2"]["Shared latent variable"]),
            (variational_posterior_dict["Modality 1"]["Private latent variable"], variational_posterior_dict["Modality 2"]["Private latent variable"]),
            (variational_posterior_dict["Modality 1"]["Auxiliary latent variable"], variational_posterior_dict["Modality 2"]["Auxiliary latent variable"]),
            (None, None), # Currently only None supported
            x_2_counts # Necessary for number of trials needs to be eliminated later only None supported
        )

        # Compute the log-likelihood (LL) for unimodal generation and crossmodal generation per modality
        if self.scale_log_likelihood:
            # Scale the log-likelihoods to be on similar scale (target is modality 1 log-likelihood)
            ll_1_unimodal = self.vae_1.log_likelihood(x_1_counts, generative_model_dict["Modality 1 unimodal"])
            ll_2_unimodal_unscaled = self.vae_2.log_likelihood(x_2_counts, generative_model_dict["Modality 2 unimodal"])

            ll_1_crossmodal = self.vae_1.log_likelihood(x_1_counts, generative_model_dict["Modality 1 crossmodal"])
            ll_2_crossmodal_unscaled = self.vae_2.log_likelihood(x_2_counts, generative_model_dict["Modality 2 crossmodal"])

            # Calculate mean scaling factors for modality 2 log-likelihoods
            with torch.no_grad():
                # Calculate current batch ratios
                # We use absolute values of means to ensure a positive scaling factor
                # and avoid issues if one LL is positive and another is negative
                scale_unimodal = torch.abs(ll_1_unimodal.mean()) / (torch.abs(ll_2_unimodal_unscaled.mean()) + 1e-8)
                scale_crossmodal = torch.abs(ll_1_crossmodal.mean()) / (torch.abs(ll_2_crossmodal_unscaled.mean()) + 1e-8)

                # Update buffers (assuming these are initialized in __init__)
                momentum = 0.99
                self.running_scale_unimodal = momentum * self.running_scale_unimodal + (1 - momentum) * scale_unimodal
                self.running_scale_crossmodal = momentum * self.running_scale_crossmodal + (1 - momentum) * scale_crossmodal

            ll_2_unimodal = ll_2_unimodal_unscaled * self.running_scale_unimodal
            ll_2_crossmodal = ll_2_crossmodal_unscaled * self.running_scale_crossmodal

            nll_1 = - (ll_1_unimodal + ll_1_crossmodal)
            nll_2 = - (ll_2_unimodal / self.running_scale_unimodal + ll_2_crossmodal / self.running_scale_crossmodal)
        else:
            ll_1_unimodal = self.vae_1.log_likelihood(x_1_counts, generative_model_dict["Modality 1 unimodal"])
            ll_2_unimodal = self.vae_2.log_likelihood(x_2_counts, generative_model_dict["Modality 2 unimodal"])
            ll_1_crossmodal = self.vae_1.log_likelihood(x_1_counts, generative_model_dict["Modality 1 crossmodal"])
            ll_2_crossmodal = self.vae_2.log_likelihood(x_2_counts, generative_model_dict["Modality 2 crossmodal"])

            nll_1 = - (ll_1_unimodal + ll_1_crossmodal)
            nll_2 = - (ll_2_unimodal + ll_2_crossmodal)

        # Compute the log priors for the shared and private latent variables of each modality
        prior_1_shared_dict = {
            "Mean": torch.zeros_like(variational_posterior_dict["Modality 1"]["Shared latent variable"]),
            "Variance": torch.ones_like(variational_posterior_dict["Modality 1"]["Shared latent variable"])
        }
        prior_1_private_dict = {
            "Mean": torch.zeros_like(variational_posterior_dict["Modality 1"]["Private latent variable"]),
            "Variance": torch.ones_like(variational_posterior_dict["Modality 1"]["Private latent variable"])
        }
        prior_2_shared_dict = {
            "Mean": torch.zeros_like(variational_posterior_dict["Modality 2"]["Shared latent variable"]),
            "Variance": torch.ones_like(variational_posterior_dict["Modality 2"]["Shared latent variable"])
        }
        prior_2_private_dict = {
            "Mean": torch.zeros_like(variational_posterior_dict["Modality 2"]["Private latent variable"]),
            "Variance": torch.ones_like(variational_posterior_dict["Modality 2"]["Private latent variable"])
        }

        log_prior_1_shared = self.gaussian_log_likelihood(
            variational_posterior_dict["Modality 1"]["Shared latent variable"],
            prior_1_shared_dict
        )
        log_prior_1_private = self.gaussian_log_likelihood(
            variational_posterior_dict["Modality 1"]["Private latent variable"],
            prior_1_private_dict
        )

        log_prior_2_shared = self.gaussian_log_likelihood(
            variational_posterior_dict["Modality 2"]["Shared latent variable"],
            prior_2_shared_dict
        )
        log_prior_2_private = self.gaussian_log_likelihood(
            variational_posterior_dict["Modality 2"]["Private latent variable"],
            prior_2_private_dict
        )

        # Log-likelihoods of private latent variables
        latent_1_private_dict = {
            "Mean": variational_posterior_dict["Modality 1"]["Private latent mean"],
            "Variance": variational_posterior_dict["Modality 1"]["Private latent variance"]
        }
        latent_2_private_dict = {
            "Mean": variational_posterior_dict["Modality 2"]["Private latent mean"],
            "Variance": variational_posterior_dict["Modality 2"]["Private latent variance"]
        }

        log_latent_1_private = self.gaussian_log_likelihood(
            variational_posterior_dict["Modality 1"]["Private latent variable"],
            latent_1_private_dict
        )
        log_latent_2_private = self.gaussian_log_likelihood(
            variational_posterior_dict["Modality 2"]["Private latent variable"],
            latent_2_private_dict
        )

        # Mixture of experts log-likelihoods for the shared latent variables
        moe_ll_1 = self.moe_log_likelihood(variational_posterior_dict, 1)
        moe_ll_2 = self.moe_log_likelihood(variational_posterior_dict, 2)

        # ELBO components per modality

        pseudo_kld_1 = self.vae_1.beta * (log_prior_1_shared + log_prior_1_private) - self.vae_1.beta * (moe_ll_1 + log_latent_1_private)
        pseudo_kld_2 = self.vae_2.beta * (log_prior_2_shared + log_prior_2_private) - self.vae_2.beta * (moe_ll_2 + log_latent_2_private)

        log_imp_weights_1 = ll_1_unimodal + pseudo_kld_1 + ll_2_crossmodal
        log_imp_weights_2 = ll_2_unimodal + pseudo_kld_2 + ll_1_crossmodal

        # IWAE ELBO computation per modality
        elbo_1 = - torch.logsumexp(log_imp_weights_1, dim=0) + torch.log(torch.tensor(log_imp_weights_1.shape[0], device=log_imp_weights_1.device, dtype=log_imp_weights_1.dtype))
        elbo_2 = - torch.logsumexp(log_imp_weights_2, dim=0) + torch.log(torch.tensor(log_imp_weights_2.shape[0], device=log_imp_weights_2.device, dtype=log_imp_weights_2.dtype))

        if self.learn_modality_weighting:
            # With learnable modality weights a balancing loss and an entropy regularization is added
            avg_weight_1 = variational_posterior_dict["Mixture of experts"]["Weighting modality 1"].mean()
            avg_weight_2 = variational_posterior_dict["Mixture of experts"]["Weighting modality 2"].mean()
            balancing_loss = 2 * ((0.5 - avg_weight_1) ** 2 + (0.5 - avg_weight_2) ** 2)
            entropy_regularization = - (avg_weight_1 * torch.log(avg_weight_1 + 1e-10) + avg_weight_2 * torch.log(avg_weight_2 + 1e-10))

            modality_weight_regularization = self.modality_weight_balancing_coeff * balancing_loss + self.modality_weight_entropy_coeff * entropy_regularization

            elbo = (
                        variational_posterior_dict["Mixture of experts"]["Weighting modality 1"] * elbo_1
                        + variational_posterior_dict["Mixture of experts"]["Weighting modality 2"] * elbo_2
            ) + modality_weight_regularization
        else:
            # During KL warm-up we do not have learnable modality weights
            modality_weight_regularization = 0.0
            elbo = 0.5 * elbo_1 + 0.5 * elbo_2 + modality_weight_regularization


        # Pseudo kld per modality
        kld_1 = - torch.logsumexp(pseudo_kld_1, dim=0) + torch.log(torch.tensor(pseudo_kld_1.shape[0], device=pseudo_kld_1.device, dtype=pseudo_kld_1.dtype))
        kld_2 = - torch.logsumexp(pseudo_kld_2, dim=0) + torch.log(torch.tensor(pseudo_kld_2.shape[0], device=pseudo_kld_2.device, dtype=pseudo_kld_2.dtype))

        return elbo, elbo_1, elbo_2, nll_1, nll_2, kld_1, kld_2


class LogisticRegressionClassifier(nn.Module):
    r"""
        Linear classifier mapping feature vectors to unnormalized class logits.
    """
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            device: str
    ) -> None:
        r"""
        Initialize a linear classifier that maps feature vectors to class logits. The output is suitable for a
        classification loss that accepts logits; the classifier does not apply softmax.

        :param input_dim: (int) Number of input features.
        :param num_classes: (int) Number of output classes.
        :param device: (str) Device setting stored on the classifier. Move the module to this device explicitly.
        """
        super(LogisticRegressionClassifier, self).__init__()

        self.linear_layer = nn.Linear(input_dim, num_classes)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Map input feature vectors through the linear layer to compute unnormalized class logits.

        :param x: (torch.Tensor) Input feature vectors with final dimension input_dim, typically (batch size, input_dim).
        :return: logits (torch.Tensor) Class logits with the same leading dimensions and final dimension num_classes.
        """

        logits = self.linear_layer(x)

        return logits





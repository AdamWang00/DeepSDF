#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(), # >= 0 e.g. 4 = normalize output of 5rd layer
        latent_in=(), # >= 0 e.g. 4 = concatenate latent+xyz to input of 5th layer
        weight_norm=False,
        xyz_in_all=False, # disabled
        use_tanh=False, # unused - last layer is always tanh
        latent_dropout=False,
        use_fourier_features=False,
        fourier_features_std=1.0,
        fourier_features_size=256, # note that the actual feature size is double this
        film_layers=() # >= 0 e.g. 4 = film output of 5th layer
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        if use_fourier_features:
            xyz_size = fourier_features_size*2
        else:
            xyz_size = 3

        if 0 in latent_in:
            dims = [latent_size + xyz_size] + dims + [4]
        else:
            dims = [xyz_size] + dims + [4]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        self.use_fourier_features = use_fourier_features
        if self.use_fourier_features:
            self.fourier_features_size = fourier_features_size
            gaussian_matrix = torch.normal(0, fourier_features_std, size=(self.fourier_features_size, 3))
            gaussian_matrix.requires_grad = False
            self.register_buffer('gaussian_matrix', gaussian_matrix)

        self.use_film = len(film_layers) > 0
        if self.use_film:
            film_layers.sort()
            self.film_layers = film_layers
            self.film_index_offsets = []
            film_index_offset = 0

        self.out_dims = []

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.latent_in:
                out_dim = dims[layer + 1] - (latent_size + xyz_size)
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= self.fourier_features_size*2

            self.out_dims.append(out_dim)
            print(layer, out_dim)

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if self.use_film and layer in self.film_layers:
                self.film_index_offsets.append(film_index_offset)
                film_index_offset += 2 * out_dim # scale and bias per feature

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        if self.use_film:
            self.film_linear = nn.Linear(latent_size, film_index_offset)

        # self.use_tanh = use_tanh
        # if use_tanh:
        #     self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]
        latent_vecs = input[:, :-3]

        if self.use_fourier_features:
            xyz = (2.*np.pi*xyz) @ torch.t(self.gaussian_matrix).cuda()
            xyz = torch.cat((torch.sin(xyz), torch.cos(xyz)), -1)

        if self.latent_dropout:
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
        
        if 0 in self.latent_in:
            x = torch.cat((latent_vecs, xyz), -1)
        else:
            x = xyz

        if self.use_film:
            film = self.film_linear(latent_vecs)
            film_counter = 0

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer > 0 and layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh - ** remove this since last layer is always tanh **
            # if layer == self.num_layers - 2 and self.use_tanh:
            #     x[:, 0] = self.tanh(x[:, 0]) # only activate sdf value with tanh
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                
                if self.use_film and layer in self.film_layers:
                    out_dim = self.out_dims[layer]
                    film_index_offset = self.film_index_offsets[film_counter]
                    film_counter += 1

                    gammas = film[:, film_index_offset:film_index_offset + out_dim]
                    betas = film[:, film_index_offset + out_dim:film_index_offset + (2 * out_dim)]
                    x = (gammas * x) + betas

                x = self.relu(x)

                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # if hasattr(self, "th"):
        x[:, 0] = self.th(x[:, 0]) # only activate sdf value with tanh

        x[:, 1:4] = x[:, 1:4] * 255 # scale up (to avoid large parameters)

        return x
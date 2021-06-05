#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class FiLM(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.relu = nn.ReLU()
    def forward(self, x, gammas, betas):
        return self.relu((gammas * x) + betas)


class FiLMedSIREN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.film = FiLM()
    def forward(self, x, gammas, betas):
        return torch.sin(self.film(x, gammas, betas))


# class MappingNetwork(nn.Module):
#     def __init__(self, latent_len, num_filmed_layers, out_dim=512):
#         nn.Module.__init__(self)
#         self.fc1 = nn.Linear(latent_len, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, num_filmed_layers * out_dim * 2) # need scales and shifts

#         self.lrelu = nn.LeakyReLU(0.2)
    
#     def forward(self, latent):
#         return self.fc4(self.lrelu(self.fc3(self.lrelu(self.fc2(self.lrelu(self.fc1(latent)))))))

# class MappingNetwork(nn.Module):
#     def __init__(self, latent_len, out_dim=256):
#         nn.Module.__init__(self)
#         self.fc1 = nn.Linear(latent_len, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 256)
#         self.fc4 = nn.Linear(256, out_dim * 2) # need scales and shifts

#         self.lrelu = nn.LeakyReLU(0.2)
    
#     def forward(self, latent):
#         return self.fc4(self.lrelu(self.fc3(self.lrelu(self.fc2(self.lrelu(self.fc1(latent)))))))

class MappingNetwork(nn.Module):
    def __init__(self, latent_len, hidden_dims, out_dim=256):
        nn.Module.__init__(self)

        dims = [latent_len] + hidden_dims + [out_dim * 2] # need scales and shifts
        self.num_layers = len(dims)

        for layer in range(0, self.num_layers - 1):
            setattr(self, "lin" + str(layer), nn.Linear(dims[layer], dims[layer + 1]))

        self.lrelu = nn.LeakyReLU(0.2)
    
    def forward(self, latent):
        x = latent
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.lrelu(x)
        return x

# def gammas_betas(mapping_output, layer_index, out_dim=512):
#     gammas = mapping_output[:, layer_index * out_dim : (layer_index + 1) * out_dim] + 1
#     betas = mapping_output[:, layer_index * out_dim : (layer_index + 1) * out_dim]

#     return gammas, betas

def gammas_betas(mapping_output, out_dim=256):
    gammas = mapping_output[:, :out_dim] + 1
    betas = mapping_output[:, out_dim:]

    return gammas, betas


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
        use_film=True,
        mapping_film_dims=[]
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []
        
        self.use_film = use_film
        if self.use_film:
            self.film = FiLM()
            # self.num_hidden_layers = len(dims)
            self.linear_dim = dims[0] # assume all hidden linear layers in decoder are the same dim
            # self.mapping_film = MappingNetwork(latent_size, self.num_hidden_layers, out_dim=self.linear_dim)
            self.mapping_film_dims = mapping_film_dims

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

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.latent_in:
                out_dim = dims[layer + 1] - (latent_size + xyz_size)
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= self.fourier_features_size*2
            print(layer, out_dim)

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))
            
            if self.use_film:
                setattr(self, "mn" + str(layer), MappingNetwork(latent_size, self.mapping_film_dims, out_dim=self.linear_dim))

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

        # if self.use_film:
        #     mapping_output = self.mapping_film(latent_vecs)

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
                
                if self.use_film:
                    mn = getattr(self, "mn" + str(layer))
                    mapping_output = mn(latent_vecs)
                    # gammas, betas = gammas_betas(mapping_output, layer, out_dim=self.linear_dim)
                    gammas, betas = gammas_betas(mapping_output, out_dim=self.linear_dim)
                    x = self.film(x, gammas, betas)

                x = self.relu(x)

                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # if hasattr(self, "th"):
        x[:, 0] = self.th(x[:, 0]) # only activate sdf value with tanh

        x[:, 1:4] = x[:, 1:4] * 255 # scale up (to avoid large parameters)

        return x
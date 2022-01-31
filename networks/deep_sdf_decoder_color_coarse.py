#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# pts (numpy array): [batch_size, 3] with 0 <= x, y, z <= 1 (i.e. in a unit box grid) to be interpolated (otherwise, if outside the unit box grid, extrapolated)
# features (torch tensor): [batch_size, D, H, W, feature_len] with D, H, W >= 2
# output: [batch_size, feature_len]
def interpolate_trilinear(pts, features):
    batch_size, depth, height, width, _ = features.shape

    # (i, j, k) are the indices of the (lower, bottom, down) feature, out of the 8 features we interpolate point between. (0, 0, 0) <= (i, j, k) <= (D - 2, H - 2, W - 2)
    pts = np.copy(pts)
    pts[:, 0] *= depth - 1
    pts[:, 1] *= height - 1
    pts[:, 2] *= width - 1
    ijk = pts.astype(np.int32)
    i = np.clip(ijk[:, 0], 0, depth - 2, out=ijk[:, 0])
    j = np.clip(ijk[:, 1], 0, height - 2, out=ijk[:, 1])
    k = np.clip(ijk[:, 2], 0, width - 2, out=ijk[:, 2])

    # features at each of the 8 corners
    batch_indices = np.arange(batch_size)
    f000 = features[batch_indices,  i   , j   ,  k   ]
    f100 = features[batch_indices, (i+1), j   ,  k   ]
    f010 = features[batch_indices,  i   ,(j+1),  k   ]
    f001 = features[batch_indices,  i   , j   , (k+1)]
    f101 = features[batch_indices, (i+1), j   , (k+1)]
    f011 = features[batch_indices,  i   ,(j+1), (k+1)]
    f110 = features[batch_indices, (i+1),(j+1),  k   ]
    f111 = features[batch_indices, (i+1),(j+1), (k+1)]

    # (x, y, z) are the "sliders" for the three dimensions, with (0, 0, 0) being full weights for the (lower, bottom, down) feature and (1, 1, 1) being full weights for the (upper, top, up) feature
    xyz = torch.Tensor(pts - ijk).cuda()
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    fxyz = (f000.T * (1 - x)*(1 - y)*(1 - z)
            + f100.T * x * (1 - y) * (1 - z)
            + f010.T * (1 - x) * y * (1 - z)
            + f001.T * (1 - x) * (1 - y) * z
            + f101.T * x * (1 - y) * z
            + f011.T * (1 - x) * y * z
            + f110.T * x * y * (1 - z)
            + f111.T * x * y * z).T
    return fxyz


# pts (numpy array): [batch_size, 3] with 0 <= x, y, z <= 1 (i.e. in a unit box grid) to be interpolated (otherwise, if outside the unit box grid, extrapolated)
# features (torch tensor): [D, H, W, feature_len] with D, H, W >= 2
# output: [batch_size, feature_len]
def interpolate_trilinear_alt(pts, features):
    depth, height, width, _ = features.shape

    # (i, j, k) are the indices of the (lower, bottom, down) feature, out of the 8 features we interpolate point between. (0, 0, 0) <= (i, j, k) <= (D - 2, H - 2, W - 2)
    pts = np.copy(pts)
    pts[:, 0] *= depth - 1
    pts[:, 1] *= height - 1
    pts[:, 2] *= width - 1
    ijk = pts.astype(np.int32)
    i = np.clip(ijk[:, 0], 0, depth - 2, out=ijk[:, 0])
    j = np.clip(ijk[:, 1], 0, height - 2, out=ijk[:, 1])
    k = np.clip(ijk[:, 2], 0, width - 2, out=ijk[:, 2])

    # features at each of the 8 corners
    f000 = features[i   , j   ,  k   ]
    f100 = features[(i+1), j   ,  k   ]
    f010 = features[ i   ,(j+1),  k   ]
    f001 = features[ i   , j   , (k+1)]
    f101 = features[(i+1), j   , (k+1)]
    f011 = features[ i   ,(j+1), (k+1)]
    f110 = features[(i+1),(j+1),  k   ]
    f111 = features[(i+1),(j+1), (k+1)]

    # (x, y, z) are the "sliders" for the three dimensions, with (0, 0, 0) being full weights for the (lower, bottom, down) feature and (1, 1, 1) being full weights for the (upper, top, up) feature
    xyz = torch.Tensor(pts - ijk).cuda()
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    fxyz = (f000.T * (1 - x)*(1 - y)*(1 - z)
            + f100.T * x * (1 - y) * (1 - z)
            + f010.T * (1 - x) * y * (1 - z)
            + f001.T * (1 - x) * (1 - y) * z
            + f101.T * x * (1 - y) * z
            + f011.T * (1 - x) * y * z
            + f110.T * x * y * (1 - z)
            + f111.T * x * y * z).T
    return fxyz


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=False, # disabled
        use_tanh=False, # unused - last layer is always tanh
        latent_dropout=False,
        use_fourier_features=False,
        fourier_features_std=1.0,
        fourier_features_size=128,
        convt3d_dims=[512],
        convt3d_kernel_sizes=[2],
        convt3d_strides=[1],
    ):
        super(Decoder, self).__init__()

        convt3d_dims = [latent_size] + convt3d_dims

        self.conv3d_t_num_layers = len(convt3d_dims)
        for layer in range(self.conv3d_t_num_layers - 1):
            in_channels = convt3d_dims[layer]
            out_channels = convt3d_dims[layer + 1]
            print("convT", layer, out_channels)
            setattr(self, "convT" + str(layer), nn.ConvTranspose3d(in_channels, out_channels, convt3d_kernel_sizes[layer], convt3d_strides[layer]))

        if use_fourier_features:
            xyz_size = fourier_features_size*2
        else:
            xyz_size = 3

        dims = [convt3d_dims[-1] + xyz_size] + dims + [4]

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
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
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

        # self.use_tanh = use_tanh
        # if use_tanh:
        #     self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(negative_slope=0.01)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3) or N x 3 if scene_latent (L) is provided
    def forward(self, input, scene_latent=None):
        if scene_latent is not None:
            return self.forward_alt(input, scene_latent)

        xyz = input[:, -3:]
        latent_vecs = input[:, :-3]

        latent_vecs = latent_vecs.view(latent_vecs.shape[0], latent_vecs.shape[1], 1, 1, 1)
        for layer in range(self.conv3d_t_num_layers - 1):
            convT = getattr(self, "convT" + str(layer))
            latent_vecs = convT(latent_vecs)
            if (layer < self.conv3d_t_num_layers - 2):
                latent_vecs = self.leaky(latent_vecs)
        
        latent_vecs = torch.permute(latent_vecs, (0, 2, 3, 4, 1)) # swaps feature (1) with DHW (2, 3, 4)
        xyz_normalized = (xyz.cpu().detach().numpy() + 1) / 2 # should generally be 0 <= x,y,z <= 1
        latent_vecs = interpolate_trilinear(xyz_normalized, latent_vecs)

        if self.use_fourier_features:
            xyz = (2.*np.pi*xyz) @ torch.t(self.gaussian_matrix).cuda()
            xyz = torch.cat((torch.sin(xyz), torch.cos(xyz)), -1)
            
            input = torch.cat((latent_vecs, xyz), 1)

            if input.shape[1] > self.fourier_features_size*2 and self.latent_dropout:
                latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
                x = torch.cat([latent_vecs, xyz], 1)
            else:
                x = input
        else:
            if input.shape[1] > 3 and self.latent_dropout:
                latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
                x = torch.cat([latent_vecs, xyz], 1)
            else:
                x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
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
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # if hasattr(self, "th"):
        x[:, 0] = self.th(x[:, 0]) # only activate sdf value with tanh

        x[:, 1:4] = x[:, 1:4] * 255 # scale up (to avoid large parameters)

        return x


    # xyz: [N, 3]
    # latent_vecs: [L]
    def forward_alt(self, xyz, latent_vec):
        latent_vec = latent_vec.view(1, latent_vec.shape[0], 1, 1, 1)
        for layer in range(self.conv3d_t_num_layers - 1):
            convT = getattr(self, "convT" + str(layer))
            latent_vec = convT(latent_vec)
            if (layer < self.conv3d_t_num_layers - 2):
                latent_vec = self.leaky(latent_vec)
        
        latent_vec = torch.permute(latent_vec[0], (1, 2, 3, 0)) # swaps feature (0) with DHW (1, 2, 3)
        # print(latent_vec.shape)
        xyz_normalized = (xyz.cpu().detach().numpy() + 1) / 2 # should generally be 0 <= x,y,z <= 1
        latent_vec = interpolate_trilinear_alt(xyz_normalized, latent_vec)

        if self.use_fourier_features:
            xyz = (2.*np.pi*xyz) @ torch.t(self.gaussian_matrix).cuda()
            xyz = torch.cat((torch.sin(xyz), torch.cos(xyz)), -1)

        if self.latent_dropout:
            latent_vec = F.dropout(latent_vec, p=0.2, training=self.training)
        
        input = torch.cat([latent_vec, xyz], 1)
        x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
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
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # if hasattr(self, "th"):
        x[:, 0] = self.th(x[:, 0]) # only activate sdf value with tanh

        x[:, 1:4] = x[:, 1:4] * 255 # scale up (to avoid large parameters)

        return x
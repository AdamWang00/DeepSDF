#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import torch
import numpy as np
from skimage import color
from scipy.stats import multivariate_normal
from scipy.spatial import cKDTree

def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def decode_sdf(decoder, latent_vector, queries):
    # num_samples = queries.shape[0]

    # if latent_vector is None:
    #     inputs = queries
    # else:
    #     latent_repeat = latent_vector.expand(num_samples, -1)
    #     inputs = torch.cat([latent_repeat.cuda(), queries], 1)

    # sdf = decoder(inputs)

    # return sdf

    sdf = decoder(queries, scene_latent=latent_vector.cuda())

    return sdf


def compute_soft_encoding_lookup_table(color_bins_lab, num_closest=8, sigma=5.0):
    '''
    compute the (B x B) lookup table for the soft-encoding of each bin, similar to the technique used in Colorful Image Colorization

    num_closest is the number of Gaussians to sample (i.e. nonzero columns) per row, including self (i.e. the number of "neighbors" would therefore be num_closest - 1)
    sigma is the standard deviation of each Gaussian centered at its corresponding bin in Lab space
    '''
    num_bins = color_bins_lab.shape[0]
    table = np.zeros((num_bins, num_bins))

    color_bins_kdtree = cKDTree(color_bins_lab)

    for r in range(num_bins):
        color_bin = color_bins_lab[r]
        _, closest_bins_idx = color_bins_kdtree.query(color_bin, k=num_closest)
        row_sum = 0

        for neighbor_idx in closest_bins_idx:
            z = multivariate_normal.pdf(color_bin, mean=color_bins_lab[neighbor_idx], cov=sigma**2)
            table[r, neighbor_idx] = z
            row_sum += z

        table[r, :] = table[r, :] / row_sum

    return table


def lab_bins_to_rgb(soft_encoded_lab_bins, bin_to_lab, annealing_temperature=0.38):
    vertex_colors = np.exp(np.log(soft_encoded_lab_bins+1e-10) / annealing_temperature)
    vertex_colors = np.divide(vertex_colors.T, np.sum(vertex_colors, axis=1)).T
    vertex_colors = vertex_colors @ bin_to_lab
    vertex_colors = color.lab2rgb(vertex_colors)
    vertex_colors = (np.clip(vertex_colors * 256, 0, 255)).astype(np.int)
    return vertex_colors
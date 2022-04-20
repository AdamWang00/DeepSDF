#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import deep_sdf
import deep_sdf.workspace as ws
import random

random.seed(0)

def load_logs(experiment_directory):

    logs = np.load(os.path.join(experiment_directory, "LipschitzBoundLogs.npy"))

    fig, ax = plt.subplots()

    num_epochs = logs.shape[0]

    to_plot = []

    for i in range(logs.shape[1]):
        to_plot.extend(
            [
                np.arange(num_epochs),
                logs[:, i],
                "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]),
            ]
        )

    ax.plot(*to_plot)

    ax.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")

    ax.grid()
    plt.show()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Plot DeepSDF lipschitz bound logs")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment "
        + "specifications in 'specs.json', and logging will be done in this directory "
        + "as well",
    )
    args = arg_parser.parse_args()

    load_logs(args.experiment_directory)

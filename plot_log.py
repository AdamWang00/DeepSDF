#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import deep_sdf
import deep_sdf.workspace as ws


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def load_logs(experiment_directory, type, is_color, is_lipschitz, is_disc, save=False):

    logs = torch.load(os.path.join(experiment_directory, ws.logs_filename))

    logging.info("latest epoch is {}".format(logs["epoch"]))

    num_iters = len(logs["loss"])
    iters_per_epoch = num_iters / logs["epoch"]

    logging.info("{} iters per epoch".format(iters_per_epoch))

    smoothed_loss_41 = running_mean(logs["loss"], 41)
    smoothed_loss_1601 = running_mean(logs["loss"], 1601)

    fig, ax = plt.subplots()

    if type == "loss":
        to_plot = [
            np.arange(num_iters) / iters_per_epoch,
            logs["loss"],
            "#82c6eb",
            np.arange(20, num_iters - 20) / iters_per_epoch,
            smoothed_loss_41,
            "#2a9edd",
            # np.arange(800, num_iters - 800) / iters_per_epoch,
            # smoothed_loss_1601,
            # "#16628b",
        ]

        if is_color:
            to_plot.extend([
                np.arange(num_iters) / iters_per_epoch,
                logs["sdf_loss"],
                "#6759ff",
                np.arange(num_iters) / iters_per_epoch,
                logs["color_loss"],
                "#ffaf59"
            ])

        if is_lipschitz:
            to_plot.extend([
                np.arange(num_iters) / iters_per_epoch,
                logs["lipschitz_loss"],
                "#378805"
            ])
        
        if is_disc:
            to_plot.extend([
                np.arange(num_iters) / iters_per_epoch,
                logs["disc_loss_log"],
                "#378805"
            ])

        ax.plot(*to_plot)

        ax.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")

    elif type == "learning_rate":
        combined_lrs = np.array(logs["learning_rate"])

        ax.plot(
            np.arange(combined_lrs.shape[0]),
            combined_lrs[:, 0],
            np.arange(combined_lrs.shape[0]),
            combined_lrs[:, 1],
        )
        ax.set(xlabel="Epoch", ylabel="Learning Rate", title="Learning Rates")

    elif type == "time":
        ax.plot(logs["timing"], "#833eb7")
        ax.set(xlabel="Epoch", ylabel="Time per Epoch (s)", title="Timing")

    elif type == "lat_mag":
        ax.plot(logs["latent_magnitude"])
        ax.set(xlabel="Epoch", ylabel="Magnitude", title="Latent Vector Magnitude")

    elif type == "param_mag":
        for _name, mags in logs["param_magnitude"].items():
            ax.plot(mags)
        ax.set(xlabel="Epoch", ylabel="Magnitude", title="Parameter Magnitude")
        ax.legend(logs["param_magnitude"].keys())

    else:
        raise Exception('unrecognized plot type "{}"'.format(type))

    ax.grid()
    if save:
        plt.savefig("log.png")
    else:
        plt.show()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Plot DeepSDF training logs")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment "
        + "specifications in 'specs.json', and logging will be done in this directory "
        + "as well",
    )
    arg_parser.add_argument("--type", "-t", dest="type", default="loss")
    arg_parser.add_argument("--color", "-c", dest="is_color", default=False, action="store_true")
    arg_parser.add_argument("--lipschitz", "-l", dest="is_lipschitz", default=False, action="store_true")
    arg_parser.add_argument("--disc", "-d", dest="is_disc", default=False, action="store_true")
    arg_parser.add_argument("--save", "-s", dest="save", default=False, action="store_true")

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    load_logs(args.experiment_directory, args.type, args.is_color, args.is_lipschitz, args.is_disc, save=args.save)

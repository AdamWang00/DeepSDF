#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import numpy as np

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf.utils import compute_soft_encoding_lookup_table

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(
                schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(
        experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    sdf_loss_log,
    color_loss_log,
    lipschitz_loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "sdf_loss": sdf_loss_log,
            "color_loss": color_loss_log,
            "lipschitz_loss": lipschitz_loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["sdf_loss"],
        data["color_loss"],
        data["lipschitz_loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, sdf_loss_log, color_loss_log, lipschitz_loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    sdf_loss_log = sdf_loss_log[: (iters_per_epoch * epoch)]
    color_loss_log = color_loss_log[: (iters_per_epoch * epoch)]
    lipschitz_loss_log = lipschitz_loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, sdf_loss_log, color_loss_log, lipschitz_loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def rgb_to_bin(r, g, b, dim=8):
    return r * dim * dim + g * dim + b


def main_function(experiment_directory, continue_from, load_ram):

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory,
                       "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory,
                            "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):

        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(
            epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(
            epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = specs["UseClamping"] if "UseClamping" in specs else True

    do_code_regularization = get_spec_with_default(
        specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(
        specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    color_loss_sdf_threshold = get_spec_with_default(
        specs, "ColorLossSdfThreshold", 100.0)
    color_loss_weight = get_spec_with_default(specs, "ColorLossWeight", 0.0)

    with_lipschitz = "LipschitzLossWeight" in specs
    if with_lipschitz:
        lipschitz_loss_weight = get_spec_with_default(specs, "LipschitzLossWeight", 0.0)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    # decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 5)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=load_ram
    )

    num_data_loader_threads = get_spec_with_default(
        specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(
        num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev",
                              1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    sdf_loss_log = []
    color_loss_log = []
    lipschitz_loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    LOG_LIPSCHITZ_BOUNDS = True
    if LOG_LIPSCHITZ_BOUNDS:
        lipschitz_bounds_log = []

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, sdf_loss_log, color_loss_log, lipschitz_loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, sdf_loss_log, color_loss_log, lipschitz_loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, sdf_loss_log, color_loss_log, lipschitz_loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    color_bins_path = specs["ColorBinsPath"]
    color_bins_key = specs["ColorBinsKey"]
    print("using color bins from", color_bins_path)
    color_soft_encoding_num_closest = get_spec_with_default(specs, "ColorSoftEncodingNumClosest", 8)
    color_soft_encoding_sigma = get_spec_with_default(specs, "ColorSoftEncodingSigma", 5.0)

    color_bins_lab = np.load(color_bins_path)[color_bins_key]
    color_bin_soft_encodings_table = compute_soft_encoding_lookup_table(color_bins_lab, num_closest=color_soft_encoding_num_closest, sigma=color_soft_encoding_sigma)
    color_bin_soft_encodings_table = torch.from_numpy(color_bin_soft_encodings_table)

    rgb_bin_proportions = torch.zeros((512))
    for sdf_data, indices in sdf_loader:
        rgb_bin_idx = sdf_data.reshape(-1, 5)[:, 4].long()
        rgb_bin_proportions += torch.sum(color_bin_soft_encodings_table[rgb_bin_idx, :], 0)

    rgb_bin_proportions /= torch.sum(rgb_bin_proportions)
    assert abs(torch.sum(rgb_bin_proportions) - 1) < 1e-5

    rgb_bin_weights = 1 / (0.5 * rgb_bin_proportions + (0.5 / 512))
    rgb_bin_weights /= torch.dot(rgb_bin_proportions, rgb_bin_weights)
    assert abs(torch.dot(rgb_bin_proportions, rgb_bin_weights) - 1) < 1e-5
    rgb_bin_weights = rgb_bin_weights.cuda()
    rgb_bin_weights.requires_grad = False

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        logging.info("epoch {}...".format(epoch))

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        # sdf_data.shape is [ScenesPerBatch, SamplesPerScene, 5]
        for sdf_data, indices in sdf_loader:

            # we want to split the batch such that each mini batch = 1 scene (i.e. 1 latent code)
            batch_split = scene_per_batch

            # Process the input data
            sdf_data = sdf_data.reshape(-1, 5)  # x, y, z, sdf_gt, color_bin_idx

            num_sdf_samples = sdf_data.shape[0]

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3]
            rgb_bin_idx_gt = sdf_data[:, 4].long()
            rgb_bin_gt = color_bin_soft_encodings_table[rgb_bin_idx_gt, :]

            if xyz.dtype == torch.float64:
                xyz = xyz.float()

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)
            sdf_gt = torch.chunk(sdf_gt, batch_split)
            rgb_bin_idx_gt = torch.chunk(rgb_bin_idx_gt, batch_split)
            rgb_bin_gt = torch.chunk(rgb_bin_gt, batch_split)

            batch_loss = 0.0
            batch_sdf_loss = 0.0
            batch_color_loss = 0.0
            batch_lipschitz_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split): # batch_split = scene_per_batch
                batch_vec = lat_vecs(indices[i]).cuda()

                # forward pass
                pred_sdf, pred_rgb_bins = decoder(xyz[i].cuda(), scene_latent=batch_vec)

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                batch_sdf_gt = sdf_gt[i].cuda()
                batch_rgb_bin_idx_gt = rgb_bin_idx_gt[i].cuda()
                batch_rgb_bin_gt = rgb_bin_gt[i].cuda()

                # include color loss for points with |sdf_gt| < threshold
                mask = torch.where(torch.abs(batch_sdf_gt) <
                                   color_loss_sdf_threshold, 1, 0).unsqueeze(1)

                batch_rgb_bin_gt = torch.mul(batch_rgb_bin_gt, mask)

                loss_sdf = loss_l1(pred_sdf, batch_sdf_gt) / num_sdf_samples

                pred_rgb_bins = torch.clamp(pred_rgb_bins, min=1e-30) # ensure pred_rgb_bins>0 to avoid div0
                loss_rgb = -torch.sum(torch.mul(rgb_bin_weights[batch_rgb_bin_idx_gt],
                    torch.sum(torch.mul(batch_rgb_bin_gt, torch.log(pred_rgb_bins)), 1))) / num_sdf_samples
                loss_rgb *= color_loss_weight

                chunk_loss = loss_sdf + loss_rgb

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vec, dim=0))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples

                    chunk_loss = chunk_loss + reg_loss.cuda()

                chunk_loss.backward()

                batch_loss += chunk_loss.item()
                batch_sdf_loss += loss_sdf.item()
                batch_color_loss += loss_rgb.item()

            if with_lipschitz:
                lipschitz_bounds = decoder.get_lipschitz_bounds()
                lipschitz_loss = lipschitz_loss_weight
                for lb in lipschitz_bounds:
                    lipschitz_loss *= lb
                lipschitz_loss = lipschitz_loss.cuda()
                lipschitz_loss.backward()
                batch_lipschitz_loss += lipschitz_loss.item()
                batch_loss += lipschitz_loss.item()

            logging.debug("loss = {}".format(batch_loss))

            loss_log.append(batch_loss)
            sdf_loss_log.append(batch_sdf_loss)
            color_loss_log.append(batch_color_loss)
            lipschitz_loss_log.append(batch_lipschitz_loss)
            # print("sdf loss", batch_sdf_loss)
            # print("rgb loss", batch_color_loss)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            if with_lipschitz and LOG_LIPSCHITZ_BOUNDS:
                temp = [lb.item() for lb in lipschitz_bounds]

            optimizer_all.step()

            if with_lipschitz and LOG_LIPSCHITZ_BOUNDS:
                print([lipschitz_bounds[i].item() - temp[i] for i in range(len(temp))])
        
        if with_lipschitz and LOG_LIPSCHITZ_BOUNDS:
            lipschitz_bounds = decoder.get_lipschitz_bounds()
            lipschitz_bounds = [lb.item() for lb in lipschitz_bounds]
            lipschitz_bounds_log.append(lipschitz_bounds)

        end = time.time()

        seconds_elapsed = end - start
        print("time:", seconds_elapsed)
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch)
                      for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                sdf_loss_log,
                color_loss_log,
                lipschitz_loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )

            if with_lipschitz and LOG_LIPSCHITZ_BOUNDS:
                np.save(os.path.join(experiment_directory, "LipschitzBoundLogs"), np.array(lipschitz_bounds_log))

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--loadram",
        "-l",
        dest="load_ram",
        default=False,
        action="store_true",
        help="Store all data in RAM for performance.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, args.load_ram)

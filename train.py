import json
import os
from datetime import datetime
from tqdm.auto import tqdm

import numpy as np
import ignite
import torch as th
from ignite.contrib.handlers import ProgressBar
from monai.data import ArrayDataset
from monai.handlers import TensorBoardStatsHandler, TensorBoardImageHandler, MeanDice
from monai.metrics import DiceMetric, LossMetric
from monai.networks.nets import UNet
from monai.transforms import Resize, EnsureChannelFirst, LoadImage, Compose, ScaleIntensity
from monai.utils import first
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluate_util import get_model2, plot_metric_over_thresh, plot_model_output, evaluate_polar_model
from train_util import dataloader_setup
from loss import TotalLoss


# following https://github.com/Project-MONAI/tutorials/blob/818673937c9c5d0b0964924b056a867238991a6a/3d_segmentation/unet_segmentation_3d_ignite.ipynb#L102
# and https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_array.py
# and https://www.kaggle.com/code/juhha1/simple-model-development-using-monai
# https://github.com/Project-MONAI/tutorials/blob/main/modules/batch_output_transform.ipynb
# https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe#scrollTo=uHAA3LUxD2b6


def train(config=None):
    starttime = datetime.now()
    now_str = starttime.strftime("%Y_%m_%d__%H_%M_%S")

    # -------------
    # --- SETUP ---
    # -------------

    if not config:
        with open("config.json", 'r') as file:
            config = json.load(file)

    device = th.device(config["cuda_name"] if th.cuda.is_available() else "cpu")

    model_config = config["model_config"]
    loss_config = config["loss_config"]

    epochs = config["epochs"]
    polar = config["polar_data_used"]
    exp_path = "experiments/" + now_str + "_" + config["experiment_name"] + "/"
    if len(config["overwrite_exp_path"]) > 0:
        exp_path = "experiments/" + config["overwrite_exp_path"] + "/"

    config["timestamp"] = now_str

    train_dataloader, val_dataloader, test_dataloader, train_polar_dataloader, val_polar_dataloader, test_polar_dataloader = dataloader_setup(
        config)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    with open(exp_path + "config.json", "w+") as outfile:
        json.dump(config, outfile, indent=4)

    # get all architectural details from config.json
    model = UNet(
        spatial_dims=model_config["spatial_dims"],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        channels=model_config["channels"],
        strides=model_config["strides"],
        kernel_size=model_config["kernel_size"],
        up_kernel_size=model_config["up_kernel_size"],
        num_res_units=model_config["num_res_units"],
        act=model_config["activation"]
    ).to(device)

    opt = th.optim.Adam(model.parameters(), 1e-3)
    loss_func = TotalLoss(loss_config)

    # ----------------
    # --- TRAINING ---
    # ----------------

    trn_dl = train_dataloader
    val_dl = val_dataloader
    tst_dl = test_dataloader
    if polar:
        trn_dl = train_polar_dataloader
        val_dl = val_polar_dataloader
        tst_dl = test_polar_dataloader

    writer = SummaryWriter(log_dir=exp_path)
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        model.train()
        step = 0
        for batch_data in tqdm(trn_dl, desc="Batches", leave=False):
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            opt.zero_grad()
            outputs = model(inputs)
            losses = loss_func(outputs, labels)
            loss = losses.sum()
            loss.backward()
            opt.step()
            epoch_len = len(trn_dl.dataset) // trn_dl.batch_size
            writer.add_scalar("train loss", loss.item(), epoch_len * epoch + step)
            writer.add_scalar("dice loss", losses[0].item(), epoch_len * epoch + step)
            writer.add_scalar("topology loss", losses[1].item(), epoch_len * epoch + step)

        model.eval()
        val_loss = 0
        for batch_data in val_dl:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            val_loss += loss.item()
        writer.add_scalar("validation loss", val_loss, epoch)
        writer.add_image("sample output channel 1", outputs[0, 1], global_step=epoch, dataformats="HW")
        writer.add_image("sample output channel 2", outputs[0, 2], global_step=epoch, dataformats="HW")

    th.save(model.state_dict(), exp_path + "model.pt")

    writer.close()

    # ------------------
    # --- EVALUATION ---
    # ------------------

    if bool(config["evaluate_after_training"]):
        model = get_model2(exp_path, config)

        img, seg = first(val_dl)
        output_images = model(img)
        if bool(loss_config["sigmoid"]):
            y_pred = th.sigmoid(output_images.detach())
        if bool(loss_config["softmax"]):
            y_pred = th.softmax(output_images.detach(), dim=1)

        plot_model_output((img,
                           y_pred[0],
                           seg),
                          exp_path + "model_output.png")

        metric = DiceMetric()
        best_metric_per_channel, best_threshold_per_channel = plot_metric_over_thresh(config,
                                                                                      metric,
                                                                                      model,
                                                                                      val_dl,
                                                                                      writer,
                                                                                      exp_path + "thresh_variation.png",
                                                                                      device)

        if polar:
            best_metric_per_channel = evaluate_polar_model(config,
                                                           best_threshold_per_channel,
                                                           metric,
                                                           model,
                                                           writer,
                                                           exp_path,
                                                           device)
            print(best_metric_per_channel)

    # --------------
    # --- FINISH ---
    # --------------

    writer.close()

    endtime = datetime.now()
    time_diff = endtime - starttime
    hours = divmod(time_diff.total_seconds(), 3600)
    minutes = divmod(hours[1], 60)
    seconds = divmod(minutes[1], 1)
    print("Training+Evaluation took %d:%d:%d" % (hours[0], minutes[0], seconds[0]))

    return best_metric_per_channel


if __name__ == "__main__":
    train()

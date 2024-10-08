import json
import os
from datetime import datetime
from tqdm.auto import tqdm

import torch as th
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.utils import first
from torch.utils.tensorboard import SummaryWriter

from evaluate_util import get_model2, plot_metric_over_thresh, plot_model_output, evaluate_normal_model, \
    evaluate_polar_model
from train_util import dataloader_setup
from loss import TotalLoss


# following https://github.com/Project-MONAI/tutorials/blob/818673937c9c5d0b0964924b056a867238991a6a/3d_segmentation/unet_segmentation_3d_ignite.ipynb#L102
# and https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_array.py
# and https://www.kaggle.com/code/juhha1/simple-model-development-using-monai
# https://github.com/Project-MONAI/tutorials/blob/main/modules/batch_output_transform.ipynb
# https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe#scrollTo=uHAA3LUxD2b6


def train(config=None, leave_tqdm_and_timestamp=True):
    starttime = datetime.now()
    now_str = starttime.strftime("%Y_%m_%d__%H_%M_%S")

    # -------------
    # --- SETUP ---
    # -------------

    if not config:
        with open("config_unet.json", 'r') as file:
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

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    with open(exp_path + "config_unet.json", "w+") as outfile:
        json.dump(config, outfile, indent=4)

    if loss_config["lambda_cdr"] >= 1e-8 and polar:
        raise Exception("cdr loss and polar data are incompatible, either use normal data set or turn off cdr loss")

    train_dataloader, val_dataloader, test_dataloader, train_polar_dataloader, val_polar_dataloader, test_polar_dataloader = dataloader_setup(
        config)

    # get all architectural details from config_unet.json
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
    loss_func = TotalLoss(config, device)

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
    for epoch in tqdm(range(epochs), desc="Epochs", leave=leave_tqdm_and_timestamp):
        model.train()
        step = 0
        for batch_data in tqdm(trn_dl, desc="Batches", leave=False):
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            opt.zero_grad()
            outputs = model(inputs)
            losses = loss_func(outputs, labels)
            loss = losses[0]
            loss.backward()
            opt.step()
            epoch_len = len(trn_dl.dataset) // trn_dl.batch_size
            writer.add_scalar("train_loss/total", loss.item(), epoch_len * epoch + step)
            writer.add_scalar("train_loss/diceCe", losses[1].item(), epoch_len * epoch + step)
            writer.add_scalar("train_loss/topology", losses[2].item(), epoch_len * epoch + step)
            writer.add_scalar("train_loss/cdr", losses[3].item(), epoch_len * epoch + step)

        model.eval()
        val_loss_total = 0
        val_loss_dice = 0
        val_loss_topology = 0
        val_loss_cdr = 0
        counter = 0
        for batch_data in val_dl:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            outputs = model(inputs)
            losses = loss_func(outputs, labels)
            val_loss_total += losses[0].item()
            val_loss_dice += losses[1].item()
            val_loss_topology += losses[2].item()
            val_loss_cdr += losses[3].item()
            counter += 1
        writer.add_scalar("validation_loss/total", val_loss_total / counter, epoch)
        writer.add_scalar("validation_loss/diceCe", val_loss_dice / counter, epoch)
        writer.add_scalar("validation_loss/topology", val_loss_topology / counter, epoch)
        writer.add_scalar("validation_loss/cdr", val_loss_cdr / counter, epoch)
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
                                                                                      exp_path,
                                                                                      device)

        if not polar:
            for evaluation_dataset in ["validation", "test", "chaksu"]:
                evaluate_normal_model(config,
                                      best_threshold_per_channel,
                                      model,
                                      writer,
                                      exp_path,
                                      evaluation_dataset,
                                      device)
        else:
            for evaluation_dataset in ["validation", "test", "chaksu"]:
                evaluate_polar_model(config,
                                     best_threshold_per_channel,
                                     model,
                                     writer,
                                     exp_path,
                                     evaluation_dataset,
                                     device)

    # --------------
    # --- FINISH ---
    # --------------

    writer.close()

    if leave_tqdm_and_timestamp:
        endtime = datetime.now()
        time_diff = endtime - starttime
        hours = divmod(time_diff.total_seconds(), 3600)
        minutes = divmod(hours[1], 60)
        seconds = divmod(minutes[1], 1)
        print("Training+Evaluation took %d:%d:%d" % (hours[0], minutes[0], seconds[0]))

    return best_metric_per_channel


if __name__ == "__main__":
    train()

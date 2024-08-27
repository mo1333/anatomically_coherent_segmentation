import json
import os
from datetime import datetime
from tqdm.auto import tqdm

import torch as th
from monai.utils import first
from torch.utils.tensorboard import SummaryWriter

from evaluate_util import get_model3, plot_model_output, evaluate_topunet_model
from train_util import dataloader_setup, TopUNet, dataloader_setup_topunet
from loss import TotalLoss, TopUNetLoss



def train(config=None, leave_tqdm_and_timestamp=True):
    starttime = datetime.now()
    now_str = starttime.strftime("%Y_%m_%d__%H_%M_%S")

    # -------------
    # --- SETUP ---
    # -------------

    if not config:
        with open("config_topunet.json", 'r') as file:
            config = json.load(file)

    device = th.device(config["cuda_name"] if th.cuda.is_available() else "cpu")

    model_config = config["model_config"]
    loss_config = config["loss_config"]

    epochs = config["epochs"]
    exp_path = "experiments/" + now_str + "_" + config["experiment_name"] + "/"
    if len(config["overwrite_exp_path"]) > 0:
        exp_path = "experiments/" + config["overwrite_exp_path"] + "/"

    config["timestamp"] = now_str

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    with open(exp_path + "config_topunet.json", "w+") as outfile:
        json.dump(config, outfile, indent=4)

    train_dataloader, val_dataloader = dataloader_setup_topunet(config)

    model = TopUNet(config=config).to(device)

    opt = th.optim.Adam(model.parameters(), 1e-3)
    loss_func = TopUNetLoss(loss_config)

    # ----------------
    # --- TRAINING ---
    # ----------------

    trn_dl = train_dataloader
    val_dl = val_dataloader

    writer = SummaryWriter(log_dir=exp_path)
    for epoch in tqdm(range(epochs), desc="Epochs", leave=leave_tqdm_and_timestamp):
        model.train()
        step = 0
        for batch_data in tqdm(trn_dl, desc="Batches", leave=False):
            step += 1
            inputs, labels = batch_data[0].to(device), [label.to(device) for label in batch_data[1]]
            opt.zero_grad()
            outputs = model(inputs)
            losses = loss_func(outputs, labels)
            loss = losses[0]
            loss.backward()
            opt.step()
            epoch_len = len(trn_dl.dataset) // trn_dl.batch_size
            writer.add_scalar("train_loss/total", loss.item(), epoch_len * epoch + step)
            writer.add_scalar("train_loss/diceCe", losses[1].item(), epoch_len * epoch + step)
            writer.add_scalar("train_loss/kl", losses[2].item(), epoch_len * epoch + step)
            writer.add_scalar("train_loss/l1", losses[3].item(), epoch_len * epoch + step)

        model.eval()
        val_loss_total = 0
        val_loss_dice = 0
        val_loss_kl= 0
        val_loss_l1 = 0
        counter = 0
        for batch_data in val_dl:
            inputs, labels = batch_data[0].to(device), [label.to(device) for label in batch_data[1]]
            outputs = model(inputs)
            losses = loss_func(outputs, labels)
            val_loss_total += losses[0].item()
            val_loss_dice += losses[1].item()
            val_loss_kl += losses[2].item()
            val_loss_l1 += losses[3].item()
            counter += 1
        writer.add_scalar("validation_loss/total", val_loss_total / counter, epoch)
        writer.add_scalar("validation_loss/diceCe", val_loss_dice / counter, epoch)
        writer.add_scalar("validation_loss/kl", val_loss_kl / counter, epoch)
        writer.add_scalar("validation_loss/l1", val_loss_l1 / counter, epoch)
        writer.add_image("sample output channel 1", outputs[0][0, 1], global_step=epoch, dataformats="HW")
        writer.add_image("sample output channel 2", outputs[0][0, 2], global_step=epoch, dataformats="HW")

    th.save(model.state_dict(), exp_path + "model.pt")

    writer.close()

    # ------------------
    # --- EVALUATION ---
    # ------------------
    best_metric_per_channel = [0]*2
    if bool(config["evaluate_after_training"]):
        model = get_model3(exp_path, config, overwrite_device_to_cpu=True)

        img, seg = first(val_dl)
        output_images = model(img)

        plot_model_output((img[:, :3] / 255,
                           output_images[0][0].detach().cpu(),
                           seg[0]),
                          exp_path + "model_output.png")

        model = get_model3(exp_path, config)

        for evaluation_dataset in ["validation", "test", "chaksu"]:
            evaluate_topunet_model(config,
                                   model,
                                   writer,
                                   exp_path,
                                   dataset=evaluation_dataset,
                                   device=device)


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

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

from evaluate_util import get_model2, plot_metric_over_thresh, plot_model_output
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
    batch_size = config["batch_size"]
    exp_path = "experiments/" + now_str + "_" + config["experiment_name"] + "/"
    if len(config["overwrite_exp_path"]) > 0:
        exp_path = "experiments/" + config["overwrite_exp_path"] + "/"

    config["timestamp"] = now_str

    transformer_train = Compose([LoadImage(image_only=True),
                                 EnsureChannelFirst(),
                                 ScaleIntensity()])

    transformer_val = Compose([LoadImage(image_only=True),
                               EnsureChannelFirst(),
                               ScaleIntensity()])

    train_image_path = "data/REFUGE2/Train/Images/"
    train_dm_path = "data/REFUGE2/Train/Disc_Masks/"
    test_image_path = "data/REFUGE2/Test/Images/"
    test_dm_path = "data/REFUGE2/Test/Disc_Masks/"
    val_image_path = "data/REFUGE2/Validation/Images/"
    val_dm_path = "data/REFUGE2/Validation/Disc_Masks/"

    train_file_names_img = sorted([train_image_path + file for file in os.listdir(train_image_path)])
    train_file_names_seg = sorted([train_dm_path + file for file in os.listdir(train_dm_path)])

    # when specified in config, we only use a certain percentage of training data for training
    if config["perc_data_used"] < 1.0:
        used_indices = np.random.choice(len(train_file_names_img),
                                        size=int(len(train_file_names_img) * config["perc_data_used"]),
                                        replace=False)
        train_file_names_img = [file for i, file in enumerate(train_file_names_img) if i in used_indices]
        train_file_names_seg = [file for i, file in enumerate(train_file_names_seg) if i in used_indices]

    train_data = ArrayDataset(img=train_file_names_img,
                              img_transform=transformer_train,
                              seg=train_file_names_seg,
                              seg_transform=transformer_train)

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=12,
                                  pin_memory=th.cuda.is_available())

    val_data = ArrayDataset(img=sorted([val_image_path + file for file in os.listdir(val_image_path)]),
                            img_transform=transformer_val,
                            seg=sorted([val_dm_path + file for file in os.listdir(val_dm_path)]),
                            seg_transform=transformer_val)

    val_dataloader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=12,
                                pin_memory=th.cuda.is_available())

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

    # trainer = ignite.engine.create_supervised_trainer(model,
    #                                                   opt,
    #                                                   loss,
    #                                                   device,
    #                                                   False,
    #                                                   output_transform=lambda x, y, y_pred, loss: (x, y, y_pred, loss))
    # writer = SummaryWriter(log_dir=exp_path)
    #
    # # Record train loss
    # train_tb_stats_handler = TensorBoardStatsHandler(log_dir=exp_path,
    #                                                  summary_writer=writer,
    #                                                  tag_name="train loss",
    #                                                  output_transform=lambda output: output[3].item())  # output[3] = loss
    # train_tb_stats_handler.attach(trainer)
    #
    # # Record validation dice metric
    # metric = MeanDice()
    # val_metric = {"validation score": metric}
    # evaluator = ignite.engine.create_supervised_evaluator(
    #     model,
    #     val_metric,
    #     device,
    #     True
    # )
    #
    # val_tb_stats_handler = TensorBoardStatsHandler(log_dir=exp_path,
    #                                                summary_writer=writer,
    #                                                tag_name="validation loss")
    # val_tb_stats_handler.attach(evaluator)
    #
    # @trainer.on(ignite.engine.Events.EPOCH_COMPLETED())
    # def run_intrain_val(engine):
    #     evaluator.run(val_dataloader)
    #
    # # Record example images from
    # train_tb_image_handler = TensorBoardImageHandler(log_dir=exp_path,
    #                                                  summary_writer=writer,
    #                                                  output_transform=lambda output: output[2][0])  # output[2] = y_pred
    #
    # train_tb_image_handler.attach(trainer)
    #
    # # Save the current model
    # checkpoint_handler = ignite.handlers.ModelCheckpoint(exp_path, "net", n_saved=1, require_empty=False)
    # trainer.add_event_handler(
    #     event_name=ignite.engine.Events.EPOCH_COMPLETED,
    #     handler=checkpoint_handler,
    #     to_save={"trainer": trainer,
    #              "net": model,
    #              "opt": opt}
    # )
    #
    # ProgressBar(persist=False).attach(trainer)
    # trainer.run(train_dataloader, epochs)

    writer = SummaryWriter(log_dir=exp_path)
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        model.train()
        step = 0
        for batch_data in tqdm(train_dataloader, desc="Batches", leave=False):
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            opt.step()
            epoch_len = len(train_data) // train_dataloader.batch_size
            writer.add_scalar("train loss", loss.item(), epoch_len * epoch + step)

        model.eval()
        val_loss = 0
        for batch_data in val_dataloader:
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

        img, seg = first(val_dataloader)
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
        best_metric_per_channel = plot_metric_over_thresh(config,
                                                          metric,
                                                          model,
                                                          val_dataloader,
                                                          writer,
                                                          exp_path + "thresh_variation.png")

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

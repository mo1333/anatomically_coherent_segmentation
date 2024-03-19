import json
import os
from datetime import datetime

import ignite
import torch as th
from ignite.contrib.handlers import ProgressBar
from monai.data import ArrayDataset
from monai.handlers import TensorBoardStatsHandler, TensorBoardImageHandler, from_engine
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Resize, EnsureChannelFirst, LoadImage, Compose, ScaleIntensity
from monai.utils import first
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluate_util import get_model, plot_metric_over_thresh, plot_model_output
from loss import TotalLoss


# following https://github.com/Project-MONAI/tutorials/blob/818673937c9c5d0b0964924b056a867238991a6a/3d_segmentation/unet_segmentation_3d_ignite.ipynb#L102
# and https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_array.py
# and https://www.kaggle.com/code/juhha1/simple-model-development-using-monai
# https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe#scrollTo=uHAA3LUxD2b6


def train():
    starttime = datetime.now()
    now_str = starttime.strftime("%Y_%m_%d__%H_%M_%S")

    # -------------
    # --- SETUP ---
    # -------------

    with open("config.json", 'r') as file:
        config = json.load(file)

    device = th.device(config["cuda_name"] if th.cuda.is_available() else "cpu")

    model_config = config["model_config"]
    loss_config = config["loss_config"]

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    image_size = config["image_size"]  # make smaller to use on Laptop
    exp_path = "experiments/" + now_str + "/"

    transformer_train = Compose([LoadImage(image_only=True),
                                 EnsureChannelFirst(),
                                 ScaleIntensity(),
                                 Resize(image_size)])

    transformer_val = Compose([LoadImage(image_only=True),
                               EnsureChannelFirst(),
                               ScaleIntensity(),
                               Resize(image_size)])

    train_image_path = "data/REFUGE2/Train/Images/"
    train_dm_path = "data/REFUGE2/Train/Disc_Masks/"
    test_image_path = "data/REFUGE2/Test/Images/"
    test_dm_path = "data/REFUGE2/Test/Disc_Masks/"
    val_image_path = "data/REFUGE2/Validation/Images/"
    val_dm_path = "data/REFUGE2/Validation/Disc_Masks/"

    train_data = ArrayDataset(img=[train_image_path + file for file in os.listdir(train_image_path)],
                              img_transform=transformer_train,
                              seg=[train_dm_path + file for file in os.listdir(train_dm_path)],
                              seg_transform=transformer_train)

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=32,
                                  pin_memory=th.cuda.is_available())

    val_data = ArrayDataset(img=[val_image_path + file for file in os.listdir(val_image_path)],
                            img_transform=transformer_val,
                            seg=[val_dm_path + file for file in os.listdir(val_dm_path)],
                            seg_transform=transformer_val)

    val_dataloader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=32,
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
    loss = TotalLoss(loss_config)

    # ----------------
    # --- TRAINING ---
    # ----------------

    trainer = ignite.engine.create_supervised_trainer(model, opt, loss, device, False)

    writer = SummaryWriter(log_dir=exp_path)

    # Record the loss
    train_tb_stats_handler = TensorBoardStatsHandler(log_dir=exp_path,
                                                     summary_writer=writer,
                                                     output_transform=lambda x: x)
    train_tb_stats_handler.attach(trainer)

    # Record example output images
    train_tb_image_handler = TensorBoardImageHandler(log_dir=exp_path,
                                                     summary_writer=writer,
                                                     batch_transform=from_engine(["image", "label"]),
                                                     output_transform=from_engine(["pred"]))

    train_tb_image_handler.attach(trainer)

    # Save the current model
    checkpoint_handler = ignite.handlers.ModelCheckpoint(exp_path, "net", n_saved=1, require_empty=False)
    trainer.add_event_handler(
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={"trainer": trainer,
                 "net": model,
                 "opt": opt}
    )

    ProgressBar(persist=False).attach(trainer)
    trainer.run(train_dataloader, epochs)

    # writer = SummaryWriter(log_dir=exp_path)
    # for epoch in tqdm(range(epochs)):
    #     model.train()
    #     epoch_loss = 0
    #     step = 0
    #     for batch_data in train_dataloader:
    #         step += 1
    #         inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
    #         opt.zero_grad()
    #         outputs = model(inputs)
    #         loss = loss_func(outputs, labels)
    #         loss.backward()
    #         opt.step()
    #         epoch_loss += loss.item()
    #         epoch_len = len(train_data) // train_dataloader.batch_size
    #         writer.add_scalar("train loss", loss.item(), epoch_len * epoch + step)
    #
    # writer.close()

    # ------------------
    # --- EVALUATION ---
    # ------------------

    if bool(config["evaluate_after_training"]):
        model, _ = get_model(exp_path, config)

        img, seg = first(val_dataloader)
        output_images = model(img)
        plot_model_output((img,
                           th.nn.functional.softmax(output_images[0].detach(), dim=0),
                           seg),
                          exp_path + "model_output.png")

        metric = DiceMetric()
        plot_metric_over_thresh(metric,
                                th.sigmoid(output_images.detach()),
                                seg,
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


if __name__ == "__main__":
    train()

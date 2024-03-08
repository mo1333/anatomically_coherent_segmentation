import os
from datetime import datetime
import json

import ignite
import torch as th
from monai.data import ArrayDataset
from monai.handlers import TensorBoardStatsHandler
from monai.losses import DiceLoss, DiceFocalLoss
from monai.networks.nets import UNet
from monai.transforms import Resize, EnsureChannelFirst, LoadImage, Compose
from monai.utils import first
from torch.utils.data import DataLoader

# following https://github.com/Project-MONAI/tutorials/blob/818673937c9c5d0b0964924b056a867238991a6a/3d_segmentation/unet_segmentation_3d_ignite.ipynb#L102
# https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe#scrollTo=uHAA3LUxD2b6


def train():
    now_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    device_str = "cuda" if th.cuda.is_available() else "cpu"
    device = th.device(device_str)

    with open("config.json", 'r') as file:
        config = json.load(file)

    model_config = config["model_config"]

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    image_size = config["image_size"]  # make smaller to use on Laptop
    exp_path = "experiments/" + now_str + "/"

    transformer = Compose([LoadImage(image_only=True),
                           EnsureChannelFirst(),
                           Resize(image_size)])

    train_image_path = "data/REFUGE2/Train/Images/"
    train_dm_path = "data/REFUGE2/Train/Disc_Masks/"
    test_image_path = "data/REFUGE2/Test/Images/"
    test_dm_path = "data/REFUGE2/Test/Disc_Masks/"
    val_image_path = "data/REFUGE2/Validation/Images/"
    val_dm_path = "data/REFUGE2/Validation/Disc_Masks/"

    train_data = ArrayDataset(img=[train_image_path + file for file in os.listdir(train_image_path)],
                              img_transform=transformer,
                              seg=[train_dm_path + file for file in os.listdir(train_dm_path)],
                              seg_transform=transformer)

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=th.cuda.is_available(),
                                  pin_memory_device=device_str)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    with open(exp_path + "config.json", "w+") as outfile:
        json.dump(config, outfile)

    model = UNet(
        spatial_dims=model_config["spatial_dims"],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        channels=model_config["channels"],
        strides=model_config["strides"],
        num_res_units=model_config["num_res_units"]
    ).to(device)

    opt = th.optim.Adam(model.parameters(), 1e-3)
    loss = DiceLoss()
    trainer = ignite.engine.create_supervised_trainer(model, opt, loss, device, False)

    # Record the loss
    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=exp_path, output_transform=lambda x: x)
    train_tensorboard_stats_handler.attach(trainer)

    # Save the current model
    checkpoint_handler = ignite.handlers.ModelCheckpoint(exp_path, "net", n_saved=1, require_empty=False)
    trainer.add_event_handler(
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={"net": model, "opt": opt},
    )

    trainer.run(train_dataloader, epochs)

if __name__ == "__main__":
    train()

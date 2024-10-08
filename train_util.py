import os
from typing import Tuple

import numpy as np
import torch as th
from monai.data import ArrayDataset
from monai.networks.nets import UNet
from monai.networks.blocks import UnetResBlock
from monai.transforms import EnsureChannelFirst, LoadImage, Compose, ScaleIntensity
from torch.utils.data import DataLoader

from dataset import DatasetTopUNet


def setup_loader(config, file_names_img, file_names_seg, transformer, shuffle, num_workers=12):
    dataset = ArrayDataset(img=file_names_img,
                           img_transform=transformer,
                           seg=file_names_seg,
                           seg_transform=transformer)

    dataloader = DataLoader(dataset,
                            batch_size=config["batch_size"],
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=th.cuda.is_available())

    return dataloader


def dataloader_setup(config):
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

    train_image_path_polar = "data_polar/REFUGE2/Train/Images/"
    train_dm_path_polar = "data_polar/REFUGE2/Train/Disc_Masks/"
    test_image_path_polar = "data_polar/REFUGE2/Test/Images/"
    test_dm_path_polar = "data_polar/REFUGE2/Test/Disc_Masks/"
    val_image_path_polar = "data_polar/REFUGE2/Validation/Images/"
    val_dm_path_polar = "data_polar/REFUGE2/Validation/Disc_Masks/"

    train_file_names_img = sorted([train_image_path + file for file in os.listdir(train_image_path)])
    train_file_names_seg = sorted([train_dm_path + file for file in os.listdir(train_dm_path)])

    train_file_names_img_polar = sorted([train_image_path_polar + file for file in os.listdir(train_image_path_polar)])
    train_file_names_seg_polar = sorted([train_dm_path_polar + file for file in os.listdir(train_dm_path_polar)])

    # when specified in config, we only use a certain percentage of training data for training
    if config["perc_data_used"] < 1.0:
        used_indices = np.random.choice(len(train_file_names_img),
                                        size=int(len(train_file_names_img) * config["perc_data_used"]),
                                        replace=False)
        train_file_names_img = [file for i, file in enumerate(train_file_names_img) if i in used_indices]
        train_file_names_seg = [file for i, file in enumerate(train_file_names_seg) if i in used_indices]

        train_file_names_img_polar = [file for i, file in enumerate(train_file_names_img_polar) if i in used_indices]
        train_file_names_seg_polar = [file for i, file in enumerate(train_file_names_seg_polar) if i in used_indices]

    train_dataloader = setup_loader(config,
                                    train_file_names_img,
                                    train_file_names_seg,
                                    transformer_train,
                                    shuffle=True,
                                    num_workers=config["num_workers"])

    val_dataloader = setup_loader(config,
                                  sorted([val_image_path + file for file in os.listdir(val_image_path)]),
                                  sorted([val_dm_path + file for file in os.listdir(val_dm_path)]),
                                  transformer_val,
                                  shuffle=False,
                                  num_workers=config["num_workers"])

    test_dataloader = setup_loader(config,
                                   sorted([test_image_path + file for file in os.listdir(test_image_path)]),
                                   sorted([test_dm_path + file for file in os.listdir(test_dm_path)]),
                                   transformer_val,
                                   shuffle=False,
                                   num_workers=config["num_workers"])

    train_polar_dataloader = setup_loader(config,
                                          train_file_names_img_polar,
                                          train_file_names_seg_polar,
                                          transformer_train,
                                          shuffle=True,
                                          num_workers=config["num_workers"])

    val_polar_dataloader = setup_loader(config,
                                        sorted(
                                            [val_image_path_polar + file for file in os.listdir(val_image_path_polar)]),
                                        sorted([val_dm_path_polar + file for file in os.listdir(val_dm_path_polar)]),
                                        transformer_val,
                                        shuffle=False,
                                        num_workers=config["num_workers"])

    test_polar_dataloader = setup_loader(config,
                                         sorted([test_image_path_polar + file for file in
                                                 os.listdir(test_image_path_polar)]),
                                         sorted([test_dm_path_polar + file for file in os.listdir(test_dm_path_polar)]),
                                         transformer_val,
                                         shuffle=False,
                                         num_workers=config["num_workers"])

    return train_dataloader, val_dataloader, test_dataloader, train_polar_dataloader, val_polar_dataloader, test_polar_dataloader


def eval_dataloader_setup(dataset):
    if dataset not in ["validation", "test", "chaksu"]:
        raise ValueError("No validation for " + dataset + " possible!")
    transformer_val = Compose([LoadImage(image_only=True),
                               EnsureChannelFirst(),
                               ScaleIntensity()])

    num_workers = 0 # num_workers greater than 0 causes troubles on some servers

    if dataset == "validation":
        image_path = "data/REFUGE2/Validation/Images/"
        dm_path = "data/REFUGE2/Validation/Disc_Masks/"
        image_path_polar = "data_polar/REFUGE2/Validation/Images/"
        dm_path_polar = "data_polar/REFUGE2/Validation/Disc_Masks/"
    elif dataset == "test":
        image_path = "data/REFUGE2/Test/Images/"
        dm_path = "data/REFUGE2/Test/Disc_Masks/"
        image_path_polar = "data_polar/REFUGE2/Test/Images/"
        dm_path_polar = "data_polar/REFUGE2/Test/Disc_Masks/"
        num_workers = 0  # test dataset with polar part seems to be incompatible with multiple workers
    elif dataset == "chaksu":
        image_path = "data/CHAKSU/Images/"
        dm_path = "data/CHAKSU/Disc_Masks/"
        image_path_polar = "data_polar/CHAKSU/Images/"
        dm_path_polar = "data_polar/CHAKSU/Disc_Masks/"
        num_workers = 0  # chaksu dataset seems to be incompatible with multiple workers

    config = {"batch_size": 1}

    val_dataloader = setup_loader(config,
                                  sorted([image_path + file for file in os.listdir(image_path)]),
                                  sorted([dm_path + file for file in os.listdir(dm_path)]),
                                  transformer_val,
                                  shuffle=False,
                                  num_workers=num_workers)

    val_polar_dataloader = setup_loader(config,
                                        sorted([image_path_polar + file for file in os.listdir(image_path_polar)]),
                                        sorted([dm_path_polar + file for file in os.listdir(dm_path_polar)]),
                                        transformer_val,
                                        shuffle=False,
                                        num_workers=num_workers)

    names = sorted(os.listdir(image_path))

    return val_dataloader, val_polar_dataloader, names


def val_topunet_dataloader_setup(dataset="validation"):
    config = {"batch_size": 1,
              "perc_data_used": 1.0}

    val_dataloader, _, names = eval_dataloader_setup(dataset)
    _, val_topunet_dataloader = dataloader_setup_topunet(config, dataset)

    return val_dataloader, val_topunet_dataloader, names


class TopUNet(th.nn.Module):
    def __init__(self, config, overwrite_device_to_cpu=False):
        super(TopUNet, self).__init__()
        model_config = config["model_config"]
        self.config = config
        self.device = th.device(
            config["cuda_name"] if th.cuda.is_available() and not overwrite_device_to_cpu else "cpu")
        self.unet = UNet(spatial_dims=model_config["spatial_dims"],
                         # additional in channels encode pixel positions, one for each spatial dimension
                         in_channels=model_config["in_channels"] + model_config["additional_in_channels"],
                         out_channels=model_config["channels"][0],
                         channels=model_config["channels"],
                         strides=model_config["strides"],
                         kernel_size=model_config["kernel_size"],
                         up_kernel_size=model_config["up_kernel_size"],
                         num_res_units=model_config["num_res_units"],
                         act=model_config["activation"])

        self.conv_m = th.nn.Sequential(
            UnetResBlock(
                spatial_dims=model_config["spatial_dims"],
                in_channels=model_config["channels"][0],
                out_channels=model_config["channels_conv_m"],
                kernel_size=model_config["kernel_size"],
                stride=1,
                norm_name='INSTANCE',
                act_name=model_config["activation"]),
            th.nn.Conv2d(
                in_channels=model_config["channels_conv_m"],
                out_channels=model_config["out_channels"],
                kernel_size=1,
                stride=1
            )
        )

        self.conv_s = th.nn.Sequential(
            UnetResBlock(
                spatial_dims=model_config["spatial_dims"],
                in_channels=model_config["channels"][0],
                out_channels=model_config["channels_conv_m"],
                kernel_size=model_config["kernel_size"],
                stride=1,
                norm_name='INSTANCE',
                act_name=model_config["activation"]),
            th.nn.Conv2d(
                in_channels=model_config["channels_conv_m"],
                out_channels=model_config["number_anatomical_layers"],
                kernel_size=1,
                stride=1
            )
        )

    def forward(self, input) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        unet_output = self.unet(input)
        pixel_wise_labeling = th.softmax(self.conv_m(unet_output), dim=1)
        qs = th.softmax(self.conv_s(unet_output),
                        dim=3)  # take row-wise soft max (column-wise in paper, but we have rotated images!)
        B, C, H, W = qs.shape
        vector = th.arange(1, W + 1).float().to(self.device)  # [1, 2, 3, ..., W]
        s = th.einsum('bchw,w->bch', qs, vector)

        # Iterate through the channels starting from the second channel
        for j in range(1, C):
            s[:, j, :] = s[:, j - 1, :] + th.nn.functional.relu(s[:, j, :] - s[:, j - 1, :])
        return pixel_wise_labeling, qs, s


def dataloader_setup_topunet(config, eval_dataset="validation"):
    training_data = DatasetTopUNet(
        "data_topunet/REFUGE2/Train/Images/",
        "data_topunet/REFUGE2/Train/Disc_Masks/",
        "data_topunet/REFUGE2/Train/q_Masks/",
        "data_topunet/REFUGE2/Train/s_Masks/",
        config
    )
    train_dataloader = DataLoader(training_data, batch_size=config["batch_size"], shuffle=True)

    if eval_dataset not in ["validation", "test", "chaksu"]:
        raise ValueError("No validation for " + eval_dataset + " possible!")

    if eval_dataset == "validation":
        path = "data_topunet/REFUGE2/Validation/"
    elif eval_dataset == "test":
        path = "data_topunet/REFUGE2/Test/"
    elif eval_dataset == "chaksu":
        path = "data_topunet/CHAKSU/"
    val_data = DatasetTopUNet(
        path + "Images/",
        path + "Disc_Masks/",
        path + "q_Masks/",
        path + "s_Masks/"
    )
    val_dataloader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)
    return train_dataloader, val_dataloader

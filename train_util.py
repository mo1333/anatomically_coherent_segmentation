import os

import numpy as np
import torch as th
from monai.data import ArrayDataset
from monai.transforms import EnsureChannelFirst, LoadImage, Compose, ScaleIntensity
from torch.utils.data import DataLoader


def setup_loader(config, file_names_img, file_names_seg, transformer, shuffle):
    dataset = ArrayDataset(img=file_names_img,
                           img_transform=transformer,
                           seg=file_names_seg,
                           seg_transform=transformer)

    dataloader = DataLoader(dataset,
                            batch_size=config["batch_size"],
                            shuffle=shuffle,
                            num_workers=12,
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
                                    shuffle=True)

    val_dataloader = setup_loader(config,
                                  sorted([val_image_path + file for file in os.listdir(val_image_path)]),
                                  sorted([val_dm_path + file for file in os.listdir(val_dm_path)]),
                                  transformer_val,
                                  shuffle=False)

    test_dataloader = setup_loader(config,
                                   sorted([test_image_path + file for file in os.listdir(test_image_path)]),
                                   sorted([test_dm_path + file for file in os.listdir(test_dm_path)]),
                                   transformer_val,
                                   shuffle=False)

    train_polar_dataloader = setup_loader(config,
                                          train_file_names_img_polar,
                                          train_file_names_seg_polar,
                                          transformer_train,
                                          shuffle=True)

    val_polar_dataloader = setup_loader(config,
                                        sorted([val_image_path_polar + file for file in os.listdir(val_image_path_polar)]),
                                        sorted([val_dm_path_polar + file for file in os.listdir(val_dm_path_polar)]),
                                        transformer_val,
                                        shuffle=False)

    test_polar_dataloader = setup_loader(config,
                                         sorted([test_image_path_polar + file for file in os.listdir(test_image_path_polar)]),
                                         sorted([test_dm_path_polar + file for file in os.listdir(test_dm_path_polar)]),
                                         transformer_val,
                                         shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, train_polar_dataloader, val_polar_dataloader, test_polar_dataloader

import os
from datetime import datetime

import ignite
import torch as th
from monai.data import ArrayDataset
from monai.handlers import TensorBoardStatsHandler
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import Resize, EnsureChannelFirst, LoadImage, Compose
from monai.utils import first
from torch.utils.data import DataLoader

# following https://github.com/Project-MONAI/tutorials/blob/818673937c9c5d0b0964924b056a867238991a6a/3d_segmentation/unet_segmentation_3d_ignite.ipynb#L102
# https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe#scrollTo=uHAA3LUxD2b6

now_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

device_str = "cuda" if th.cuda.is_available() else "cpu"
device = th.device(device_str)
epochs = 1
batch_size = 8
new_image_size = (16, 16)  # make smaller to use on Laptop
exp_path = "experiments/" + now_str + "/"

transformer = Compose([LoadImage(image_only=True),
                       EnsureChannelFirst(),
                       Resize(new_image_size)])

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

im, seg = first(train_dataloader)

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

opt = th.optim.Adam(model.parameters(), 1e-3)
loss = DiceLoss(sigmoid=True)
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

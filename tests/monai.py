import numpy as np
import torch as th

# build https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe#scrollTo=uHAA3LUxD2b6
# and https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb
# but with a UNet class implementation
# and think aout how to inlcude the topology loss

loss = th.nn.CrossEntropyLoss()

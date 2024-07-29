import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np

class DatasetTopUNet(Dataset):
    def __init__(self, input_path, seg_target_path, q_target_path, s_target_path, config={"perc_data_used": 1.0}):
        self.input_path = input_path
        self.seg_target_path = seg_target_path
        self.q_target_path = q_target_path
        self.s_target_path = s_target_path
        self.file_names = sorted(os.listdir(input_path))

        if config["perc_data_used"] < 1.0:
            used_indices = np.random.choice(len(self.file_names),
                                            size=int(len(self.file_names) * config["perc_data_used"]),
                                            replace=False)
            reduced_file_names = [self.file_names[i] for i in used_indices]
            self.file_names = sorted(reduced_file_names)



    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = read_image(self.input_path + self.file_names[index])

        x_coords = torch.arange(1, img.size(2) + 1)
        x_coords = x_coords.expand(img.size(1), x_coords.size(0)) / img.size(2)
        x_coords = x_coords.unsqueeze(dim=0)

        y_coords = torch.arange(1, img.size(1) + 1)
        y_coords = y_coords.expand(y_coords.size(0), img.size(2)) / img.size(1)
        y_coords = y_coords.unsqueeze(dim=0)

        img_coords = torch.cat((img, x_coords, y_coords), dim=0)

        seg = read_image(self.seg_target_path + self.file_names[index])
        q = read_image(self.q_target_path + self.file_names[index]).float()[1:] / 255
        s = np.load(self.s_target_path + self.file_names[index][:-3] + "npy")
        s = torch.from_numpy(s).float()

        return img_coords, (seg, q, s)



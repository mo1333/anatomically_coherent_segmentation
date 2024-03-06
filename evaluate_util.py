import json
import torch as th
from monai.handlers import CheckpointLoader
from monai.networks.nets import UNet


def get_model(exp_path):
    with open(exp_path + "model_config.json", 'r') as file:
        model_config = json.load(file)

    model = UNet(
        spatial_dims=model_config["spatial_dims"],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        channels=model_config["channels"],
        strides=model_config["strides"],
        num_res_units=model_config["num_res_units"]
    )
    opt = th.optim.Adam(model.parameters(), 1e-3)
    save_dict = {
        "net": model,
        "opt": opt
    }
    map_location = "cpu"
    handler = CheckpointLoader(load_path=exp_path, load_dict=save_dict, map_location=map_location, strict=True)

    return model, opt


import torch as th
from monai.handlers import CheckpointLoader
from monai.networks.nets import UNet


def get_model(exp_path, model_config):

    # make sure loading is backwards compatible
    if "activation" not in model_config.keys():
        model_config["activation"] = "PReLU"
    if "kernel_size" not in model_config.keys():
        model_config["kernel_size"] = 3
    if "up_kernel_size" not in model_config.keys():
        model_config["up_kernel_size"] = 3

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
    )

    opt = th.optim.Adam(model.parameters(), 1e-3)
    save_dict = {
        "net": model,
        "opt": opt
    }
    handler = CheckpointLoader(load_path=exp_path, load_dict=save_dict, map_location="cpu", strict=True)

    return model, opt


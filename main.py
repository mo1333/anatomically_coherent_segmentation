import json
import os
from tqdm.auto import tqdm

from simulated_data_shortage import simulate

def go_through_configs():
    config_paths = os.listdir("configs/")
    for path in tqdm(config_paths, desc="experiments"):
        with open("configs/" + path + "/config_sds.json", 'r') as file:
            config_sds = json.load(file)

        if config_sds["model"] == "unet":
            with open("configs/" + path + "/config_unet.json", 'r') as file:
                config = json.load(file)
        elif config_sds["model"] == "topunet":
            with open("configs/" + path + "/config_topunet.json", 'r') as file:
                config = json.load(file)

        simulate(config_sds, config, False)


if __name__ == "__main__":
    go_through_configs()
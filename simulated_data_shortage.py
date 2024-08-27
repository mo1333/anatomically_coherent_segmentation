import json
import numpy as np
import pandas as pd

from train import train as train_unet
from train_topunet import train as train_topunet


def simulate(config_sds=None, config_model=None, leave_tqdm_and_timestamp=True):
    if not config_sds:
        with open("config_sds.json", 'r') as file:
            config_sds = json.load(file)

    experiment_name = config_sds["experiment_name"]
    data_shortage_config = config_sds["data_shortage_config"]

    best_metric_per_percentage = []

    percentages = np.linspace(data_shortage_config["start_percentage"],
                              data_shortage_config["end_percentage"],
                              num=data_shortage_config["number_of_experiments"])
    percentages = [round(percentage, 2) for percentage in percentages]

    config = config_model
    if config_sds["model"] == "unet":
        train = train_unet
        if not config_model:
            with open("config_unet.json", 'r') as file:
                config = json.load(file)

    elif config_sds["model"] == "topunet":
        train = train_topunet
        if not config_model:
            with open("config_topunet.json", 'r') as file:
                config = json.load(file)

        else:
            raise Exception("Model " + config_sds["model"] + " not implemented! Pick either 'unet' or 'topunet'")

    for percentage in percentages:
        config["experiment_name"] = experiment_name + "_" + str(int(percentage * 100))
        config["overwrite_exp_path"] = experiment_name + "/" + str(int(percentage * 100))
        config["perc_data_used"] = percentage
        best_metric_per_percentage.append(train(config, leave_tqdm_and_timestamp))

    joined_frame = [[perc, m[0], m[1]] for perc, m in zip(percentages, best_metric_per_percentage)]
    output_frame = pd.DataFrame(joined_frame, columns=['percentage', 'optic_cup_performance', 'optic_disc_performance'])
    output_frame.to_csv("experiments/" + experiment_name + "/simulated_data_shortage_output.csv")


if __name__ == "__main__":
    simulate()

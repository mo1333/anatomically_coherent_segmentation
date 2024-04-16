import json
import numpy as np

from train import train

with open("config.json", 'r') as file:
    config = json.load(file)

experiment_name = config["experiment_name"]

percentages = np.linspace(0.1, 1.0, num=10)
for percentage in percentages:
    config["experiment_name"] = experiment_name + "_" + str(int(percentage*100))
    config["perc_data_used"] = percentage
    train(config)
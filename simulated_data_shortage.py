import json
import numpy as np
import pandas as pd

from train import train

with open("config.json", 'r') as file:
    config = json.load(file)

experiment_name = config["experiment_name"]

best_metric_per_percentage = []

percentages = np.linspace(0.1, 1.0, num=10)
for percentage in percentages:
    config["experiment_name"] = experiment_name + "_" + str(int(percentage*100))
    config["overwrite_exp_path"] = experiment_name + "/" + str(int(percentage*100))
    config["perc_data_used"] = percentage
    best_metric_per_percentage.append(train(config))

joined_frame = [[perc, m[0], m[1]] for perc, m in zip(percentages, best_metric_per_percentage)]
output_frame = pd.DataFrame(joined_frame, columns=['percentage', 'optic_cup_performance', 'optic_disc_performance'])
output_frame.to_csv("experiments/" + experiment_name + "/simulated_data_shortage_output.csv")


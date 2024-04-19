import os.path

import ignite
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from monai.handlers import CheckpointLoader
from monai.networks.nets import UNet

from loss import TotalLoss


def get_model(exp_path, config):
    # make sure loading is backwards compatible
    model_config = config["model_config"]
    loss_config = config["loss_config"]
    if "activation" not in model_config.keys():
        model_config["activation"] = "PReLU"
    if "kernel_size" not in model_config.keys():
        model_config["kernel_size"] = 3
    if "up_kernel_size" not in model_config.keys():
        model_config["up_kernel_size"] = 3

    device = "cpu"

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

    loss = TotalLoss(loss_config)

    files = os.listdir(exp_path)
    checkpoint_name = [checkpoint for checkpoint in files if checkpoint.endswith(".pt")][-1]
    trainer = ignite.engine.create_supervised_trainer(model, opt, loss, device, False)
    handler = CheckpointLoader(load_path=exp_path + checkpoint_name, load_dict=save_dict, map_location="cpu",
                               strict=True)

    handler(trainer)

    return model, opt


def get_model2(exp_path, config):
    model_config = config["model_config"]
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

    model.load_state_dict(th.load(exp_path + "model.pt"))
    model.eval()

    return model


def plot_model_output(sample, save_name):
    fig, ((img1, img2), (img3, img4), (img5, img6), (img7, img8)) = plt.subplots(4, 2, figsize=(16, 16))

    img1.set_title("original image")
    img1.imshow(sample[0][0].permute(1, 2, 0))
    img1.set_axis_off()

    img2.set_title("model output")
    img2.imshow(sample[1].permute(1, 2, 0))
    img2.set_axis_off()

    desired_output_imgs = [img3, img5, img7]
    actual_output_imgs = [img4, img6, img8]

    for i, img in enumerate(desired_output_imgs):
        img.set_title("desired output, channel %d" % (i))
        pos = img.imshow(sample[2][0, i], cmap="gray", vmin=0, vmax=1)
        img.set_axis_off()
        fig.colorbar(pos, ax=img)

    for i, img in enumerate(actual_output_imgs):
        img.set_title("model output, channel %d" % (i))
        pos = img.imshow(sample[1][i], cmap="gray", vmin=0, vmax=1)
        img.set_axis_off()
        fig.colorbar(pos, ax=img)

    plt.savefig(save_name)
    plt.show()


def plot_metric_over_thresh(config, metric, model, val_dataloader, writer, save_name):
    loss_config = config["loss_config"]
    channels_of_interest = [1, 2]
    fig, (plots) = plt.subplots(len(channels_of_interest),
                                3,
                                figsize=(10, 8))

    best_metric_per_channel = []

    for plot, j in zip(plots, channels_of_interest):
        best_metric = -1
        best_thresh = -1
        thresh_list = np.arange(0, 1, 0.01)
        m_list = []
        for thresh in thresh_list:
            m = 0
            counter = 0
            for batch_data in val_dataloader:
                inputs, labels = batch_data[0], batch_data[1]
                outputs = model(inputs)
                if bool(loss_config["sigmoid"]):
                    y_pred = th.sigmoid(outputs)
                if bool(loss_config["softmax"]):
                    y_pred = th.softmax(outputs, dim=1)
                y_pred_only1channel = th.unsqueeze(th.tensor(y_pred[:, j] >= thresh), 1)
                y_true_only1channel = th.unsqueeze(labels[:, j], 1)
                m += th.mean(metric(y_pred_only1channel,
                                    y_true_only1channel))
                counter += 1
            m = m / counter  # get the mean instead of sum for easy to interpret average metric

            m_list.append(m)
            if best_metric < m:
                best_metric = m
                best_thresh = thresh

        best_metric_per_channel.append(best_metric)
        for i in range(len(list(thresh_list))):
            writer.add_scalar("Dice Score Channel " + str(j), m_list[i], i)
        plot[0].plot(thresh_list, m_list)
        plot[0].set_title("metric over threshold")

        plot[1].imshow(y_pred[0, j] >= best_thresh,
                       cmap="gray")  # take the first image in the batch and show thresholded version of model output
        plot[1].set_axis_off()

        plot[2].imshow(labels[0, j], cmap="gray")
        plot[2].set_axis_off()
    plt.savefig(save_name)
    plt.show()

    return best_metric_per_channel

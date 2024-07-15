import os.path
import pickle

from tqdm.auto import tqdm
import ignite
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from monai.handlers import CheckpointLoader
from monai.networks.nets import UNet
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import DiceMetric

from loss import TotalLoss, get_vertical_diameter
from train_util import val_dataloader_setup



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
    plt.close(fig)


def plot_metric_over_thresh(config, metric, model, val_dataloader, writer, exp_path, device=th.device("cpu")):
    save_name_plot = exp_path + "thresh_variation.png"
    save_name_haus = exp_path + "hausdorff.pickle"
    save_name_dice = exp_path + "dice.pickle"
    save_name_diameters = exp_path + "diameters.pickle"
    loss_config = config["loss_config"]
    device_model = model.to(device)

    haus = HausdorffDistanceMetric(reduction=None)
    dice = DiceMetric(reduction=None)

    channels_of_interest = [1, 2]
    fig, (plots) = plt.subplots(len(channels_of_interest),
                                3,
                                figsize=(10, 8))

    best_metric_per_channel = []
    best_threshold_per_channel = []
    hausdorff_metric_per_channel = []
    dice_metric_per_channel = []
    diameters_per_channel = {"label": [], "pred": []}
    for plot, j in zip(plots, channels_of_interest):
        best_metric = -1
        best_thresh = -1
        thresh_list = np.linspace(0, 1, 100)
        m_list = []
        y_pred = []
        y_true = []
        for batch_data in val_dataloader:
            inputs, labels = batch_data[0].to(device), batch_data[1]
            output = device_model(inputs)
            if bool(loss_config["sigmoid"]):
                output = th.sigmoid(output)
            if bool(loss_config["softmax"]):
                output = th.softmax(output, dim=1)
            y_pred.append(output.detach().cpu().numpy())
            y_true.append(labels.numpy())

        y_pred = np.array(y_pred)
        y_pred = np.vstack(y_pred)  # merge all batches to get (#samples, 3, 512, 512) as shape
        y_true = np.array(y_true)
        y_true = np.vstack(y_true)

        y_true_only1channel = th.unsqueeze(th.tensor(y_true[:, j] >= 0.5), 1)
        for thresh in tqdm(thresh_list, desc="Finding threshold for channel %d" % j, leave=False):
            y_pred_only1channel = th.unsqueeze(th.tensor(y_pred[:, j] >= thresh), 1)
            """
            think about this later: When we did the rescaling of images, the label matrix got non-integer entries, 
            due to interpolation. This is why we need to write >= 0.5, since the DiceMetric wants to see
            binarized inputs. This is repeated wherever y_true is needed
            """

            m = th.mean(metric(y_pred_only1channel,
                               y_true_only1channel))
            m_list.append(m)
            if best_metric < m:
                best_metric = m
                best_thresh = thresh

        y_pred_only1channel = th.unsqueeze(th.tensor(y_pred[:, j] >= best_thresh), 1)
        best_metric_per_channel.append(best_metric)
        best_threshold_per_channel.append(best_thresh)

        hausdorff_metric_per_channel.append(haus(y_pred_only1channel, y_true_only1channel))
        dice_metric_per_channel.append(dice(y_pred_only1channel, y_true_only1channel))

        label_diameter = get_vertical_diameter(y_true_only1channel[:, 0])
        pred_diameter = get_vertical_diameter(y_pred_only1channel[:, 0])

        diameters_per_channel["label"].append(label_diameter)
        diameters_per_channel["pred"].append(pred_diameter)

        for i in range(len(list(thresh_list))):
            writer.add_scalar("Dice Score Channel " + str(j), m_list[i], i)
        plot[0].plot(thresh_list, m_list)
        plot[0].set_title("metric over threshold")

        # take the first image and show thresholded version of model output
        sample_image = y_pred[0, j] >= best_thresh
        sample_label = y_true[0, j] >= 0.5
        sample_image_torch = th.from_numpy(sample_image)[None, None, :, :]
        sample_label_torch = th.from_numpy(sample_label)[None, None, :, :]
        plot[1].set_title("prediction channel \n Sample dice score: %.3f" % metric(sample_image_torch,
                                                                                   sample_label_torch))
        plot[1].imshow(sample_image >= best_thresh, cmap="gray")
        plot[1].set_axis_off()

        plot[2].set_title("ground-truth")
        plot[2].imshow(sample_label, cmap="gray")  # and compare it to its corresponding label
        plot[2].set_axis_off()
    plt.savefig(save_name_plot)
    plt.show()
    plt.close(fig)

    with open(save_name_haus, "wb") as handle:
        pickle.dump(hausdorff_metric_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_name_dice, "wb") as handle:
        pickle.dump(dice_metric_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_name_diameters, "wb") as handle:
        pickle.dump(diameters_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return best_metric_per_channel, best_threshold_per_channel


def evaluate_polar_model(config, best_threshold_per_channel, metric, model, writer, save_name, device=th.device("cpu")):
    with open("data_polar/REFUGE2/Validation/settings.pickle", "rb") as handle:
        settings_dict = pickle.load(handle)
    plt.figure(3)
    val_dataloader, polar_val_dataloader, names = val_dataloader_setup()
    device_model = model.to(device)
    loss_config = config["loss_config"]
    channels_of_interest = [1, 2]
    metrics = [[] for _ in range(len(channels_of_interest))]
    for j, (og_batch, polar_batch) in enumerate(zip(val_dataloader, polar_val_dataloader)):
        og_image, og_labels = og_batch[0], og_batch[1]
        polar_image, polar_labels = polar_batch[0].to(device), polar_batch[1]
        output = device_model(polar_image)
        if bool(loss_config["sigmoid"]):
            output = th.sigmoid(output)
        if bool(loss_config["softmax"]):
            output = th.softmax(output, dim=1)
        output = output.detach().cpu().numpy()[0]
        output_cartesian = settings_dict[names[j]].convertToCartesianImage(np.transpose(output, (2, 1, 0)))
        output_cartesian = np.transpose(output_cartesian, (2, 0, 1))
        output_cartesian = np.expand_dims(output_cartesian, axis=0)

        # ------------------------------------
        if j == 0:
            plt.imshow(og_image[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(save_name + "og_image.png")
            plt.imshow(og_labels[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(save_name + "og_labels.png")
            plt.imshow(polar_image.detach().cpu()[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(save_name + "polar_image.png")
            plt.imshow(polar_labels[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(save_name + "polar_labels.png")
            plt.imshow(np.transpose(output, (1, 2, 0)))
            plt.title(names[j])
            plt.savefig(save_name + "output.png")
            plt.imshow(np.transpose(output_cartesian[0], (1, 2, 0)))
            plt.title(names[j])
            plt.savefig(save_name + "output_cartesian.png")
        # ------------------------------------

        for i, (channel, thresh) in enumerate(zip(channels_of_interest, best_threshold_per_channel)):
            output_only1channel = th.unsqueeze(th.tensor(output_cartesian[:, channel] >= thresh), 1)
            y_true_only1channel = th.unsqueeze(og_labels[:, channel], 1)
            metrics[i].append(th.mean(metric(output_only1channel, y_true_only1channel)))
            writer.add_image("final output channel " + str(channel), output_only1channel[0, 0], global_step=j,
                             dataformats="HW")
            writer.add_image("desired output channel " + str(channel), y_true_only1channel[0, 0], global_step=j,
                             dataformats="HW")
    metric_per_channel = [np.mean(m) for m in metrics]
    return metric_per_channel

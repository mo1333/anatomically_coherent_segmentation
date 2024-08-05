import os.path
import pickle
import re

from tqdm.auto import tqdm
import ignite
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch as th
from monai.handlers import CheckpointLoader
from monai.networks.nets import UNet
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import DiceMetric

from loss import TotalLoss, get_vertical_diameter
from train_util import eval_dataloader_setup, TopUNet, val_topunet_dataloader_setup


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


def get_model3(exp_path, config, overwrite_device_to_cpu=False):
    model = TopUNet(config=config, overwrite_device_to_cpu=overwrite_device_to_cpu)

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
    loss_config = config["loss_config"]
    device_model = model.to(device)

    channels_of_interest = [1, 2]
    fig, (plots) = plt.subplots(len(channels_of_interest),
                                3,
                                figsize=(10, 8))

    best_metric_per_channel = []
    best_threshold_per_channel = []
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

        best_metric_per_channel.append(best_metric)
        best_threshold_per_channel.append(best_thresh)

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

    return best_metric_per_channel, best_threshold_per_channel


def evaluate_normal_model(config, best_threshold_per_channel, model, writer, exp_path, dataset="validation",
                          device=th.device("cpu")):
    save_name_haus = exp_path + dataset + "_hausdorff.pickle"
    save_name_dice = exp_path + dataset + "_dice.pickle"
    save_name_diameters = exp_path + dataset + "_diameters.pickle"

    plt.figure(3)
    haus = HausdorffDistanceMetric(reduction=None)
    dice = DiceMetric(reduction=None)
    val_dataloader, _, names = eval_dataloader_setup(dataset)
    device_model = model.to(device)
    loss_config = config["loss_config"]
    channels_of_interest = [1, 2]
    dice_metric_per_channel = [[] for _ in range(len(channels_of_interest))]
    hausdorff_metric_per_channel = [[] for _ in range(len(channels_of_interest))]
    diameters_per_channel = {"label": [[], []], "pred": [[], []]}
    for j, batch in enumerate(val_dataloader):
        image, labels = batch[0].to(device), batch[1].to(device)
        output = device_model(image)
        if bool(loss_config["sigmoid"]):
            output = th.sigmoid(output)
        if bool(loss_config["softmax"]):
            output = th.softmax(output, dim=1)
        output = output.detach().cpu().numpy()[0]

        # ------------------------------------
        if j == 0:
            plt.imshow(image[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(exp_path + "image.png")

            plt.imshow(labels[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(exp_path + "labels.png")

            plt.imshow(np.transpose(output, (1, 2, 0)))
            plt.title(names[j])
            plt.savefig(exp_path + "output.png")

        # ------------------------------------

        for i, (channel, thresh) in enumerate(zip(channels_of_interest, best_threshold_per_channel)):
            output_only1channel = th.unsqueeze(th.tensor(output[:, channel] >= thresh), 1)
            y_true_only1channel = th.unsqueeze(labels[:, channel], 1)
            dice_metric_per_channel[i].append(th.mean(dice(output_only1channel, y_true_only1channel)))
            hausdorff_metric_per_channel[i].append(th.mean(haus(output_only1channel, y_true_only1channel)))

            label_diameter = get_vertical_diameter(y_true_only1channel[:, 0])
            pred_diameter = get_vertical_diameter(output_only1channel[:, 0])

            diameters_per_channel["label"][i].append(label_diameter)
            diameters_per_channel["pred"][i].append(pred_diameter)

            writer.add_image("final output channel " + str(channel), output_only1channel[0, 0], global_step=j,
                             dataformats="HW")
            writer.add_image("desired output channel " + str(channel), y_true_only1channel[0, 0], global_step=j,
                             dataformats="HW")

    with open(save_name_haus, "wb") as handle:
        pickle.dump(hausdorff_metric_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_name_dice, "wb") as handle:
        pickle.dump(dice_metric_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_name_diameters, "wb") as handle:
        pickle.dump(diameters_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_polar_model(config, best_threshold_per_channel, model, writer, exp_path, dataset="validation",
                         device=th.device("cpu")):

    if dataset not in ["validation", "test", "chaksu"]:
        raise ValueError("No validation for " + dataset + " possible!")

    if dataset == "validation":
        settings_path = "data_polar/REFUGE2/Validation/"
    elif dataset == "test":
        settings_path = "data_polar/REFUGE2/Test/"
    elif dataset == "chaksu":
        settings_path = "data_polar/CHAKSU/"

    with open(settings_path + "settings.pickle", "rb") as handle:
        settings_dict = pickle.load(handle)

    save_name_haus = exp_path + dataset + "_hausdorff.pickle"
    save_name_dice = exp_path + dataset + "_dice.pickle"
    save_name_diameters = exp_path + dataset + "_diameters.pickle"

    plt.figure(3)
    haus = HausdorffDistanceMetric(reduction=None)
    dice = DiceMetric(reduction=None)
    val_dataloader, polar_val_dataloader, names = eval_dataloader_setup(dataset)
    device_model = model.to(device)
    loss_config = config["loss_config"]
    channels_of_interest = [1, 2]
    dice_metric_per_channel = [[] for _ in range(len(channels_of_interest))]
    hausdorff_metric_per_channel = [[] for _ in range(len(channels_of_interest))]
    diameters_per_channel = {"label": [[], []], "pred": [[], []]}
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
            plt.savefig(exp_path + "og_image.png")

            plt.imshow(og_labels[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(exp_path + "og_labels.png")

            plt.imshow(polar_image.detach().cpu()[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(exp_path + "polar_image.png")

            plt.imshow(polar_labels[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(exp_path + "polar_labels.png")

            plt.imshow(np.transpose(output, (1, 2, 0)))
            plt.title(names[j])
            plt.savefig(exp_path + "output.png")

            plt.imshow(np.transpose(output_cartesian[0], (1, 2, 0)))
            plt.title(names[j])
            plt.savefig(exp_path + "output_cartesian.png")
        # ------------------------------------

        for i, (channel, thresh) in enumerate(zip(channels_of_interest, best_threshold_per_channel)):
            output_only1channel = th.unsqueeze(th.tensor(output_cartesian[:, channel] >= thresh), 1)
            y_true_only1channel = th.unsqueeze(og_labels[:, channel], 1)
            dice_metric_per_channel[i].append(th.mean(dice(output_only1channel, y_true_only1channel)))
            hausdorff_metric_per_channel[i].append(th.mean(haus(output_only1channel, y_true_only1channel)))

            label_diameter = get_vertical_diameter(y_true_only1channel[:, 0])
            pred_diameter = get_vertical_diameter(output_only1channel[:, 0])

            diameters_per_channel["label"][i].append(label_diameter)
            diameters_per_channel["pred"][i].append(pred_diameter)

            writer.add_image("final output channel " + str(channel), output_only1channel[0, 0], global_step=j,
                             dataformats="HW")
            writer.add_image("desired output channel " + str(channel), y_true_only1channel[0, 0], global_step=j,
                             dataformats="HW")

    with open(save_name_haus, "wb") as handle:
        pickle.dump(hausdorff_metric_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_name_dice, "wb") as handle:
        pickle.dump(dice_metric_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_name_diameters, "wb") as handle:
        pickle.dump(diameters_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_topunet_model(config, model, exp_path, dataset="validation", device=th.device("cpu")):
    if dataset not in ["validation", "test", "chaksu"]:
        raise ValueError("No validation for " + dataset + " possible!")

    if dataset == "validation":
        settings_path = "data_polar/REFUGE2/Validation/"
    elif dataset == "test":
        settings_path = "data_polar/REFUGE2/Test/"
    elif dataset == "chaksu":
        settings_path = "data_polar/CHAKSU/"

    with open(settings_path + "settings.pickle", "rb") as handle:
        settings_dict = pickle.load(handle)

    save_name_haus = exp_path + dataset + "_hausdorff.pickle"
    save_name_dice = exp_path + dataset + "_dice.pickle"
    save_name_diameters = exp_path + dataset + "_diameters.pickle"

    val_dataloader, val_topunet_dataloader, names = val_topunet_dataloader_setup(dataset)
    device_model = model.to(device)

    plt.figure(3)
    dice = DiceMetric(reduction=None)
    haus = HausdorffDistanceMetric(reduction=None)
    channels_of_interest = [1, 2]
    dice_metric_per_channel = [[] for _ in range(len(channels_of_interest))]
    hausdorff_metric_per_channel = [[] for _ in range(len(channels_of_interest))]
    diameters_per_channel = {"label": [[], []], "pred": [[], []]}
    for j, (og_batch, topunet_batch) in enumerate(zip(val_dataloader, val_topunet_dataloader)):
        og_image, og_labels = og_batch[0], og_batch[1]
        topunet_image, topunet_labels = topunet_batch[0].to(device), topunet_batch[1]
        output_image, output_q, output_s = device_model(topunet_image)

        output_image = th.softmax(output_image, dim=1)
        output_image = output_image.detach().cpu().numpy()[0]

        s = output_s.detach().cpu().numpy()[0]

        pred = np.arange(1, output_image.shape[1] + 1).reshape(1, -1)
        pred = np.expand_dims(np.repeat(pred, output_image.shape[2], axis=0), axis=2)
        pred = np.repeat(pred, 3, axis=2)  # 3 channel image: 0.. background, 1.. cup, 2.. disc

        pred[:, :, 0] = pred[:, :, 0] > s[1].reshape(-1, 1)  # fill background with ones where no disc is
        for i in channels_of_interest:
            pred[:, :, i] = pred[:, :, i] <= s[i - 1].reshape(-1, 1)
        pred = pred.astype(np.uint8) * 255

        output_cartesian = settings_dict[names[j]].convertToCartesianImage(pred)
        output_cartesian = np.transpose(output_cartesian, (2, 0, 1))
        output_cartesian = np.expand_dims(output_cartesian, axis=0)

        zero_padding = np.zeros(shape=(output_image.shape[1], output_image.shape[2], 1))

        # ------------------------------------
        if j == 0:
            plt.imshow(og_image[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(exp_path + "og_image.png")

            plt.imshow(og_labels[0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(exp_path + "og_labels.png")

            plt.imshow(topunet_image.detach().cpu()[0].permute(1, 2, 0)[:, :, :3].numpy() / 255)
            plt.title(names[j])
            plt.savefig(exp_path + "topunet_image.png")

            plt.imshow(topunet_labels[0][0].permute(1, 2, 0).numpy())
            plt.title(names[j])
            plt.savefig(exp_path + "topunet_seg_labels.png")

            temp_img = topunet_labels[1][0].permute(1, 2, 0).numpy()
            plt.imshow(np.concatenate((zero_padding, temp_img), axis=2))
            plt.title(names[j])
            plt.savefig(exp_path + "topunet_q_labels.png")

            plt.imshow(np.transpose(output_image, (1, 2, 0)))
            plt.title(names[j])
            plt.savefig(exp_path + "output_seg.png")

            temp_img = np.transpose(output_q.detach().cpu().numpy()[0], (1, 2, 0))
            plt.imshow(np.concatenate((zero_padding, temp_img), axis=2))
            plt.title(names[j])
            plt.savefig(exp_path + "output_q.png")

            plt.imshow(pred)
            plt.title(names[j])
            plt.savefig(exp_path + "output_using_s.png")

            plt.imshow(np.transpose(output_cartesian[0], (1, 2, 0)))
            plt.title(names[j])
            plt.savefig(exp_path + "output_cartesian.png")
        # ------------------------------------
        for i, channel in enumerate(channels_of_interest):
            output_only1channel = th.unsqueeze(th.tensor(output_cartesian[:, channel] >= 127), 1)
            y_true_only1channel = th.unsqueeze(og_labels[:, channel], 1)
            dice_metric_per_channel[i].append(th.mean(dice(output_only1channel, y_true_only1channel)))
            hausdorff_metric_per_channel[i].append(th.mean(haus(output_only1channel, y_true_only1channel)))

            label_diameter = get_vertical_diameter(y_true_only1channel[:, 0])
            pred_diameter = get_vertical_diameter(output_only1channel[:, 0])

            diameters_per_channel["label"][i].append(label_diameter)
            diameters_per_channel["pred"][i].append(pred_diameter)

    with open(save_name_haus, "wb") as handle:
        pickle.dump(hausdorff_metric_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_name_dice, "wb") as handle:
        pickle.dump(dice_metric_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_name_diameters, "wb") as handle:
        pickle.dump(diameters_per_channel, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_metrics(exp_path):
    perc_experiments = [exp_path + path + "/" for path in os.listdir(exp_path) if not (path.endswith(".csv"))]
    # Since the string ordering is '1', '10', '2',... we have to sort by the last integer in the path:
    sorted_perc_experiments = sorted(perc_experiments, key=lambda s: int(re.findall(r'(\d+)', s)[-1]))
    dices = []
    hauss = []
    diameters = []
    for perc_exp_path in sorted_perc_experiments:
        with open(perc_exp_path + "dice.pickle", "rb") as handle:
            dices.append(pickle.load(handle))

        with open(perc_exp_path + "hausdorff.pickle", "rb") as handle:
            hauss.append(pickle.load(handle))

        with open(perc_exp_path + "diameters.pickle", "rb") as handle:
            diameters.append(pickle.load(handle))

    cdrs = []
    for diam in diameters:
        cdr_label = np.array([diam["label"][0][i] / diam["label"][1][i] for i in range(len(diam["label"][0]))])
        cdr_pred = np.array([diam["pred"][0][i] / diam["pred"][1][i] for i in range(len(diam["label"][0]))])
        cdrs.append({"label": cdr_label, "pred": cdr_pred})
    return np.array(dices), np.array(hauss), cdrs


def sds_file_handling(exp_path):
    if os.path.isfile(
            exp_path + "simulated_data_shortage_output.csv"):  # check whether we have just a single experiment, or a collection
        meta_data = pd.read_csv(exp_path + "simulated_data_shortage_output.csv")

        dices, hauss, cdrs = get_metrics(exp_path)
        dice_dic = {perc: dices[i] for i, perc in enumerate(meta_data["percentage"])}
        haus_dic = {perc: hauss[i] for i, perc in enumerate(meta_data["percentage"])}

        return meta_data["percentage"], dice_dic, haus_dic, cdrs

    else:
        sub_exp_paths = [exp_path + path + "/" for path in os.listdir(exp_path) if
                         not (path.endswith(".csv") or path.endswith(".pickle") or path.endswith(".gitignore"))]
        meta_data = [pd.read_csv(path + "simulated_data_shortage_output.csv") for path in sub_exp_paths]

        metrics = [get_metrics(path) for path in sub_exp_paths]
        dices = [m[0] for m in metrics]
        hauss = [m[1] for m in metrics]
        cdrs = [m[2] for m in metrics]

        joinced_dices = np.concatenate(dices, axis=2)
        joinced_hauss = np.concatenate(hauss, axis=2)
        joined_cdrs = []
        for i, _ in enumerate(cdrs[0]):
            joined_labels = np.concatenate([cdr[i]["label"] for cdr in cdrs])
            joined_preds = np.concatenate([cdr[i]["pred"] for cdr in cdrs])
            joined_cdrs.append({"label": joined_labels, "pred": joined_preds})

        # assumption: all experiments were run with the same percentage range
        dice_dic = {perc: joinced_dices[i] for i, perc in enumerate(meta_data[0]["percentage"])}
        haus_dic = {perc: joinced_hauss[i] for i, perc in enumerate(meta_data[0]["percentage"])}
        cdrs_dic = {perc: joined_cdrs[i] for i, perc in enumerate(meta_data[0]["percentage"])}

        return meta_data[0]["percentage"], dice_dic, haus_dic, cdrs_dic


def plot_dice_mean_comparison(experiments, labels):
    experiments_colors = [("b", "cornflowerblue"), ("green", "limegreen"), ("orangered", "coral")]
    exp_paths = ["experiments/" + name + "/" for name in experiments]

    plt.rcParams["figure.figsize"] = (10, 4)

    for i in range(len(experiments)):
        percentages, dice_dic, _, _ = sds_file_handling(exp_paths[i])
        whole_dice = np.array([dice_dic[percentages[i]] for i in range(percentages.shape[0])]).squeeze()
        plt.errorbar(percentages + i / 2000,
                     whole_dice.mean(axis=2)[:, 1],
                     linestyle="-", color=experiments_colors[i][0], label=labels[i] + ", disc")
        plt.errorbar(percentages + i / 2000,
                     whole_dice.mean(axis=2)[:, 0],
                     linestyle="--", color=experiments_colors[i][1], label=labels[i] + ", cup")

    plt.xlabel("fraction of used training data")
    plt.ylabel("dice score")
    plt.legend(loc=4)  # loc 4 ... lower right
    plt.show()


def plot_haus_mean_comparison(experiments, labels):
    experiments_colors = [("b", "cornflowerblue"), ("green", "limegreen"), ("orangered", "coral")]
    exp_paths = ["experiments/" + name + "/" for name in experiments]

    plt.rcParams["figure.figsize"] = (10, 4)

    for i in range(len(experiments)):
        percentages, _, haus_dic, _ = sds_file_handling(exp_paths[i])
        whole_haus = np.array([haus_dic[percentages[i]] for i in range(percentages.shape[0])]).squeeze()
        plt.errorbar(percentages + i / 2000,
                     whole_haus.mean(axis=2)[:, 1],
                     linestyle="-", color=experiments_colors[i][0], label=labels[i] + ", disc")
        plt.errorbar(percentages + i / 2000,
                     whole_haus.mean(axis=2)[:, 0],
                     linestyle="--", color=experiments_colors[i][1], label=labels[i] + ", cup")

    plt.xlabel("fraction of used training data")
    plt.ylabel("hausdorff distance")
    plt.legend(loc=1)  # loc 1 ... upper right
    plt.show()


def plot_cdr_mae_mean_comparison(experiments, labels):
    experiments_colors = [("b", "cornflowerblue"), ("green", "limegreen"), ("orangered", "coral")]
    exp_paths = ["experiments/" + name + "/" for name in experiments]

    plt.rcParams["figure.figsize"] = (10, 4)

    for i in range(len(experiments)):
        percentages, _, _, cdrs = sds_file_handling(exp_paths[i])
        maes = np.array([np.abs(cdrs[perc]["label"] - cdrs[perc]["pred"]) for perc in percentages]).squeeze()
        plt.errorbar(percentages + i / 2000,
                     maes.mean(axis=1),
                     linestyle="-", color=experiments_colors[i][0], label=labels[i])

    plt.xlabel("fraction of used training data")
    plt.ylabel("MAE of vCDR")
    plt.legend(loc=1)  # loc 1 ... upper right
    plt.show()


def plot_dice_violin_comparison(experiments, labels_exp, width_default=0.04,
                                shift_default=30, show_disc=True, show_cup=True,
                                show_means=False, show_extrema=False):
    experiments_colors = [("b", "cornflowerblue"), ("green", "limegreen"), ("orangered", "coral")]
    exp_paths = ["experiments/" + name + "/" for name in experiments]
    patches = []
    labels = []
    plt.rcParams["figure.figsize"] = (15, 5)
    percentages, _, _, _ = sds_file_handling(exp_paths[0])
    width = width_default * list(percentages)[-1]  # widths depends on the percentage area we look at
    shift_factor = shift_default / list(percentages)[-1]  # as well as how much we want to shift the experiments

    for i in range(len(experiments)):
        percentages, dice_dic, _, _ = sds_file_handling(exp_paths[i])
        whole_dice = np.array([dice_dic[percentages[i]] for i in range(percentages.shape[0])]).squeeze()
        if show_disc:
            violin = plt.violinplot(whole_dice[:, 1].T,
                                    positions=[p + i / shift_factor for p in percentages],
                                    widths=width,
                                    showmeans=show_means,
                                    showextrema=show_extrema,
                                    side="high")
            for pc in violin['bodies']:
                pc.set_facecolor(experiments_colors[i][0])
                pc.set_edgecolor(experiments_colors[i][0])
            # for partname in ('cbars','cmins','cmaxes'):
            #     vp = violin[partname]
            #     vp.set_edgecolor(experiments_colors[i][0])
            patches.append(mpatches.Patch(color=experiments_colors[i][0]))
            labels.append(labels_exp[i] + ", disc")

        if show_cup:
            violin = plt.violinplot(whole_dice[:, 0].T,
                                    positions=[p + i / shift_factor for p in percentages],
                                    widths=width,
                                    showmeans=False,
                                    showextrema=False,
                                    side="high")
            for pc in violin['bodies']:
                pc.set_facecolor(experiments_colors[i][1])
                pc.set_edgecolor(experiments_colors[i][1])
            # for partname in ('cbars','cmins','cmaxes'):
            #     vp = violin[partname]
            #     vp.set_edgecolor(experiments_colors[i][1])
            patches.append(mpatches.Patch(color=experiments_colors[i][1]))
            labels.append(labels_exp[i] + ", cup")

    plt.xlabel("fraction of used training data")
    plt.ylabel("dice score")
    plt.legend(patches, labels, loc=4)  # loc 4 ... lower right
    # plt.savefig("comparison.png")
    plt.show()


def plot_haus_violin_comparison(experiments, labels_exp, width_default=0.04,
                                shift_default=30, show_disc=True, show_cup=True,
                                show_means=False, show_extrema=False):
    experiments_colors = [("b", "cornflowerblue"), ("green", "limegreen"), ("orangered", "coral")]
    exp_paths = ["experiments/" + name + "/" for name in experiments]
    patches = []
    labels = []
    plt.rcParams["figure.figsize"] = (15, 5)
    percentages, _, _, _ = sds_file_handling(exp_paths[0])
    width = width_default * list(percentages)[-1]  # widths depends on the percentage area we look at
    shift_factor = shift_default / list(percentages)[-1]  # as well as how much we want to shift the experiments

    for i in range(len(experiments)):
        percentages, _, haus_dic, _ = sds_file_handling(exp_paths[i])
        whole_haus = np.array([haus_dic[percentages[i]] for i in range(percentages.shape[0])]).squeeze()
        if show_disc:
            violin = plt.violinplot(whole_haus[:, 1].T,
                                    positions=[p + i / shift_factor for p in percentages],
                                    widths=width,
                                    showmeans=show_means,
                                    showextrema=show_extrema,
                                    side="high")
            for pc in violin['bodies']:
                pc.set_facecolor(experiments_colors[i][0])
                pc.set_edgecolor(experiments_colors[i][0])
            # for partname in ('cbars','cmins','cmaxes'):
            #     vp = violin[partname]
            #     vp.set_edgecolor(experiments_colors[i][0])
            patches.append(mpatches.Patch(color=experiments_colors[i][0]))
            labels.append(labels_exp[i] + ", disc")

        if show_cup:
            violin = plt.violinplot(whole_haus[:, 0].T,
                                    positions=[p + i / shift_factor for p in percentages],
                                    widths=width,
                                    showmeans=False,
                                    showextrema=False,
                                    side="high")
            for pc in violin['bodies']:
                pc.set_facecolor(experiments_colors[i][1])
                pc.set_edgecolor(experiments_colors[i][1])
            # for partname in ('cbars','cmins','cmaxes'):
            #     vp = violin[partname]
            #     vp.set_edgecolor(experiments_colors[i][1])
            patches.append(mpatches.Patch(color=experiments_colors[i][1]))
            labels.append(labels_exp[i] + ", cup")

    plt.xlabel("fraction of used training data")
    plt.ylabel("hausdorff distance")
    plt.legend(patches, labels, loc=1)  # loc 1 ... upper right
    # plt.savefig("comparison.png")
    plt.show()


def plot_cdr_mae_violin_comparison(experiments, labels_exp, width_default=0.04, shift_default=30,
                                   show_means=False, show_extrema=False):
    experiments_colors = [("b", "cornflowerblue"), ("green", "limegreen"), ("orangered", "coral")]
    exp_paths = ["experiments/" + name + "/" for name in experiments]
    patches = []
    plt.rcParams["figure.figsize"] = (15, 5)
    percentages, _, _, _ = sds_file_handling(exp_paths[0])
    width = width_default * list(percentages)[-1]  # widths depends on the percentage area we look at
    shift_factor = shift_default / list(percentages)[-1]  # as well as how much we want to shift the experiments

    for i in range(len(experiments)):
        percentages, _, _, cdrs = sds_file_handling(exp_paths[i])
        maes = np.array([np.abs(cdrs[perc]["label"] - cdrs[perc]["pred"]) for perc in percentages]).squeeze()
        violin = plt.violinplot(maes.T,
                                positions=[p + i / shift_factor for p in percentages],
                                widths=width,
                                showmeans=show_means,
                                showextrema=show_extrema,
                                side="high")
        for pc in violin['bodies']:
            pc.set_facecolor(experiments_colors[i][0])
            pc.set_edgecolor(experiments_colors[i][0])
        # for partname in ('cbars','cmins','cmaxes'):
        #     vp = violin[partname]
        #     vp.set_edgecolor(experiments_colors[i][0])
        patches.append(mpatches.Patch(color=experiments_colors[i][0]))

    plt.xlabel("fraction of used training data")
    plt.ylabel("MAE of vCDR")
    plt.legend(patches, labels_exp, loc=1)  # loc 1 ... upper right
    # plt.savefig("comparison.png")
    plt.show()

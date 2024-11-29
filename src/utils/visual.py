r"""This module provides visualization functions."""
import os
from typing import Tuple, Optional, List, Callable
import io
import re

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import PIL
import imageio.v2 as imageio


def adjust_line_breaks_naive(formula: str, threshold: int = 6) -> str:
    r"""
    For a long LaTeX expression of a PDE, add line breaks to fit the length of
    the video title. Ignoring parenthesis.
    """
    formula = formula.replace('-', '+-')
    term_list = formula.split('+')
    n_lines = 1 + (len(term_list) - 1) // threshold
    term_cut_idx = np.linspace(0, len(term_list), n_lines + 1, dtype=int)
    line_list = []
    for i in range(len(term_cut_idx) - 1):
        line_list.append('+'.join(term_list[term_cut_idx[i]:term_cut_idx[i+1]]))
    formula = '$\n$+'.join(line_list)
    formula = formula.replace('+-', '-')
    return formula


def adjust_line_breaks(formula: str, threshold: int = 6, depth: int = 1) -> str:
    r"""
    For a long LaTeX expression of a PDE, add line breaks recursively to fit
    the length of the video title.
    """
    # Stop condition (Maximum cut times)
    if depth >= 10:
        return formula

    # Count the number of plus signs
    plus_count = formula.count('+') + formula.count('-') \
        + formula.count('sin') + formula.count('cos')

    if plus_count > threshold:
        # Find positions of all plus signs
        plus_positions = [m.start() for m in re.finditer(r'\+|-|sin|cos', formula)]
        # Determine the cut position
        cut_index = plus_positions[min(len(plus_positions)-1, 6)]

        # Check if the insertion point is within parentheses
        def inside_parentheses(text, index):
            stack = []
            for i, char in enumerate(text):
                if char == '(':
                    stack.append('(')
                elif char == ')':
                    if stack:
                        stack.pop()
                if i == index:
                    break
            return bool(stack)

        # Adjust the position if inside parentheses (Only depth=1)
        if depth == 1:
            while (inside_parentheses(formula, cut_index)
                   or formula[cut_index] not in ['+', '-']):
                cut_index -= 1

        # Adjust the position if the cut position is not at sign (+ or -)
        while formula[cut_index] not in ['+', '-']:
            cut_index -= 1

        # Insert the line break at the adjusted midpoint position recursively
        if cut_index > 0:
            formula = formula[:cut_index] + "$\n" \
                + adjust_line_breaks("$" + formula[cut_index:], depth=depth+1)

    return formula


def wrap_long_latex(formula: str) -> str:
    r"""
    For a long LaTeX expression of a PDE, add line breaks to fit the length of
    the plotting title.
    """
    # Split the title by existing line breaks
    lines = formula.split('\n')

    # Process each line
    processed_lines = [adjust_line_breaks(line) for line in lines]

    # Join the processed lines back together
    return '\n'.join(processed_lines)


def plot_1d(u_label: NDArray[float],
            u_predict: NDArray[float],
            file_name: str,
            title: str = "",
            save_dir: Optional[str] = None) -> None:
    r"""
    Plot the 1D image containing the label and the prediction.

    Args:
        u_label (numpy.ndarray): The label of the 1D image.
        u_predict (numpy.ndarray): The prediction of the 1D image.
        file_name (str): The name of the saved file.
        title (str): The title of the plot. Default: "".
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """

    plt.rcParams['figure.figsize'] = [6.4, 4.8]
    fig = plt.figure()
    ax_ = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_.plot(u_label, ls='-', c="blue", label="Reference")
    ax_.plot(u_predict, ls=':', c="red", label="Predict")
    ax_.legend()

    fig.suptitle(title)
    fig.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    plt.close()


def plot_2d(u_label: NDArray[float],
            u_predict: NDArray[float],
            file_name: str,
            title: str = "",
            save_dir: Optional[str] = None,
            shape: Optional[Tuple[int]] = None) -> None:
    r"""
    Plot the 2D image containing the label and the prediction.

    Args:
        u_label (NDArray[float]): The label of the 2D image.
        u_predict (NDArray[float]): The prediction of the 2D image.
        file_name (str): The name of the saved file.
        title (str): The title of the plot. Default: "".
        shape (Optional[Tuple[int]]): The shape of the input arrays. Default:
            None, do not reshape the input arrays.
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    if shape is not None:
        u_label = u_label.reshape(shape)
        u_predict = u_predict.reshape(shape)
    u_error = np.abs(u_label - u_predict)

    vmin_u = u_label.min()
    vmax_u = u_label.max()
    vmin_error = u_error.min()
    vmax_error = u_error.max()
    vmin = [vmin_u, vmin_u, vmin_error]
    vmax = [vmax_u, vmax_u, vmax_error]

    sub_titles = ["Reference", "Predict", "Error"]

    fig = plt.figure(figsize=(9.6, 3.2))
    gs_ = gridspec.GridSpec(2, 6)
    slice_ = [gs_[0:2, 0:2], gs_[0:2, 2:4], gs_[0:2, 4:6]]
    for i, data in enumerate([u_label, u_predict, u_error]):
        ax_ = fig.add_subplot(slice_[i])

        img = ax_.imshow(
            data.T, vmin=vmin[i],
            vmax=vmax[i],
            cmap=plt.get_cmap("jet"),
            origin='lower')

        ax_.set_title(sub_titles[i], fontsize=10)
        plt.xticks(())
        plt.yticks(())

        aspect = 20
        pad_fraction = 0.5
        divider = make_axes_locatable(ax_)
        width = axes_size.AxesY(ax_, aspect=1 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        cb_ = plt.colorbar(img, cax=cax)
        cb_.ax.tick_params(labelsize=6)

    gs_.tight_layout(fig, pad=1.0, w_pad=3.0, h_pad=1.0)

    title = wrap_long_latex(title)
    fig.suptitle(title, y=1.1)
    fig.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    plt.close()


def plot_2dxn(u_list: List[NDArray[float]],
              file_name: str,
              title: str = "",
              save_dir: Optional[str] = None) -> None:
    r"""
    Plot the images for partially-observed inverse problems.

    Args:
        u_list (list): A list of numpy.ndarrays containing the label, noisy (optional),
            observed (optional), prediction1, and prediction2.
        file_name (str): The name of the saved file.
        title (str): The title of the plot. Default: "".
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    n_plots = len(u_list)
    u_label = u_list[0]
    vmin_u = u_label.min()
    vmax_u = u_label.max()

    if n_plots == 3:
        sub_titles = ["Reference", "Predict1", "Predict2"]
    elif n_plots == 4:
        sub_titles = ["Reference", "Observed", "Predict1", "Predict2"]
    elif n_plots == 5:
        sub_titles = ["Reference", "Noisy", "Observed", "Predict1", "Predict2"]
    else:
        raise NotImplementedError

    plt.rcParams['figure.figsize'] = [9.6, 3.2]
    fig = plt.figure()
    gs_ = gridspec.GridSpec(2, 2 * n_plots)
    slice_ = [gs_[0:2, 2*i:2*i+2] for i in range(n_plots)]
    for i, data in enumerate(u_list):
        ax_ = fig.add_subplot(slice_[i])

        img = ax_.imshow(
            data.T, vmin=vmin_u,
            vmax=vmax_u,
            cmap=plt.get_cmap("jet"),
            origin='lower',
            interpolation='none')

        ax_.set_title(sub_titles[i], fontsize=10)
        plt.xticks(())
        plt.yticks(())

        aspect = 20
        pad_fraction = 0.5
        divider = make_axes_locatable(ax_)
        width = axes_size.AxesY(ax_, aspect=1 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        cb_ = plt.colorbar(img, cax=cax)
        cb_.ax.tick_params(labelsize=6)

    gs_.tight_layout(fig, pad=1.0, w_pad=3.0, h_pad=1.0)

    fig.suptitle(title, y=1.1)
    fig.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    plt.close()


def plot_l2_error_and_num_nodes(data: NDArray[float],
                                file_name: str,
                                save_dir: Optional[str] = None) -> None:
    r"""
    Plot the line chart, where the X axis represents the number of
    nodes in the graphormer and the Y axis represents L2 error.

    Args:
        data (numpy.ndarray): A numpy array of shape (size, 2), where size is
            the number of experiments. The first column represents the number
            of nodes in the graphormer, and the second column represents the L2
            error.
        file_name (str): The name of the saved file.
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    # data preprocessing
    num_nodes = data[:, 0]  # [size]
    l2_error = data[:, 1]  # [size]

    dic = {}
    for idx, num_node in enumerate(num_nodes):
        if num_node in dic:
            dic[num_node].append(l2_error[idx])
        else:
            dic[num_node] = [l2_error[idx]]

    for key, val in dic.items():
        dic[key] = np.array(val).mean()

    sorted_keys = sorted(dic.keys())
    points = []
    for key in sorted_keys:
        points.append([key, dic[key]])
    points = np.array(points)
    x_pts = [int(i) for i in points[:, 0]]
    y_pts = points[:, 1]

    # draw a line plot
    plt.figure(figsize=(6.8, 4.8))
    plt.plot(x_pts, y_pts, marker='o', markerfacecolor='red',
             markeredgecolor='red', markersize=5, ls=':')
    plt.xlabel("number of nodes")
    plt.ylabel("L2 error")
    plt.xticks(x_pts)
    plt.savefig(os.path.join(save_dir, f"line_{file_name}"))
    plt.close()

    # draw a scatter plot
    plt.figure(figsize=(6.8, 4.8))
    plt.scatter(num_nodes, l2_error, marker='o', s=1)
    plt.xlabel("number of nodes")
    plt.ylabel("L2 error")
    plt.xticks(x_pts)
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, f"scatter_{file_name}"))
    plt.close()


def plot_l2_error_and_epochs(data: list,
                             file_name: str,
                             save_dir: Optional[str] = None) -> None:
    r"""
    Plot the line chart, where the X axis represents epoch and the Y axis
    represents L2 error.

    Args:
        data (list): A list of numpy.ndarrays containing epoch, train_l2_error,
            and test_l2_error.
        file_name (str): The name of the saved file.
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    plt.figure(figsize=(6.8, 4.8))

    plt.plot(data[0], data[1], '--', color='b', label='train')
    plt.plot(data[0], data[2], '-', color='r', label='test')

    xticks_step = max(1, len(data[0]) // 5)
    xticks = [data[0][i] for i in range(0, len(data[0]), xticks_step)]
    plt.xticks(xticks)

    plt.xlabel('epochs')
    plt.ylabel('L2 error')
    plt.yscale('log')

    if min(data[1]) < 0.01 or min(data[2]) < 0.01:
        plt.ylim(0.001, 1.5)
    else:
        plt.ylim(0.01, 1.5)

    plt.legend()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()


def plot_l2_error_histogram(data: NDArray[float],
                            file_name: str,
                            save_dir: Optional[str] = None) -> None:
    r"""
    Plot a histogram of the L2 error distribution, where X represents
    the L2 error and the y-axis represents the frequency.

    Args:
        data (numpy.ndarray): A numpy array of shape (size, 1).
        file_name (str): The name of the saved file.
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    plt.figure(figsize=(6.8, 4.8))

    log_data = np.log10(data)

    plt.hist(log_data, bins=50, alpha=0.5, color='b')

    plt.title('L2 error Distribution')
    plt.xlabel('L2 error')
    plt.ylabel('Frequency')
    plt.xlim(log_data.min(), log_data.max())

    xticks = np.linspace(log_data.min(), log_data.max(), 5)
    xticks_label = [f"{10**x:.4f}" for x in xticks]
    plt.xticks(xticks, labels=xticks_label)

    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()


def plot_inverse_coef(label: NDArray[float],
                      pred: NDArray[float],
                      file_name: str,
                      save_dir: Optional[str] = None) -> None:
    r"""
    Plot the results of the inverted equation coefficients, with the X axis
    representing ground truth and the Y axis representing the inverted
    coefficients.

    Args:
        label (numpy.ndarray): A numpy array of shape (size).
        pred (numpy.ndarray): A numpy array of shape (size).
        file_name (str): The name of the saved file.
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    plt.figure(figsize=(6, 6))

    range_ = [min(label.min(), np.percentile(pred, 5)),
              max(label.max(), np.percentile(pred, 95))]
    plt.plot(range_, range_, linewidth='1')

    mae = np.abs(label - pred)

    plt.scatter(label, pred, s=8, c=mae, cmap='jet')
    plt.xlabel(r"Ground Truth", fontsize=16)
    plt.ylabel(r"Recovered", fontsize=16)
    plt.grid(alpha=0.3)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    plt.close()


def plot_noise_ic(ic_gt: NDArray[float],
                  ic_noisy: NDArray[float],
                  file_name: str,
                  save_dir: Optional[str] = None) -> None:
    r"""Plot the (ground truth / noise) initial condition for the inverse problem."""
    plt.figure(figsize=(6.8, 4.8))

    plt.plot(ic_gt, label='IC_gt')
    plt.plot(ic_noisy, label='IC_noisy')
    plt.xlabel('x')

    plt.legend()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()


def plot_2d_snapshots(u_pred: NDArray[float],
                      t_coord: NDArray[float],
                      x_ext: NDArray[float],
                      y_ext: NDArray[float]) -> None:
    r"""
    Plot PDEformer inference results on several time-steps.

    Arguments:
        u_pred: Shape [n_snap, dim1, dim2, n_vars].
        t_coord: Shape [n_snap].
        x_ext: Shape [dim1, dim2].
        y_ext: Shape [dim1, dim2].
    """
    n_t, _, _, n_vars = u_pred.shape
    size_unit = min(5, 10 / n_t)
    _, axes = plt.subplots(n_vars, n_t, squeeze=False,
                           figsize=(n_t * size_unit, n_vars * size_unit))
    for idx_var in range(n_vars):
        vmax = np.max(u_pred[..., idx_var])
        vmin = np.min(u_pred[..., idx_var])
        for idx_t in range(n_t):
            ax_ = axes[idx_var, idx_t]
            img = ax_.pcolormesh(
                x_ext, y_ext, u_pred[idx_t, :, :, idx_var],
                vmin=vmin, vmax=vmax, cmap="turbo")
            ax_.set_xlim(0, 1)
            ax_.set_ylim(0, 1)
            ax_.set_aspect(1)
            # ax_.set_xticks([])
            # ax_.set_yticks([])
        plt.colorbar(img, ax=axes[idx_var])
    for idx_t, time in enumerate(t_coord):
        axes[0, idx_t].set_title(f"$t={time:.2f}$")


def video_2d_old(label: NDArray[float],
                 predict: NDArray[float],
                 file_name: str,
                 title: str = "",
                 plot_num_t: int = 25,
                 save_dir: Optional[str] = None) -> None:
    r"""Generate a video for 2D predicted solutions."""
    # label.shape: [T, H, W, C]
    # predict.shape: [T, H, W, C]
    sample_t, sample_x, sample_y, num_channel = np.shape(label)

    predict = np.reshape(predict, [sample_t, sample_x, sample_y, num_channel])
    error = np.abs(label - predict)  # [T, H, W, C]

    # vmin = [label[..., i].min() for i in range(num_channel)] # [C]
    # vmax = [label[..., i].max() for i in range(num_channel)] # [C]
    vmin = [np.percentile(label[..., i], 5) for i in range(num_channel)]  # [C]
    vmax = [np.percentile(label[..., i], 95)
            for i in range(num_channel)]  # [C]

    vmin_error = [error[..., i].min() for i in range(num_channel)]  # [C]
    vmax_error = [error[..., i].max() for i in range(num_channel)]  # [C]

    if sample_t <= plot_num_t:
        times = list(range(0, sample_t))
    else:
        times = list(range(0, sample_t, sample_t // plot_num_t))

    title = wrap_long_latex(title)
    title_hight = 1 + len(title.split('\n')) // 2

    sub_titles = ["Reference", "Predict", "Error"]

    images = []
    # size = (480 * 1.1 * 3, 480 * num_channel)
    # plt.rcParams['figure.figsize'] = [s / 100 for s in size]
    for time in times:
        fig = plt.figure(figsize=(12, 3 * num_channel + title_hight))
        gs_ = gridspec.GridSpec(num_channel, 3)

        plt.suptitle(f"{title}\n$T={time}$", fontsize=26)

        for i, data_2d in enumerate([label[time], predict[time], error[time]]):
            for j in range(num_channel):
                ax_ = fig.add_subplot(gs_[j, i])

                if i in [0, 1]:
                    img = ax_.imshow(
                        data_2d[..., j].T, vmin=vmin[j],
                        vmax=vmax[j],
                        cmap=plt.get_cmap("turbo"),
                        origin='lower')
                else:
                    img = ax_.imshow(
                        data_2d[..., j].T, vmin=vmin_error[j],
                        vmax=vmax_error[j],
                        cmap=plt.get_cmap("turbo"),
                        origin='lower')

                if j == 0:
                    ax_.set_title(sub_titles[i], fontsize=24)

                plt.axis('off')

                aspect = 20
                pad_fraction = 0.5
                divider = make_axes_locatable(ax_)
                width = axes_size.AxesY(ax_, aspect=1 / aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                cb_ = plt.colorbar(img, cax=cax)
                cb_.ax.tick_params(labelsize=20)

        # gs_.tight_layout(fig, pad=3.0, w_pad=3.0, h_pad=3.0)
        gs_.tight_layout(fig)

        # save image to memory buffer
        buffer_ = io.BytesIO()  # memory buffer
        fig.savefig(buffer_, format="jpg")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)

        images.append(np.asarray(image))

        buffer_.close()  # release memory buffer
        plt.close()

    images = np.stack(images)  # [T, H, W, C]

    imageio.mimsave(os.path.join(save_dir, file_name),
                    images, 'GIF', duration=0.1)


def video_2dxn(u_list: List[NDArray[float]],
               file_name: str,
               title: str = "",
               shape: Optional[Tuple[int]] = None,
               plot_num_t: int = 25,
               save_dir: Optional[str] = None) -> None:
    r"""Generate a video for 2D partially-observed inverse problem data."""
    n_plots = len(u_list)
    if shape is not None:
        u_list = [np.reshape(u, shape) for u in u_list]
    label = u_list[0]  # [T, H, W, C].
    sample_t, _, _, num_channel = np.shape(label)

    vmin = [np.percentile(label[..., i], 5) for i in range(num_channel)]  # [C]
    vmax = [np.percentile(label[..., i], 95) for i in range(num_channel)]  # [C]

    title = wrap_long_latex(title)
    title_hight = 1 + len(title.split('\n')) // 2

    if n_plots == 3:
        sub_titles = ["Reference", "Predict1", "Predict2"]
    elif n_plots == 4:
        sub_titles = ["Reference", "Observed", "Predict1", "Predict2"]
    elif n_plots == 5:
        sub_titles = ["Reference", "Noisy", "Observed", "Predict1", "Predict2"]
    else:
        raise NotImplementedError

    if sample_t <= plot_num_t:
        times = list(range(0, sample_t))
    else:
        times = list(range(0, sample_t, sample_t // plot_num_t))

    images = []
    for time in times:
        fig = plt.figure(figsize=(4 * n_plots, 3 * num_channel + title_hight))
        gs_ = gridspec.GridSpec(num_channel, n_plots)

        plt.suptitle(f"{title}\n$T={time}$", fontsize=26)

        for i, data_2d in enumerate(u_list):
            for j in range(num_channel):
                ax_ = fig.add_subplot(gs_[j, i])

                img = ax_.imshow(data_2d[time, ..., j].T, vmin=vmin[j], vmax=vmax[j],
                                 cmap="turbo", origin='lower', interpolation='none')

                if j == 0:
                    ax_.set_title(sub_titles[i], fontsize=24)

                plt.axis('off')

                aspect = 20
                pad_fraction = 0.5
                divider = make_axes_locatable(ax_)
                width = axes_size.AxesY(ax_, aspect=1 / aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                cb_ = plt.colorbar(img, cax=cax)
                cb_.ax.tick_params(labelsize=20)

        gs_.tight_layout(fig)

        # save image to memory buffer
        buffer_ = io.BytesIO()  # memory buffer
        fig.savefig(buffer_, format="jpg")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)

        images.append(np.asarray(image))

        buffer_.close()  # release memory buffer
        plt.close()

    images = np.stack(images)  # [T, H, W, C]

    imageio.mimsave(os.path.join(save_dir, file_name),
                    images, 'GIF', duration=0.1)


def video_2d(label: NDArray[float],
             predict: NDArray[float],
             file_name: str,
             coords: Optional[NDArray[float]] = None,
             title: str = "",
             plot_num_t: int = 25,
             save_dir: Optional[str] = None) -> None:
    r"""Generate a video for 2D predicted solutions."""
    # arrays to plot
    # shape of 'label' and 'predict': [T, H, W, C]
    error = np.abs(label - predict)  # [T, H, W, C]
    n_t, _, _, n_channel = label.shape
    if n_channel > 5:
        raise ValueError(f"Too many channels ({n_channel})!")

    # coordinates, shape [H, W, 2]
    if coords is None:
        x_coord = np.linspace(0, 1, label.shape[1] + 1)[:-1]
        y_coord = np.linspace(0, 1, label.shape[2] + 1)[:-1]
        coords = np.stack(np.meshgrid(x_coord, y_coord, indexing="ij"))

    # titles
    title = wrap_long_latex(title)
    title_hight = 1 + len(title.split('\n')) // 2
    sub_titles = ["Reference", "Predict", "Error"]

    # initial plot
    fig = plt.figure(figsize=(12, 3.5 * n_channel + title_hight))
    gs_ = gridspec.GridSpec(n_channel, 3)
    img_list = [[None for _ in range(3)] for _ in range(n_channel)]
    for i in range(n_channel):
        vmax = np.percentile(label[..., i], 95)
        vmin = np.percentile(label[..., i], 5)
        for j, data_2d in enumerate([label[0], predict[0], error[0]]):
            if j == 2:  # override vmax, vmin for plotting error
                vmax = error[..., i].max()
                vmin = 0.

            ax_ = fig.add_subplot(gs_[i, j])
            img_list[i][j] = ax_.pcolormesh(
                coords[..., 0], coords[..., 1], data_2d[..., i],
                vmin=vmin, vmax=vmax, cmap="turbo")

            ax_.set_xlim(0, 1)
            ax_.set_ylim(0, 1)
            ax_.set_aspect(1)
            # ax_.set_axis_off()
            ax_.set_xticks([])
            ax_.set_yticks([])
            if i == 0:
                ax_.set_title(sub_titles[j], fontsize=24)

            aspect = 20
            pad_fraction = 0.5
            divider = make_axes_locatable(ax_)
            width = axes_size.AxesY(ax_, aspect=1 / aspect)
            pad = axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            cb_ = plt.colorbar(img_list[i][j], cax=cax)
            cb_.ax.tick_params(labelsize=20)

    if n_t > 1:
        delta_t = 1 / (n_t - 1)
        plt.suptitle(f"{title} ($t=0.00$)", fontsize=26)
    else:
        plt.suptitle(title, fontsize=26)
    gs_.tight_layout(fig)

    # animation
    def update(frame):
        for i in range(n_channel):
            for j, data_2d in enumerate([label[frame], predict[frame], error[frame]]):
                img_list[i][j].set_array(data_2d[..., i])
        if n_t > 1:
            plt.suptitle(f"{title} ($t={frame * delta_t:.2f}$)", fontsize=26)
    update(0)
    frames = np.linspace(0, label.shape[0] - 1, plot_num_t + 1, dtype=int)
    anim = FuncAnimation(fig, update, frames=frames)
    if save_dir is not None:
        file_name = os.path.join(save_dir, file_name)
    anim.save(file_name)


def video_2d_realtime(u_at_t_fn: Callable[[float], NDArray[float]],
                      t_coord: NDArray[float],
                      x_ext: NDArray[float],
                      y_ext: NDArray[float]) -> FuncAnimation:
    r"""
    Generate a video for 2D predicted solutions, in which the solution
    snapshots are computed on-the-fly.

    Arguments:
        u_at_t_fn: Function that accepts time t, and outputs the predicted
            solution at t, in the form of an array with shape
            [dim1, dim2, n_vars].
        t_coord: Shape [n_snap].
        x_ext: Shape [dim1, dim2].
        y_ext: Shape [dim1, dim2].
    """
    u_init = u_at_t_fn(t_coord[0])
    _, _, n_vars = u_init.shape

    fig, axes = plt.subplots(1, n_vars, squeeze=False)
    axes = axes[0]  # [1, n_vars] -> [n_vars]
    vmin = np.empty(n_vars)
    vmax = np.empty(n_vars)
    img_list = []
    for idx_var in range(n_vars):
        ui_init = u_init[:, :, idx_var]
        vmin[idx_var] = ui_init.min()
        vmax[idx_var] = ui_init.max()
        img = axes[idx_var].pcolormesh(x_ext, y_ext, ui_init, cmap="turbo")
        axes[idx_var].set_xlim(0, 1)
        axes[idx_var].set_ylim(0, 1)
        axes[idx_var].set_aspect(1)
        plt.colorbar(img, ax=axes[idx_var])
        img_list.append(img)

    plt.suptitle(f"$t={t_coord[0]:.2f}$")

    frame_list = [u_init]

    def update(frame: int) -> None:
        if frame < len(frame_list):  # cached case
            snapshot = frame_list[frame]
        elif frame == len(frame_list):
            snapshot = u_at_t_fn(t_coord[frame])
            frame_list.append(snapshot)
            for idx_var in range(n_vars):
                vmin[idx_var] = min(vmin[idx_var],
                                    np.min(snapshot[:, :, idx_var]))
                vmax[idx_var] = max(vmax[idx_var],
                                    np.max(snapshot[:, :, idx_var]))
                img_list[idx_var].set_clim(vmin=vmin[idx_var],
                                           vmax=vmax[idx_var])

        for idx_var in range(n_vars):
            img_list[idx_var].set_array(snapshot[:, :, idx_var])
        plt.suptitle(f"$t={t_coord[frame]:.2f}$")

    anim = FuncAnimation(fig, update, frames=range(len(t_coord)))
    return anim


def _test_video_2d():
    r"""Unit test of 'video_2d'."""
    resolution = 16
    phi_coord = np.linspace(0, 2 * np.pi, resolution + 1).reshape(-1, 1)
    r_coord = np.linspace(0, 0.4, resolution).reshape(1, -1)
    x_coord = 0.5 + r_coord * np.cos(phi_coord)
    y_coord = 0.5 + r_coord * np.sin(phi_coord)
    coords = np.stack([x_coord, y_coord], axis=-1)
    label = np.stack([np.cos(t) * x_coord for t in np.linspace(0, 1, resolution)])
    label = np.stack([label, label], axis=-1)
    video_2d(label, label, "test.gif", coords, title="latex")


if __name__ == "__main__":  # unit test
    _test_video_2d()

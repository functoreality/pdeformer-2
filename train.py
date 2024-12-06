r"""Train the model."""
import time
import argparse
import math
from typing import Dict, Any

import numpy as np
import mindspore as ms
from mindspore import dtype as mstype
from mindspore import ops, nn, Tensor, context
from mindspore.communication import init, get_rank
from mindspore.amp import DynamicLossScaler, auto_mixed_precision

from src.data import load_dataset, split_data_tuple
from src.core import (calculate_l2_error, L2ErrorRecord, EvalErrorRecord,
                      LossFunction, get_lr, get_optimizer)
from src.utils import load_config, init_record, AllGather, set_seed
from src.utils.visual import video_2d
from src.cell import get_model


def parse_args():
    r"""Parse input args"""
    parser = argparse.ArgumentParser(description="pde foundation model")
    parser.add_argument("--mode", type=str, default="GRAPH",
                        choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False,
                        choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="Ascend",
                        choices=["GPU", "Ascend", "CPU"],
                        help=("The target device to run. "
                              "Supporting 'Ascend', 'GPU', 'CPU'."))
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument('--no_distributed', action='store_true',
                        help='unenable distributed training (data parallel)')
    parser.add_argument('--no_data_sink', action='store_true',
                        help='unenable data sink during training')
    parser.add_argument("--config_file_path", "-c", type=str, required=True,
                        help="Path of the configuration YAML file.")

    input_args = parser.parse_args()
    input_args.distributed = not input_args.no_distributed
    input_args.data_sink = not input_args.no_data_sink

    return input_args


def plot_solution(label_plot: Tensor,
                  pred_plot: Tensor,
                  coordinate: Tensor,
                  data_info: Dict[str, Any],
                  img_name: str,
                  description: str = "") -> None:
    r"""Comparative plot of the ground-truth and the predicted solutions."""
    # convert arrays
    label_plot = label_plot.asnumpy().astype(np.float32)
    pred_plot = pred_plot.asnumpy().astype(np.float32)
    coordinate = coordinate.asnumpy().astype(np.float32)

    # reshape arrays
    shape = (data_info["n_t_grid"], data_info["n_x_grid"],
             data_info["n_y_grid"], label_plot.shape[-1])
    label_plot = label_plot.reshape(shape)
    pred_plot = pred_plot.reshape(shape)
    coordinate = coordinate.reshape(shape[:-1] + (4,))  # [nt, nx, ny, 4]
    coordinate = coordinate[0, :, :, 1:3]  # [nx, ny, 2]

    # make the plot
    pde_latex = data_info["pde_latex"]
    title = (pde_latex + f"\n${data_info['var_latex']}$ ("
             f"{data_info['idx_var']}) of sample {data_info['idx_pde']}, "
             + description)
    # title = pde_latex + "\n" + description  # used with `eval_plot_all_vars`
    record.visual(video_2d, label_plot, pred_plot, img_name,
                  coords=coordinate, title=title,
                  save_dir=record.video2d_dir)
    # record.print(f"LaTeX: {pde_latex}, coef: {data_info['coef_dict']}")


def eval_loop(dataset_iter, dataset, img_name="result", plot_num=1):
    r"""Evaluation loop for a single dataset"""
    eval_error = []
    l2_error = []
    worst_eval_error = 0.
    model.set_train(False)
    plot_idx = 0
    eval_loss_fn = LossFunction(config.eval, reduce_mean=False)
    for data_tuple in dataset_iter:
        input_tuple, label, data_idx = split_data_tuple(data_tuple)  # (tuple, tensor, tensor)
        coordinate = input_tuple[-1]
        pred = model(*input_tuple)  # [bsz, num_points, dim_out]

        eval_error_tmp = eval_loss_fn(pred, label, coordinate).asnumpy()  # [bsz]
        eval_error_tmp = eval_error_tmp.clip(0, 5)
        eval_error.extend(eval_error_tmp.tolist())
        l2_error_tmp = calculate_l2_error(pred, label)  # [bsz]
        l2_error.extend(l2_error_tmp.tolist())

        # update the worst sample
        worst_idx = int(np.argmax(eval_error_tmp))
        if eval_error_tmp[worst_idx] > worst_eval_error:
            worst_eval_error = eval_error_tmp[worst_idx]
            worst_label = label[worst_idx]
            worst_pred = pred[worst_idx]
            worst_coord = coordinate[worst_idx]
            worst_data_idx = data_idx[worst_idx]

        # plot label vs. pred
        for in_batch_idx in range(data_idx.shape[0]):
            if plot_idx >= plot_num:
                break
            plot_data_idx = int(data_idx[in_batch_idx])
            description = (f"loss: {eval_error_tmp[in_batch_idx]:.4f}, "
                           f"nRMSE: {l2_error_tmp[in_batch_idx]:.4f}")
            plot_solution(label[in_batch_idx],
                          pred[in_batch_idx],
                          coordinate[in_batch_idx],
                          dataset.get_pde_info(plot_data_idx),
                          f"{img_name}_{plot_idx}.gif",
                          description=description)
            plot_idx += 1

    eval_error = Tensor(eval_error).astype(mstype.float32)  # [datasize]
    l2_error = Tensor(l2_error).astype(mstype.float32)  # [datasize]

    # distributed training (data parallel)
    if use_ascend and args.distributed:
        # [num_devices, datasize] => [num_devices * datasize]
        eval_error_np = all_gather(eval_error).flatten().asnumpy()
        l2_error_np = all_gather(l2_error).flatten().asnumpy()

        # select worst sample across devices
        worst_eval_errors = all_gather(Tensor([worst_eval_error]))  # [num_devices]
        worst_idx = int(np.argmax(worst_eval_errors.asnumpy()))
        worst_eval_error = worst_eval_errors[worst_idx].asnumpy()
        # [*] -> [1, *] -> [num_devices, *] -> [*]
        worst_label = all_gather(worst_label.expand_dims(0))[worst_idx]
        worst_pred = all_gather(worst_pred.expand_dims(0))[worst_idx]
        worst_coord = all_gather(worst_coord.expand_dims(0))[worst_idx]
        worst_data_idx = all_gather(worst_data_idx.expand_dims(0))[worst_idx]
    else:
        eval_error_np = eval_error.asnumpy()
        l2_error_np = l2_error.asnumpy()

    # plot the worst sample
    if plot_num > 0:
        worst_data_idx = int(worst_data_idx)
        worst_l2_error = calculate_l2_error(worst_pred.expand_dims(0),
                                            worst_label.expand_dims(0))
        description = (f"loss: {worst_eval_error.item():.4f}, "
                       f"nRMSE: {worst_l2_error.item():.4f}")
        plot_solution(worst_label, worst_pred, worst_coord,
                      dataset.get_pde_info(worst_data_idx),
                      f"{img_name}_worst-{worst_data_idx}.gif",
                      description=description)
        # record.print(f"worst sample: data_idx {worst_data_idx}, "
        #              f"eval_error {worst_eval_error}, LaTeX\n  {pde_latex}")

    return eval_error_np, l2_error_np


def eval_plot_all_vars(dataset_iter, dataset, img_name="result", plot_num=1):
    r"""
    When using `eval_loop` to plot PDEformer predictions, only one PDE unknown
    variable is shown at a time. This function can be used to show the
    predictions of all variables in an image. Before substituting it for
    `eval_loop`, please make sure that:
    1. The script is not run in parallel mode.
    2. Set `config.eval.total_batch_size` to be equal to the number of
        variables of the PDE (eg. 1 for `dcr`/`wave`, 2 for `elasticsteady`,
        3 for `swe`/`dcdcr`).
    """
    model.set_train(False)
    plot_idx = 0
    for data_tuple in dataset_iter:
        input_tuple, label, data_idx = split_data_tuple(data_tuple)  # (tuple, tensor, tensor)
        coordinate = input_tuple[-1]
        pred = model(*input_tuple)  # [bsz, num_points, dim_out]
        l2_error_tmp = calculate_l2_error(pred, label)  # [bsz]

        # plot label vs. pred
        if plot_idx >= plot_num:
            break
        plot_data_idx = int(data_idx[0])
        description = f"nRMSE: {np.mean(l2_error_tmp):.4f}"
        plot_solution(label.transpose((1, 2, 0)),  # [bsz, n_pts, 1] -> [n_pts, 1, bsz]
                      pred.transpose((1, 2, 0)),
                      coordinate[0],
                      dataset.get_pde_info(plot_data_idx),
                      f"{img_name}_{plot_idx}.gif",
                      description=description)
        plot_idx += 1

    fake_l2_error_np = np.full(1, -1)
    return fake_l2_error_np, fake_l2_error_np


def eval_dataset_dict(epoch, dataset_dict, prefix="train"):
    r"""Evaluation loop for multiple pde datasets"""
    eval_error_record = EvalErrorRecord()
    l2_error_record = L2ErrorRecord()

    plot_num_per_type = config.eval.plot_num_per_type
    for pde_type in dataset_dict:
        # make the plots distributed uniformly over all datasets
        num_datasets = len(dataset_dict[pde_type])
        if num_datasets == 0:
            continue
        plot_nums = [plot_num_per_type // num_datasets] * num_datasets
        for i in range(plot_num_per_type % num_datasets):
            plot_nums[i] += 1

        for datafile, (dataset_iter, dataset) in dataset_dict[pde_type].items():
            cur_plot_num = plot_nums.pop(0)
            img_name = f"{prefix}_epoch-{epoch}_{pde_type}_{datafile}"
            eval_error, l2_error = eval_loop(
                dataset_iter, dataset, img_name=img_name, plot_num=cur_plot_num)
            eval_error_dict = eval_error_record.append(pde_type, datafile, eval_error)
            # record.print(f"Epoch {epoch}: {prefix} {pde_type} {datafile} "
            #              + eval_error_record.dict2str(eval_error_dict))
            record.add_dict(epoch, eval_error_dict,
                            prefix=f"{prefix}_{pde_type}_{datafile}")
            l2_error_dict = l2_error_record.append(pde_type, datafile, l2_error)
            # record.print(f"Epoch {epoch}: {prefix} {pde_type} {datafile} "
            #              + l2_error_record.dict2str(l2_error_dict))
            record.add_dict(epoch, l2_error_dict,
                            prefix=f"{prefix}_{pde_type}_{datafile}")

        eval_error_dict = eval_error_record.reduce(pde_type)
        record.print(f"Epoch {epoch}: {prefix} {pde_type} all "
                     + eval_error_record.dict2str(eval_error_dict))
        record.add_dict(epoch, eval_error_dict, prefix=f"{prefix}_{pde_type}_all")
        l2_error_dict = l2_error_record.reduce(pde_type)
        record.print(f"Epoch {epoch}: {prefix} {pde_type} all "
                     + l2_error_record.dict2str(l2_error_dict))
        record.add_dict(epoch, l2_error_dict, prefix=f"{prefix}_{pde_type}_all")

        # Release memory in the last evaluation loop.
        if epoch >= config.train.epochs:
            dataset_dict[pde_type] = {}

    eval_error_dict = eval_error_record.reduce("all")
    record.print(f"Epoch {epoch}: {prefix} all all "
                 + eval_error_record.dict2str(eval_error_dict))
    record.add_dict(epoch, eval_error_dict, prefix=f"{prefix}_all_all")
    l2_error_dict = l2_error_record.reduce("all")
    record.print(f"Epoch {epoch}: {prefix} all all "
                 + l2_error_record.dict2str(l2_error_dict))
    record.add_dict(epoch, l2_error_dict, prefix=f"{prefix}_all_all")

    return eval_error_dict, l2_error_dict


def eval_model(epoch):
    r"""Evaluate the model with both train data and test data."""
    eval_dataset_dict(epoch, train_iter_dict, prefix="train")
    eval_error_test, l2_error_test = eval_dataset_dict(epoch, test_iter_dict, prefix="test")

    return eval_error_test, l2_error_test


def train():
    r"""Train the model."""
    # auto mixed precision
    if use_ascend:
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')

    # Evaluation before training
    eval_error_test, l2_error_test = eval_model(epoch=0)
    eval_error_best = eval_error_test['eval_error_mean']
    l2_error_best = l2_error_test['l2_error_mean']
    if config.train.epochs <= 0:
        record.print(f"eval_error_mean: {eval_error_best:>7f}")
        return

    # loss function
    loss_fn = LossFunction(config.train, reduce_mean=True)

    # optimizer
    steps_per_epoch = dataset_train.get_dataset_size()
    lr_var = get_lr(steps_per_epoch, config.train)
    optimizer = get_optimizer(lr_var, model, config.train)

    # gradient postprocessing
    if use_ascend and args.distributed:
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters)
    else:
        def grad_reducer(x):
            return x
    grad_clip_value = config.train.get("grad_clip_value", -1)

    # define forward function
    def forward_fn(input_tuple, label):
        pred = model(*input_tuple)
        coordinate = input_tuple[-1]  # tensor
        loss = loss_fn(pred, label, coordinate)

        # auto mixed precision
        if use_ascend:
            loss = loss_scaler.scale(loss)

        return loss, pred

    # define gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # train one step
    @ms.jit
    def train_step(*data_tuple):
        input_tuple, label, _ = split_data_tuple(data_tuple)  # (tuple, tensor, tensor)
        (loss, pred), grads = grad_fn(input_tuple, label)

        # auto mixed precision
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            grads = loss_scaler.unscale(grads)

        if grad_clip_value > 0:
            grads = ops.clip_by_global_norm(grads, clip_norm=grad_clip_value)
        grads = grad_reducer(grads)  # distributed training (data parallel)
        loss = ops.depend(loss, optimizer(grads))

        return loss, pred

    # data sink
    if args.data_sink:
        sink_process = ms.data_sink(train_step, dataset_train, 1)
    else:
        dataset_iter = dataset_train.create_tuple_iterator()
    dataset_train_size = dataset_train.get_dataset_size()

    # training loop
    record.print('training...')
    print_interval = math.ceil(config.train.epochs / 2500)
    for epoch in range(1, 1 + config.train.epochs):
        model.set_train()
        loss_all = []
        if args.data_sink:
            # data sink
            for _ in range(dataset_train_size):
                loss, _ = sink_process()
                loss_all.append(loss.asnumpy())
        else:
            # not data sink
            for data_tuple in dataset_iter:
                loss, _ = train_step(*data_tuple)
                loss_all.append(loss.asnumpy())

        if (epoch - 1) % print_interval == 0:
            loss = np.mean(loss_all)  # [dataset_train_size] -> []
            record.print(f"Epoch {epoch}: loss {loss:>10f}")
            record.add_scalar("train/loss", loss, epoch)
            # Update the logical-physical mapping for dynamic datasets.
            data_updater(record.print)

        # Evaluation
        if epoch % config.eval.interval == 0 or epoch == config.train.epochs:
            # save last checkpoint
            record.save_ckpt(model, 'model_last.ckpt')

            eval_error_test, l2_error_test = eval_model(epoch=epoch)

            # save best checkpoint
            if eval_error_best > eval_error_test['eval_error_mean']:
                eval_error_best = eval_error_test['eval_error_mean']

            if l2_error_best > l2_error_test['l2_error_mean']:
                l2_error_best = l2_error_test['l2_error_mean']
                record.save_ckpt(model, 'model_best.ckpt')

    record.print(f"best eval_error_mean: {eval_error_best:>7f}")
    record.print(f"best l2_error_mean: {l2_error_best:>7f}")
    record.print("training done!")


if __name__ == "__main__":
    # seed
    set_seed(123456)

    # args
    args = parse_args()

    # mindspore context
    context.set_context(
        save_graphs=args.save_graphs,
        save_graphs_path=args.save_graphs_path,
        device_target=args.device_target)
    if args.mode.upper().startswith("GRAPH"):
        context.set_context(mode=context.GRAPH_MODE)
    else:
        context.set_context(mode=context.PYNATIVE_MODE)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    # context.set_context(max_call_depth=10000)  # for larger models

    # compute_type
    compute_type = mstype.float16 if use_ascend else mstype.float32

    # distributed training (data parallel)
    rank_id = None
    if use_ascend:
        if args.distributed:
            init()  # enable HCCL
            context.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True)
            rank_id = get_rank()
            all_gather = AllGather()  # nn.cell for ops.ALLGather()
        else:
            context.set_context(device_id=args.device_id)
        # context.set_context(deterministic="ON")

    # load config file
    config, config_str = load_config(args.config_file_path)

    # init record
    record = init_record(use_ascend, rank_id, args, config, config_str)

    # dataset
    record.print(f"Loading {config.data.type} data...")
    (dataset_train, data_updater, train_iter_dict, test_iter_dict) = load_dataset(config)

    # model
    model = get_model(config, record, compute_type)

    # train
    start_time = time.time()
    try:
        train()
    except Exception as err:
        data_updater("terminate")
        record.print("Exception detected.")
        record.close()
        raise err

    data_updater("terminate")
    record.print(f"End-to-End total time: {time.time() - start_time} s")
    record.close()

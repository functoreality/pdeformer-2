r"""
Recover a function (the source term, or the wave velocity field) of a PDE using
gradient descent based on the pre-trained PDEformer model.
"""
import argparse
import math
from typing import Tuple, Dict, Any
from omegaconf import DictConfig
import numpy as np
from numpy.typing import NDArray

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import ops, nn, Tensor, context
from mindspore.common.initializer import initializer, Uniform, One
from mindspore.amp import DynamicLossScaler, auto_mixed_precision

from src.data.load_inverse_data import get_inverse_data, inverse_observation
from src.utils import load_config, init_record, set_seed
from src.utils.visual import plot_2d, video_2dxn
from src.core import LossFunction, get_lr_list
from src.cell import get_model


def parse_args():
    r"""Parse input args."""
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
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument('--distributed', action='store_true',
                        help='enable distributed training (data parallel)')
    parser.add_argument("--config_file_path", "-c", type=str, required=True,
                        help="Path of the configuration YAML file.")
    return parser.parse_args()


class InverseProblem:
    r"""
    Using gradient descent to solve the inverse problem of recovering a
    function-valued term in a PDE.

    Args:
        pde_idx (int): The index of the PDE to solve.
        data_tuple (tuple): The model's input and output data packaged in batch
            form have the same PDE structure, but different initial conditions
            (and different source terms for the wave equation).
        inverse_config (dict): The configuration of the inverse problem.
        data_info (dict): A dictionary of data information.
    """

    def __init__(self,
                 pde_idx: int,
                 data_tuple: Tuple[Tensor],
                 inverse_config: DictConfig,
                 data_info: Dict[str, Any]) -> None:
        self.pde_idx = pde_idx
        self.data_tuple = data_tuple
        self.data_info = data_info
        node_function = data_tuple.node_function

        # shape of node_function is [n_graph, num_function, num_points_function, 5]
        self.n_graph, self.num_function, self.num_points_function, _ = node_function.shape
        # [n_graph, n_txyz, 1] -> [n_graph, n_t, n_x, n_y, n_z, 1]
        self.u_label = data_tuple.u_label.asnumpy().reshape((
            self.n_graph, data_info["n_t_grid"], data_info["n_x_grid"],
            data_info["n_y_grid"], data_info["n_z_grid"], 1))

        recovered = initializer(Uniform(scale=0.01),
                                [1, 1, self.num_points_function, 1],
                                mstype.float32)
        self.recovered = ms.Parameter(recovered, name='function_valued_term')

        # Shape is [n_graph, num_function, num_points_function, 5].
        mask = np.zeros(node_function.shape).astype(bool)
        function_node_id = config.inverse.func.function_node_id
        mask[:, function_node_id, :, -1] = True
        self.mask = Tensor(mask, dtype=mstype.bool_)

        self.gt_np = node_function.asnumpy()[0, function_node_id, :, -1]  # [num_points_function]

        # add noise and spatial-temporal subsampling
        u_noisy, u_obs_plot, u_obs, coordinate_obs = inverse_observation(
            config.inverse.observation, self.u_label, data_tuple.coordinate.asnumpy())
        self.u_noisy = u_noisy  # [n_graph, n_t, n_x, n_y, n_z, 1]
        self.u_obs_plot = u_obs_plot  # [n_graph, n_t, n_x, n_y, n_z, 1]
        # These are the only tensor-valued attributes of this class object.
        # Shape is [n_graph, n_txyz_obs_pts, 1].
        self.u_obs = Tensor(u_obs, dtype=mstype.float32)
        # Shape is [n_graph, n_txyz_obs_pts, 4].
        self.coordinate_obs = Tensor(coordinate_obs, dtype=mstype.float32)

        # regularization_loss
        self.regularize_type = inverse_config.func.function_regularize.type
        self.regularize_weight = inverse_config.func.function_regularize.weight
        self.fwi = inverse_config.func.fwi

    def get_data_tuple(self, is_train: bool = True) -> Tuple[Tensor]:
        r"""
        Get the data tuple for training or testing.

        Args:
            is_train (bool): Whether to get the training data or testing data.

        Returns:
            A data tuple for training or testing.

        """
        (node_type, node_scalar, node_function, in_degree, out_degree,
         attn_bias, spatial_pos, coordinate_gt, _) = self.data_tuple

        recovered_repeat = self.recovered  # [1, 1, num_points_function, 1]
        node_function_ = ops.select(self.mask, recovered_repeat, node_function)

        if is_train:
            coordinate = self.coordinate_obs  # [n_graph, num_obs_point, 2]
        else:
            coordinate = coordinate_gt  # [n_graph, num_point, 2]

        data_tuple = (node_type, node_scalar, node_function_, in_degree, out_degree,
                      attn_bias, spatial_pos, coordinate, self.u_obs)
        return data_tuple

    def regularization_loss(self) -> Tensor:
        r"""Calculate the regularization loss."""
        # Specific for 2D case, spatial resolution 128.
        recovered_f = self.recovered.reshape((
            self.data_info["n_x_grid"], self.data_info["n_y_grid"]))
        dx_f = 128 * recovered_f.diff(axis=0)  # [n_x - 1, n_y]
        dy_f = 128 * recovered_f.diff(axis=1)  # [n_x, n_y - 1]
        fwi_pen = ops.maximum(recovered_f - 4, 0.) +\
            ops.maximum(0.01 - recovered_f, 0.)
        if self.regularize_type == "L1":
            penalty = ops.mean(ops.abs(dx_f)) + ops.mean(ops.abs(dy_f))
        elif self.regularize_type == "squareL2":
            penalty = ops.mean(ops.square(dx_f)) + ops.mean(ops.square(dy_f))
        elif self.regularize_type == "L2":
            penalty = ops.sqrt(ops.mean(ops.square(dx_f))
                               + ops.mean(ops.square(dy_f)) + 1e-6)
        else:
            raise ValueError(f"Unknown regularize_type '{self.regularize_type}'!")
        if self.fwi:
            penalty += ops.mean(fwi_pen) * 10**5
        return self.regularize_weight * penalty

    def compare(self, enable_plot: bool = False) -> float:
        r"""
        Compare the ground-truth and the recovered function values.

        Args:
            enable_plot (bool): Whether to plot the comparison results.

        Returns:
            The nRMSE between the ground-truth and the recovered function values.
        """
        recovered_np = self.recovered.asnumpy().flatten()  # [num_points_function]

        gt_norm = np.linalg.norm(self.gt_np, ord=2, axis=0, keepdims=False)
        rmse = np.linalg.norm(recovered_np - self.gt_np, ord=2, axis=0, keepdims=False)
        nrmse = rmse / (gt_norm + 1e-6)

        if enable_plot:
            record.visual(
                plot_2d, self.gt_np, recovered_np,
                f"inverse-{self.pde_idx}.png",
                shape=(self.data_info["n_x_grid"], self.data_info["n_y_grid"]),
                title=self.data_info["pde_latex"],
                save_dir=record.inverse_dir)

        return nrmse.item()

    def visual(self, model: nn.Cell) -> None:
        r"""
        Visualization of the real equation solution (label), noisy equation solution (noisy),
        observed equation solution with coordinate subsampling (obs),
        model's predicted solution given ground-truth equation function value (raw_pred), and
        model's predicted solution given the recovered equation function value (pred).

        Args:
            model (nn.Cell): The model to predict the solution.

        Returns:
            None.
        """
        def get_pred(data_tuple: Tuple[Tensor], i_sample: int) -> NDArray[float]:
            input_tuple = (tensor[[i_sample]] for tensor in data_tuple[:-1])
            pred = model(*input_tuple)  # [1, num_point, 1]
            pred = pred.asnumpy().astype(np.float32)
            return pred[0]  # [num_point, 1]

        txy_grid_shape = (self.data_info["n_t_grid"],
                          self.data_info["n_x_grid"],
                          self.data_info["n_y_grid"],
                          1)
        for plot_idx in range(config.inverse.plot_num_per_cls):
            file_name = f"compare-{self.pde_idx}-{plot_idx}.gif"
            raw_pred_plot = get_pred(self.data_tuple, plot_idx)
            pred_plot = get_pred(self.get_data_tuple(False), plot_idx)
            plot_list = [self.u_label[plot_idx], self.u_noisy[plot_idx],
                         self.u_obs_plot[plot_idx], raw_pred_plot, pred_plot]
            record.visual(video_2dxn, plot_list, file_name,
                          title=self.data_info["pde_latex"],
                          shape=txy_grid_shape,
                          save_dir=record.video2d_dir)


def inverse(model: nn.Cell) -> None:
    r"""
    Solve the inverse problem that recovers the function-valued term in a PDE
    from the observed data using gradient descent based on the pre-trained model.
    """
    # loss function
    loss_fn = LossFunction(config.inverse, reduce_mean=True)

    # learning rate
    lr_var = get_lr_list(1,
                         config.inverse.func.epochs,
                         config.inverse.func.learning_rate,
                         config.inverse.func.lr_scheduler.type,
                         config.inverse.func.lr_scheduler.milestones,
                         config.inverse.func.lr_scheduler.lr_decay)

    # auto mixed precision
    if use_ascend:
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')

    nrmse_all = []
    pde_cases = config.inverse.pde_cases
    if isinstance(pde_cases, int):
        pde_cases = range(pde_cases)
    for pde_idx in pde_cases:
        data_tuple, data_info = get_inverse_data(config, pde_idx)
        record.print(f"PDE {pde_idx}: {data_info.get('pde_latex')}\n  "
                     f"coefs: {data_info.get('coef_dict')}")
        invp = InverseProblem(pde_idx, data_tuple, config.inverse, data_info)

        # optimizer
        params = [{'params': [invp.recovered], 'lr': lr_var,
                   'weight_decay': config.inverse.func.weight_decay}]
        optimizer = nn.Adam(params)

        # distributed training (data parallel)
        if use_ascend and args.distributed:
            grad_reducer = nn.DistributedGradReducer(optimizer.parameters)
        else:
            def grad_reducer(x_in):
                return x_in

        # define forward function
        get_data_tuple = invp.get_data_tuple
        regularization_loss = invp.regularization_loss

        def forward_fn():
            data_tuple = get_data_tuple(is_train=True)  # pylint: disable=W0640
            input_tuple = data_tuple[:-1]  # tuple
            label = data_tuple[-1]  # tensor
            pred = model(*input_tuple)
            coordinate = input_tuple[-1]
            loss = loss_fn(pred, label, coordinate) + regularization_loss()  # pylint: disable=W0640

            # auto mixed precision
            if use_ascend:
                loss = loss_scaler.scale(loss)

            return loss, pred

        # define gradient function
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        # function for one step of training
        @ms.jit
        def train_step():
            (loss, pred), grads = grad_fn()  # pylint: disable=W0640

            grads = ops.clip_by_global_norm(grads, clip_norm=1.0)
            # distributed training (data parallel)
            grads = grad_reducer(grads)  # pylint: disable=W0640

            # auto mixed precision
            if use_ascend:
                loss = loss_scaler.unscale(loss)
                grads = loss_scaler.unscale(grads)

            loss = ops.depend(loss, optimizer(grads))  # pylint: disable=W0640

            return loss, pred

        # training loop
        print_interval = math.ceil(config.inverse.func.epochs / 100)
        if invp.fwi:
            initial_lst = [0.4 * i for i in range(1,10)]
        else:
            initial_lst = [None]
        loss_lst = []
        recovered_lst = []
        for coef in initial_lst:
            if coef is not None:
                recovered = ops.ones((1, 1, invp.num_points_function, 1))
                invp.recovered.set_data(coef * recovered)
                num_init = int(coef/0.4)
            else:
                num_init = 1
            print_interval = math.ceil(config.inverse.func.epochs / 100)
            for epoch in range(1, 1 + config.inverse.func.epochs):
                loss, _ = train_step()

                if (epoch - 1) % print_interval == 0 or epoch == config.inverse.func.epochs:
                    # record
                    loss = loss.asnumpy().item()
                    if epoch == config.inverse.func.epochs:
                        nrmse = invp.compare(enable_plot=False)
                        nrmse_all.append(nrmse)
                    else:
                        nrmse = invp.compare(enable_plot=False)
                    record.print(f"PDE {pde_idx},initialize {num_init} epoch {epoch}: loss {loss:>10f} nrmse {nrmse:>7f}")
                    record.add_scalar(f"train_pde-{pde_idx}/loss", loss, epoch)
                    record.add_scalar(f"train_pde-{pde_idx}/nrmse", nrmse, epoch)
                    if loss > 10**5 or math.isnan(loss) or math.isinf(loss):
                        loss = 10**5
                        break
            loss_lst.append(loss)
            recovered_lst.append(invp.recovered.asnumpy())
        min_idex = loss_lst.index(min(loss_lst))
        recovered_best = recovered_lst[min_idex]
        invp.recovered=ms.Parameter(recovered_best, name='function_valued_term')
        invp.compare(enable_plot=True)

    nrmse_mean = np.array(nrmse_all).mean()
    record.print(f"nrmse_mean: {nrmse_mean:>7f}")

    record.print("inversion done!")


if __name__ == "__main__":
    # seed
    set_seed(123456)

    # args
    args = parse_args()

    # mindspore context
    context.set_context(
        save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
        device_target=args.device_target, device_id=args.device_id)
    if args.mode.upper().startswith("GRAPH"):
        context.set_context(mode=context.GRAPH_MODE)
    else:
        context.set_context(mode=context.PYNATIVE_MODE)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    # compute_type
    compute_type = mstype.float16 if use_ascend else mstype.float32

    # load config file
    config, config_str = load_config(args.config_file_path)

    # record
    record = init_record(use_ascend, 0, args, config, config_str, inverse_problem=True)

    # model
    model_ = get_model(config, record, compute_type)

    # inverse
    try:
        inverse(model_)
    except Exception as err:
        record.close()
        raise err

    record.close()

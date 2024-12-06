r"""
Recover scalar coefficients in a PDE using particle swarm optimization based on
the pre-trained PDEformer model.
"""
import argparse
import math
from typing import Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray
from mindspore import dtype as mstype
from mindspore import ops, Tensor, context, nn

from src.data.load_inverse_data import get_inverse_data, inverse_observation
from src.data.pde_dag import NODE_TYPE_DICT
from src.core import LossFunction
from src.utils import load_config, init_record, set_seed
from src.utils.visual import plot_inverse_coef, plot_2d, video_2dxn
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
    parser.add_argument("--config_file_path", "-c", type=str, required=True,
                        help="Path of the configuration YAML file.")
    return parser.parse_args()


class DataPSO:
    r"""
    Data preprocessing for inverse problems.

    Args:
        pde_idx (int): The index of the PDE to be solved.
        data_tuple (Tuple[Tensor]): The model's input and output data packaged
            in batch form have the same PDE structure, but different ICs.
        data_info (Dict[str, Any]): A dictionary of data information.
        enable_inverse_nu (bool): Whether to enable the inverse of viscosity
            coefficient math:`\nu`. Default: False.
        num_coef_inverse (int): The number of coefficients to be recovered. Default: 1.
    """

    def __init__(self,
                 pde_idx: int,
                 data_tuple: Tuple[Tensor],
                 data_info: Dict[str, Any],
                 enable_inverse_nu: bool = False,
                 num_coef_inverse: int = 1) -> None:
        self.pde_idx = pde_idx
        self.input_tuple = data_tuple[:-1]  # tuple
        self.data_info = data_info

        # Shape is [n_graph, num_scalar, 1].
        self.node_scalar_gt = data_tuple.node_scalar.asnumpy()
        # Shape is [n_graph, num_function, num_points_function, 5].
        self.node_function = data_tuple.node_function.asnumpy()
        self.n_graph, self.num_scalar, _ = self.node_scalar_gt.shape
        # [n_graph, n_txyz, 1] -> [n_graph, n_t, n_x, n_y, n_z, 1]
        self.u_label = data_tuple.u_label.asnumpy().reshape((
            self.n_graph, data_info["n_t_grid"], data_info["n_x_grid"],
            data_info["n_y_grid"], data_info["n_z_grid"], 1))

        # mask of PDE coefficients to be recovered
        node_type = data_tuple.node_type.asnumpy()  # [n_graph, n_node, 1]
        if node_type.shape[-1] != 1:
            raise ValueError(
                f"'node_dim' must be equal to 1, but got {node_type.shape[-1]}.")
        # Shape is [n_graph, num_scalar, 1].
        self.mask = np.equal(node_type[:, :self.num_scalar, :],
                             NODE_TYPE_DICT["coef"])
        num_true = np.cumsum(self.mask, axis=1)
        self.num_coef_inverse = num_coef_inverse
        if enable_inverse_nu:
            self.mask[num_true > num_coef_inverse] = False
        else:
            self.mask[np.logical_or(
                num_true == 1, num_true > num_coef_inverse + 1)] = False
        self.cond = Tensor(self.mask, mstype.bool_)  # [n_graph, num_scalar, 1]

        self.num_coef_without_pad = np.count_nonzero(self.mask[0])  # int
        if self.num_coef_without_pad == 0:
            return

        # add noise and spatial-temporal subsampling
        u_noisy, u_obs_plot, u_obs, coordinate_obs = inverse_observation(
            config.inverse.observation, self.u_label, data_tuple.coordinate.asnumpy())
        self.u_noisy = u_noisy  # [n_graph, n_txyz, 1]
        self.u_obs_plot = u_obs_plot  # [n_graph, n_txyz, 1]
        # These are the only tensor-valued attributes of this class object.
        # Shapes are [n_graph, n_tx_obs_pts, 1] and [n_graph, n_tx_obs_pts, 4].
        self.u_obs = Tensor(u_obs, dtype=mstype.float32)
        self.coordinate_obs = Tensor(coordinate_obs, dtype=mstype.float32)

        if config.inverse.coef.enable_nu:
            self.nu_pos = np.argmax(self.mask)  # record the nu scalar position

    def get_input_tuple(self,
                        coef_with_pad: NDArray[float],
                        full_coord: bool = False) -> Tuple[Tensor]:
        r"""
        Construct the model's input tuple based on the given equation coefficients.

        Args:
            coef_with_pad (NDArray[float]): Coefficients with padding,
                the shape of tensor is :math:`(num\_scalar)`.
            full_coord (bool): Whether to use the full coordinate information. Default: False.

        Returns:
            intpu_tuple (Tuple[Tensor]): Input tuple for PDEformer.
        """
        (node_type, node_scalar, node_function, in_degree, out_degree, attn_bias,
         spatial_pos, coordinate) = self.input_tuple

        # [num_scalar] -> [n_graph, num_scalar, 1]
        coef_with_pad = np.repeat(coef_with_pad[np.newaxis, :, np.newaxis],
                                  repeats=self.n_graph, axis=0)
        # [n_graph, num_scalar, 1]
        coef_with_pad = Tensor(coef_with_pad, dtype=mstype.float32)
        # [n_graph, num_scalar, 1]
        node_scalar = ops.select(self.cond, coef_with_pad, node_scalar)

        if not full_coord:
            coordinate = self.coordinate_obs  # [n_graph, num_obs_point, 4]

        input_tuple = (node_type, node_scalar, node_function, in_degree,
                       out_degree, attn_bias, spatial_pos, coordinate)
        return input_tuple

    def compare(self, recovered: NDArray[float]) -> Tuple[NDArray[float]]:
        r"""
        Compare the recovered equation coefficients with the ground truth.

        Args:
            recovered (NDArray[float]): Recovered equation coefficients with padding,
                the shape of tensor is math:`(num\_scalar)`.

        Returns:
            tuple: The ground truth without padding, the recovered equation
                coefficients without padding, and the mean absolute error
                between them.
        """
        mask = self.mask[0, :, 0]  # [num_scalar]
        ground_truth = self.node_scalar_gt[0, :, 0]  # [num_scalar]

        label = np.where(mask, ground_truth, 0)  # [num_scalar]
        pred = np.where(mask, recovered, 0)  # [num_scalar]
        mae = np.sum(np.abs(pred - label)) / self.num_coef_without_pad  # []

        ground_truth = ground_truth[mask]  # [num_coef_without_pad]
        recovered = recovered[mask]  # [num_coef_without_pad]

        return ground_truth, recovered, mae

    def visual(self, model: nn.Cell, coef_with_pad: NDArray[float]) -> None:
        r"""
        Visualization of the real equation solution (label), noisy equation solution (noisy),
        observed equation solution with coordinate subsampling (obs),
        model's predicted solution given ground truth equation coefficients (raw_pred), an
        model's predicted solution given the recovered equation coefficients (pred).
        """
        def get_pred(input_tuple: Tuple[Tensor], i_sample: int) -> NDArray[float]:
            input_tuple = (tensor[[i_sample]] for tensor in input_tuple)
            pred = model(*input_tuple)  # [1, num_point, 1]
            pred = pred.asnumpy().astype(np.float32)
            return pred[0]  # [num_point, 1]

        txy_grid_shape = (self.data_info["n_t_grid"],
                          self.data_info["n_x_grid"],
                          self.data_info["n_y_grid"],
                          1)
        for plot_idx in range(config.inverse.plot_num_per_cls):
            file_name = f"compare-{self.pde_idx}-{plot_idx}.gif"
            raw_pred_plot = get_pred(self.input_tuple, plot_idx)
            pred_plot = get_pred(self.get_input_tuple(
                coef_with_pad, full_coord=True), plot_idx)
            plot_list = [self.u_label[plot_idx], self.u_noisy[plot_idx],
                         self.u_obs_plot[plot_idx], raw_pred_plot, pred_plot]
            record.visual(video_2dxn, plot_list, file_name,
                          title=self.data_info["pde_latex"],
                          shape=txy_grid_shape,
                          save_dir=record.video2d_dir)

            ic_gt = self.u_label[plot_idx, 0]
            ic_noisy = self.node_function[plot_idx, 0, :, -1]
            # warning: IC for wave is ignored
            record.visual(
                plot_2d, ic_gt, ic_noisy,
                f"IC-compare-{plot_idx}.png",
                shape=(self.data_info["n_x_grid"], self.data_info["n_y_grid"]),
                title=self.data_info["pde_latex"],
                save_dir=record.inverse_dir)


class InversePSO:
    r"""
    Particle swarm optimization (PSO) algorithm for solving the inverse
    problem of recovering the equation coefficients.

    Args:
        model (nn.Cell): The pre-trained model used for inverse problem.
        data (DataPSO): The data preprocessing object.
        loss_fn (nn.Cell): The loss function used for calculating the fitness of the population.
        pop_size (int): The size of the population. Default: 20.
        max_gen (int): The maximum number of generations. Default: 200.
        omega (float): Inertia weight. Default: 1.0.
        c_1 (float): Cognitive weight. Default: 1.49445.
        c_2 (float): Social weight. Default: 1.49445.
        vel_min (float): Minimum velocity. Default: -1.
        vel_max (float): Maximum velocity. Default: 1.
        pop_min (float): Minimum value of the population. Default: -3.
        pop_max (float): Maximum value of the population. Default: 3.
    """

    def __init__(self,
                 model: nn.Cell,
                 data: DataPSO,
                 loss_fn: nn.Cell,
                 pop_size: int = 20,
                 max_gen: int = 200,
                 omega: float = 1.0,
                 c_1: float = 1.49445,
                 c_2: float = 1.49445,
                 vel_min: float = -1,
                 vel_max: float = 1,
                 pop_min: float = -3,
                 pop_max: float = 3) -> None:
        self.model = model
        self.data = data
        self.loss_fn = loss_fn
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.omega = omega
        self.c_1 = c_1
        self.c_2 = c_2
        self.vel_min = vel_min
        self.vel_max = vel_max
        self.pop_min = pop_min
        self.pop_max = pop_max

        # population initialization
        self.pop = np.random.uniform(
            self.pop_min, self.pop_max, (self.pop_size, self.data.num_scalar))
        if config.inverse.coef.enable_nu:
            self.pop[:, [self.data.nu_pos]] = np.random.uniform(
                0, 1, (self.pop_size, 1))

        self.vel = np.random.uniform(-1, 1,
                                     (self.pop_size, self.data.num_scalar))
        fitness = self.cal_fitness()  # [pop_size]
        min_idx = np.argmin(fitness)  # int
        self.p_best = self.pop  # [pop_size, num_scalar]
        self.p_fitness_best = fitness  # [pop_size]
        self.g_best = self.pop[min_idx]  # [num_scalar]
        self.g_fitness_best = fitness[min_idx]  # float

    def cal_fitness(self) -> NDArray[float]:
        r"""
        Calculate the fitness of the current population. The fitness is the
        loss between the model's predicted solution on the observed data and
        the real solution, and the smaller the better.
        """
        fitness = []
        for i in range(self.pop_size):
            input_tuple = self.data.get_input_tuple(self.pop[i])
            label = self.data.u_obs
            pred = self.model(*input_tuple)
            coordinate = input_tuple[-1]
            loss = self.loss_fn(pred, label, coordinate)  # [1]
            fitness.append(loss.asnumpy())

        return np.array(fitness, dtype=np.float32)  # [pop_size]

    def update(self) -> None:
        r"""
        Update population. The PSO algorithm relies on two main formulas:
        the velocity update formula and the position update formula. The
        velocity update formula adjusts the velocity of each particle based on
        its own best position and the global best position, while the position
        update formula adjusts the position of each particle based on its own
        velocity.

        Velocity Update Formula:
            math:`v_{i}^{k+1} = \omega \cdot v_{i}^{k} + c_1 \cdot r_1 \cdot
                (p_{best,i} - x_{i}^{k}) + c_2 \cdot r_2 \cdot
                (g_{best} - x_{i}^{k})`

            - math:`v_{i}^{k+1}` represents the velocity of particle math:`i`
                  at iteration math:`k+1`.
            - math:`\omega` is the inertia weight that controls the impact of
                the previous velocity on the current velocity. A higher
                math:`\omega` allows for more global exploration, while a lower
                math:`\omega` focuses on local exploration.
            - math:`c_1` and math:`c_2` are learning factors that determine the
                step size towards the individual best and global best
                solutions, respectively.
            - math:`r_1` and math:`r_2` are random numbers between math:`[0,1]`
                that introduce randomness in the algorithm.
            - math:`p_{best,i}` is the best position found by particle math:`i` so far.
            - math:`g_{best}` represents the best position found by any
                particle in the swarm so far.
            - math:`x_{i}^{k}` is the position of particle math:`i` at iteration math:`k`.

        Position Update Formula:
            math:`x_{i}^{k+1} = x_{i}^{k} + 0.5 \cdot v_{i}^{k+1}`

            - math:`x_{i}^{k+1}` represents the new position of particle
                math:`i` at iteration math:`k+1`.

        These two formulas govern the movement of particles in the search
        space. During each iteration, particles adjust their velocities and
        positions based on their own best positions and the global best
        position, aiming to converge towards the optimal solution.
        """
        # update velocity
        self.vel = self.omega * self.vel  # [pop_size, num_scalar]
        # [pop_size, num_scalar]
        self.vel += self.c_1 * np.random.rand(1) * (self.p_best - self.pop)
        self.vel += self.c_2 * np.random.rand(1) * \
            (self.g_best.reshape(1, self.data.num_scalar) -
             self.pop)  # [pop_size, num_scalar]
        # [pop_size, num_scalar]
        self.vel[self.vel > self.vel_max] = self.vel_max
        # [pop_size, num_scalar]
        self.vel[self.vel < self.vel_min] = self.vel_min

        # update position
        self.pop = self.pop + 0.5 * self.vel  # [pop_size, num_scalar]
        # [pop_size, num_scalar]
        self.pop[self.pop > self.pop_max] = self.pop_max
        # [pop_size, num_scalar]
        self.pop[self.pop < self.pop_min] = self.pop_min
        if config.inverse.coef.enable_nu:
            self.pop[:, self.data.nu_pos:self.data.nu_pos+1] = \
                np.clip(self.pop[:, self.data.nu_pos:self.data.nu_pos+1], 0, 1)
            # Avoid nu gathers to zero
            self.pop[:, self.data.nu_pos:self.data.nu_pos+1] += \
                np.random.uniform(0, 0.005, (self.pop_size, 1)) * \
                np.random.randint(0, 4, (self.pop_size, 1))

        # update the optimal position of each particle
        fitness = self.cal_fitness()  # [pop_size]
        flag = fitness < self.p_fitness_best  # [pop_size]
        self.p_fitness_best[flag] = fitness[flag]  # [pop_size]
        self.p_best[flag, :] = self.pop[flag, :]  # [pop_size, num_scalar]

        # update the global optimal position
        min_idx = np.argmin(fitness)  # 1
        fitness_tmp = fitness[min_idx]  # 1
        if fitness_tmp < self.g_fitness_best:
            self.g_fitness_best = fitness_tmp  # 1
            self.g_best = self.pop[min_idx]  # [num_scalar]

    def run(self) -> Tuple[NDArray[float]]:
        r"""Optimization process of population."""
        print_interval = math.ceil(self.max_gen / 10)
        (node_type, node_scalar, node_function, in_degree, out_degree, attn_bias,
         spatial_pos, coordinate) = self.data.input_tuple
        coordinate = self.data.coordinate_obs
        input_gt = (node_type, node_scalar, node_function, in_degree, out_degree, attn_bias,
                    spatial_pos, coordinate)
        pred = self.model(*input_gt)
        label = self.data.u_obs
        loss = self.loss_fn(pred, label, coordinate).asnumpy()
        info = f'PDE {self.data.pde_idx} ground truth fitness {loss:>7f}'
        record.print(info)
        for i in range(self.max_gen):
            self.update()

            info = f"PDE {self.data.pde_idx} iter-{i}: fitness {self.g_fitness_best:>7f}"
            if i % print_interval == 0 or i + 1 == self.max_gen:
                ground_truth, recovered, mae = self.data.compare(self.g_best)
                info = info + f" mae {mae:>7f}"
            record.print(info)
        inverse_fail = self.g_fitness_best > loss * 1.05

        return ground_truth, recovered, mae, inverse_fail

    def visual(self) -> None:
        r"""
        Visualization of the real equation solution (label), noisy equation solution (noisy),
        observed equation solution with coordinate subsampling (obs),
        model's predicted solution given ground truth equation coefficients (raw_pred), an
        model's predicted solution given the recovered equation coefficients (pred).
        """
        self.data.visual(self.model, self.g_best)


def inverse(model: nn.Cell) -> None:
    r"""
    Solve the inverse problem that recovers the equation coefficients from the observed data
    using particle swarm optimization based on the pre-trained model.
    """
    # loss function for calculate fitness
    loss_fn = LossFunction(config.inverse, reduce_mean=True)

    coef_ground_truth = []
    coef_recovered = []
    coef_gt_w_idx = []
    coef_recover_w_idx = []
    coef_mae = []
    idx_list = []
    symbol_list = []
    fail_lst = []
    pde_cases = config.inverse.pde_cases
    if isinstance(pde_cases, int):
        pde_cases = list(range(pde_cases))
    for pde_idx in pde_cases:
        # inversion for each pde sample, and each pde sample has different ICs.
        data_tuple, data_info = get_inverse_data(config, pde_idx)
        record.print(f"PDE {pde_idx}: {data_info.get('pde_latex')}\n  "
                     f"coefs: {data_info.get('coef_dict')}")

        data = DataPSO(pde_idx, data_tuple, data_info,
                       enable_inverse_nu=config.inverse.coef.enable_nu,
                       num_coef_inverse=config.inverse.coef.num_coef)
        if data.num_coef_without_pad == 0:  # no need to inverse
            continue

        pso = InversePSO(model, data, loss_fn,
                         pop_size=config.inverse.coef.pso.pop_size,
                         max_gen=config.inverse.coef.pso.max_gen,
                         pop_min=-config.inverse.coef.coef_scale,
                         pop_max=config.inverse.coef.coef_scale)

        ground_truth, recovered, mae, inverse_fail = pso.run()
        coef_ground_truth.extend(ground_truth.tolist())
        coef_recovered.extend(recovered.tolist())
        coef_mae.append(mae)
        coef_gt_w_idx.append(np.array(ground_truth, dtype=np.float32))
        coef_recover_w_idx.append(np.array(recovered, dtype=np.float32))
        idx_list.append(pde_idx)
        fail_lst.append(inverse_fail)
        if 'coef_dict' not in data_info:
            raise KeyError(
                f"'coef_dict' not found in data_info for PDE {pde_idx}.")
        symbol_list.append(data_info['coef_dict'])

        if pde_idx == pde_cases[-1]:
            pso.visual()

    coef_ground_truth = np.array(coef_ground_truth, dtype=np.float32)
    coef_recovered = np.array(coef_recovered, dtype=np.float32)
    coef_mae = np.array(coef_mae, dtype=np.float32)

    # save the results
    pkl_dict = {"coef_ground_truth": coef_ground_truth,
                "coef_recovered": coef_recovered,
                "coef_mae": coef_mae,
                "coef_gt_w_idx": coef_gt_w_idx,
                "coef_recover_w_idx": coef_recover_w_idx,
                "pde_idx_list": idx_list,
                "symbol_list": symbol_list,
                "inverse_fail": inverse_fail}
    record.save_pickle(pkl_dict, file_name="inverse.pkl")

    for idx in range(len(idx_list)):
        record.print(f"Recover PDE idx {idx}:")
        record.print(f"PDE coef all: {symbol_list[idx]}")
        record.print(f"PDE coef reference: {coef_gt_w_idx[idx]}")
        record.print(f"PDE coef recovered: {coef_recover_w_idx[idx]}")
        record.print(f"Inverse fitness > ground truth fitness: {inverse_fail}")

    # print the results
    record.print(f"All: params.mean: {str(coef_recovered.mean())}, "
                 f"params.min: {str(coef_recovered.min())}, "
                 f"params.max: {str(coef_recovered.max())}\n    "
                 f"mae_mean {coef_mae.mean():>7f} mae_max {coef_mae.max():>7f}")
    record.visual(plot_inverse_coef, coef_ground_truth, coef_recovered,
                  "inverse_coef.png", save_dir=record.inverse_dir)
    record.print("inversion done!")


if __name__ == "__main__":
    # seed
    set_seed(123456)

    # args
    args = parse_args()

    # mindspore context
    context.set_context(
        mode=context.GRAPH_MODE if args.mode.upper().startswith(
            "GRAPH") else context.PYNATIVE_MODE,
        save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
        device_target=args.device_target, device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    # compute_type
    compute_type = mstype.float16 if use_ascend else mstype.float32

    # load config file
    config, config_str = load_config(args.config_file_path)

    # record
    record = init_record(use_ascend, 0, args, config,
                         config_str, inverse_problem=True)

    # model
    model_ = get_model(config, record, compute_type)

    # inverse
    try:
        inverse(model_)
    finally:
        record.close()

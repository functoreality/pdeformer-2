r"""Loading datasets containing one specific PDE (single_pde), mainly PDEBench datasets."""
from typing import Tuple, Union, List, Dict, Any, Callable

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..env import float_dtype
from ..utils_dataload import Dataset, datasets2loader, concat_datasets
from ..multi_pde.dataloader import StaticDatasetFakeUpdater
from .basics import pde_type_class_dict, SinglePDEInputFileDataset
# update 'pde_type_class_dict'
from . import dataset_cart1, dataset_cart2, dataset_scat1

_data_modules = [dataset_cart1, dataset_cart2, dataset_scat1]


class PDEOutputDataset(Dataset):
    r"""Base class for loading the PDE solution data for different models."""
    DATA_COLUMN_NAMES = ["input_field", "input_scalar", "coordinates", "u_label"]

    def __init__(self,
                 config: DictConfig,
                 input_dataset: SinglePDEInputFileDataset,
                 n_samples: int,
                 test: bool = False,
                 for_eval: bool = False) -> None:
        super().__init__()
        self.input_dataset = input_dataset
        self.n_samples = n_samples
        self.test = test
        if for_eval:
            # self.data_augment = False
            self.num_txyz_samp_pts = -1
        else:
            # self.data_augment = config.data.augment
            self.num_txyz_samp_pts = config.train.num_txyz_samp_pts

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray]:
        if self.test:
            idx_pde = len(self.input_dataset) - 1 - idx_pde
        idx_pde = int(idx_pde)  # np.int64 -> int
        input_field, input_scalar, coord, u_label = self.input_dataset[idx_pde]
        return (input_field.astype(float_dtype),
                input_scalar.astype(float_dtype),
                coord.astype(float_dtype),
                u_label.astype(float_dtype))

    def __len__(self) -> int:
        return self.n_samples

    def get_pde_info(self, idx_data: int) -> Dict[str, Any]:
        r"""Get a dictionary containing the information of the current PDE."""
        idx_pde = idx_data
        if self.test:
            idx_pde = len(self.input_dataset) - 1 - idx_pde
        data_info = self.input_dataset.get_pde_info(idx_pde)
        return data_info

    def add_model_config_(self, config: DictConfig, idx_pde: int = 0) -> None:
        r"""Add config options related to model hyperparameters."""
        # nothing to do by default


class NO2DModelPDEDataset(PDEOutputDataset):
    r"""
    Base class for loading the PDE solution data for 2D neural operators, in
    which different time-steps are treated as different output channels.
    """
    DATA_COLUMN_NAMES = ["grid_in", "coordinate", "u_label"]

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field, input_scalar, coordinate, u_label = super().__getitem__(idx_pde)
        n_x, n_y, _ = input_field.shape
        # [n_scalars] -> [n_y, n_scalars] -> [n_x, n_y, n_scalars]
        input_scalar = np.repeat(input_scalar[np.newaxis], n_y, axis=0)
        input_scalar = np.repeat(input_scalar[np.newaxis], n_x, axis=0)
        grid_in = np.concatenate([input_field, input_scalar],
                                 axis=-1)  # [n_x, n_y, n_fields+n_scalars]
        _, n_x, n_y, _, _ = u_label.shape
        # [n_t, n_x, n_y, n_z=1, n_vars] -> [n_x, n_y, n_z, n_t, n_vars]
        u_label = np.transpose(u_label, (1, 2, 3, 0, 4))
        # [n_x, n_y, n_z=1, n_t, n_vars] -> [n_x, n_y, n_t * n_vars]
        u_label = u_label.reshape(n_x, n_y, -1)
        return grid_in, coordinate, u_label

    def add_model_config_(self, config: DictConfig, idx_pde: int = 0) -> None:
        input_field, input_scalar, _, u_label = self.input_dataset[idx_pde]
        _, _, n_fields = input_field.shape
        n_scalars, = input_scalar.shape
        n_t, n_x, n_y, _, n_vars = u_label.shape

        config.fno3d.in_channels = n_fields + n_scalars
        config.fno3d.out_channels = n_vars
        config.fno3d.resolution = [n_t, n_x, n_y]


class NO3DModelPDEDataset(PDEOutputDataset):
    r"""
    Base class for loading the PDE solution data for 3D neural operators, in
    which the temporal axis is treated as a new spatial axis.
    """
    DATA_COLUMN_NAMES = ["grid_in", "coordinate", "u_label"]

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field, input_scalar, coordinate, u_label = super().__getitem__(idx_pde)
        # [n_t, n_x, n_y, n_z=1, n_vars] -> [n_t, n_x, n_y, n_vars]
        u_label = u_label.squeeze(-2)
        n_t_grid = u_label.shape[0]

        n_x, n_y, _ = input_field.shape
        # [n_scalars] -> [n_y, n_scalars] -> [n_x, n_y, n_scalars]
        input_scalar = np.repeat(input_scalar[np.newaxis], n_y, axis=0)
        input_scalar = np.repeat(input_scalar[np.newaxis], n_x, axis=0)
        grid_in = np.concatenate([input_field, input_scalar],
                                 axis=-1)  # [n_x, n_y, n_fields+n_scalars]
        # [n_x, n_y, n_f+n_s] -> [n_t, n_x, n_y, n_f+n_s]
        grid_in = np.repeat(grid_in[np.newaxis], n_t_grid, axis=0)
        return grid_in, coordinate, u_label

    add_model_config_ = NO2DModelPDEDataset.add_model_config_


class INRModelPDEDataset(PDEOutputDataset):
    r"""Base class for loading the PDE solution data for INRs."""

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field, input_scalar, coordinate, u_label = super().__getitem__(idx_pde)

        # subsampled coordinate, u_label
        # Shape is [n_t, n_x, n_y, n_z, 4 + n_vars] (Cartesian)
        # or [n_t, n_xyz, 4 + n_vars] (scattered).
        txyz_u = np.concatenate([coordinate, u_label], axis=-1)
        if self.num_txyz_samp_pts == -7:  # sampling time-steps
            num_t_pts = txyz_u.shape[0]
            t_sample_idx = np.random.randint(num_t_pts)
            txyz_u = txyz_u[t_sample_idx]  # [n_t, ...] -> [...]
        # [..., 4+n_vars] -> [*, 4+n_vars]
        txyz_u = txyz_u.reshape((-1, txyz_u.shape[-1]))
        if self.num_txyz_samp_pts > 0:
            num_txyz_pts = txyz_u.shape[0]
            txyz_sample_idx = np.random.randint(
                0, num_txyz_pts, self.num_txyz_samp_pts)
            txyz_u = txyz_u[txyz_sample_idx, :]
        coordinate = txyz_u[:, :4]  # [n_txyz_pts, 4]
        u_label = txyz_u[:, 4:]  # [n_txyz_pts, n_vars]

        return input_field, input_scalar, coordinate, u_label


class DeepONetPDEDataset(INRModelPDEDataset):
    r"""Loading PDE dataset for DeepONet."""
    DATA_COLUMN_NAMES = ["branch_in", "trunk_in", "u_label"]

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field, input_scalar, coordinate, u_label = super().__getitem__(idx_pde)
        trunk_in = coordinate
        # downsample [n_x, n_y, n_fields] -> [n_x//2, n_y//2, n_fields]
        input_field = input_field[::2, ::2, :]
        branch_in = np.concatenate([input_field.ravel(), input_scalar])
        return branch_in, trunk_in, u_label

    def add_model_config_(self, config: DictConfig, idx_pde: int = 0) -> None:
        branch_in, trunk_in, u_label = self[idx_pde]
        _, config.deeponet.trunk_dim_in = trunk_in.shape
        config.deeponet.branch_dim_in, = branch_in.shape
        _, config.deeponet.n_vars = u_label.shape


class CNNDeepONetPDEDataset(INRModelPDEDataset):
    r"""
    Loading PDE dataset for CNNDeepONet.
    The main difference from DeepONetPDEDataset is that the branch input
    maintains its 2D grid structure for CNN processing.
    """
    DATA_COLUMN_NAMES = ["branch_in", "trunk_in", "u_label"]

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field, input_scalar, coordinate, u_label = super().__getitem__(idx_pde)
        trunk_in = coordinate

        # Keep the 2D grid structure for CNN processing
        n_x, n_y, _ = input_field.shape
        # Add input_scalar as additional channels to the 2D grid
        # [n_scalars] -> [n_y, n_scalars] -> [n_x, n_y, n_scalars]
        input_scalar = np.repeat(input_scalar[np.newaxis], n_y, axis=0)
        input_scalar = np.repeat(input_scalar[np.newaxis], n_x, axis=0)
        # [n_x, n_y, n_fields] -> [n_x, n_y, n_fields + n_scalars]
        branch_in = np.concatenate([input_field, input_scalar], axis=-1)
        branch_in = branch_in.transpose(2, 0, 1)  # HWC -> CHW

        return branch_in, trunk_in, u_label

    def add_model_config_(self, config: DictConfig, idx_pde: int = 0) -> None:
        branch_in, trunk_in, u_label = self[idx_pde]
        _, config.deeponet.trunk_dim_in = trunk_in.shape
        config.deeponet.branch_dim_in, _, _ = branch_in.shape
        _, config.deeponet.n_vars = u_label.shape


class PDEformerPDEDataset(INRModelPDEDataset):
    r"""Loading PDE dataset for PDEformer."""
    DATA_COLUMN_NAMES = ['node_type', 'node_scalar', 'node_function',
                         'in_degree', 'out_degree', 'attn_bias',
                         'spatial_pos', 'coordinate', 'u_label']

    def __getitem__(self, idx_data: int) -> Tuple[NDArray[float]]:
        idx_pde, idx_var = divmod(idx_data, self.input_dataset.n_vars)

        input_field, input_scalar, coordinate, u_label = super().__getitem__(idx_pde)
        dag_tuple = self.input_dataset.get_pde_dag_info(
            idx_pde, idx_var, input_field, input_scalar)
        ui_label = u_label[:, [idx_var]]  # [n_txyz_pts, 1]

        data_tuple = (*dag_tuple, coordinate, ui_label)
        return data_tuple

    def __len__(self):
        return self.n_samples * self.input_dataset.n_vars

    def get_pde_info(self, idx_data: int) -> Dict[str, Any]:
        idx_pde, idx_var = divmod(idx_data, self.input_dataset.n_vars)
        if self.test:
            idx_pde = len(self.input_dataset) - 1 - idx_pde
        data_info = self.input_dataset.get_pde_info(idx_pde, idx_var)
        return data_info


def preprocess_single_pde(config: DictConfig,
                          print_fn: Callable[[str], None] = print) -> None:
    r"""Preprocess datasets of a single PDE when required, eg. data interpolation."""
    if config.data.type != "single_pde":
        return  # nothing to do

    input_dataset_cls = pde_type_class_dict[config.data.single_pde.param_name]
    params = config.data.single_pde.train
    if "test" in config.data.single_pde:
        params.extend(config.data.single_pde.test)
    for pde_param in params:
        input_dataset_cls.preprocess_data(config, pde_param, print_fn)
    print_fn("All single PDE datasets are preprocessed.")


def get_dataset(config: DictConfig,
                pde_type: str,
                pde_param: Union[float, List[float]],
                n_samples: int,
                test: bool,
                for_eval: bool) -> Dataset:
    r"""Obtain PDE solution dataset for the current network model."""
    # input dataset (file handling)
    input_dataset_cls = pde_type_class_dict[pde_type]
    input_dataset = input_dataset_cls(config, pde_param)
    input_dataset.post_init(for_eval)

    # output dataset (model-specific)
    model_type = config.model_type.lower()
    if model_type in ["pdeformer", "pdeformer-2", "pdeformer2", "pf"]:
        data_cls = PDEformerPDEDataset
    elif model_type == "deeponet":
        data_cls = DeepONetPDEDataset
    elif model_type == "cnn_deeponet":
        data_cls = CNNDeepONetPDEDataset
    elif model_type in ["fno2d", "unet2d"]:
        data_cls = NO2DModelPDEDataset
    elif model_type in ["fno3d", "sno3d"]:
        data_cls = NO3DModelPDEDataset
    else:
        raise NotImplementedError(f"unknown model_type: {model_type}")

    return data_cls(config, input_dataset, n_samples, test, for_eval)


def gen_loader_dict(config: DictConfig,
                    n_samples: int,
                    pde_param_list: Union[List[float], List[List[float]]],
                    batch_size: int,
                    test: bool = False) -> Dict[str, Dict[str, Tuple]]:
    r"""
    Generate a dictionary containing the dataloaders (`BatchDataset` class
    objects in MindSpore) for the training or testing datasets.
    """
    for_eval = True
    shuffle = not for_eval
    pde_type = config.data.single_pde.param_name

    def dataloader_from_param(pde_param):
        dataset = get_dataset(config, pde_type, pde_param,
                              n_samples, test, for_eval)
        dataloader = datasets2loader(
            [dataset], batch_size, shuffle, config.data.num_workers,
            create_iter=True)
        return (dataloader, dataset)

    if config.eval.get("dataset_per_type", -1) >= 0:
        pde_param_list = pde_param_list[:config.eval.dataset_per_type]
    param_loader_dict = {pde_param: dataloader_from_param(pde_param)
                         for pde_param in pde_param_list}
    return {pde_type: param_loader_dict}


class RegularizedFineTuneDataset(Dataset):
    r"""
    To avoid overfitting when fine-tuning PDEformer on small datasets, we
    include the pre-training (multi_pde) dataset during the fine-tuning stage
    as a regularization.
    Each sample in `dataset`, with probability `regularize_ratio`, is replaced
    by a randomly selected sample from `regularize_dataset`.
    """
    DATA_COLUMN_NAMES = ['node_type', 'node_scalar', 'node_function',
                         'in_degree', 'out_degree', 'attn_bias',
                         'spatial_pos', 'coordinate', 'u_label']

    def __init__(self,
                 datasets: List[Dataset],
                 regularize_datasets: List[Dataset],
                 regularize_ratio: float) -> None:
        self.dataset = concat_datasets(datasets)
        self.regularize_dataset = concat_datasets(regularize_datasets)
        self.regularize_ratio = regularize_ratio
        self.num_regularize_data = len(self.regularize_dataset)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        if np.random.rand() < self.regularize_ratio:
            idx_reg = np.random.randint(self.num_regularize_data)
            return self.regularize_dataset[idx_reg]
        return self.dataset[idx_pde]

    def __len__(self) -> int:
        return len(self.dataset)


def single_pde_dataset(config: DictConfig) -> Tuple:
    r"""
    Generate dataloaders for the single_pde datasets.

    Args:
        config (DictConfig): Training configurations.

    Returns:
        dataloader_train (BatchDataset): Data loader for the training dataset.
        data_updater (Callable): A function that does nothing.
        train_loader_dict (Dict[str, Dict[str, Tuple]]): A nested
            dictionary containing the data iterator instances of the training
            dataset for the evaluation, in which the random operations (data
            augmentation, spatio-temporal subsampling, etc) are disabled. Here,
            to be more specific, 'Tuple' actually refers to
            'Concatenate[TupleIterator, Dataset]'.
        test_loader_dict (Dict[str, Dict[str, Tuple]]): Similar to
            `train_loader_dict`, but for the testing dataset.
    """
    num_samples_train = config.data.num_samples_per_file.train
    num_samples_test = config.data.num_samples_per_file.test
    train_params = config.data.single_pde.train
    test_params = config.data.single_pde.get("test", train_params)

    train_datasets = [get_dataset(
        config, config.data.single_pde.param_name, pde_param,
        num_samples_train, test=False, for_eval=False,
    ) for pde_param in train_params]
    train_datasets[0].add_model_config_(config)
    regularize_ratio = config.data.single_pde.get("regularize_ratio", 0.)
    if regularize_ratio > 0 and config.model_type == "pdeformer":
        # include regularization dataset (custom multi_pde utilized in
        # pre-training) during the fine-tuning stage
        if not config.train.num_txyz_samp_pts > 0:
            raise ValueError("When 'regularize_ratio' is positive, "
                             "'num_txyz_samp_pts' should be positive as well.")
        reg_dataset_dict = StaticDatasetFakeUpdater(
            config,
            config.data.multi_pde.train,
            config.data.num_samples_per_file.regularize
        ).train_dataset_dict

        # {pde_type: [dataset]} -> [dataset]
        regularize_datasets = []
        for dataset_list in reg_dataset_dict.values():
            regularize_datasets.extend(dataset_list)
        finetune_dataset = RegularizedFineTuneDataset(
            train_datasets, regularize_datasets, regularize_ratio)
        train_datasets = [finetune_dataset]
    dataloader_train = datasets2loader(
        train_datasets, config.train.total_batch_size, True,
        config.data.num_workers, create_iter=False)

    train_loader_dict = gen_loader_dict(
        config, num_samples_train, train_params,
        batch_size=config.eval.total_batch_size)
    test_loader_dict = gen_loader_dict(
        config, num_samples_test, test_params,
        batch_size=config.eval.total_batch_size, test=True)

    def data_updater(*_):
        return  # doing nothing
    out_tuple = (dataloader_train, data_updater, train_loader_dict, test_loader_dict)
    return out_tuple

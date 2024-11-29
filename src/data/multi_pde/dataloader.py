r"""Loading custom dataset consisting of multiple PDEs (multi_pde)."""
import os
import shutil
import time
from typing import Tuple, Dict, List, Union, Callable

from omegaconf import DictConfig
from mindspore.dataset import BatchDataset
from mindspore.communication import get_rank

from ..utils_dataload import datasets2loader
from . import pde_types
from .datasets import get_pde_dataset_cls, MultiPDEDatasetBase


def gen_dataloader(config: DictConfig,
                   n_samples: int,
                   datafile_dict: Dict[str, pde_types.FileListType],
                   batch_size: int) -> BatchDataset:
    r"""
    Generate the dataloader (`BatchDataset` class object in MindSpore) for
    the training dataset.
    """
    shuffle = True

    # {pde_type: [filename]} -> [filename] -> dataloader
    datasets = []
    for pde_type, file_list in datafile_dict.items():
        pde_dataset_cls = get_pde_dataset_cls(pde_type)
        file_list = pde_types.gen_file_list(file_list)
        for filename in file_list:
            datasets.append(pde_dataset_cls(
                config, filename, n_samples, test=False, for_eval=False))

    return datasets2loader(datasets, batch_size, shuffle,
                           config.data.num_workers, create_iter=False)


def gen_loader_dict(config: DictConfig,
                    n_samples: int,
                    datafile_dict: Dict[str, pde_types.FileListType],
                    batch_size: int,
                    test: bool = False) -> Dict[str, Dict[str, Tuple]]:
    r"""
    Generate a nested dictionary containing the dataloaders (tuple_iterators in
    MindSpore) for the training or testing datasets.
    """
    shuffle = False

    def dataloader_from_file(pde_type, filename):
        pde_dataset_cls = get_pde_dataset_cls(pde_type)
        dataset = pde_dataset_cls(
            config, filename, n_samples, test, for_eval=True)
        dataloader = datasets2loader(
            [dataset], batch_size, shuffle, config.data.num_workers,
            create_iter=True)
        return (dataloader, dataset)

    # {pde_type: [filename]} -> {pde_type: {filename: (dataloader, dataset)}}
    loader_dict = {}
    for pde_type, file_list in datafile_dict.items():
        file_list = pde_types.gen_file_list(file_list)
        if config.eval.get("dataset_per_type", -1) >= 0:
            file_list = file_list[:config.eval.dataset_per_type]
        loader_dict[pde_type] = {fname: dataloader_from_file(pde_type, fname)
                                 for fname in file_list}
    return loader_dict


def static_multi_pde_dataset(config: DictConfig) -> Tuple:
    r"""
    Generate dataloaders for the custom multi_pde datasets.

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
    train_file_dict = config.data.multi_pde.train

    dataloader_train = gen_dataloader(
        config, num_samples_train, train_file_dict,
        batch_size=config.train.total_batch_size)
    train_loader_dict = gen_loader_dict(
        config, num_samples_train, train_file_dict,
        batch_size=config.eval.total_batch_size)

    if "test" in config.data.multi_pde.keys():
        test_file_dict = config.data.multi_pde.test
    else:
        min_total_samples = num_samples_train + num_samples_test
        for type_loader_dict in train_loader_dict.values():
            for _, dataset in type_loader_dict.values():
                min_total_samples = min(min_total_samples, dataset.total_samples)
        if num_samples_train + num_samples_test > min_total_samples:
            raise ValueError(
                "When the test set is not specified, the sum of "
                f"'num_samples_train' ({num_samples_train}) and "
                f"'num_samples_test' ({num_samples_test}) should not "
                f"exceed the total number of samples "
                f"({min_total_samples}).")
        test_file_dict = train_file_dict

    test_loader_dict = gen_loader_dict(
        config, num_samples_test, test_file_dict,
        batch_size=config.eval.total_batch_size, test=True)

    def data_updater(*args):
        return args  # doing nothing
    out_tuple = (dataloader_train, data_updater, train_loader_dict, test_loader_dict)
    return out_tuple


class StaticDatasetFakeUpdater:
    r"""
    Maintaining the nested dictionary containing the training (or testing)
    dataset objects.
    """
    train_dataset_dict: Dict[str, List[MultiPDEDatasetBase]]
    eval_dataset_dict: Dict[str, List[MultiPDEDatasetBase]]

    def __init__(self,
                 config: DictConfig,
                 datafile_dict: Dict[str, pde_types.FileListType],
                 n_samples: int,
                 test: bool = False) -> None:
        self.train_dataset_dict = {}
        self.eval_dataset_dict = {}
        eval_per_type = config.eval.dataset_per_type
        if eval_per_type < 0:
            eval_per_type = None

        for pde_type, file_list in datafile_dict.items():
            pde_dataset_cls = get_pde_dataset_cls(pde_type)
            # FileListType -> List[str]
            file_list = pde_types.gen_file_list(file_list)
            self.eval_dataset_dict[pde_type] = [
                pde_dataset_cls(config, filename, n_samples,
                                test=test, for_eval=True)
                for filename in file_list[:eval_per_type]]
            if test:
                continue
            self.train_dataset_dict[pde_type] = [
                pde_dataset_cls(config, filename, n_samples)
                for filename in file_list]

    def __call__(self, print_fn: Union[str, Callable[[str], None]]) -> None:
        pass  # Nothing to do by default

    def train_dataloader(self,
                         batch_size: int,
                         num_workers: int) -> BatchDataset:
        r"""
        Generate the dataloader (`BatchDataset` class object in MindSpore) for
        the training dataset.
        """
        shuffle = True

        # {pde_type: [dataset]} -> [dataset] -> dataloader
        datasets = []
        for dataset_list in self.train_dataset_dict.values():
            datasets.extend(dataset_list)
        return datasets2loader(datasets, batch_size, shuffle,
                               num_workers, create_iter=False)

    def eval_loader_dict(self,
                         batch_size: int,
                         num_workers: int = 1) -> Dict[str, Dict[int, Tuple]]:
        r"""
        Generate a nested dictionary containing the dataloaders
        (`TupleIterator` in MindSpore) tailored for model evaluation, applied
        to both the training and the testing datasets.

        Returns: loader_dict (Dict[str, Dict[int, Tuple]])
            format: {pde_type: {idx: (dataloader, dataset)}}
        """
        shuffle = False

        def loader_tuple(dataset: MultiPDEDatasetBase) -> Tuple:
            dataloader = datasets2loader(
                [dataset], batch_size, shuffle, num_workers, create_iter=True)
            return (dataloader, dataset)

        loader_dict = {}
        for pde_type, dset_list in self.eval_dataset_dict.items():
            loader_dict[pde_type] = {i: loader_tuple(dataset)
                                     for i, dataset in enumerate(dset_list)}
        return loader_dict


class DynamicDatasetUpdater(StaticDatasetFakeUpdater):
    r"""
    Maintaining the nested dictionary containing the training (or testing)
    dataset objects, and update the mapping from the logical dataset objects to
    the (dynamically updated) real dataset files.
    """
    map_info_dir: str
    distributed_name: str
    datafile_dict: Dict[str, List[str]]
    DYN_MANAGER_INSTRUCTION = (
        "Before you start training, please make sure the process "
        "'dynamic_dataset_maintainer.py' is running. You can start this "
        "process by running the following command in another terminal:\n\n\t "
        "python3 dynamic_dataset_manager.py -c CONFIG_PATH\n")

    def __init__(self,  # pylint: disable=super-init-not-called
                 config: DictConfig,
                 pde_type_list: List[str],
                 n_samples: int) -> None:
        self.map_info_dir = os.path.join(
            config.data.path, pde_types.DYN_DSET_COMM_DIR)
        try:  # data parallel case
            self.distributed_name = f"used_by_rank{get_rank()}"
        except RuntimeError:
            self.distributed_name = "used_by_single"

        self.datafile_dict = {}
        self.train_dataset_dict = {}
        self.eval_dataset_dict = {}
        eval_per_type = config.eval.dataset_per_type
        if eval_per_type < 0:
            eval_per_type = None

        for pde_type in pde_type_list:
            # Read file_list from DATA_PATH/dyn_dset_comm/init/PDE_TYPE
            comm_file_path = os.path.join(self.map_info_dir, "init", pde_type)
            if not os.path.exists(comm_file_path):
                raise FileNotFoundError(f"Cannot read file {comm_file_path}. "
                                        + self.DYN_MANAGER_INSTRUCTION)
            with open(comm_file_path, "r", encoding="UTF-8") as comm_file:
                file_list = comm_file.read().split("\n")
            self.datafile_dict[pde_type] = file_list
            pde_dataset_cls = get_pde_dataset_cls(pde_type)
            self.train_dataset_dict[pde_type] = [
                pde_dataset_cls(config, filename, n_samples, test=False)
                for filename in file_list]
            self.eval_dataset_dict[pde_type] = [
                pde_dataset_cls(config, filename, n_samples,
                                test=False, for_eval=True)
                for filename in file_list[:eval_per_type]]

            # Mark data file occupation by creating communication files:
            # DATA_PATH/dyn_dset_comm/file2rank/FILENAME/DISTRIBUTED_NAME
            for filename in file_list:
                comm_file_path = os.path.join(self.map_info_dir, "file2rank",
                                              filename, self.distributed_name)
                with open(comm_file_path, "w", encoding="UTF-8") as comm_file:
                    comm_file.write("")

        # Inform initialization of this logical dataset by creating
        # communication directory DATA_PATH/dyn_dset_comm/init_done
        comm_file_dir = os.path.join(self.map_info_dir, "init_done")
        os.makedirs(comm_file_dir, exist_ok=True)

    def __call__(self, print_fn: Union[str, Callable[[str], None]]) -> None:
        # training done case
        if isinstance(print_fn, str) and print_fn == "terminate":
            # Mark training termination by creating communication directory
            # DATA_PATH/dyn_dset_comm/terminate/
            comm_file_dir = os.path.join(self.map_info_dir, "terminate")
            os.makedirs(comm_file_dir, exist_ok=True)
            return

        # normal update case
        time_start = time.time()
        n_updated = 0
        for pde_type, file_list in self.datafile_dict.items():
            for i, filename in enumerate(file_list):
                # Get name of the current data from communication file
                # DATA_PATH/dyn_dset_comm/logical2file/PDE_TYPE/IDX
                comm_file_path = os.path.join(
                    self.map_info_dir, "logical2file", pde_type, str(i))
                with open(comm_file_path, "r", encoding="UTF-8") as comm_file:
                    filename_new = comm_file.read()
                if filename_new == filename:
                    continue

                # Apply the update
                n_updated += 1
                file_list[i] = filename_new
                self.train_dataset_dict[pde_type][i].use_datafile(filename_new)
                if i < len(self.eval_dataset_dict[pde_type]):
                    self.eval_dataset_dict[pde_type][i].use_datafile(filename_new)

                # Update data file occupation by moving communication files:
                # DATA_PATH/dyn_dset_comm/file2rank/FILENAME/DISTRIBUTED_NAME
                comm_file_path = os.path.join(self.map_info_dir, "file2rank",
                                              filename, self.distributed_name)
                target_path = os.path.join(self.map_info_dir, "file2rank",
                                           filename_new, self.distributed_name)
                shutil.move(comm_file_path, target_path)

        if n_updated > 0:
            time_elapsed = time.time() - time_start
            print_fn(f"Updated {n_updated} dataset(s) in {time_elapsed:.1f}s.")


def dynamic_multi_pde_dataset(config: DictConfig) -> Tuple:
    r"""
    Generate dataloaders for the custom multi_pde datasets.

    Args:
        config (DictConfig): Training configurations.

    Returns:
        dataloader_train (BatchDataset): Data loader for the training dataset.
        data_updater (Callable): A function to update the mappings between the
            logical datasets and the physical data files for dynamic datasets.
        train_loader_dict (Dict[str, Dict[str, Tuple]]): A nested
            dictionary containing the data iterator instances of the training
            dataset for the evaluation, in which the random operations (data
            augmentation, spatio-temporal subsampling, etc) are disabled. Here,
            to be more specific, 'Tuple' actually refers to
            'Concatenate[TupleIterator, Dataset]'.
        test_loader_dict (Dict[str, Dict[str, Tuple]]): Similar to
            `train_loader_dict`, but for the testing dataset.
    """
    # dict of train/test data file
    num_samples_train = config.data.num_samples_per_file.train
    num_samples_test = config.data.num_samples_per_file.test
    train_file_dict = config.data.multi_pde.train
    if "test" in config.data.multi_pde:
        test_file_dict = config.data.multi_pde.test
    elif num_samples_train + num_samples_test > 1000:
        raise ValueError(
            "When the test set is not specified, the sum of "
            f"'num_samples_train' ({num_samples_train}) and "
            f"'num_samples_test' ({num_samples_test}) should not exceed "
            "1000.")
    else:
        test_file_dict = train_file_dict

    # training dataloaders
    if "dynamic" in config.data and config.data.dynamic.enabled:
        data_updater = DynamicDatasetUpdater(
            config, list(train_file_dict), num_samples_train)
    else:
        data_updater = StaticDatasetFakeUpdater(
            config, train_file_dict, num_samples_train)
    dataloader_train = data_updater.train_dataloader(
        config.train.total_batch_size, config.data.num_workers)
    train_loader_dict = data_updater.eval_loader_dict(
        config.eval.total_batch_size,
        # config.data.num_workers)
        num_workers=1)

    if "test" in config.data.multi_pde.keys():
        test_file_dict = config.data.multi_pde.test
    else:
        min_total_samples = num_samples_train + num_samples_test
        for type_loader_dict in train_loader_dict.values():
            for _, dataset in type_loader_dict.values():
                min_total_samples = min(min_total_samples, dataset.total_samples)
        if num_samples_train + num_samples_test > min_total_samples:
            raise ValueError(
                "When the test set is not specified, the sum of "
                f"'num_samples_train' ({num_samples_train}) and "
                f"'num_samples_test' ({num_samples_test}) should not "
                f"exceed the total number of samples "
                f"({min_total_samples}).")
        test_file_dict = train_file_dict

    test_updater = StaticDatasetFakeUpdater(
        config, test_file_dict, num_samples_test, test=True)
    test_loader_dict = test_updater.eval_loader_dict(
        config.eval.total_batch_size,
        # config.data.num_workers)
        num_workers=1)

    out_tuple = (dataloader_train, data_updater, train_loader_dict, test_loader_dict)
    return out_tuple

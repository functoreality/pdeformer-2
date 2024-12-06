r"""
Managing the dynamic buffer dataset for large-scale multi_pde training. To be
more specific, we assume that the training process loads data from the local
disk. When the size of the complete training dataset exceeds the capacity of
the local disk, we have to treat the local disk as a "buffer" that stores only
a subset of the complete training dataset, and dynamically replace the existing
data in this subset by new data not in this subset.

The complete training dataset may be 1. stored in a remote disk with large
capacity (in the case of the authors, the OBS server), or 2. compressed
(currently not implemented). If users want to use this script to execute
dynamic buffered training, you may need to inherit the class
'DataFileManagerBase', and implement your own methods to prepare new data on
the local disk.

The training script loads data from a fixed set of "logical datasets". During
the whole training process, a logical dataset may correspond to different
physical data files that are prepared in the local disk. This script maintains
the mapping relations from the logical datasets to the physical data files.
"""
import time
import os
import sys
import shutil
import logging
import argparse
from subprocess import check_output
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict, List

from omegaconf import DictConfig

from src.utils import load_config
from src.data.multi_pde.pde_types import (
    get_pde_info_cls, gen_file_list, DYN_DSET_COMM_DIR,
    DAG_INFO_DIR, FileListType)

FileNum = namedtuple("FileNum", ("n_logical_dataset", "max_inactive_files"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
cli_handler = logging.StreamHandler(sys.stdout)
cli_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(cli_handler)


class DataFileManagerBase(ABC):
    r"""
    A base class of a data file manager, responsible for data file preparation,
    preprocessing and deleting.
    """

    def __init__(self, data_path: str, remove_type: str) -> None:
        self.data_path = data_path
        self.remove_type = remove_type.lower()
        os.makedirs(os.path.join(data_path, DAG_INFO_DIR), exist_ok=True)

    def rearrange_file_list(self, file_list: List[str]) -> List[str]:
        r"""
        Getting a rearranged list of data files, putting existing local files
        before the unprepared ones.
        """
        exist_list = []
        nonexist_list = []
        for filename in file_list:
            file_path = os.path.join(self.data_path, filename + ".hdf5")
            if os.path.exists(file_path):
                exist_list.append(filename)
            else:
                nonexist_list.append(filename)
        return exist_list + nonexist_list

    def require(self,
                pde_type: str,
                filename: str,
                config: DictConfig) -> None:
        r"""
        Requiring access to a specific data file. May involve data downloading
        and preprocessing.
        """
        file_path = os.path.join(self.data_path, filename + ".hdf5")
        if not os.path.exists(file_path):
            self._prepare_data_file(pde_type, filename, config)

        # Preprocess DAG info
        pde_info_cls = get_pde_info_cls(pde_type)
        suffix = pde_info_cls.dag_file_suffix(config)
        file_path = os.path.join(config.data.path, DAG_INFO_DIR,
                                 filename + suffix)
        if not os.path.exists(file_path):
            self._prepare_dag_info(pde_info_cls, filename, config)

    def remove(self, filename: str) -> None:
        r"""
        Remove a specific data file that it is (temporarily) no longer used.
        """
        if self.remove_type == "none":
            return  # nothing to do
        file_path = os.path.join(self.data_path, filename + ".hdf5")
        os.remove(file_path)
        if self.remove_type != "file+dag":
            return
        dag_dir = os.path.join(self.data_path, DAG_INFO_DIR)
        dag_file_list = os.listdir(dag_dir)
        for dag_file in dag_file_list:
            if dag_file.startswith(filename):
                os.remove(os.path.join(dag_dir, dag_file))

    @staticmethod
    @abstractmethod
    def _prepare_data_file(pde_type: str,
                           filename: str,
                           config: DictConfig) -> None:
        r"""Prepare a new data file on the local disk."""

    @staticmethod
    def _prepare_dag_info(pde_info_cls: type,
                          filename: str,
                          config: DictConfig) -> None:
        r"""Prepare the preprocessed DAG information of a new data file."""
        pde_info_cls.gen_dag_info(filename, config, logger.debug)


class OBSDataFileManager(DataFileManagerBase):
    r"""
    A data file manager that prepares new training data files by downloading
    from the remote OBS server.
    """

    @staticmethod
    def _prepare_data_file(pde_type: str,
                           filename: str,
                           config: DictConfig) -> None:
        file_path = os.path.join(config.data.path, filename + ".hdf5")

        # Download from OBS
        obs_path_dict = config.data.dynamic.obs_path
        type_key = pde_type
        if type_key not in obs_path_dict:
            type_key = type_key.split("_")[0]  # "dcrLgK_sJ3" -> "dcrLgK"
        if type_key not in obs_path_dict and type_key.endswith("LgK"):
            type_key = type_key[:-3]  # "dcrLgK" -> "dcr"
        obs_file = os.path.join(obs_path_dict[type_key], filename + ".hdf5")
        logger.debug("Downloading %s -> %s.", obs_file, file_path)
        obsutil = config.data.dynamic.obsutil
        os.system(f"{obsutil} cp {obs_file} {file_path} > /dev/null")

    @classmethod
    def _prepare_dag_info(cls,
                          pde_info_cls: type,
                          filename: str,
                          config: DictConfig) -> None:
        suffix = pde_info_cls.dag_file_suffix(config)
        file_path = os.path.join(config.data.path, DAG_INFO_DIR,
                                 filename + suffix)

        # check OBS file
        obs_file = "/".join([config.data.dynamic.obs_path.dag_info,
                             filename + suffix])
        obsutil = config.data.dynamic.obsutil
        obs_out = check_output([obsutil, "ls", "-s", obs_file], text=True)
        if obs_out[-2] == "1":
            logger.debug("Downloading %s -> %s.", obs_file, file_path)
            os.system(f"{obsutil} cp {obs_file} {file_path} > /dev/null")
            # check_output([obsutil, "cp", obs_file, file_path])
        elif obs_out[-2] == "0":
            super()._prepare_dag_info(pde_info_cls, filename, config)
            if config.data.dynamic.upload_dag:
                logger.debug("Uploading %s -> %s.", file_path, obs_file)
                os.system(f"{obsutil} cp {file_path} {obs_file}")
        else:
            raise RuntimeError(f"Unexpected 'obs_out':\n{obs_out}.")


DataFileManager = OBSDataFileManager


class SingleTypeMappingManager:
    r"""
    Maintaining the mapping between logical datasets and physical data files
    for a single PDE type.
    """

    def __init__(self,
                 config_data: DictConfig,
                 pde_type: str,
                 file_num: FileNum) -> None:
        # dynamically updated attributes
        file_list = config_data.multi_pde.train[pde_type]
        self.file_list = gen_file_list(file_list)
        self.active_file_bottom_idx = 0  # range [0, max_inactive_files)
        self.bottom_dataset_idx = 0  # range [0, n_logical_dataset)

        # set-once attributes
        self.pde_type = pde_type
        self.map_info_dir = os.path.join(config_data.path, DYN_DSET_COMM_DIR)
        self.n_logical_dataset = min(file_num.n_logical_dataset,
                                     len(self.file_list))
        self.max_inactive_files = min(file_num.max_inactive_files,
                                      self.n_logical_dataset)
        n_unprepared_files = (len(self.file_list) - self.n_logical_dataset
                              - self.max_inactive_files)
        self.disable_remove = n_unprepared_files <= 0

    def init_mapping(self,
                     file_mgr: DataFileManagerBase,
                     config: DictConfig) -> None:
        r"""Initialize the mapping from logical datasets to real data files."""
        # Prepare directory for communication files:
        # DATA_PATH/dyn_dset_comm/logical2file/PDE_TYPE/IDX
        comm_file_dir = os.path.join(
            self.map_info_dir, "logical2file", self.pde_type)
        os.makedirs(comm_file_dir)
        # DATA_PATH/dyn_dset_comm/file2rank/FILENAME/DISTRIBUTED_NAME
        for filename in self.file_list:
            comm_file_dir = os.path.join(
                self.map_info_dir, "file2rank", filename)
            os.makedirs(comm_file_dir)

        self.file_list = file_mgr.rearrange_file_list(self.file_list)
        for i in range(self.n_logical_dataset):
            filename = self.file_list[i]

            # Prepare the data files required to initialize logical datasets
            file_mgr.require(self.pde_type, filename, config)

            # map the first logical dataset to the new file by writing
            # communication file DATA_PATH/dyn_dset_comm/logical2file/PDE_TYPE/IDX
            comm_file_path = os.path.join(
                self.map_info_dir, "logical2file", self.pde_type, str(i))
            with open(comm_file_path, "w", encoding="UTF-8") as comm_file:
                comm_file.write(filename)

        # Write initializing file_list to communication file
        # DATA_PATH/dyn_dset_comm/init/PDE_TYPE
        init_file_list = self.file_list[:self.n_logical_dataset]
        comm_file_path = os.path.join(self.map_info_dir, "init", self.pde_type)
        with open(comm_file_path, "w", encoding="UTF-8") as comm_file:
            comm_file.write("\n".join(init_file_list))

    def get_a_new_file(self,
                       file_mgr: DataFileManagerBase,
                       config: DictConfig) -> bool:
        r"""
        Download and preprocess a new data file, as described in
        `docs/images/dynamic_dataset_logic-3.png`.
        """
        if self.active_file_bottom_idx >= self.max_inactive_files:
            # Need to wait until the existing mapped files are occupied by the
            # logical datasets.
            return False  # no new file obtained
        if self.n_logical_dataset == len(self.file_list):
            # All files are being loaded, no need to get a new one.
            return True

        # prepare a new data file
        file_idx = self.active_file_bottom_idx + self.n_logical_dataset
        filename = self.file_list[file_idx]
        file_mgr.require(self.pde_type, filename, config)
        logger.debug("Data file ready: (%s) %s", self.pde_type, filename)

        # map the first logical dataset to the new file by writing
        # communication file DATA_PATH/dyn_dset_comm/logical2file/PDE_TYPE/IDX
        comm_file_path = os.path.join(
            self.map_info_dir, "logical2file", self.pde_type,
            str(self.bottom_dataset_idx))
        with open(comm_file_path, "w", encoding="UTF-8") as comm_file:
            comm_file.write(filename)

        # update class attributes
        self.active_file_bottom_idx += 1
        self.bottom_dataset_idx += 1
        if self.bottom_dataset_idx >= self.n_logical_dataset:
            self.bottom_dataset_idx -= self.n_logical_dataset

        return True

    def remove_unused_files(self, file_mgr: DataFileManagerBase) -> None:
        r"""
        Remove data files that are no longer used by the logical datasets (for
        all ranks, if distributed training is employed), as described in
        `docs/images/dynamic_dataset_logic-2.png`.
        """
        while self._remove_one_unused_file(file_mgr):
            pass  # loop until all unused files are moved

    def _remove_one_unused_file(self, file_mgr: DataFileManagerBase) -> bool:
        r"""
        Remove a data file that is no longer used by the logical datasets (for
        all ranks, if distributed training is employed), as described in
        `docs/images/dynamic_dataset_logic-2.png`.
        """
        if self.disable_remove or self.active_file_bottom_idx == 0:
            return False  # nothing need to remove

        filename = self.file_list[0]

        # Check data file occupation by counting communication files:
        # DATA_PATH/dyn_dset_comm/file2rank/FILENAME/DISTRIBUTED_NAME
        comm_file_dir = os.path.join(self.map_info_dir, "file2rank", filename)
        if os.listdir(comm_file_dir):  # list non-empty
            return False  # still occupied by some logical dataset

        # Apply removal
        file_mgr.remove(filename)
        logger.debug("Data file removed: (%s) %s", self.pde_type, filename)
        # Move first element to last, i.e. [a,b,c,..,f] -> [b,c,..,f,a].
        self.file_list.append(self.file_list.pop(0))
        self.active_file_bottom_idx -= 1
        return True


class OverallMappingManager:
    r"""
    Maintaining the mapping between logical datasets and physical data files
    for all PDE types.
    """

    def __init__(self, config_data: DictConfig) -> None:
        file_num_dict = self._get_file_num_dict(config_data)
        self.type_mgr_list = [SingleTypeMappingManager(config_data, pde_type,
                                                       file_num_dict[pde_type])
                              for pde_type in config_data.multi_pde.train]
        self.file_mgr = DataFileManager(
            config_data.path, config_data.dynamic.remove_type)
        # self.current_type_idx = 0  # range [0, len(type_mgr_list))
        self.map_info_dir = os.path.join(config_data.path, DYN_DSET_COMM_DIR)
        # self.n_processes = config_data.dynamic.n_processes

    def init_mapping(self, config: DictConfig) -> None:
        r"""Initialize the mapping from logical datasets to real data files."""
        # remove the existing communication directory
        if os.path.exists(self.map_info_dir):
            shutil.rmtree(self.map_info_dir)
            # If another dynamic-dataset manager process is running, wait until
            # it raises error and quits.
            time.sleep(2 * config.data.dynamic.wait_time)

        # Prepare test datasets.
        self.prepare_test_data(config)

        # Prepare directory for communication files:
        # DATA_PATH/dyn_dset_comm/init/PDE_TYPE
        os.makedirs(os.path.join(self.map_info_dir, "init"))

        # Initialize mappings of all PDE types.
        for type_mgr in self.type_mgr_list:
            type_mgr.init_mapping(self.file_mgr, config)

        # Inform manager initialization by creating communication directory
        # DATA_PATH/dyn_dset_comm/mgr_init
        os.makedirs(os.path.join(self.map_info_dir, "mgr_init"))

        logging.info("Dynamic-dataset manager initialized. "
                     "You can start the training process.")

    def logical_datasets_unprepared(self) -> bool:
        r"""Check whether the logical datasets need to be prepared."""
        # Detect preparation of logical datasets by checking communication
        # directory DATA_PATH/dyn_dset_comm/init_done
        comm_file_dir = os.path.join(self.map_info_dir, "init_done")
        if os.path.exists(comm_file_dir):
            # We cannot simply remove the 'init' dir: Need to wait until
            # logical datasets of all MPI ranks are prepared.
            # shutil.rmtree(os.path.join(self.map_info_dir, "init"))
            logging.info("Detected initialized logical datasets.")
            return False

        return True

    def training_ongoing(self) -> bool:
        r"""Check whether the training script is still running."""
        # Detecting training termination by checking communication directory
        # DATA_PATH/dyn_dset_comm/terminate/
        comm_file_dir = os.path.join(self.map_info_dir, "terminate")
        if not os.path.exists(comm_file_dir):
            return True
        # remove the current communication directory
        shutil.rmtree(self.map_info_dir)
        logging.info("Detected training done.")
        return False

    def get_a_new_file(self, config: DictConfig) -> bool:
        r"""
        Download and preprocess a new data file, as described in
        `docs/images/dynamic_dataset_logic-3.png`.
        """
        num_type_mgr = len(self.type_mgr_list)
        for i in range(num_type_mgr):
            if not self.type_mgr_list[i].get_a_new_file(self.file_mgr, config):
                continue
            # move the visited type_mgr's to the end of the list
            self.type_mgr_list = (
                self.type_mgr_list[i+1:] + self.type_mgr_list[:i+1])
            return True
        return False

    def remove_unused_files(self) -> None:
        r"""
        Remove data files that are no longer used by the logical datasets (for
        all ranks, if distributed training is employed), as described in
        `docs/images/dynamic_dataset_logic-2.png`.
        """
        for type_mgr in self.type_mgr_list:
            type_mgr.remove_unused_files(self.file_mgr)

    def prepare_test_data(self, config: DictConfig) -> None:
        r"""Prepare the (static) data files for the test dataset."""
        if "test" not in config.data.multi_pde:
            raise ValueError(
                "For dynamic multi_pde dataset, the test datasets must be "
                "specified in the configuration YAML file, and be disjoint "
                "from the training datasets.")
        for pde_type, filenames in config.data.multi_pde.test.items():
            file_list = gen_file_list(filenames)
            for filename in file_list:
                self.file_mgr.require(pde_type, filename, config)

        logging.info("All test datasets are prepared.")

    @staticmethod
    def _get_file_num_dict(config_data: DictConfig) -> Dict[str, FileNum]:
        r"""
        Get a dictionary specifying the number of data files used for each type
        of PDE.
        """
        default_num = FileNum(**config_data.dynamic.type_default)
        file_num_dict = {pde_type: default_num
                         for pde_type in config_data.multi_pde.train}
        if "type_custom" not in config_data.dynamic:
            return file_num_dict
        unknown_types = []
        for custom_type in config_data.dynamic.type_custom:  # iter over list
            custom_num = FileNum(custom_type.n_logical_dataset,
                                 custom_type.max_inactive_files)
            for pde_type in custom_type.types:
                if pde_type in file_num_dict:
                    file_num_dict[pde_type] = custom_num
                else:
                    unknown_types.append(pde_type)
        if unknown_types:  # list non-empty
            logger.warning("Unknown custom PDE types: %s.", unknown_types)
        return file_num_dict


class WaitTimeManager:
    r"""Wait until the prescribed time interval is reached."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.target_time = time.time() + self.interval

    def __call__(self) -> None:
        time.sleep(self.interval)
        self.target_time = time.time() + self.interval

    @property
    def proceed(self) -> bool:
        r"""Determine whether the prescribed waiting time is reached."""
        return time.time() >= self.target_time

    def wait_proceed(self) -> None:
        r"""Wait until the prescribed waiting time is reached."""
        time_left = self.target_time - time.time()
        if time_left > 0.:
            time.sleep(time_left)
        self.target_time = time.time() + self.interval


def prepare_static_dataset(config: DictConfig) -> None:
    r"""Main function to prepare the static dataset for training."""
    file_mgr = DataFileManager(config.data.path, "none")

    def process_file_dict(file_dict: Dict[str, FileListType]) -> None:
        for pde_type, filenames in file_dict.items():
            file_list = gen_file_list(filenames)
            for filename in file_list:
                file_mgr.require(pde_type, filename, config)

    process_file_dict(config.data.multi_pde.train)
    process_file_dict(config.data.multi_pde.get("test", {}))

    # Inform data being ready by creating communication directory
    # DATA_PATH/dyn_dset_comm/mgr_init
    os.makedirs(os.path.join(config.data.path, DYN_DSET_COMM_DIR, "mgr_init"))
    logging.info("Static dataset prepared. You can start training.")


def main() -> None:
    r"""Main function to prepare the training dataset."""
    # load config
    parser = argparse.ArgumentParser(
        description="Manager of the dynamic training dataset.")
    parser.add_argument("--config_file_path", "-c", type=str, required=True,
                        help="Path of the configuration YAML file.")
    args = parser.parse_args()
    config, _ = load_config(args.config_file_path)
    if config.data.dynamic.log_detail:
        logger.setLevel(logging.DEBUG)

    # static (non-dynamic) case
    if not config.data.dynamic.enabled:
        prepare_static_dataset(config)
        return

    # initialize
    mapping_mgr = OverallMappingManager(config.data)
    mapping_mgr.init_mapping(config)
    waiter = WaitTimeManager(config.data.dynamic.wait_time)

    # wait for logical datasets
    while mapping_mgr.logical_datasets_unprepared():
        mapping_mgr.get_a_new_file(config)
        waiter()

    # main loop during training
    while mapping_mgr.training_ongoing():
        mapping_mgr.remove_unused_files()
        mapping_mgr.get_a_new_file(config)
        waiter()


if __name__ == "__main__":
    main()

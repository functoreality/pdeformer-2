r"""Record experimental results and various outputs."""
import os
import sys
import stat
import time
import logging
import shutil
import csv
from argparse import Namespace

import pickle
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from mindspore import SummaryRecord, Tensor, save_checkpoint, nn

from .visual import plot_l2_error_and_epochs


def create_logger(path: str = "./log.log") -> logging.Logger:
    r"""
    Create a logger to save the experimental results.

    Args:
        path (str): The path to save the log file. Default: "./log.log".

    Returns:
        logging.Logger: The logger object.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = path
    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class Record:
    r"""
    Record experimental results and various outputs.
    This class provides methods to record experimental results, such as
    checkpoints, scalars, dictionaries, tables, and plots. It also provides
    utility functions to copy files and save models and data in different
    formats.

    Args:
        root_dir (str): The root directory where all the experimental results
            will be saved.
        enable_record (bool): Whether to enable the recording of experimental
            results. Default: False.
        enable_summary (bool): Whether to enable mindspore SummaryRecord.
            Default: True.
        enable_plot (bool): Whether to enable plotting of results. Default: True.
        enable_table (bool): Whether to enable saving results in table format.
            Default: True.
        inverse_problem (bool): Whether to enable recording for the inverse
            problem. Default: False.
    """

    def __init__(self,
                 root_dir: str,
                 enable_record: bool = False,
                 enable_summary: bool = True,
                 enable_plot: bool = True,
                 enable_table: bool = True,
                 inverse_problem: bool = False) -> None:
        self.enable_record = enable_record
        self.record_dir = os.path.join(root_dir, time.strftime('%Y-%m-%d-%H-%M-%S'))
        self.ckpt_dir = os.path.join(self.record_dir, 'ckpt')
        self.pkl_dir = os.path.join(self.record_dir, 'pkl')
        self.image2d_dir = os.path.join(self.record_dir, 'image2d')
        self.video2d_dir = os.path.join(self.record_dir, 'video2d')
        self.inverse_dir = os.path.join(self.record_dir, 'inverse')
        self.table_dir = os.path.join(self.record_dir, 'table')

        if not enable_record:
            return

        # create directory for save current experimental results
        os.makedirs(self.record_dir, exist_ok=True)

        # checkpoint
        os.makedirs(self.ckpt_dir, exist_ok=True)  # sub-directory for checkpoint

        # pickle
        os.makedirs(self.pkl_dir, exist_ok=True)  # sub-directory for pickle

        # logger
        self.logger = create_logger(path=os.path.join(self.record_dir, "results.log"))

        # Mindinsight SummaryRecord
        self.enable_summary = enable_summary
        if enable_summary:
            log_dir = os.path.join("summary", self.record_dir.replace("/", "#"))
            self.summary = SummaryRecord(log_dir)

        # visual
        self.enable_plot = enable_plot
        if enable_plot:
            os.makedirs(self.image2d_dir, exist_ok=True)  # sub-directory for image2d
            os.makedirs(self.video2d_dir, exist_ok=True)  # sub-directory for video2d
            if inverse_problem:
                os.makedirs(self.inverse_dir, exist_ok=True)

        # table
        self.enable_table = enable_table
        if enable_table:
            os.makedirs(self.table_dir, exist_ok=True)  # sub-directory for table
            self.dic = {}

    def print(self, info: str) -> None:
        r"""Print information to the console and log file."""
        if self.enable_record:
            self.logger.info(info)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        r"""Record the change of 'tag' with step."""
        if self.enable_record:
            if self.enable_summary:
                self.summary.add_value("scalar", tag, Tensor([value]))
                self.summary.record(step)
                self.summary.flush()

            if self.enable_table:
                self._append(tag, value)
                self._append(tag + " step", step)

    def add_dict(self, step: int, dic: dict, prefix: str = "") -> None:
        r"""Record the changes of items in 'dic' with step."""
        if self.enable_record:
            if self.enable_summary:
                for key, value in dic.items():
                    if prefix == "":
                        tag = key
                    else:
                        tag = f"{prefix}/{key}"

                    self.summary.add_value("scalar", tag, Tensor([value]))
                    self.summary.record(step)

                self.summary.flush()

            if self.enable_table:
                for key, value in dic.items():
                    if prefix == "":
                        tag = key
                    else:
                        tag = f"{prefix}/{key}"

                    self._append(tag, value)
                    self._append(tag + " step", step)

    def close(self, table_name: str = "result.csv") -> None:
        r"""Save the contents of 'self.summary' and 'self.dic' to a disk file."""
        if self.enable_record:
            if self.enable_table:
                df = pd.DataFrame({key: pd.Series(value)
                                   for key, value in self.dic.items()})
                file_path = os.path.join(self.table_dir, table_name)
                df.to_csv(file_path)

                if "test_all_all/l2_error_mean step" in df.columns:
                    data = []
                    data.append(df["test_all_all/l2_error_mean step"].dropna().tolist())
                    data.append(df["train_all_all/l2_error_mean"].dropna().tolist())
                    data.append(df["test_all_all/l2_error_mean"].dropna().tolist())
                    self.visual(plot_l2_error_and_epochs,
                                data,
                                "l2-error_and_epochs.png",
                                save_dir=self.table_dir)

            if self.enable_summary:
                self.summary.close()

    def copy_file(self, src_file_path: str, dest_file_name: str = "config.yaml") -> None:
        r"""Copy src_file to dst_file."""
        if self.enable_record:
            shutil.copyfile(src_file_path, os.path.join(self.record_dir, dest_file_name))

    def save_ckpt(self, model: nn.Cell, file_name: str = "model.ckpt") -> None:
        r"""Save model's weights to ckeckpoint file."""
        if self.enable_record:
            save_checkpoint(model, os.path.join(self.ckpt_dir, file_name))

    def save_table(self, data: list, file_name: str = "l2_error_t.csv") -> None:
        r"""Save data as a CSV format table file."""
        if self.enable_record and self.enable_table:
            file_path = os.path.join(self.table_dir, file_name)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(file_path, flags, modes), 'w') as f:
                f_csv = csv.writer(f)
                f_csv.writerows(data)

    def save_pickle(self, data: dict, file_name: str = "inverse.pkl") -> None:
        r"""Save data as a pickle file."""
        if self.enable_record:
            file_path = os.path.join(self.pkl_dir, file_name)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            modes = stat.S_IWUSR | stat.S_IRUSR
            with os.fdopen(os.open(file_path, flags, modes), 'wb') as f:
                pickle.dump(data, f)

    def visual(self, visual_func: callable, *args, **kwargs) -> None:
        r"""
        Plot the results using the specified function.

        Args:
            visual_func (callable): The function to plot.
            *args: The arguments of the function.
            **kwargs: The keyword arguments of the function.

        Returns:
            None.
        """
        if self.enable_record:
            visual_func(*args, **kwargs)

    def _append(self, key: str, value: float) -> None:
        r"""Save key-value pairs to dic."""
        if key in self.dic:
            self.dic[key].append(value)
        else:
            self.dic[key] = [value]


def init_record(rank_id: int,
                args: Namespace,
                config: DictConfig,
                inverse_problem: bool = False) -> Record:
    r'''
    Initialize the record object.

    Args:
        rank_id (int): The rank id of the current process.
        args (dict): The arguments of the current program.
        config (dict): The configuration of the current program.
        inverse_problem (bool): Whether to enable recording for the inverse
            problem. Default: False.

    Returns:
        Record: The record object.
    '''
    if hasattr(args, "distributed"):
        distributed = args.distributed
    else:
        distributed = False
    enable_record = not (distributed and rank_id > 0)
    record = Record(config.record_dir,
                    enable_record=enable_record,
                    enable_summary=False,
                    inverse_problem=inverse_problem)
    # copy the config file to the record directory
    record.copy_file(args.config_file_path, "config.yaml")
    record.print("Configuration:\n" + OmegaConf.to_yaml(config))
    record.print(f"PID: {os.getpid()}, device: {args.device_target}")
    return record

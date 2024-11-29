r"""This module provides functions to compute and record metrics."""
from typing import Dict
import numpy as np
from numpy.typing import NDArray
from mindspore import Tensor


def calculate_l2_error(pred: Tensor, label: Tensor) -> NDArray[float]:
    r"""
    Computes the relative L2 loss.

    Args:
        pred (Tensor): The shape of tensor is math:`(bsz, \ldots)`.
        label (Tensor): The shape of tensor is math:`(bsz, \ldots)`.

    Returns:
        Tensor: The relative L2 loss. The shape of tensor is math:`(bsz)`.
    """
    pred = pred.view((label.shape[0], -1)).asnumpy()  # [bsz, *]
    label = label.view((label.shape[0], -1)).asnumpy()  # [bsz, *]

    error_norm = np.linalg.norm(pred - label, ord=2, axis=1, keepdims=False)  # [bsz]
    label_norm = np.linalg.norm(label, ord=2, axis=1, keepdims=False)  # [bsz]
    relative_l2_error = error_norm / (label_norm + 1.0e-6)  # [bsz]
    relative_l2_error = relative_l2_error.clip(0, 5)  # [bsz]

    return relative_l2_error


class EvalErrorRecord:
    r"""Records the Evaluate errors on different datasets."""

    def __init__(self) -> None:
        self.dict = {}

    @staticmethod
    def dict2str(eval_error_dict: Dict[str, float]) -> str:
        r"""
        Converts the eval error dictionary to a string.

        Args:
            eval_error_dict (Dict[str, float]): The eval error dictionary.

        Returns:
            str: The string of the eval error dictionary.
        """
        tmp = ""
        for key, value in eval_error_dict.items():
            tmp += f"{key}: {value:>7f} "
        return tmp

    def append(self, pde: str, param: str, eval_error: NDArray[float]) -> Dict[str, float]:
        r"""
        Appends the eval error of a specific dataset to the record.

        Args:
            pde (str): The name of the PDE.
            param (str): The name of the PDE parameter.
            eval_error (NDArray[float]): The eval error of the dataset. The shape
                of tensor is :math:`(bsz)`.

        Returns:
            eval_error_dict (Dict[str, float]): The eval error of the specific dataset.
        """
        centered_min = np.percentile(eval_error, 1)
        centered_max = np.percentile(eval_error, 99)
        centered_mean = eval_error.clip(centered_min, centered_max).mean()

        eval_error_dict = {
            "eval_error_mean": eval_error.mean(),
            # "eval_error_min": eval_error.min(),
            "eval_error_max": eval_error.max(),
            "eval_error_centered_mean": centered_mean,
            # "eval_error_centered_min": centered_min,
            "eval_error_centered_max": centered_max,
        }

        if pde in self.dict:
            self.dict[pde][param] = eval_error_dict
        else:
            self.dict[pde] = {param: eval_error_dict}

        return eval_error_dict

    def reduce(self, pde: str) -> Dict[str, float]:
        r"""
        Reduces the eval error of each PDE which contains the different
        parameters to a single value.

        Args:
            pde (str): The name of the PDE.

        Returns:
            eval_error_dict (Dict[str, float]): The eval error of the specific PDE.
        """
        mean_tmp = []
        # min_tmp = []
        max_tmp = []
        centered_mean_tmp = []
        # centered_min_tmp = []
        centered_max_tmp = []

        if pde == "all":
            for value in self.dict.values():
                for sub_value in value.values():
                    mean_tmp.append(sub_value["eval_error_mean"])
                    # min_tmp.append(sub_value["eval_error_min"])
                    max_tmp.append(sub_value["eval_error_max"])
                    centered_mean_tmp.append(sub_value["eval_error_centered_mean"])
                    # centered_min_tmp.append(sub_value["eval_error_centered_min"])
                    centered_max_tmp.append(sub_value["eval_error_centered_max"])
        else:
            for value in self.dict[pde].values():
                mean_tmp.append(value["eval_error_mean"])
                # min_tmp.append(value["eval_error_min"])
                max_tmp.append(value["eval_error_max"])
                centered_mean_tmp.append(value["eval_error_centered_mean"])
                # centered_min_tmp.append(value["eval_error_centered_min"])
                centered_max_tmp.append(value["eval_error_centered_max"])

        if mean_tmp:  # i.e. mean_tmp is not an empty list
            eval_error_dict = {
                "eval_error_mean": np.mean(mean_tmp),
                # "eval_error_min": np.min(min_tmp),
                "eval_error_max": np.max(max_tmp),
                "eval_error_centered_mean": np.mean(centered_mean_tmp),
                # "eval_error_centered_min": np.min(centered_min_tmp),
                "eval_error_centered_max": np.max(centered_max_tmp),
            }
        else:
            eval_error_dict = {"eval_error_mean": -1.0,
                               "eval_error_max": -1.0,
                               "eval_error_centered_mean": -1.0,
                               "eval_error_centered_max": -1.0}

        return eval_error_dict


class L2ErrorRecord:
    r"""Records the L2 errors on different datasets."""

    def __init__(self) -> None:
        self.dict = {}

    @staticmethod
    def dict2str(l2_error_dict: Dict[str, float]) -> str:
        r"""
        Converts the L2 error dictionary to a string.

        Args:
            l2_error_dict (Dict[str, float]): The L2 error dictionary.

        Returns:
            str: The string of the L2 error dictionary.
        """
        tmp = ""
        for key, value in l2_error_dict.items():
            tmp += f"{key}: {value:>7f} "
        return tmp

    def append(self, pde: str, param: str, l2_error: NDArray[float]) -> Dict[str, float]:
        r"""
        Appends the L2 error of a specific dataset to the record.

        Args:
            pde (str): The name of the PDE.
            param (str): The name of the PDE parameter.
            l2_error (NDArray[float]): The L2 error of the dataset. The shape
                of tensor is :math:`(bsz)`.

        Returns:
            l2_error_dict (Dict[str, float]): The L2 error of the specific dataset.
        """
        centered_min = np.percentile(l2_error, 1)
        centered_max = np.percentile(l2_error, 99)
        centered_mean = l2_error.clip(centered_min, centered_max).mean()

        l2_error_dict = {
            "l2_error_mean": l2_error.mean(),
            # "l2_error_min": l2_error.min(),
            "l2_error_max": l2_error.max(),
            "l2_error_centered_mean": centered_mean,
            # "l2_error_centered_min": centered_min,
            "l2_error_centered_max": centered_max,
        }

        if pde in self.dict:
            self.dict[pde][param] = l2_error_dict
        else:
            self.dict[pde] = {param: l2_error_dict}

        return l2_error_dict

    def reduce(self, pde: str) -> Dict[str, float]:
        r"""
        Reduces the L2 error of each PDE which contains the different parameters to a single value.

        Args:
            pde (str): The name of the PDE.

        Returns:
            l2_error_dict (Dict[str, float]): The L2 error of the specific PDE.
        """
        mean_tmp = []
        # min_tmp = []
        max_tmp = []
        centered_mean_tmp = []
        # centered_min_tmp = []
        centered_max_tmp = []

        if pde == "all":
            for value in self.dict.values():
                for sub_value in value.values():
                    mean_tmp.append(sub_value["l2_error_mean"])
                    # min_tmp.append(sub_value["l2_error_min"])
                    max_tmp.append(sub_value["l2_error_max"])
                    centered_mean_tmp.append(sub_value["l2_error_centered_mean"])
                    # centered_min_tmp.append(sub_value["l2_error_centered_min"])
                    centered_max_tmp.append(sub_value["l2_error_centered_max"])
        else:
            for value in self.dict[pde].values():
                mean_tmp.append(value["l2_error_mean"])
                # min_tmp.append(value["l2_error_min"])
                max_tmp.append(value["l2_error_max"])
                centered_mean_tmp.append(value["l2_error_centered_mean"])
                # centered_min_tmp.append(value["l2_error_centered_min"])
                centered_max_tmp.append(value["l2_error_centered_max"])

        if mean_tmp:  # i.e. mean_tmp is not an empty list
            l2_error_dict = {
                "l2_error_mean": np.mean(mean_tmp),
                # "l2_error_min": np.min(min_tmp),
                "l2_error_max": np.max(max_tmp),
                "l2_error_centered_mean": np.mean(centered_mean_tmp),
                # "l2_error_centered_min": np.min(centered_min_tmp),
                "l2_error_centered_max": np.max(centered_max_tmp),
            }
        else:
            l2_error_dict = {"l2_error_mean": -1.0,
                             "l2_error_max": -1.0,
                             "l2_error_centered_mean": -1.0,
                             "l2_error_centered_max": -1.0}

        return l2_error_dict

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Preprocessing custom multi_pde data."""
import argparse
import time

from src.data.multi_pde import preprocess_dag_info
from src.data.single_pde import preprocess_single_pde
from src.utils.load_yaml import load_config


def timed_print(content: str) -> None:
    r"""Print message with the current time."""
    print(time.strftime("%H:%M:%S ") + content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDEformer data.")
    parser.add_argument("--config_file_path", "-c", type=str, required=True,
                        help="Path of the configuration YAML file.")
    args = parser.parse_args()
    config = load_config(args.config_file_path)
    preprocess_dag_info(config, timed_print)
    preprocess_single_pde(config, timed_print)

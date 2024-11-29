#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Preprocessing custom multi_pde data."""
import argparse

from src.data.multi_pde import preprocess_dag_info
from src.utils.load_yaml import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDEformer data.")
    parser.add_argument("--config_file_path", "-c", type=str,
                        default="test_config.yaml")
    args = parser.parse_args()
    config, _ = load_config(args.config_file_path)
    preprocess_dag_info(config)

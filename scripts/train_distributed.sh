#!/bin/bash

# path to the config file
config_path=configs/pretrain/model-L_small-data.yaml

# preprocess data
python preprocess_data.py --config_file_path $config_path

# train model with 8 Ascend NPUs
mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python train.py --config_file_path $config_path
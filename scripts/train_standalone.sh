#!/bin/bash

# path to the config file
config_path=configs/pretrain/model-S_standalone.yaml

# train model with a single device (Ascend by default)
python train.py --config_file_path $config_path --no_distributed --device_id 0

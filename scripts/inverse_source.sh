#!/bin/bash

# path to the config file
config_path=configs/inverse/inverse_source.yaml

# run the inversion process
python inverse_function.py --config_file_path $config_path --device_id 0

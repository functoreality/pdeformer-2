#!/bin/bash

# path to the config file
config_path=configs/inference/model-L.yaml

python -m src.ui.dcr -c $config_path -d -nx 48 -ny 48

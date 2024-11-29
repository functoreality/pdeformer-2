#!/bin/bash

# ============================================================================
# Train with dynamic buffer dataset, in the automatic manner. Starting such
# training successfully could be a bit difficult for the users, but we still
# decided to release this script to help users understand our large-scale
# pretraining process. And, in general, we do not expect that users would need
# to execute such large-data training like we do :)
#
# About dynamic-dataset training: We assume that the training process loads
# data from the local disk. In large-scale multi_pde pretraining, the size of
# the complete training dataset typically exceeds the capacity of the local
# disk. To tackle with this, we treat the local disk as a "buffer" that stores
# only a subset of the complete training dataset, and dynamically replace the
# existing data in this subset by new data not in this subset.
#
# The complete training dataset is stored in a remote disk with large capacity.
# If users want to execute dynamic buffered training, you may need to modify
# the Python script 'dynamic_dataset_manager.py', inherit the class
# 'DataFileManagerBase', and implement your own methods to download new data to
# the local disk.
# ============================================================================

# path to the training data. Should be same as that set in the config file.
DATA_PATH=path/to/your/data_download
# path to the config file
CONFIG_PATH=configs/pretrain/model-M_full-data.yaml

if [ ! -d $DATA_PATH ]; then
  echo "I cannot find 'DATA_PATH' $DATA_PATH. Please set it manually in the Bash script."
  exit 0
fi
echo "data_path: "$DATA_PATH
echo "config_file_path: "$CONFIG_PATH
MGR_INIT_DIR=$DATA_PATH/dyn_dset_comm/mgr_init

# Start dynamic-dataset manager (running on the background), and wait until it is ready.
[ -d $MGR_INIT_DIR ] && rmdir $MGR_INIT_DIR
nohup python3 dynamic_dataset_manager.py -c $CONFIG_PATH > dynamic_dataset_manager.out &
until [ -d $MGR_INIT_DIR ]; do
  echo "$(date) -- Waiting for initialization of dynamic-dataset manager..."
  sleep 30
done

# Start the main training process (with 8 Ascend NPUs).
mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python3 train.py -c $CONFIG_PATH 2> exp_train.err

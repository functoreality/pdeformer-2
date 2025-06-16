## File Directory

```text
./
│  dynamic_dataset_manager.py                    # Dynamic dataset buffer management program, starts before pretraining when handling large datasets exceeding local disk capacity
│  inverse_function.py                           # Code for inverse problems, estimating functions (source terms, wave equation velocity field) in PDEs
│  inverse_scalar.py                             # Code for inverse problems, estimating scalars (equation coefficients) in PDEs
│  PDEformer_inference.ipynb                     # English version of interactive notebook for model inference
│  PDEformer_inference_CN.ipynb                  # Chinese version of interactive notebook for model inference
│  pip-requirements.txt                          # Python dependency list
│  preprocess_data.py                            # Preprocesses data, generates computation graphs, and saves results in new auxiliary data files
│  README.md                                     # English version of the documentation
│  README_CN.md                                  # Chinese version of the documentation
│  train.py                                      # Model training code
├─configs                                        # Configuration files directory
│   │  full_config_example.yaml                  # Full configuration file example, covering all options
│   ├─baseline                                   # Configuration for training baseline models
│   │      ins-pipe_fno2d_20.yaml                # FNO2D trained on INS-Pipe dataset with 20 samples
│   │      ins-tracer_cnn-deeponet_900.yaml      # DeepONet (CNN) trained on INS-Tracer dataset with 900 samples
│   │      sine-gordon_unet2d_1.yaml             # U-Net2D trained on Sine-Gordon dataset with 1 samples
│   │      wave-c-sines_deeponet_4.yaml          # DeepONet (MLP) trained on Wave-C-Sines dataset with 4 samples
│   │      wave-gauss_fno3d_80.yaml              # FNO3D trained on Wave-Gauss dataset with 80 samples
│   ├─finetune                                   # Configuration for fine-tuning models
│   │      pdebench-swe-rdb_model-M.yaml         # Fine-tuning a pretrained model of size M on the PDEBench shallow water (radial dam break) dataset
│   │      ins-tracer_model-M.yaml               # Fine-tuning a pretrained model of size M on the INS-Tracer dataset
│   ├─inference                                  # Configurations for loading pretrained models for inference
│   │      model-L.yaml                          # Configuration for loading a size-L pretrained model for inference
│   │      model-M.yaml                          # Configuration for loading a size-M pretrained model for inference
│   │      model-S.yaml                          # Configuration for loading a size-S pretrained model for inference
│   ├─inverse                                    # Configurations for testing pretrained models on inverse problems
│   │      inverse_fullwaveform.yaml             # Configuration for inverting source functions (wave equation velocity field)
│   │      inverse_function.yaml                 # Configuration for inverting source functions (equation source terms)
│   │      inverse_scalar.yaml                   # Configuration for inverting scalar coefficients in equations
│   └─pretrain                                   # Configurations for pretraining models
│          model-L_small-data.yaml               # Pretraining a size-L model with partial data
│          model-S_standalone.yaml               # Pretraining a size-S model with partial data and single-GPU training
│          model-M_full-data.yaml                # Pretraining a size-M model with full data, using dynamic dataset buffering
├─docs                                           # Additional documentation
│      DATASET.md                                # Dataset documentation (English)
│      DATASET_CN.md                             # Dataset documentation (Chinese)
│      FILE_TREE.md                              # File directory structure documentation (English)
│      FILE_TREE_CN.md                           # File directory structure documentation (Chinese)
│      PDE_DAG.md                                # PDE computation graph documentation (English)
│      PDE_DAG_CN.md                             # PDE computation graph documentation (Chinese)
│      images                                    # Images used in README and other documentation
├─scripts                                        # Shell scripts for training, fine-tuning, and solving inverse problems
│      inverse_fullwaveform.sh                   # Script for inverting wave equation velocity fields
│      inverse_function.sh                       # Script for inverting source terms in equations
│      inverse_scalar.sh                         # Script for inverting scalar coefficients in equations
│      run_ui.sh                                 # Start GUI demonstration
│      train_distributed.sh                      # Script for multi-GPU distributed training
│      train_dynamic_dataset.sh                  # Script for large-scale training with dynamic dataset buffering
│      train_standalone.sh                       # Script for single-GPU training
└─src                                            # Core code directory
    │  inference.py                              # Script for generating predictions for custom PDEs
    ├─cell                                       # Code related to model architectures
    │  │  basic_block.py                         # Basic module blocks
    │  │  env.py                                 # Configurations not suitable for config.yaml
    │  │  lora.py                                # LoRA fine-tuning modules
    │  │  wrapper.py                             # Encapsulation interface for model selection
    │  ├─baseline                                # Baseline model architectures
    │  │      activation.py                      # Various activation functions for FNO
    │  │      check_func.py                      # Function input validation for FNO
    │  │      deeponet.py                        # DeepONet network architecture
    │  │      dft.py                             # Discrete Fourier Transform module for FNO
    │  │      fno.py                             # FNO network architecture
    │  │      unet2d.py                          # U-Net network architecture
    │  └─pdeformer                               # PDEFormer model architectures
    │      │  function_encoder.py                # Function encoder module
    │      │  pdeformer.py                       # PDEFormer network architecture
    │      ├─graphormer                          # Graphormer network architecture
    │      │      graphormer_encoder.py          # Encoder module
    │      │      graphormer_encoder_layer.py    # Layer-level information module
    │      │      graphormer_layer.py            # Module for encoding node self-information and connectivity
    │      │      multihead_attention.py         # Multi-head attention module
    │      └─inr_with_hypernet                   # INR + HyperNet model architectures
    │              mfn.py                        # MFN + HyperNet module
    │              siren.py                      # Siren + HyperNet module
    │              poly_inr.py                   # Poly-INR + HyperNet module
    ├─core                                       # Core modules for network training
    │      losses.py                             # Loss function module
    │      lr_scheduler.py                       # Learning rate scheduler module
    │      metric.py                             # Evaluation metrics module
    │      optimizer.py                          # Optimizer module
    ├─data                                       # Data loading code
    │  │   env.py                                # Stable configurations unsuitable for config.yaml, including precision, constants, and switches
    │  │   load_inverse_data.py                  # Data loading for inverse problems
    │  │   pde_dag.py                            # General module for generating Directed Acyclic Graphs and graph data based on PDE forms
    │  │   utils_dataload.py                     # General module for various dataset loading tasks
    │  │   wrapper.py                            # Wrapper functions for various dataset loaders
    │  ├─multi_pde                               # Data loader for multiple PDE forms (forward problems)
    │  │      boundary_v2_from_dict.py           # DAG generation for boundary conditions, interface for GUI toolkit
    │  │      boundary_v2.py                     # DAG generation for boundary conditions, current version
    │  │      boundary.py                        # DAG generation for boundary conditions, deprecated version
    │  │      dataloader.py                      # Operations for loading data, including training/test splits, batching, and dynamic buffer loading
    │  │      datasets.py                        # File reading for various PDE categories
    │  │      pde_types.py                       # Generates DAG and LaTeX expressions for various PDE categories
    │  │      terms_from_dict.py                 # DAG generation for PDE terms, GUI toolkit interface
    │  │      terms.py                           # DAG generation for PDE terms
    │  └─single_pde                              # Data loader for single PDE forms (forward problems)
    │         basics.py                          # Common basic modules
    │         dataloader.py                      # Operations for loading data, using different data format for different networks
    │         dataset_cart1.py                   # Dataset on Cartesian-grid 1
    │         dataset_cart2.py                   # Dataset on Cartesian-grid 2
    │         dataset_scat1.py                   # Dataset using scattered points
    ├─ui                                         # GUI utilities
    │      basics.py                             # Common basic modules
    │      database.py                           # Database for various PDE terms
    │      dcr.py                                # Main program for DCR GUI
    │      elements.py                           # Some elements used in GUI
    │      pde_types.py                          # Generates DAG and LaTeX expressions for various PDE categories
    │      utils.py                              # Utility functions for symbolic expressions and plotting
    │      widgets.py                            # GUI widgets for various PDE terms
    └─utils                                      # Utility functions for training and result recording
           load_yaml.py                          # YAML configuration file loader
           record.py                             # Experiment result recording
           tools.py                              # Miscellaneous utility functions
           visual.py                             # Visualization tools
```

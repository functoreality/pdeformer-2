## 文件目录结构

```text
./
│  dynamic_dataset_manager.py                    # 数据集动态缓冲区管理程序，在大数据量（超出本地磁盘容量）预训练开始前启动
│  inverse_function.py                           # 反问题代码，估计 PDE 中的函数（源项、波方程速度场）
│  inverse_scalar.py                             # 反问题代码，估计 PDE 中的标量（方程系数）
│  PDEformer_inference.ipynb                     # 英文版模型预测交互式 notebook
│  PDEformer_inference_CN.ipynb                  # 中文版模型预测交互式 notebook
│  pip-requirements.txt                          # Python 依赖库
│  preprocess_data.py                            # 预处理数据、生成计算图，并将结果保存到新建的辅助数据文件中
│  README.md                                     # 英文版说明文档
│  README_CN.md                                  # 中文版说明文档
│  train.py                                      # 模型训练代码
├─configs                                        # 配置文件目录
│   │  full_config_example.yaml                  # 完整配置文件示例，包含了所有可能的选项
│   ├─finetune                                   # 微调模型训练配置
│   │      pdebench-swe-rdb_model-M.yaml         # 加载预训练的 M 规模模型在 PDEBench 浅水波（径向溃坝）数据上微调
│   ├─inference                                  # 加载预训练的模型参数用于推理
│   │      model-L.yaml                          # 加载预训练的 L 规模模型用于推理
│   │      model-M.yaml                          # 加载预训练的 M 规模模型用于推理
│   │      model-S.yaml                          # 加载预训练的 S 规模模型用于推理
│   ├─inverse                                    # 预训练模型反问题测试配置
│   │      inverse_fullwaveform.yaml             # 反演源项函数（波方程波速场）
│   │      inverse_function.yaml                 # 反演源项函数（方程源项）
│   │      inverse_scalar.yaml                   # 反演方程标量系数
│   └─pretrain                                   # 预训练模型训练配置
│          model-L_small-data.yaml               # L 规模模型预训练，使用部分数据
│          model-S_standalone.yaml               # S 规模模型预训练，使用部分数据、单卡训练
│          model-M_full-data.yaml                # M 规模模型预训练，使用完整数据，并启用数据集动态缓冲区
├─docs                                           # 附加说明文档
│      DATASET.md                                # 数据集说明文档（英文）
│      DATASET_CN.md                             # 数据集说明文档（中文）
│      FILE_TREE.md                              # 文件目录结构说明文档（英文）
│      FILE_TREE_CN.md                           # 文件目录结构说明文档（中文）
│      PDE_DAG.md                                # PDE 计算图说明文档（英文）
│      PDE_DAG_CN.md                             # PDE 计算图说明文档（中文）
│      images                                    # 保存 README 等各文档中的图片
├─scripts                                        # 用于启动训练、微调、反问题求解的 shell 脚本
│      inverse_fullwaveform.sh                   # 反演波方程波速场
│      inverse_function.sh                       # 反演方程源项场
│      inverse_scalar.sh                         # 反演方程标量系数
│      train_distributed.sh                      # 多卡分布式训练
│      train_dynamic_dataset.sh                  # 启用数据集动态缓冲区的大规模训练
│      train_standalone.sh                       # 单卡训练
└─src                                            # 基础代码目录
    │  inference.py                              # 用于针对自定义 PDE 生成预测解
    ├─cell                                       # 模型架构相关的代码
    │  │  basic_block.py                         # 基本单元模块
    │  │  wrapper.py                             # 封装后的模型选择接口
    │  ├─baseline                                # Baseline 模型架构
    │  │      activation.py                      # FNO 可用的多种激活函数
    │  │      check_func.py                      # FNO 函数入参检查
    │  │      deeponet.py                        # DeepONet 网络架构
    │  │      dft.py                             # FNO 涉及的离散傅里叶变换模块
    │  │      fno.py                             # FNO 网络架构
    │  └─pdeformer                               # PDEFormer 模型架构
    │      │  function_encoder.py                # 函数编码器模块
    │      │  pdeformer.py                       # PDEFormer 网络架构
    │      ├─graphormer                          # Graphormer 网络架构
    │      │      graphormer_encoder.py          # 编码器模块
    │      │      graphormer_encoder_layer.py    # 层级信息模块
    │      │      graphormer_layer.py            # 编码节点自身信息、节点连通信息模块
    │      │      multihead_attention.py         # 多头注意力模块
    │      └─inr_with_hypernet                   # INR + HyperNet 模型架构
    │              mfn.py                        # MFN + HyperNet 模块
    │              siren.py                      # Siren + HyperNet 模块
    │              poly_inr.py                   # Poly-INR + HyperNet 模块
    ├─core                                       # 网络训练核心模块
    │      losses.py                             # 损失函数模块
    │      lr_scheduler.py                       # 学习率衰减模块
    │      metric.py                             # 评估指标模块
    │      optimizer.py                          # 优化器模块
    ├─data                                       # 数据加载代码
    │  │   env.py                                # 已稳定、不宜放在 config.yaml 中的一系列配置，如整形、浮点型数据精度，以及各种常量、开关
    │  │   load_inverse_data.py                  # 反问题数据加载
    │  │   load_single_pde.py                    # 只包含单一方程形式的数据集加载（正问题）
    │  │   pde_dag.py                            # 根据 PDE 形式生成有向无环图及相应图数据的通用模块
    │  │   utils_dataload.py                     # 不同数据集加载的通用模块
    │  │   wrapper.py                            # 不同数据集加载的封装函数
    │  └─multi_pde                               # 同时包含多种方程形式的数据集加载（正问题）
    │         boundary_v2_from_dict.py           # 边界条件的计算图 DAG 生成，仅用于 GUI 套件的接口
    │         boundary_v2.py                     # 边界条件的计算图 DAG 生成，当前使用版本
    │         boundary.py                        # 边界条件的计算图 DAG 生成，已停用的历史版本
    │         dataloader.py                      # 数据加载相关操作，包括训练测试数据区分、batch、动态缓冲区数据集读取
    │         datasets.py                        # 针对各种 PDE 形式大类的数据文件读取
    │         pde_types.py                       # 针对各种 PDE 形式大类生成计算图 DAG 与 LaTeX 表达式
    │         terms_from_dict.py                 # PDE 中各项的计算图 DAG 生成，仅用于 GUI 套件的接口
    │         terms.py                           # PDE 中各项的计算图 DAG 生成
    └─utils                                      # 训练与结果记录需要用到的工具函数
           load_yaml.py                          # 读取 YAML 配置文件
           record.py                             # 记录实验结果
           tools.py                              # 其他工具函数
           visual.py                             # 可视化工具函数
```

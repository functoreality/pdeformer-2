ENGLISH | [简体中文](README_CN.md)

# PDEformer-2: A Foundation Model for Two-Dimensional PDEs

## Overview

Partial differential equations (PDEs) are closely related to numerous physical phenomena and engineering applications, covering multiple fields such as airfoil design, electromagnetic field simulation, and stress analysis.
In these practical applications, solving PDE often requires repeated iterations.
Although traditional PDE solving algorithms are highly accurate, they often consume a significant amount of computational resources and time.
The neural operator methods proposed in recent years, based on deep learning, have greatly improved the speed of solving PDEs.
However, they pose difficulties to generalize to new forms of PDE, and often encounter problems such as high training costs and limited data size.

We develop the PDEformer model series to address the above issues.
This is a class of end-to-end solution prediction models that can directly handle almost **any form of PDE**, eliminating the need for customized architecture design and training for different PDEs, thereby significantly reducing model deployment costs and improving solution efficiency.
The [PDEformer-1](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/pdeformer1d) model developed for one-dimensional PDEs has been open-sourced previously.
The current PDEformer-2 model for two-dimensional PDEs, pretrained on a dataset of **approximately 40TB**, can directly handle 2D PDEs with different **computational domains, boundary conditions, number of variables, and time dependencies**, and quickly obtain predicted solutions at **any spatio-temporal location**.
In addition, as a **differentiable** surrogate model for solving forward problems, PDEformer-2 can also be used to solve various **inverse problems**, estimating scalar coefficients, source term fields, or wave velocity fields based on **noisy** spatio-temporal **scatter** observations.
This has laid a promising foundation for the model to support research on numerous physical phenomena and engineering applications in fields such as fluids and electromagnetics.

## Methodology

We consider two-dimensional PDEs defined on $(t,r)\in[0,1]\times\Omega$ of the generic form

$$\mathcal{F}(u_1,u_2,\dots,c_1,c_2,\dots,s_1(r),s_2(r),\dots)=0\text{ in }\Omega,$$
$$\mathcal{B}_i(u_1,u_2,\dots,c_{i1},c_{i2},\dots,s_{i1}(r),s_{i2}(r),\dots)=0\text{ on }\Gamma_i,$$

where $r=(x,y)\in\Omega\subseteq[0,1]^2$ is the spatial coordinate, $c_1,c_2,\dots,c_{11},c_{12},\dots \in \mathbb{R}$ are real-valued coefficients, $s_1(r),s_2(r)\dots,s_{11}(r),\dots$ are scalar functions (which may serve as initial conditions, boundary values or coefficient fields in the equation), and $u_1,u_2,\dots:[0,1]\times\Omega\to\R$ are unknown field variables to be solved in the equation.
The boundary conditions are indexed by $i=1,2,\dots$.
Here, we assume that each of the operators $\mathcal{F},\mathcal{B}_1,\mathcal{B}_2,\dots$ admits a symbolic expression, which may involve differential and algebraic operations.
The goal of PDEformer-2 is to construct a surrogate model of the solution mapping
$$(\Omega,\mathcal{F},c_1,\dots,s_1(r),\dots,\Gamma_1,\mathcal{B}_1,c_{11},\dots,s_{11}(r),\dots)\mapsto(u_1,u_2,\dots),$$
The input of this solution mapping includes the location of the computational domain $\Omega$ and the boundaries $\Gamma_1,\Gamma_2,\dots$,
the symbolic expressions of the interior operator $\mathcal{F}$ and boundary operators $\mathcal{B}_1,\mathcal{B}_2,\dots$,
as well as the numeric information $c_1,\dots,c_{11},\dots,s_1(r),\dots,s_{11}(r),\dots$ involved,
and the output includes all components of the predicted solution, i.e., $u_1,u_2,\dots:[0,1]\times\Omega\to\R$.
Taking the (single component) advection equation $u_t+(cu)_x+u_y=0$, $u(0,r)=g(r)$ on $\Omega=[0,1]^2$ with periodic boundary conditions as an example:

![](docs/images/PDEformerV2Arch.png)

As shown in the figure, PDEformer-2 first formulates the symbolic expression of the PDE as a computational graph, and makes use of a scalar encoder and a function encoder to embed the numeric information of the PDE into the node features of the computational graph.
Then, PDEformer-2 encodes this computational graph using a graph Transformer, and decodes the resulting latent vectors using an implicit neural representation (INR) to obtain the predicted values of each solution component of PDE at specific spatio-temporal coordinates.
A more detailed interpretation of the working principle of the model can be found in the introduction of [PDEformer-1](https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/pdeformer1d).
 
In terms of the complex domain shapes and boundary locations that may appear in two-dimensional equations, PDEformer-2 represents them as signed distance functions (SDFs), and embeds this information into the computational graph using the function encoder.
The example shown in the following figure demonstrates the way of using computational graphs to represent Dirichlet boundary conditions on a square domain:

![](docs/images/DAG-BC-Dirichlet.png)

## Installation

Please first make sure that MindSpore is successfully installed, as instructed in the [Installation Tutorial](https://www.mindspore.cn/install).
Other dependencies can be installed using the following command:

```bash
pip3 install -r pip-requirements.txt
```

## Model Running

We provide configuration files for PDEformer models with different numbers of parameters in the [configs/inference](configs/inference) folder.
The details are as follows:

| Model | Parameters | Configuration File | Checkpoint File |
| ---- | ---- | ---- | ---- |
| PDEformer2-S | 27.75M | [configs/inference/model-S.yaml](configs/inference/model-S.yaml) | [model-S.ckpt](https://ai.gitee.com/functoreality/PDEformer2-S/blob/master/model-S.ckpt) |
| PDEformer2-M | 71.07M | [configs/inference/model-M.yaml](configs/inference/model-M.yaml) | [model-M.ckpt](https://ai.gitee.com/functoreality/PDEformer2-M/blob/master/model-M.ckpt) |
| PDEformer2-L | 82.65M | [configs/inference/model-L.yaml](configs/inference/model-L.yaml) | [model-L.ckpt](https://ai.gitee.com/functoreality/PDEformer2-L/blob/master/model-L.ckpt) |

### Inference Example

The example code below demonstrates how to use PDEformer-2 to predict the solution of a given PDE,
taking the nonlinear conservation law $u_{t}+(u^2)_x+(-0.3u)_y=0$ (with periodic boundary conditions) as the example.
Before running, it is necessary to download the pretrained PDEformer weights `model-M.ckpt` from [Gitee AI](https://ai.gitee.com/functoreality/PDEformer2-M/blob/master/model-M.ckpt),
and change the value of the `model.load_ckpt` entry in [configs/inference/model-M.yaml](configs/inference/model-M.yaml) to the path of the corresponding weight file.

```python
import numpy as np
from mindspore import context
from src import load_config, get_model, PDENodesCollector
from src.inference import infer_plot_2d, x_fenc, y_fenc

# Basic Settings
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
config, _ = load_config("configs/inference/model-M.yaml")
model = get_model(config)

# Specify the PDE to be solved
pde = PDENodesCollector()
u = pde.new_uf()
u_ic = np.sin(2 * np.pi * x_fenc) * np.cos(4 * np.pi * y_fenc)
pde.set_ic(u, u_ic, x=x_fenc, y=y_fenc)
pde.sum_eq0(pde.dt(u), pde.dx(pde.square(u)), pde.dy(-0.3 * u))

# Predict the solution using PDEformer (with spatial resolution 32) and plot
pde_dag = pde.gen_dag(config)
x_plot, y_plot = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
u_pred = infer_plot_2d(model, pde_dag, x_plot, y_plot)
```

For more examples, please refer to the interactive notebook [PDEformer_inference.ipynb](PDEformer_inference.ipynb).

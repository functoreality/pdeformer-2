# 数据集文件

## 预训练

我们计划发布预训练数据集。
然而，由于数据集较大（约 40TB），发布过程可能会较为复杂，并需要一些时间。

以下是我们数据集中使用的 PDE 形式的描述：

### 扩散-对流-反应（DiffConvecReac2D, dcr）

PDE 形式如下：

$$u_t+Lu+f_0(u)+s(r)+f_1(u)_x+f_2(u)_y=0,$$

$u(0,r)=g(r)$ ， $t\in[0,1]$ ， $r=(x,y)\in[0,1]^2$ 。
在边界 $\Gamma_i$ 上，施加的边界条件为周期性条件或一般形式 $B_iu(r)=0$ ，其中 $r\in\Gamma_i$ 。

这里，空间二阶项 $Lu$ 以相等概率随机选择以下形式：
- 非散度形式 $Lu=-a(r)\Delta u$
- 因式分解形式 $Lu=-\sqrt{a(r)}\nabla\cdot(\sqrt{a(r)}\nabla u)$
- 散度形式 $Lu=-\nabla\cdot(a(r)\nabla u)$

其中 $a(r)$ 是随机标量或随机场， $r=(x,y,z)$ 表示空间坐标。

取 $f_i(u) = \sum_{k=1}^3c_{i0k}u^k + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$，$i=0,1,2$，其中$J_0+J_1+J_2\le J$为随机生成 。

每个边界算子$B_iu$以相等概率取以下形式：
- Robin 边界条件（D 型） $B_iu = u + b_i(r)\partial u/\partial n + c_i(r)$
- Robin 边界条件（N 型） $B_iu = a_i(r)u + \partial u/\partial n + c_i(r)$

其中，每个系数场 $a_i(r),b_i(r),c_i(r)$ 随机取值为零、一、随机标量或随机场。注意，当 $a_i(r)$或$b_i(r)$ 为零时，边界条件将分别退化为 Dirichlet 型或 Neumann 型。此外，可根据初始条件设置 $c_i(r)$ 。

### 波动方程

PDE 形式如下：

$$u_{tt}+\mu(r)u_t+Lu+f_0(u)+s(r)+f_1(u)_x+f_2(u)_y=0,$$

$u(0,r)=g(r)$ ， $u_t(0,r)=h(r)$ ， $t\in[0,1]$ ， $r=(x,y)\in[0,1]^2$ 。
在边界 $\Gamma_i$ 上，施加的边界条件为周期性条件或一般形式 $B_iu(r)=0$ ，其中 $r\in\Gamma_i$ 。

每个边界算子 $B_iu$ 以相等概率取以下形式：
- Robin 边界条件（D 型） $B_iu = u + b_i(r)\partial u/\partial n + c_i(r)$
- Robin 边界条件（N 型） $B_iu = a_i(r)u + \partial u/\partial n + c_i(r)$
- （广义）Mur 型 $B_iu = u_t + a_i(r)u + b_i(r)\partial u/\partial n + c_i(r)$

其中，每个系数场 $a_i(r),b_i(r),c_i(r)$ 随机取值为零、一、随机标量或随机场。注意，当 $a_i(r)$ 或 $b_i(r)$ 为零时，边界条件将分别退化为 Dirichlet 型或 Neumann 型。此外，可根据初始条件设置 $c_i(r)$ 。

其余设置与 DCR 数据集相同。

### 多变量 DCR 方程（MCompn2D, mvdcr）

PDE 形式如下：

$$\partial_tu_i + L_iu_i + \boldsymbol{f}_0(u)_i + s_i(r) + \partial_x\boldsymbol{f}_1(u)_i + \partial_y\boldsymbol{f}_2(u)_i = 0,$$

$u_i(0,r)=g_i(r)$ ， $t\in[0,1]$ ， $r=(x,y)\in[0,1]^2$ ，
$0 \le i,j,k \le d_u-1$，$j \le k$ 。
为简化，采用周期性边界条件。

我们取：

$$\boldsymbol{f}_l(u)_i = \sum_j a_{lij}u_j + \sum_{j,k}b_{lijk}u_ju_k$$

$l=0,1,2$ 。系数 $a,b$ 为稀疏数组，总共最多包含 $3d_u$ 个非零项。

注意，当 $d_u=3$ 时，对于特定随机系数选择，PDE 形式包含二维 Maxwell 方程（TE 或 TM 形式）在均匀介质与周期边界条件下的情形。

### 散度约束 DCR（DivConstrDCR2D, dcdcr）

PDE 形式如下：

$$\partial_tu_i + L_iu_i + \boldsymbol{f}_0(u)_i + s_i(r) + \partial_x\boldsymbol{f}_1(u)_i + \partial_y\boldsymbol{f}_2(u)_i + (-c_i)p + (\nabla p)_i = 0,$$

其中 $i=0,1$ ，并添加散度约束：

$$\partial_xu_0 + \partial_yu_1 + c_0u_0 + c_1u_1 + c_2 = 0。$$

部分数据集初始条件满足此约束（有效初始条件，icV），其他数据集的初始条件为高斯随机场（无约束初始条件，icA）。

与多变量 DCR 的区别：
- DCR 涉及的变量数固定为 $d_u=2$ 。
- 增加了压力变量 $p$ 及散度约束方程。
- 部分数据集的初始条件来自非 GRF 分布。

我们注意到，我们的数据集在特定的参数选取下包含周期性边界条件的二维不可压缩的 Navier-Stokes (NS) 方程（守恒形式）。
也就是说，对于特定的随机系数选择，PDE 的形式成为不可压 NS 方程：

$$\partial_tu_0-\nu\Delta u_0+\partial_x(u_0^2)+\partial_y(u_0u_1)+\partial_xp=0,$$
$$\partial_tu_1-\nu\Delta u_1+\partial_x(u_0u_1)+\partial_y(u_1^2)+\partial_yp=0,$$
$$\partial_xu_0+\partial_yu_1=0$$

### 多变量波动方程 (MCWave2D, mvwave)

$$\partial_{tt}u_i + \mu_i(r)\partial_tu_i + L_iu_i + \boldsymbol{f}_0(u)_i + s_i(r) + \partial_x\boldsymbol{f}_1(u)_i + \partial_y\boldsymbol{f}_2(u)_i = 0,$$

其余部分与多变量 DCR 相同。

### 散度约束波动方程 (DivConstrWave2D, dcwave)

$$\partial_{tt}u_i + \mu_i(r)\partial_tu_i + L_iu_i + \boldsymbol{f}_0(u)_i + s_i(r) + \partial_x\boldsymbol{f}_1(u)_i + \partial_y\boldsymbol{f}_2(u)_i + (-c_i)p + (\nabla p)_i = 0,$$

其余部分与多变量波动方程相同。

### 浅水波方程 (SWE2D, swe)

PDE 形式为

$$h_t + L_hh + f_h + s_h(r) + ((h+H(r))u)_x + ((h+H(r))v)_y = 0,$$
$$u_t + L_uu + f_u + s_u(r) + uu_x + vu_y + g_1h_x = 0,$$
$$v_t + L_vv + f_v + s_v(r) + uv_x + vv_y + g_2h_y = 0,$$

$\eta(0,r)=g_\eta(r)$ 对于 $\eta\in\{h,u,v\}$ ， $t\in[0,1]$ ，
$r=(x,y)\in[0,1]^2$ 。
简单起见，我们采用周期性边界条件。
我们取 $[f_h;f_u;f_v] = \boldsymbol{f}_0([h;u;v])$ ，其中 $\boldsymbol{f}_0$ 与多变量 DCR/波动方程相同。
初始水高度 $g_h(r)$ 取为正随机场。
水的基础高度 $H(r)$ 是非负的，并有一定概率取为零。

### 稳态弹性方程 (ElasticSteady2D, elasticsteady)

PDE 形式为

$$(\lambda(r)(u_x+v_y)+\mu_2(r)u_x)_x+\left(\frac{1}{2}\mu_2(r)(u_y+v_x)\right)_y+f_1(r)=0,$$
$$\left(\frac{1}{2}\mu_2(r)(u_y+v_x)\right)_x+(\lambda(r)(u_x+v_y)+\mu_2(r)v_y)_y+f_2(r)=0$$

边界条件从 Dirichlet 和 Neumann 中随机选择，分别独立于变量 $u,v$ 和正方形的四条边。

边界值从零、随机常数和一维随机场中随机选择。

$\lambda(r)$ 和 $\mu_2(r)$ 分别从随机正常数和二维随机场中随机选择。

外力 $f_i(r)$ 从零、随机常数和二维随机场中随机选择。

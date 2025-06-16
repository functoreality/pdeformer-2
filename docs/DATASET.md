# Dataset Files

## Pretraining

We are planning to release our pretraining dataset.
However, as the dataset is large (about 40TB), it would not be that direct, and may take some time for us.

Here is a description of the PDE forms used in our dataset:

### Diffusion-Convection-Reaction (DiffConvecReac2D, dcr)

The PDE takes the form

$$u_t+Lu+f_0(u)+s(r)+f_1(u)_x+f_2(u)_y=0,$$

$u(0,r)=g(r)$, $t\in[0,1]$, $r=(x,y)\in[0,1]^2$.
On edge $\Gamma_i$, the boundary condition imposed is either periodic or of
the general form $B_iu(r)=0$ for $r\in\Gamma_i$.

Here, the spatial second-order term $Lu$ is randomly selected from
the non-divergence form $Lu=-a(r)\Delta u$, the factored form
$Lu=-\sqrt a(r)\nabla\cdot(\sqrt a(r)\nabla u)$, and the divergence form
$Lu=-\nabla\cdot(a(r)\nabla u)$ with equal probability, where $a(r)$ is
taken to be a random scalar or a random field, and $r=(x,y,z)$ denotes the
spatial coordinates.

We take $f_i(u) = \sum_{k=1}^3c_{i0k}u^k + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$
for $i=0,1,2$, where $J_0+J_1+J_2\le J$ are randomly generated.

Each boundary operator $B_iu$ is taken to be Robin with D-type
$B_iu = u + b_i(r)\partial u/\partial n + c_i(r)$, or Robin with N-type
$B_iu = a_i(r)u + \partial u/\partial n + c_i(r)$, with equal probability.
Each of the coefficient field $a_i(r),b_i(r),c_i(r)$ is taken to be zero,
one, a random scalar, or a random field with certain probability. Note that
when $a_i(r)$ or $b_i(r)$ equals zero, the boundary condition would
degenerate to the Dirichlet type or the Neumann type. We may also set
$c_i(r)$ to meet the initial condition.

### Wave

The PDE takes the form

$$u_{tt}+\mu(r)u_t+Lu+f_0(u)+s(r)+f_1(u)_x+f_2(u)_y=0,$$

$u(0,r)=g(r)$, $u_t(0,r)=h(r)$, $t\in[0,1]$, $r=(x,y)\in[0,1]^2$.
On edge $\Gamma_i$, the boundary condition imposed is either periodic or of
the general form $B_iu(r)=0$ for $r\in\Gamma_i$.

Each boundary operator $B_iu$ is taken to be Robin with D-type
$B_iu = u + b_i(r)\partial u/\partial n + c_i(r)$, Robin with N-type
$B_iu = a_i(r)u + \partial u/\partial n + c_i(r)$, or (generalized) Mur type
$B_iu = u_t + a_i(r)u + b_i(r)\partial u/\partial n + c_i(r)$, with equal
probability.
Each of the coefficient field $a_i(r),b_i(r),c_i(r)$ is taken to be zero,
one, a random scalar, or a random field with certain probability. Note that
when $a_i(r)$ or $b_i(r)$ equals zero, the boundary condition would
degenerate to the Dirichlet type or the Neumann type. We may also set
$c_i(r)$ to meet the initial condition.

The remaining settings are the same as the DCR dataset.

### Multi-Variable DCR (MCompn2D, mvdcr)

The PDE takes the form

$$\partial_tu_i + L_iu_i + \boldsymbol{f}_0(u)_i + s_i(r) + \partial_x\boldsymbol{f}_1(u)_i + \partial_y\boldsymbol{f}_2(u)_i = 0,$$

$u_i(0,r)=g_i(r)$, $t\in[0,1]$, $r=(x,y)\in[0,1]^2$,
$0 \le i,j,k \le d_u-1$, $j \le k$.
Periodic boundary conditions are employed for simplicity.

We take

$$\boldsymbol{f}_l(u)_i = \sum_ja_{lij}u_j + \sum_{j,k}b_{lijk}u_ju_k$$

for $l=0,1,2$. The coefficients $a,b$ are sparse arrays, with a total of at
most $3d_u$ non-zero entries.

We note that for the case of $d_u=3$, the two-dimensional Maxwell's equation (TE
or TM form) with homogeneous media and periodic boundaries is in-distribution.
That is, for a specific choice of the random coefficients, the form of the PDE
becomes Maxwell's equations.

### Divergence-Constrained DCR (DivConstrDCR2D, dcdcr)

$$\partial_tu_i + L_iu_i + \boldsymbol{f}_0(u)_i + s_i(r) + \partial_x\boldsymbol{f}_1(u)_i + \partial_y\boldsymbol{f}_2(u)_i + (-c_i)p + (\nabla p)_i = 0,$$

in which $i=0,1$, with additional divergence constraint
$\partial_xu_0 + \partial_yu_1 + c_0u_0 + c_1u_1 + c_2 = 0$.
The initial condition of some datasets comply with this constraint
(valid initial condition, icV),
and are generated as Gaussian random fields (GRFs) for the other datasets
(arbitrary initial condition, icA).

Differences from MV-DCR equation:

* The number of DCR variables is fixed to $d_u=2$.
* The additional pressure variable $p$ along with the divergence constraint equation are added.
* For part of the dataset, the initial condition are from another (non-GRF) distribution.

We note that the two-dimensional incompressible Navier-Stokes (NS) equation
(conservation form) with periodic boundary condition is in-distribution.
That is, for a specific choice of the random coefficients, the form of the PDE
becomes the NS equation:

$$\partial_tu_0-\nu\Delta u_0+\partial_x(u_0^2)+\partial_y(u_0u_1)+\partial_xp=0,$$
$$\partial_tu_1-\nu\Delta u_1+\partial_x(u_0u_1)+\partial_y(u_1^2)+\partial_yp=0,$$
$$\partial_xu_0+\partial_yu_1=0.$$

### Multi-Variable Wave, (MCWave2D, mvwave)

$$\partial_{tt}u_i + \mu_i(r)\partial_tu_i + L_iu_i + \boldsymbol{f}_0(u)_i + s_i(r) + \partial_x\boldsymbol{f}_1(u)_i + \partial_y\boldsymbol{f}_2(u)_i = 0,$$

and the rest is the same as MV-DCR.

### Divergence-Constrained Wave (DivConstrWave2D, dcwave)

$$\partial_{tt}u_i + \mu_i(r)\partial_tu_i + L_iu_i + \boldsymbol{f}_0(u)_i + s_i(r) + \partial_x\boldsymbol{f}_1(u)_i + \partial_y\boldsymbol{f}_2(u)_i + (-c_i)p + (\nabla p)_i = 0,$$

and the rest is the same as MV-Wave.

### Shallow-Water Equation (SWE2D, swe)

The PDE takes the form

$$h_t + L_hh + f_h + s_h(r) + ((h+H(r))u)_x + ((h+H(r))v)_y = 0,$$
$$u_t + L_uu + f_u + s_u(r) + uu_x + vu_y + g_1h_x = 0,$$
$$v_t + L_vv + f_v + s_v(r) + uv_x + vv_y + g_2h_y = 0,$$

$\eta(0,r)=g_\eta(r)$ for $\eta\in\{h,u,v\}$, $t\in[0,1]$,
$r=(x,y)\in[0,1]^2$.
Periodic boundary conditions are employed for simplicity.
We take $[f_h;f_u;f_v] = \boldsymbol{f}_0([h;u;v])$ with $\boldsymbol{f}_0$ being the same as that of
multi-variable DCR/Wave equations.
The initial water height $g_h(r)$ is taken to be a positive random field.
The base height of the water $H(r)$ is non-negative, and taken to be zero with certain probability.

### Steady-State Elasticity (ElasticSteady2D, elasticsteady)

The PDE takes the form

$$(\lambda(r)(u_x+v_y)+\mu_2(r)u_x)_x+\left(\frac{1}{2}\mu_2(r)(u_y+v_x)\right)_y+f_1(r)=0,$$
$$\left(\frac{1}{2}\mu_2(r)(u_y+v_x)\right)_x+(\lambda(r)(u_x+v_y)+\mu_2(r)v_y)_y+f_2(r)=0.$$

The boundary condition is randomly chosen from Dirichlet and Neumann,
independently for the two variables $u,v$ and the four edges of the square.

The boundary values are randomly chosen from zero, a random constant, and a
1D random field.

Each of $\lambda(r)$ and $\mu_2(r)$ is randomly chosen from a random positive
constant and a 2D random field.

The external force $f_i(r)$ is randomly chosen from zero, a random constant,
and a 2D random field.

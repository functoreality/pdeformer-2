r"""
Representing the PDE in the form of a computational graph, which includes both
the symbolic and numeric information inherent in a PDE. This essentially
constructs a directed acyclic graph (DAG).
"""
from typing import Tuple, List, Optional, Union
from collections import namedtuple

import numpy as np
from numpy.typing import NDArray
import networkx as nx
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from .env import USE_GLOBAL_NODE, int_dtype, float_dtype

# 'uf': unknown field; 'ic': initial condition; 'cf': coefficient field;
# 'bv': boundary value (deprecated); 'sdf': signed distance function;
# These node types are not used in the current version: 'eval', 'dn', 'avg_int'
VAR_NODE_TYPES = ['uf']
COEF_NODE_TYPES = ['coef']
FUNCTION_NODE_TYPES = ['ic', 'cf', 'bv', 'sdf', 'eval']
OPERATOR_NODE_TYPES = ['add', 'mul', 'eq0',
                       'dt', 'dx', 'dy', 'dz', 'dn', 'avg_int',
                       'neg', 'square', 'exp10', 'log10', 'sin', 'cos']
RESERVED_NODE_TYPES = [f'Reserved{i}' for i in range(16)]
FUNCTION_BRANCH_NODE_TYPES = [f'Branch{i}' for i in range(16)]
INR_NODE_TYPES = [f'Mod{i}' for i in range(32)]
# During batch training, we may need to add 'pad' nodes (with index 0) to
# make sure all graphs in a batch have the same number of nodes.
DAG_NODE_TYPES = (['pad'] + VAR_NODE_TYPES + COEF_NODE_TYPES + FUNCTION_NODE_TYPES
                  + OPERATOR_NODE_TYPES + RESERVED_NODE_TYPES
                  + FUNCTION_BRANCH_NODE_TYPES + INR_NODE_TYPES)
NODE_TYPE_DICT = {t: i for (i, t) in enumerate(DAG_NODE_TYPES)}

NODE_COLORS = {"pde": [0.5, 1, 0.5], "aux": [0.5, 0.5, 1],
               "pad": [0.6, 0.6, 0.6]}
DAGInfoTuple = namedtuple("DAGInfoTuple", (
    'node_type', 'node_scalar', 'node_function', 'in_degree', 'out_degree',
    'attn_bias', 'spatial_pos'))
PfDataTuple = namedtuple(
    "PfDataTuple", DAGInfoTuple._fields + ('coordinate', 'u_label'))


class ModNodeSwapper:
    r"""
    In a pde_dag, the first inr nodes correspond to the first uf node (i.e. the
    first variable/component of the PDE). When we want to let them correspond
    to another uf node, this class will help us to reorder the nodes.
    Specifically, we swap (a) the first inr nodes and (b) inr nodes
    corresponding to the component with index `idx_var`. Note that we only have
    to modify `spatial_pos`, swapping its rows and columns, and generate
    `attn_bias` accordingly. All the remaining pde_dag data are left unchanged.
    """

    def __init__(self, uf_num_mod: int, n_vars: int) -> None:
        inds_old = np.arange(uf_num_mod)  # eg. [0,..,4] when uf_num_mod==5
        inds_old = np.expand_dims(inds_old, axis=0)  # [[0,..,4]]
        inds_old = np.repeat(inds_old, n_vars, axis=0)  # [[0,..,4],..,[0,..4]]
        # inds_old: [[0,..,4],..,[0,..,4]]; inds_new: [[0,..,4],..,[10,..,14]]
        inds_new = np.arange(uf_num_mod * n_vars).reshape(
            (n_vars, uf_num_mod))
        # Shape is [n_vars, 2 * uf_num_mod].
        self.inds_l = np.concatenate([inds_old, inds_new], axis=-1)
        self.inds_r = np.concatenate([inds_new, inds_old], axis=-1)

    def apply_(self, spatial_pos: NDArray[int], idx_var: int,
               node_type: Optional[NDArray[int]] = None) -> None:
        r"""
        Swap the modulation nodes in `spatial_pos`, making those corresponding
        to the unknown field (uf) with index `idx_var` moved to the first
        places. Note that the operation is done in-place, i.e. the changes are
        applied directly to the `spatial_pos` NumPy array.
        """
        if idx_var == 0:
            return  # nothing to do
        inds_l = self.inds_l[idx_var]  # [2 * uf_num_mod]
        inds_r = self.inds_r[idx_var]  # [2 * uf_num_mod]
        spatial_pos[inds_l] = spatial_pos[inds_r]
        spatial_pos[:, inds_l] = spatial_pos[:, inds_r]
        if node_type is not None:  # check node type is correct
            inds_r0 = inds_r[0]
            if inds_l[0] != 0 or node_type[inds_r0, 0] != node_type[0, 0]:
                raise RuntimeError("'ModNodeSwapper' not working as expected.")


class PDEAsDAG:
    r"""
    Represent a PDE (partial differential equation) in the form of a DAG
    (directed acyclic graph), including PDE nodes and auxiliary nodes. The
    attributes shall include the inputs to PDEformer, eg. 'node_type',
    'node_scalar', 'attn_bias' and 'spatial_pos'.

    Args:
        config (DictConfig): PDEformer configuration.
        node_list (List[Tuple[Str]]): PDE nodes in the DAG. Each element in the
            list takes the form ``(node_name, node_type,
                predecessor_node_1_name, predecessor_node_2_name, ...)``.
            Please make sure that all the nodes involved have distinct names.
        node_scalar (List[float], Optional): Scalar values assigned to the
            nodes in the DAG. The length should be equal to that of
            `node_list`.
        node_function (List[NDArray[float]], Optional): Function values of the
            initial condition (IC) $g(x)$, coefficient field (CF) $c(x)$,
            varying coefficient (VC) $c(t)$, etc. The length should be equal to
            the number of nodes with type 'ic', 'cf' or 'vc' in `node_list`.
            Each array in the list stores the values of the form
            :math:`\{(x_i, s(x_i))\}`, and should have the same shape
            :math:`(num\_pts, 2)`.

    Attributes:
        node_type (NDArray[int]): The type number of each node, shape
            :math:`(n\_node, 1)`.
        node_scalar (NDArray[float]): The scalar value of each node, shape
            :math:`(num\_scalar, 1)`.
        node_function (NDArray[float]): The function value of each node,
            shape :math:`(num\_function, num\_pts, 2)`.
        in_degree (NDArray[int]): The in-degree of each node, shape :math:`(n\_node)`.
        out_degree (NDArray[int]): The out-degree of each node, shape :math:`(n\_node)`.
        attn_bias (NDArray[float]): The attention bias matrix of the graph,
            shape :math:`(n\_node, n\_node)`.
        spatial_pos (NDArray[int]): The spatial position (shortest path length)
            from each node to each other node, shape :math:`(n\_node, n\_node)`.

    Here, :math:`num\_scalar` is given by the configuration option
    `config.data.pde_dag.max_n_scalar_nodes`, and :math:`num\_function` is
    `config.data.pde_dag.max_n_function_nodes`. Denote :math:`N` to be the
    number of branches (`config.model.function_encoder.num_branches`), we shall
    have :math:`n\_node = num\_scalar + num\_function \times N`.

    Note that PDEformer by default predicts only the first variable (i.e. the
    first component) of the PDE. If the user wants to get the prediction of the
    variable indexed by :math:`i` (count starting from 0), consider using
        ``spatial_pos, attn_bias = pde_dag.get_spatial_pos_attn_bias(i)``
    instead of the direct version `pde_dag.spatial_pos` or `pde_dag.attn_bias`
    as the inputs to PDEformer. (This makes no difference if :math:`i=0`.)

    Example:
        >>> # u_t + u = 0
        >>> pde_nodes = [('u', 'uf'),
        ...              ('u_t', 'dt', 'u'),
        ...              ('u_t+u', 'add', 'u_t', 'u'),
        ...              ('my_eqn', 'eq0', 'u_t+u')]
        >>> pde_dag = PDEAsDAG(config, pde_nodes)
        >>> pde_dag.plot()  # plot the resulting DAG
    """

    def __init__(self,
                 config: DictConfig,
                 node_list: List[Tuple[str]],
                 node_scalar: List[float] = None,
                 node_function: List[np.ndarray] = None) -> None:
        num_spatial = config.model.graphormer.num_spatial
        disconn_attn_bias = float(config.data.pde_dag.disconn_attn_bias)
        # Number of INR modulation nodes for each uf node.
        if config.model.multi_inr.enable and config.model.multi_inr.separate_latent:
            uf_num_mod = config.model.inr.num_layers + config.model.inr2.num_layers - 2
        else:
            uf_num_mod = config.model.inr.num_layers - 1
        function_num_branches = config.model.function_encoder.num_branches
        max_n_scalar_nodes = config.data.pde_dag.max_n_scalar_nodes
        max_n_function_nodes = config.data.pde_dag.max_n_function_nodes

        # process node_list to build DAG
        dag, n_vars, n_functions, n_scalar_pad = self._build_dag(
            node_list, uf_num_mod, function_num_branches, max_n_scalar_nodes)

        n_functions_to_add = max_n_function_nodes - n_functions
        if n_functions_to_add < 0:
            raise ValueError(
                "The number of function nodes involved in the PDE computational "
                f"graph ({n_functions}) exceeds max_n_function_nodes set in "
                f"config ({max_n_function_nodes})!")
        pad_len = n_functions_to_add * function_num_branches

        # node_type
        node_type = np.array([attr_dict['typeid']
                              for node, attr_dict in dag.nodes.data()])
        padding_mask = node_type == 0  # [n_dag_node]
        node_type = np.pad(node_type, (0, pad_len)).astype(int_dtype)
        self.node_type = node_type[:, np.newaxis]  # [n_node, 1]

        # node_scalar
        if node_scalar is None:
            node_scalar = np.zeros([max_n_scalar_nodes], dtype=float_dtype)
        else:
            if len(node_scalar) != len(node_list):
                raise ValueError(
                    f"The length of 'node_scalar' ({len(node_scalar)}) should "
                    f"be equal to that of 'node_list' ({len(node_list)})!")
            node_scalar = np.array(node_scalar, dtype=float_dtype)
            n_mod_nodes = n_vars * uf_num_mod
            # [n_raw_nodes] -> [max_n_scalar_nodes]
            node_scalar = np.pad(node_scalar, (n_mod_nodes, n_scalar_pad))
        # Shape is [max_n_scalar_nodes, 1].
        self.node_scalar = node_scalar[:, np.newaxis]

        # node_function
        if node_function is None:
            n_pts = 1
            node_function = np.zeros([max_n_function_nodes, n_pts, 2],
                                     dtype=float_dtype)
        else:
            if len(node_function) != n_functions:
                raise ValueError(
                    f"The length of 'node_function' ({len(node_function)}) should be "
                    "equal to the number of function nodes involved in the PDE "
                    f"computational graph ({n_functions})!")
            node_function = np.array(node_function, dtype=float_dtype)
            node_function = np.pad(
                node_function, ((0, n_functions_to_add), (0, 0), (0, 0)))
        # Shape is [max_n_function_nodes, n_pts, 2].
        self.node_function = node_function

        # spatial_pos, attn_bias
        shortest_path_len = nx.floyd_warshall_numpy(dag)
        # Need +1 because value 0 is reserved for padded nodes.
        spatial_pos = 1 + shortest_path_len.clip(
            0, num_spatial - 2).astype(int_dtype)
        spatial_pos[padding_mask] = 0
        spatial_pos[:, padding_mask] = 0
        self.spatial_pos = np.pad(spatial_pos, ((0, pad_len), (0, pad_len)))
        self.attn_bias = self.get_attn_bias(
            self.node_type, self.spatial_pos, disconn_attn_bias)
        if n_vars > 1:
            self.mod_node_swapper = ModNodeSwapper(uf_num_mod, n_vars)
            self.disconn_attn_bias = disconn_attn_bias

        # in_degree, out_degree
        # Need +1 because value 0 is reserved for padded nodes.
        in_degree = 1 + np.array([d for node, d in dag.in_degree()])
        in_degree[padding_mask] = 0
        self.in_degree = np.pad(in_degree, (0, pad_len)).astype(int_dtype)
        out_degree = 1 + np.array([d for node, d in dag.out_degree()])
        out_degree[padding_mask] = 0
        self.out_degree = np.pad(out_degree, (0, pad_len)).astype(int_dtype)

        self._dag = dag
        self.n_vars = n_vars

        # validate config
        if len(DAG_NODE_TYPES) > config.model.graphormer.num_node_type:
            raise ValueError("'num_node_type' is too small.")
        if self.in_degree.max() >= config.model.graphormer.num_in_degree:
            raise ValueError("'num_in_degree' is too small.")
        if self.out_degree.max() >= config.model.graphormer.num_out_degree:
            raise ValueError("'num_out_degree' is too small.")

    @staticmethod
    def _build_dag(node_list: List[Tuple[str]],
                   uf_num_mod: int,
                   function_num_branches: int,
                   max_n_scalar_nodes: int) -> Tuple:
        r"""Build DAG from the given node list."""
        mod_node_list = []
        function_branch_node_list = []
        edge_list = []
        n_vars = 0
        n_functions = 0
        for node, type_, *predecessors in node_list:
            edge_list.extend([(node_p, node) for node_p in predecessors])
            if type_ in VAR_NODE_TYPES:
                n_vars += 1
                for j in range(uf_num_mod):
                    node_new = f'{node}:Mod{j}'
                    mod_node_list.append((node_new, f'Mod{j}'))
                    edge_list.append((node_new, node))
                    # if j > 0:  # ModSeq
                    #     edge_list.append((node_new, f'{node}:Mod{j - 1}'))
            elif type_ in FUNCTION_NODE_TYPES:
                n_functions += 1
                for j in range(function_num_branches):
                    node_new = f'{node}:Branch{j}'
                    function_branch_node_list.append((node_new, f'Branch{j}'))
                    if type_ in ['ic', 'bv']:
                        edge_list.append((node, node_new))
                    elif type_ in ['cf', 'sdf']:
                        edge_list.append((node_new, node))
                    else:
                        raise NotImplementedError

        if n_vars < 1:
            raise ValueError("There should be at least one 'uf' node in the PDE.")

        # pad scalar nodes
        dag_node_list = mod_node_list + node_list
        n_scalar_pad = max_n_scalar_nodes - len(dag_node_list)
        if n_scalar_pad < 0:
            raise ValueError(
                f"Target scalar node number ({max_n_scalar_nodes}) should not be "
                f"less than the number of existing nodes ({len(dag_node_list)})!")
        if NODE_TYPE_DICT['pad'] != 0:
            raise RuntimeError("Node type 'pad' not indexed as zero.")
        dag_node_list.extend([(f'pad{j}', 'pad')
                              for j in range(n_scalar_pad)])
        dag_node_list.extend(function_branch_node_list)

        # create DAG
        dag = nx.DiGraph()
        dag.add_nodes_from([
            (node, {'type': type_, 'typeid': NODE_TYPE_DICT[type_]})
            for node, type_, *predecessors in dag_node_list])
        dag.add_edges_from(edge_list)

        out_tuple = (dag, n_vars, n_functions, n_scalar_pad)
        return out_tuple

    @staticmethod
    def get_attn_bias(node_type: NDArray[int],
                      spatial_pos: NDArray[int],
                      disconn_attn_bias: float) -> NDArray[float]:
        r"""Get the NumPy array `attn_bias` corresponding to `spatial_pos`."""
        n_node, _ = node_type.shape
        attn_bias = np.zeros([n_node, n_node], dtype=float_dtype)
        # optionally disable attention between node pairs that are not
        # connected in the DAG
        connect_mask = spatial_pos == np.max(spatial_pos)  # [n_node, n_node]
        connect_mask = np.logical_and(connect_mask, connect_mask.T)
        attn_bias[connect_mask] = disconn_attn_bias
        # disable attention to padded nodes
        padding_mask = node_type[:, 0] == 0  # [n_node]
        if USE_GLOBAL_NODE:
            attn_bias = np.pad(attn_bias, ((1, 0), (1, 0)))
            padding_mask = np.pad(padding_mask, (1, 0))
        attn_bias[:, padding_mask] = -np.inf
        return attn_bias

    def get_spatial_pos_attn_bias(self, idx_var: int = 0) -> Tuple[NDArray]:
        r"""
        Get the `spatial_pos` and `attn_bias` numpy arrays, in which the
        auxiliary modulation nodes corresponding to the unknown field (uf) with
        index `idx_var` are moved to the first places.
        """
        if idx_var >= self.n_vars:
            raise ValueError(f"PDE Variable index ({idx_var}) out of range")
        if idx_var == 0:
            return (self.spatial_pos, self.attn_bias)
        spatial_pos = np.copy(self.spatial_pos)
        # swap the rows and columns of 'spatial_pos'
        self.mod_node_swapper.apply_(spatial_pos, idx_var, self.node_type)
        attn_bias = self.get_attn_bias(
            self.node_type, spatial_pos, self.disconn_attn_bias)
        return (spatial_pos, attn_bias)

    def plot(self, label: str = 'type', hide: str = 'pad') -> None:
        r"""
        Create a plot of the current directed acyclic graph (DAG).

        Arguments:
            label (str): Display mode of graph node labels. Available values:
                ["name", "type", "name+type", "none", ""]. Default: "type".
            hide (str): Node type classes not to be shown. Available values:
                ["none", "pad", "aux" (experimental)]. Default: "pad".
        """
        nodes_data = self._dag.nodes.data()

        # Generate node labels. labels_dict: Dict[str, str]
        if label == 'name':
            labels_dict = {node: node for node, attr_dict in nodes_data}
        elif label == 'type':
            labels_dict = {node: attr_dict['type']
                           for node, attr_dict in nodes_data}
        elif label == 'name+type':
            labels_dict = {node: f"{node}({attr_dict['type']})"
                           for node, attr_dict in nodes_data}
        elif label in ['', 'none']:
            labels_dict = {node: '' for node, attr_dict in nodes_data}
        else:
            raise NotImplementedError(
                "Supported values of 'label' include ['name', 'type', "
                f"'name+type', 'none'], but got '{label}'.")

        # Partition plotted and hidden nodes, with colors
        plot_nodes, hidden_nodes, plot_colors = [], [], []
        for node, attr_dict in nodes_data:
            node_type = attr_dict['type']
            if node_type == 'pad':
                if hide == 'none':
                    plot_nodes.append(node)
                    plot_colors.append(NODE_COLORS['pad'])
                else:
                    hidden_nodes.append(node)
            elif node_type in FUNCTION_BRANCH_NODE_TYPES or node_type in INR_NODE_TYPES:
                if hide == 'aux':
                    hidden_nodes.append(node)
                else:
                    plot_nodes.append(node)
                    plot_colors.append(NODE_COLORS['aux'])
            else:
                plot_nodes.append(node)
                plot_colors.append(NODE_COLORS['pde'])

        # Hide edges if necessary
        if hide == 'aux':
            plot_node_set = set(plot_nodes)
            edge_list = {(node_p, node) for (node_p, node) in self._dag.edges()
                         if node_p in plot_node_set and node in plot_node_set}
        else:
            edge_list = self._dag.edges()

        # Plot the DAG
        plt.figure(figsize=(6, 6))
        pos = nx.shell_layout(self._dag, nlist=[plot_nodes, hidden_nodes])
        nx.draw_networkx_nodes(self._dag, pos, plot_nodes, node_color=plot_colors)
        nx.draw_networkx_edges(self._dag, pos, edge_list)
        nx.draw_networkx_labels(self._dag, pos, labels_dict)
        plt.show()

    def n_nodes_with_pad(self) -> int:
        r"""Number of nodes in the current DAG, including padded nodes."""
        return self.node_type.shape[0]


class PDENode:
    r"""
    Wrapper of the DAG nodes generated by the `PDENodesCollector` class.

    Args:
        name (Str): Name of the current node.
        src_pde (PDENodesCollector): The `PDENodesCollector` class instance
            from which the current node is generated.
    """

    def __init__(self, name: str, src_pde) -> None:
        self.name = name
        self.src_pde = src_pde
        self.cache_dict = {}

    def __neg__(self):
        return self.src_pde.neg(self)

    def __add__(self, node2):
        if node2 == 0:
            return self
        return self.src_pde.sum(node2, self)

    __radd__ = __add__

    def __mul__(self, node2):
        if node2 == 0:
            return 0
        if node2 == 1:
            return self
        if node2 == -1:
            return self.src_pde.neg(self)
        if node2 is self:
            return self.square
        return self.src_pde.prod(node2, self)

    __rmul__ = __mul__

    def __sub__(self, node2):
        # note that node2 could be a number
        return self.__add__(-node2)

    @property
    def dt(self):
        r"""Create a new node that represents the temporal derivative of this node."""
        if "dt" not in self.cache_dict:
            self.cache_dict["dt"] = self.src_pde.dt(self)
        return self.cache_dict["dt"]

    @property
    def dx(self):
        r"""Create a new node that represents the x-derivative of this node."""
        if "dx" not in self.cache_dict:
            self.cache_dict["dx"] = self.src_pde.dx(self)
        return self.cache_dict["dx"]

    @property
    def dy(self):
        r"""Create a new node that represents the y-derivative of this node."""
        if "dy" not in self.cache_dict:
            self.cache_dict["dy"] = self.src_pde.dy(self)
        return self.cache_dict["dy"]

    @property
    def dz(self):
        r"""Create a new node that represents the z-derivative of this node."""
        if "dz" not in self.cache_dict:
            self.cache_dict["dz"] = self.src_pde.dz(self)
        return self.cache_dict["dz"]

    @property
    def square(self):
        r"""Create a new node that represents the square of this node."""
        if "square" not in self.cache_dict:
            self.cache_dict["square"] = self.src_pde.square(self)
        return self.cache_dict["square"]

    @property
    def cubic(self):
        r"""Create a new node that represents the cubic of this node."""
        if "cubic" not in self.cache_dict:
            self.cache_dict["cubic"] = self.square * self
        return self.cache_dict["cubic"]


NULL_NODE = None
PDENodeType = Union[PDENode, None, float, int]
ExtendedFunc = namedtuple("ExtendedFunc", "sdf ext_values")
# types: NDArray[float], Union[float, NDArray[float]]


class PDENodesCollector:
    r"""
    This class enables specifying a PDE with its computational graph
    representation via Python scripts.

    Examples:
    ---------
    The nonlinear conservation law $u_t + (u^2)_x + (-0.3u)_y = 0$ on
    $(x,y)\in[0,1]^2$ with initial condition $u(0,x,y) = \sin(2\pi x)$:
        >>> ic_field = np.sin(2 * np.pi * x_fenc)  # define initial condition
        >>> pde = PDENodesCollector()
        >>> u = pde.new_uf()  # specify an unknown field
        >>> pde.set_ic(u, ic_field, x=x_fenc, y=y_fenc)  # specify IC
        >>> pde.sum_eq0(u.dt, u.square.dx, (-0.3 * u).dy)

    The last line above shows the simplified expression. The corresponding full
    version is:
        >>> c = pde.new_coef(-0.3)
        >>> term1 = pde.dx(pde.square(u))
        >>> term2 = pde.dy(pde.prod(c, u))
        >>> pde.sum_eq0(pde.dt(u), term1, term2)

    The periodic boundary condition is employed by default. For non-periodic
    cases, please use the method `new_domain(...)` to specify the shape of the
    computational domain as well as the location of the boundaries, and use the
    method `bc_sum_eq0(...)` to specify the boundary condition.

    After the overall PDE is specified, we may pass in the PDEformer
    configurations and construct the DAG data, as follows:
        >>> pde_dag = pde.gen_dag(config)

    Users may plot the resulting DAG by executing
        >>> pde_dag.plot()

    Note 1:
    If a summation involves three or more summands, it is not recommended to
    write
        >>> term_sum = term1 + term2 + term3

    (at least for the current version), which is equivalent to
        >>> term_sum = pde.sum(pde.sum(term1, term2), term3)

    Write instead
        >>> term_sum = pde.sum(term1, term2, term3)

    Note 2:
    It is not recommended to create multiple nodes with the same semantic
    meaning. For example, it is not recommended to write
        >>> pde.sum_eq0(pde.dt(u), 0.1 * pde.dx(u), -pde.dx(pde.dx(u)))

    since each statement `pde.dx(u)` will create a new node representing
    :math:`u_x`. The better practice is to store such a node (that will be used
    multiple times) as a separate variable, like
        >>> dx_u = pde.dx(u)
        >>> pde.sum_eq0(pde.dt(u), 0.1 * dx_u, -pde.dx(dx_u))

    Also, you may make use of the `u.dx` expression as:
        >>> pde.sum_eq0(pde.dt(u), 0.1 * u.dx, -pde.dx(u.dx))

    whose functionality is equivalent to that of
        >>> dx_u = u.dx
        >>> pde.sum_eq0(pde.dt(u), 0.1 * dx_u, -dx_u.dx)

    User may check the resulting DAG by creating a plot of it, as have just
    been introduced above.
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        self.dim = dim
        self.node_list = []
        self.node_scalar = []
        self.node_function = []

    def gen_dag(self, config: DictConfig) -> PDEAsDAG:
        r"""
        Generate the directed acyclic graph (DAG) as well as its structural
        data according to the configuration.
        """
        return PDEAsDAG(config, self.node_list, self.node_scalar, self.node_function)

    def new_domain(self,
                   sdf_values: NDArray[float],
                   *,  # keyword-only arguments afterwards
                   t: Union[float, NDArray[float]] = -1.,
                   x: Union[float, NDArray[float]] = -1.,
                   y: Union[float, NDArray[float]] = -1.,
                   z: Union[float, NDArray[float]] = -1.) -> PDENode:
        r"""
        Specify a new computational domain defined by SDF (signed distance
        function).
        """
        self._add_function(sdf_values, t=t, x=x, y=y, z=z)
        return self._add_node('sdf')

    def new_uf(self, domain: Optional[PDENode] = None) -> PDENode:
        r"""Specify a new unknown field variable."""
        if domain is None:
            return self._add_node('uf')
        return self._add_node('uf', predecessors=[domain.name])

    def new_coef(self, value: float) -> PDENode:
        r"""Specify a new PDE coefficient."""
        return self._add_node('coef', scalar=value)

    def new_coef_field(self,
                       field_values: NDArray[float],
                       *,  # keyword-only arguments afterwards
                       t: Union[float, NDArray[float]] = -1.,
                       x: Union[float, NDArray[float]] = -1.,
                       y: Union[float, NDArray[float]] = -1.,
                       z: Union[float, NDArray[float]] = -1.) -> PDENode:
        r"""Specify a new non-constant PDE coefficient."""
        self._add_function(field_values, t=t, x=x, y=y, z=z)
        return self._add_node('cf')

    def set_ic(self,
               src_node: PDENode,
               field_values: NDArray[float],
               *,  # keyword-only arguments afterwards
               x: Union[float, NDArray[float]] = -1.,
               y: Union[float, NDArray[float]] = -1.,
               z: Union[float, NDArray[float]] = -1.) -> None:
        r"""Specify initial condition."""
        self._add_function(field_values, t=0., x=x, y=y, z=z)
        self._add_node('ic', predecessors=[src_node.name])

    def set_bv(self,
               src_node: PDENode,
               boundary_sdf: Union[PDENode, NDArray[float]],
               field_values: Union[float, NDArray[float]],
               *,  # keyword-only arguments afterwards
               t: Union[float, NDArray[float]] = -1.,
               x: Union[float, NDArray[float]] = -1.,
               y: Union[float, NDArray[float]] = -1.,
               z: Union[float, NDArray[float]] = -1.) -> None:
        r"""
        Specify boundary values. The mathematical meaning is similar to
        'bc_sum_eq0', but the resulting DAG has a bit more connectivity.
        """
        if isinstance(boundary_sdf, np.ndarray):
            boundary_sdf = self.new_domain(boundary_sdf, t=t, x=x, y=y, z=z)
        self._add_function(field_values, t=t, x=x, y=y, z=z)
        self._add_node('bv', predecessors=[src_node.name, boundary_sdf.name])

    def dt(self, src_node: PDENodeType) -> PDENodeType:
        r"""Create a new node that represents the temporal derivative of the source node."""
        return self._unary(src_node, 'dt')

    def dx(self, src_node: PDENodeType) -> PDENodeType:
        r"""Create a new node that represents the x-derivative of the source node."""
        return self._unary(src_node, 'dx')

    def dy(self, src_node: PDENodeType) -> PDENodeType:
        r"""Create a new node that represents the y-derivative of the source node."""
        return self._unary(src_node, 'dy')

    def dz(self, src_node: PDENodeType) -> PDENodeType:
        r"""Create a new node that represents the z-derivative of the source node."""
        return self._unary(src_node, 'dz')

    def dn_sum_list(self,
                    src_node: PDENode,
                    domain: PDENode,
                    coef: Union[float, PDENode] = 1.) -> List[PDENode]:
        r"""
        Get a list of nodes, whose summation corresponds to the out-normal
        derivative of `src_node` $c\frac{du}{dn}$.
        """
        if self.dim is None:
            raise RuntimeError(
                "When using 'dn_sum_list', please specify the dimensionality "
                "of the PDE on initialization. Eg. For 2D PDE, initialize the "
                "class object as `pde = PDENodesCollector(dim=2)`.")
        if np.isscalar(coef):
            if coef == 0.:
                return []  # normal derivative being zero
            if coef != 1.:
                # float -> PDENode to avoid repeated creation of this coef node
                coef = self.new_coef(coef)
        sum_list = [self.prod(coef, src_node.dx, domain.dx)]
        if self.dim >= 2:
            sum_list.append(self.prod(coef, src_node.dy, domain.dy))
        if self.dim >= 3:
            sum_list.append(self.prod(coef, src_node.dz, domain.dz))
        return sum_list

    def neg(self, src_node: PDENodeType) -> PDENodeType:
        r"""Create a new node that represents the negation of the source node."""
        return self._unary(src_node, 'neg')

    def square(self, src_node: PDENodeType) -> PDENodeType:
        r"""Create a new node that represents the square of the source node."""
        return self._unary(src_node, 'square')

    def sin(self, src_node: PDENodeType) -> PDENodeType:
        r"""Create a new node that represents the sine of the source node."""
        return self._unary(src_node, 'sin')

    def cos(self, src_node: PDENodeType) -> PDENodeType:
        r"""Create a new node that represents the cosine of the source node."""
        return self._unary(src_node, 'cos')

    def exp10(self, src_node: PDENodeType) -> PDENodeType:
        r"""Create a new node that represents the exponential of the source node."""
        return self._unary(src_node, 'exp10')

    def log10(self, src_node: PDENodeType) -> PDENodeType:
        r"""Create a new node that represents the logarithm of the source node."""
        return self._unary(src_node, 'log10')

    def sum(self, *src_nodes) -> PDENodeType:
        r"""
        Create a new node that represents the summation of the existing source
        nodes, with type 'add'.
        """
        return self._multi_predecessor('add', src_nodes, ignore_node=0.)

    def prod(self, *src_nodes) -> PDENodeType:
        r"""
        Create a new node that represents the product of the existing source
        nodes, with type 'mul'.
        """
        if 0 in src_nodes:
            return 0
        return self._multi_predecessor('mul', src_nodes, ignore_node=1.)

    def sum_eq0(self, *src_nodes) -> PDENodeType:
        r"""Add an equation: the sum of all input nodes equals zero."""
        sum_node = self._multi_predecessor('add', src_nodes, ignore_node=0.)
        return self._unary(sum_node, 'eq0')

    def bc_sum_eq0(self, boundary_sdf: PDENode, *src_nodes) -> PDENodeType:
        r"""
        Add a boundary condition, requiring the sum of all input nodes being
        equal to zero on the boundary. The functionality is similar to
        'set_bv', but the resulting DAG may seem more natural, resembling that
        of the interior PDE without introducing a new type of node ('bv').
        """
        sum_node = self._multi_predecessor('add', src_nodes, ignore_node=0.)
        return self._add_node(
            'eq0', predecessors=[sum_node.name, boundary_sdf.name])

    def _add_node(self,
                  type_: str,
                  predecessors: Optional[List[PDENode]] = None,
                  scalar: float = 0.,
                  name: str = "") -> PDENode:
        r"""Add a new node to the DAG."""
        if name == "":
            name = type_ + str(len(self.node_list))
        if predecessors is None:
            self.node_list.append((name, type_))
        else:
            self.node_list.append((name, type_, *predecessors))
        if not np.isscalar(scalar):
            raise ValueError(f"PDE node receives non-scalar value {scalar}.")
        self.node_scalar.append(scalar)
        return PDENode(name, self)

    def _add_function(self,
                      f_values: NDArray[float],
                      *,  # keyword-only arguments afterwards
                      t: Union[float, NDArray[float]] = -1.,
                      x: Union[float, NDArray[float]] = -1.,
                      y: Union[float, NDArray[float]] = -1.,
                      z: Union[float, NDArray[float]] = -1.) -> None:
        r"""
        Add a new node representing a function to the DAG. '-1' values of
        t/x/y/z indicate that the function is independent of this variable.
        """
        function = np.stack(np.broadcast_arrays(t, x, y, z, f_values), axis=-1)
        # [*, 5] -> [n_pts, 5]
        function = function.reshape((-1, 5))
        self.node_function.append(function)

    def _unary(self, src_node: PDENodeType, node_type: str) -> PDENodeType:
        r"""Add a new node representing a unary operation to the DAG."""
        if np.isscalar(src_node):
            raise TypeError(
                "Unexpected usage. When applying unary operations to a scalar,"
                " please compute the numerical value directly, instead of"
                " creating a new node in the computational graph.")
        if src_node is NULL_NODE or np.isscalar(src_node):
            return NULL_NODE
        return self._add_node(node_type, predecessors=[src_node.name])

    def _multi_predecessor(self,
                           node_type: str,
                           src_nodes: Tuple,
                           *,
                           ignore_node: PDENodeType = NULL_NODE) -> PDENodeType:
        r"""
        Input:
            node_type (str): choices {"add", "mul"}
            src_nodes (Tuple): (node1, node2, ..) or ([node1, node2, ..], )
                or [[node1, node2, ..]], where node1, node2, .. are
                of type Union[PDENode, int, float, NoneType].
            ignore_node (Union[None, float]): Node types to be ignored.
        Output:
            out_node (PDENodeType): The newly created node.
        """
        # ([node1, node2, ..], ) -> [node1, node2, ..]
        if len(src_nodes) == 1 and isinstance(src_nodes[0], (list, tuple)):
            src_nodes, = src_nodes
        # remove NULL_NODE or ignored entries
        src_nodes = [node for node in src_nodes
                     if node not in [NULL_NODE, ignore_node]]
        # type check; convert number entries into 'coef' nodes in the DAG
        for i, node in enumerate(src_nodes):
            if np.isscalar(node):
                src_nodes[i] = self.new_coef(node)
            elif not isinstance(node, PDENode):
                raise ValueError("Nodes involved in sum/prod should have type"
                                 + " {PDENode, float, int, NoneType}, but got"
                                 + f" {type(node)}")
        if len(src_nodes) == 0:  # pylint: disable=C1801
            return NULL_NODE
        if len(src_nodes) == 1:
            return src_nodes[0]  # single predecessor, no new node to create
        src_node_names = [node.name for node in src_nodes]

        return self._add_node(node_type, predecessors=src_node_names)


def merge_extended_func_soft(func_list: List[ExtendedFunc]) -> ExtendedFunc:
    r"""
    Given $f_i: X_i\to\R$, each extended to the whole square domain, returns
    another extended function which equals $f_i$ on $X_i$.
    """
    # Both have shape [n_func, n_x, n_y].
    sdf_all = np.stack(np.broadcast_arrays(
        *(sdf for (sdf, ext_values) in func_list)))
    values_all = np.stack(np.broadcast_arrays(
        *(ext_values for (sdf, ext_values) in func_list)))

    # extended values given by a weighted average
    inv_sdf = 1. / np.maximum(1e-8, sdf_all)  # negative values also ignored
    inv_sdf /= inv_sdf.sum(axis=0, keepdims=True)
    ext_values = np.sum(values_all * inv_sdf, axis=0)  # [n_x, n_y]

    sdf = sdf_all.min(axis=0)
    return ExtendedFunc(sdf, ext_values)


def merge_extended_func_hard(func_list: List[ExtendedFunc]) -> ExtendedFunc:
    r"""
    Given $f_i: X_i\to\R$, each extended to the whole square domain, returns
    another extended function which equals $f_i$ on $X_i$. Hard extension is
    employed, and the extended function thus typically contains discontinuity.
    """
    (sdf, ext_values) = func_list[0]
    sdf_out = sdf.copy()
    values_out = ext_values.copy()
    for (sdf, ext_values) in func_list[1:]:
        update_mask = sdf < sdf_out
        sdf_out = np.where(update_mask, sdf, sdf_out)
        values_out = np.where(update_mask, ext_values, values_out)
    return ExtendedFunc(sdf_out, values_out)


merge_extended_func = merge_extended_func_hard

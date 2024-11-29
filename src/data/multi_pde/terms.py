r"""
Generate DAG nodes as well as LaTeX representations for the basic terms
involved in the custom multi_pde dataset.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Union, Tuple, List, Dict

import numpy as np
from numpy.typing import NDArray
import h5py
from scipy.interpolate import CubicSpline

from ..pde_dag import PDENodesCollector, PDENode


class LaTeXSum:
    r"""
    Storing a summation of a PDE, with each term represented in LaTeX.

    Attributes:
        term_list (List[str]): A list of the terms (in LaTeX) to be summed.
        coef_dict (OrderedDict[str, float]): A dictionary containing the values
            of the PDE coefficients.
    """

    def __init__(self, keep_all_coef: bool, coef_dict: OrderedDict) -> None:
        self.term_list = []
        self.coef_dict = coef_dict  # note that this is not a deep copy
        self.keep_all_coef = keep_all_coef

    # def __iadd__(self, latex_sum_new) -> None:
    #     self.term_list.extend(latex_sum_new.term_list)
    #     self.coef_dict.update(latex_sum_new.coef_dict)

    def add_term(self, term_latex: str) -> None:
        r"""Add a summand term involved in the summation."""
        if term_latex != "":
            self.term_list.append(term_latex)

    def add_coef(self, symbol: str, value: float) -> None:
        r"""Add a PDE coefficient."""
        self.coef_dict[symbol] = value

    # def merge_coefs(self, latex_sum_new) -> None:
    #     r"""Merge the dictionary of PDE coefficients with another instance."""
    #     self.coef_dict.update(latex_sum_new.coef_dict)

    def strip_sum(self, prefix: str = "", postfix: str = "") -> str:
        r"""Get the overall LaTeX expression of summation."""
        sum_str = "+".join(self.term_list)
        if sum_str == "":
            return ""
        sum_str = sum_str.replace("+-", "-")
        sum_str = prefix + sum_str + postfix
        self.term_list = []  # clear existing summands
        return sum_str

    def add_term_with_coef(self,
                           coef_value: float,
                           coef_symbol: str,
                           term_latex: str) -> None:
        r"""
        Add the LaTeX expression for the multiplication of a coefficient and a
        specific term.
        """
        if self.keep_all_coef or coef_value not in [0., 1., -1.]:
            if (term_latex != "" and term_latex[0].isalpha()
                    and coef_symbol[0] == "\\" and coef_symbol[-1].isalpha()):
                sep = " "
            else:
                sep = ""
            self.add_term(coef_symbol + sep + term_latex)
            self.add_coef(coef_symbol, coef_value)
        elif coef_value != 0.:
            if term_latex == "":
                term_latex = "1"
            if coef_value < 0.:
                term_latex = "-" + term_latex
            self.add_term(term_latex)


class PDETermBase(ABC):
    r""" Abstract basic class of a PDE term. """

    @abstractmethod
    def __init__(self,
                 hdf5_group: Union[h5py.Group, h5py.Dataset],
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        self.keep_all_coef = keep_all_coef

    @abstractmethod
    def nodes(self, pde: PDENodesCollector) -> Union[float, PDENode]:
        r"""Generate the node representation for this PDE term."""
        return 0.

    @abstractmethod
    def add_latex(self, latex_sum: LaTeXSum, symbol: str) -> None:
        r"""
        Generate the mathematical representation (a list of terms represented
        in LaTeX) as well as a dictionary of coefficient values for this PDE
        term.

        Args:
            symbol (str): LaTeX expression of the current term.
        """


class ScalarCoef(PDETermBase):
    r"""
    One scalar coefficient in a PDE, each entry set to zero or one with certain
    probability.
    """
    value: float

    def __init__(self,
                 hdf5_group: h5py.Dataset,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.value = hdf5_group[idx_pde]

    def nodes(self, pde: PDENodesCollector) -> PDENode:
        if self.keep_all_coef:
            return pde.new_coef(self.value)
        return self.value

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum: LaTeXSum,
                  symbol: str,
                  term_latex: str = "") -> None:
        latex_sum.add_term_with_coef(self.value, symbol, term_latex)


class LgNonNegScalarCoef(ScalarCoef):
    r"""
    Non-negative scalar coefficient in a PDE, written in the form $10^a$ to
    avoid small values.
    """

    def nodes(self, pde: PDENodesCollector) -> PDENode:
        if self.value < 0.:
            raise ValueError("Cannot set `lg_val` for negative values.")
        if self.value == 0.:
            return 0
        if self.value == 1. and not self.keep_all_coef:
            return 1
        return pde.exp10(pde.new_coef(np.log10(self.value)))


class CoefArray(PDETermBase):
    r"""
    An array containing multiple coefficients in a PDE, each entry set to zero
    or one with certain probability.
    """
    value: NDArray[float]

    def __init__(self,
                 hdf5_group: h5py.Dataset,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.value = hdf5_group[idx_pde]

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              idx: Union[int, Tuple[int]] = ()) -> PDENode:
        value = self._get_value(idx)
        if self.keep_all_coef:
            return pde.new_coef(value)
        return value

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum: LaTeXSum,
                  symbol: str,
                  term_latex: str = "",
                  idx: Union[int, Tuple[int]] = ()) -> None:
        value = self._get_value(idx)
        latex_sum.add_term_with_coef(value, symbol, term_latex)

    def _get_value(self, idx: Union[int, Tuple[int]]) -> float:
        r"""Obtain the scalar value."""
        value = self.value[idx]
        if np.size(value) > 1:
            raise ValueError("'array[idx]' should be a scalar rather than an array.")
        return value


class FieldCoef(PDETermBase):
    r"""Coefficient field term involved in a PDE"""

    def __init__(self,
                 hdf5_group: h5py.Dataset,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.field = hdf5_group[idx_pde]

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              coord_dict: Dict[str, NDArray[float]] = None) -> PDENode:
        if coord_dict is None:
            raise ValueError("Input 'coord_dict' should be given.")
        return pde.new_coef_field(self.field, **coord_dict)

    def add_latex(self, latex_sum: LaTeXSum, symbol: str) -> None:
        latex_sum.add_term(symbol + "(r)")


class LgNonNegFieldCoef(FieldCoef):
    r"""
    Non-negative coefficient field term involved in a PDE, in the form
    $10^{a(r)}$ to avoid small values.
    """

    def nodes(self,
              pde: PDENodesCollector,
              coord_dict: Dict[str, NDArray[float]] = None) -> PDENode:
        if coord_dict is None:
            raise ValueError("Input 'coord_dict' should be given.")
        return pde.exp10(pde.new_coef_field(np.log10(self.field), **coord_dict))


class ConstOrField(PDETermBase):
    r"""
    Coefficient term involved in a PDE, which can be a zero, a real number
    (scalar) or a spatial-varying field.
    """
    ZERO_COEF = 0
    UNIT_COEF = 1
    SCALAR_COEF = 2
    FIELD_COEF = 3

    def __init__(self,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.coef_type = hdf5_group["coef_type"][idx_pde]
        self.field = hdf5_group["field"][idx_pde]

    def __add__(self, other: Union['ConstOrField', int, float]) -> 'ConstOrField':
        if np.isscalar(other) or other.coef_type != other.FIELD_COEF:
            if isinstance(other, ConstOrField):
                other = other._get_value()  # pylint: disable=protected-access
            field = self.field + other
            value = field.flat[0]
            if self.coef_type == self.FIELD_COEF:
                coef_type = self.FIELD_COEF
            elif np.abs(value) < 1e-6:
                coef_type = self.ZERO_COEF
            elif np.abs(value - 1) < 1e-6:
                coef_type = self.UNIT_COEF
            else:
                coef_type = self.SCALAR_COEF
            return ConstOrField._from_field_and_type(
                field, coef_type, self.keep_all_coef)

        field = self.field + other.field
        coef_type = self.FIELD_COEF
        return ConstOrField._from_field_and_type(
            field, coef_type, self.keep_all_coef)

    __radd__ = __add__

    def __mul__(self, other: Union['ConstOrField', int, float]) -> 'ConstOrField':
        if np.isscalar(other):
            field = self.field * other
            if self.coef_type == self.UNIT_COEF and other != 1:
                coef_type = self.SCALAR_COEF
            elif other == 0:
                coef_type = self.ZERO_COEF
            else:
                coef_type = self.coef_type
            return ConstOrField._from_field_and_type(
                field, coef_type, self.keep_all_coef)

        field = self.field * other.field
        if self.ZERO_COEF in (self.coef_type, other.coef_type):
            coef_type = self.ZERO_COEF
        elif self.FIELD_COEF in (self.coef_type, other.coef_type):
            coef_type = self.FIELD_COEF
        elif self.SCALAR_COEF in (self.coef_type, other.coef_type):
            coef_type = self.SCALAR_COEF
        else:
            coef_type = self.UNIT_COEF
        return ConstOrField._from_field_and_type(
            field, coef_type, self.keep_all_coef)

    __rmul__ = __mul__

    def __sub__(self, other: Union['ConstOrField', int, float]) -> 'ConstOrField':
        return self + (-other)

    def __rsub__(self, other: Union[int, float]) -> 'ConstOrField':
        return other + (-self)

    def __neg__(self) -> 'ConstOrField':
        field = -self.field
        if self.coef_type == self.UNIT_COEF:
            coef_type = self.SCALAR_COEF
        else:
            coef_type = self.coef_type
        return ConstOrField._from_field_and_type(
            field, coef_type, self.keep_all_coef)

    def __truediv__(self, other: Union['ConstOrField', int, float]) -> 'ConstOrField':
        if not isinstance(other, ConstOrField):
            if other == 0.:
                raise ZeroDivisionError("Cannot divide by zero.")
            return self * (1. / other)
        return self * other.inverse()

    def __rtruediv__(self, other: Union[int, float]) -> 'ConstOrField':
        return other * self.inverse()

    @property
    def type_char(self) -> str:
        r"""Get the character representing the type of the coefficient."""
        if self.coef_type == self.ZERO_COEF:
            return "0"
        if self.coef_type == self.UNIT_COEF:
            return "1"
        if self.coef_type == self.SCALAR_COEF:
            return "c"
        if self.coef_type == self.FIELD_COEF:
            return "f"
        raise RuntimeError

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              coord_dict: Dict[str, NDArray[float]] = None) -> PDENode:
        if self.coef_type == self.FIELD_COEF:
            if coord_dict is None:
                raise ValueError("Input 'coord_dict' should be given.")
            return pde.new_coef_field(self.field, **coord_dict)

        value = self._get_value()
        if self.keep_all_coef:
            return pde.new_coef(value)
        return value

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum: LaTeXSum,
                  symbol: str,
                  term_latex: str = "") -> None:
        if self.coef_type == self.FIELD_COEF:
            latex_sum.add_term(symbol + "(r)" + term_latex)
        else:
            value = self._get_value()
            latex_sum.add_term_with_coef(value, symbol, term_latex)

    def inverse(self) -> 'ConstOrField':
        r"""Get the inversion of the current field."""
        if self.coef_type == self.FIELD_COEF:
            if np.any(self.field <= 0.):
                raise ValueError("Cannot take the inverse of non-positive field.")
            coef_type = self.FIELD_COEF
        elif self.coef_type == self.ZERO_COEF:
            raise ZeroDivisionError("Cannot divide by zero.")
        else:
            coef_type = self.coef_type
        return ConstOrField._from_field_and_type(
            1. / self.field, coef_type, self.keep_all_coef)

    @staticmethod
    def _from_field_and_type(field: NDArray[float],
                             coef_type: int,
                             keep_all_coef: bool) -> 'ConstOrField':
        instance = ConstOrField.__new__(ConstOrField)
        instance.field = field
        instance.coef_type = coef_type
        instance.keep_all_coef = keep_all_coef
        return instance

    def _get_value(self) -> float:
        r"""Obtain the scalar value."""
        if self.coef_type == self.ZERO_COEF:
            return 0.
        if self.coef_type == self.UNIT_COEF:
            return 1.
        if self.coef_type == self.SCALAR_COEF:
            return self.field.flat[0]
        raise RuntimeError("unexpected 'coef_type'!")


class ReNamedConstOrField(ConstOrField):
    r"""Identify the name of dataset of ConstOrField."""

    def __init__(self,  # pylint: disable=super-init-not-called
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool,
                 coef_type_name: str = "coef_type",
                 field_name: str = "field") -> None:
        self.coef_type = hdf5_group[coef_type_name][idx_pde]
        self.field = hdf5_group[field_name][idx_pde]
        self.keep_all_coef = keep_all_coef


class LgNonNegConstOrField(ConstOrField):
    r"""
    Non-negative coefficient term involved in a PDE, which can be a zero, a
    real number (scalar) or a spatial-varying field. Written in the form
    $10^{a(r)}$.
    """

    def nodes(self,
              pde: PDENodesCollector,
              coord_dict: Dict[str, NDArray[float]] = None) -> PDENode:
        if self.coef_type == self.FIELD_COEF:
            # following LgNonNegFieldCoef
            if coord_dict is None:
                raise ValueError("Input 'coord_dict' should be given.")
            return pde.exp10(pde.new_coef_field(
                np.log10(self.field), **coord_dict))

        # following LgNonNegScalarCoef
        value = self._get_value()
        if value < 0.:
            raise ValueError("Cannot set `lg_val` for negative values.")
        if value == 0.:
            return 0
        if value == 1. and not self.keep_all_coef:
            return 1
        return pde.exp10(pde.new_coef(np.log10(value)))


class TimeIndepForce(PDETermBase):
    r"""Time-independent force term involved in a PDE"""

    def __init__(self,
                 hdf5_group: h5py.Dataset,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        field = hdf5_group["field"][idx_pde]  # Shape: (*, n_vars)
        self.n_vars = field.shape[-1]
        self.fields = [field[..., i] for i in range(self.n_vars)]

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              idx_var: int,
              coord_dict: Dict[str, NDArray[float]],
              neg: bool = False) -> PDENode:
        field = self.fields[idx_var]
        if neg:
            field = -field
        return pde.new_coef_field(field, **coord_dict)

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum_list: List[LaTeXSum],
                  symbol: str = "f") -> None:
        for i in range(self.n_vars):
            latex_sum = latex_sum_list[i]
            latex_sum.add_term(symbol + "_{" + str(i) + "}(r)")


class TimeDepForce(PDETermBase):
    r"""
    Time-dependent force term involved in a PDE, with the form
    $f(r, t) = fr(r)ft(t)$.
    """

    def __init__(self,
                 hdf5_group: h5py.Dataset,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        fr_field = hdf5_group["fr/field"][idx_pde]
        self.n_vars = fr_field.shape[-1]
        self.fr_fields = [fr_field[..., i] for i in range(self.n_vars)]
        self.ft_field = hdf5_group["ft/field"][idx_pde]

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              idx_var: int,
              coord_dict: Dict[str, NDArray[float]],
              t_coord: NDArray[float],
              neg: bool = False) -> PDENode:
        fr_field = self.fr_fields[idx_var]
        if neg:
            fr_field = -fr_field
        fr_node = pde.new_coef_field(fr_field, **coord_dict)
        # TODO: this is a temporary solution, which should be replaced.
        t_coord_new = np.linspace(t_coord[0], t_coord[-1], coord_dict["y"].size)
        ft_func = CubicSpline(t_coord, self.ft_field, axis=0)
        ft_field = ft_func(t_coord_new)
        # # Method 1:
        # # Extend the ft(t) to a 2D field ft(x, t).
        # ft_node = pde.new_coef_field(
        #     np.tile(ft_field, (coord_dict["x"].size, 1)),
        #     x=coord_dict["x"].reshape(-1, 1),
        #     t=t_coord_new.reshape(1, -1))
        # Method 2:
        # repeat the t_coord and ft(t) for coord_dict["x"].size times
        ft_node = pde.new_coef_field(
            np.repeat(ft_field, coord_dict["x"].size),
            t=np.repeat(t_coord_new, coord_dict["x"].size))
        return pde.prod(fr_node, ft_node)

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum_list: List[LaTeXSum],
                  symbol: str = "f") -> None:
        for i in range(self.n_vars):
            latex_sum = latex_sum_list[i]
            term_latex = symbol + "r_{" + str(i) + "}(r)" + symbol + "t(t)"
            latex_sum.add_term(term_latex)


class StiffnessMatrix2D2Comp(PDETermBase):
    r"""
    Stiffness matrix in 2-D 2-Component ElasticWave equation.
    """
    ISOTROPY = 0
    SUPPORTED_TYPES = [ISOTROPY]
    n_vars: int = 2
    value_cls: type = ConstOrField

    def __init__(self,
                 hdf5_group: h5py.Dataset,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.values = []
        stiff_type = hdf5_group.get("stiff_type", self.ISOTROPY)
        if stiff_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported 'stiff_type' {stiff_type}.")
        for key in hdf5_group:
            self.values.append(self.value_cls(
                hdf5_group[key], idx_pde, keep_all_coef))
        self.dof = len(self.values)
        if stiff_type == self.ISOTROPY:
            if self.dof != 2:
                raise ValueError(f"Expected 2 DOF for isotropy, but got "
                                 f"{self.dof}.")
            # store the Lame parameters instead
            e_ = self.values[0]  # Young's modulus
            nu_ = self.values[1]  # Poisson's ratio
            g_2 = e_ / (1 + nu_)
            lamb = g_2 * nu_ / (1 - nu_)
            self.values = [g_2, lamb]
        self.stiff_type = stiff_type

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_list: List[PDENode],
              coord_dict: Dict[str, NDArray[float]]) -> List[List[PDENode]]:
        if len(u_list) != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, but got "
                             f"{len(u_list)}.")
        if self.stiff_type == self.ISOTROPY:
            # doubled 1st Lame parameter
            g_2 = self.values[0].nodes(pde, coord_dict)
            lamb = self.values[1].nodes(pde, coord_dict)  # 2st Lame parameter
            # diagonal entries of strain tensor
            eps_xx = u_list[0].dx
            eps_yy = u_list[1].dy
            # stress tensor
            term1 = lamb * (eps_xx + eps_yy)
            s_xx = term1 + g_2 * eps_xx
            s_yy = term1 + g_2 * eps_yy
            s_xy = pde.prod(0.5, g_2, u_list[0].dy + u_list[1].dx)
        else:
            raise RuntimeError(f"Unexpected 'stiff_type' {self.stiff_type}.")

        return [[s_xx, s_xy], [s_xy, s_yy]]

    def add_latex(self, latex_sum: LaTeXSum, symbol: str) -> None:  # pylint: disable=unused-argument
        raise NotImplementedError


class PolySinusNonlinTerm(PDETermBase):
    r"""
    The non-linear term $f(u)$ in the PDE, in the form
    $f(u) = \sum_{k=1}^3c_{0k}u^k + \sum_{j=1}^Jc_{j0}h_j(c_{j1}u+c_{j2}u^2)$,
    where $h_j\in\{\sin,\cos\}$.
    """
    value: NDArray[float]

    def __init__(self,
                 hdf5_group: h5py.Dataset,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.value = hdf5_group[idx_pde]

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_node: PDENode = None) -> List[PDENode]:
        if u_node is None:
            raise ValueError("Argument 'u_node' must be specified.")
        sum_list = []

        def term(j, k):
            coef = self.value[j, k]  # float
            if self.keep_all_coef:
                coef = pde.new_coef(coef)
            elif coef == 0.:
                return 0.
            if k == 0:
                return coef
            if k == 1:
                return coef * u_node
            if k == 2:
                return coef * u_node.square
            if k == 3:
                return coef * u_node.cubic
            raise ValueError(f"Unexpected 'k' {k}.")

        # polynomial part
        for k in range(1, 4):
            sum_list.append(term(0, k))

        # sinusoidal part
        for j in range(1, self.value.shape[0]):
            if self.value[j, 0] == 0:
                # ignore this sinusoidal term even when keep_all_coef=True
                continue
            sinus_sum_list = []
            sinus_sum_list.append(term(j, 1))
            sinus_sum_list.append(term(j, 2))
            op_j = pde.sum(sinus_sum_list)
            if self.value[j, 3] > 0:
                hj_op = pde.sin(op_j)
            else:
                hj_op = pde.cos(op_j)
            if None in [term(j, 0), hj_op]:
                print(f"term(j, 0)={term(j, 0)}, hj_op={hj_op}")
            if hj_op is not None:
                sum_list.append(term(j, 0) * hj_op)

        return sum_list

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum: LaTeXSum,
                  symbol: str = "c",
                  u_symbol: str = "u",
                  i: Union[int, str] = "") -> None:
        def coef(j, k):
            return f"{symbol}_{{{i}{j}{k}}}"

        # polynomial part
        j = 0 if self.value.shape[0] > 1 else ""
        latex_sum.add_term_with_coef(self.value[0, 1], coef(j, 1), u_symbol)
        latex_sum.add_term_with_coef(self.value[0, 2], coef(j, 2), u_symbol + "^2")
        latex_sum.add_term_with_coef(self.value[0, 3], coef(j, 3), u_symbol + "^3")

        # sinusoidal part
        for j in range(1, self.value.shape[0]):
            if self.value[j, 0] == 0:
                continue
            opj_sum = LaTeXSum(self.keep_all_coef, latex_sum.coef_dict)
            opj_sum.add_term_with_coef(self.value[j, 1], coef(j, 1), u_symbol)
            opj_sum.add_term_with_coef(self.value[j, 2], coef(j, 2), u_symbol + "^2")
            hj_latex = r"\sin(" if self.value[j, 3] > 0 else r"\cos("
            hj_latex = opj_sum.strip_sum(hj_latex, ")")
            latex_sum.add_term_with_coef(self.value[j, 0], coef(j, 0), hj_latex)
            # latex_sum.merge_coefs(opj_sum)


class HomSpatialOrder2Term(PDETermBase):
    r"""
    Spatial second-order term, whose form is randomly selected from
    the non-divergence form $Lu=-a\Delta u$, the factored form
    $Lu=-\sqrt a\nabla\cdot(\sqrt a\nabla u)$, and the divergence form
    $Lu=-\nabla\cdot(a\nabla u)$ with equal probability. Here $a$ is
    taken to be a random non-negative real number.
    Note that these three forms are mathematically equivalent since $a$ has no
    spatial dependency, and we distinguish them only because they correspond to
    different DAG representations for PDEformer.
    """
    NON_DIV_FORM = 0
    FACTORED_FORM = 1
    DIV_FORM = 2
    LG_COEF: bool = False

    def __init__(self,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.diff_type = hdf5_group["diff_type"][idx_pde]
        self.value = hdf5_group["value"][idx_pde]

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_node: PDENode = None) -> PDENode:
        c_or_c2 = self.value
        if c_or_c2 == 0. and (self.LG_COEF or not self.keep_all_coef):
            return 0.

        if u_node is None:
            raise ValueError("Argument 'u_node' must be specified.")
        if c_or_c2 == 1. and not self.keep_all_coef:
            return -(u_node.dx.dx + u_node.dy.dy)

        if self.diff_type == self.FACTORED_FORM:
            c_or_c2 = np.sqrt(c_or_c2)  # c rather than c^2

        if self.LG_COEF:
            c_or_c2 = pde.exp10(pde.new_coef(np.log10(c_or_c2)))
        else:
            c_or_c2 = pde.new_coef(c_or_c2)

        # compute the 2nd-order differential term
        if self.diff_type == self.NON_DIV_FORM:
            return -(c_or_c2 * (u_node.dx.dx + u_node.dy.dy))
        div_term = pde.dx(c_or_c2 * u_node.dx) + pde.dy(c_or_c2 * u_node.dy)
        if self.diff_type == self.FACTORED_FORM:
            return -(c_or_c2 * div_term)
        if self.diff_type == self.DIV_FORM:
            return -div_term
        raise NotImplementedError

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum: LaTeXSum,
                  symbol: str = "a",
                  u_symbol: str = "u") -> None:
        c_or_c2 = self.value
        if c_or_c2 == 0. and not self.keep_all_coef:
            return
        if c_or_c2 == 1. and not self.keep_all_coef:
            latex_sum.add_term(rf"-\Delta {u_symbol}")
            return

        if self.diff_type == self.FACTORED_FORM:
            c_or_c2 = np.sqrt(c_or_c2)  # c rather than c^2

        if self.LG_COEF:
            latex_sum.add_coef(symbol, np.log10(c_or_c2))
            symbol = "10^{" + symbol + "}"
        else:
            latex_sum.add_coef(symbol, c_or_c2)

        # compute the 2nd-order differential term
        if self.diff_type == self.NON_DIV_FORM:
            latex_sum.add_term(rf"-{symbol}\Delta {u_symbol}")
        elif self.diff_type == self.FACTORED_FORM:
            latex_sum.add_term(
                rf"-{symbol}\nabla\cdot({symbol}\nabla {u_symbol})")
        elif self.diff_type == self.DIV_FORM:
            latex_sum.add_term(rf"-\nabla\cdot({symbol}\nabla {u_symbol})")


class LgCoefHomSpatialOrder2Term(HomSpatialOrder2Term):
    r"""
    Same as `HomSpatialOrder2Term`, but with the coefficients written in the
    exponential form $10^a$.
    """
    LG_COEF: bool = True


class InhomSpatialOrder2Term(PDETermBase):
    r"""
    Spatial second-order term, whose form is randomly selected from
    the non-divergence form $Lu=-a(r)\Delta u$, the factored form
    $Lu=-\sqrt a(r)\nabla\cdot(\sqrt a(r)\nabla u)$, and the divergence form
    $Lu=-\nabla\cdot(a(r)\nabla u)$ with equal probability. Here $a(r)$ is
    taken to be a non-negative random field.
    """
    NON_DIV_FORM = 0
    FACTORED_FORM = 1
    DIV_FORM = 2
    LG_COEF: bool = False

    def __init__(self,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.diff_type = hdf5_group["diff_type"][idx_pde]
        self.field = hdf5_group["field"][idx_pde]

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_node: PDENode = None,
              coord_dict: Dict[str, NDArray[float]] = None) -> PDENode:
        if u_node is None:
            raise TypeError("Argument 'u_node' must be specified.")
        if coord_dict is None:
            raise TypeError("Input 'coord_dict' should be given.")

        if self.diff_type == self.FACTORED_FORM:
            c_or_c2 = self.field
        else:
            c_or_c2 = np.sqrt(self.field)

        if self.LG_COEF:
            c_or_c2 = pde.exp10(pde.new_coef_field(np.log10(c_or_c2),
                                                   **coord_dict))
        else:
            c_or_c2 = pde.new_coef_field(c_or_c2, **coord_dict)

        # compute the 2nd-order differential term
        if self.diff_type == self.NON_DIV_FORM:
            return -(c_or_c2 * (u_node.dx.dx + u_node.dy.dy))
        div_term = pde.dx(c_or_c2 * u_node.dx) + pde.dy(c_or_c2 * u_node.dy)
        if self.diff_type == self.FACTORED_FORM:
            return -(c_or_c2 * div_term)
        if self.diff_type == self.DIV_FORM:
            return -div_term
        raise NotImplementedError

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum: LaTeXSum,
                  symbol: str = "a",
                  u_symbol: str = "u") -> None:
        symbol = symbol + "(r)"
        if self.LG_COEF:
            symbol = "10^{" + symbol + "}"

        # compute the 2nd-order differential term
        if self.diff_type == self.NON_DIV_FORM:
            latex_sum.add_term(rf"-{symbol}\Delta {u_symbol}")
        elif self.diff_type == self.FACTORED_FORM:
            latex_sum.add_term(
                rf"-{symbol}\nabla\cdot({symbol}\nabla {u_symbol})")
        elif self.diff_type == self.DIV_FORM:
            latex_sum.add_term(rf"-\nabla\cdot({symbol}\nabla {u_symbol})")


class LgCoefInhomSpatialOrder2Term(InhomSpatialOrder2Term):
    r"""
    Same as `InhomSpatialOrder2Term`, but with the coefficients written in the
    base-10 exponential form $10^{a(r)}$.
    """
    LG_COEF: bool = True


class MultiComponentLinearTerm(PDETermBase):
    r"""
    Linear term (in vector form) for multi-component PDEs:
        $$f(u)_i = \sum_ja_{ij}u_j,$$
    where $0 \le i,j \le d_u-1$.
    Coefficient matrix $a_{ij}$ stored in the COOrdinate format.
    """

    def __init__(self,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        coo_len = hdf5_group["coo_len"][idx_pde]
        self.coo_i = hdf5_group["coo_i"][idx_pde, :coo_len]
        self.coo_j = hdf5_group["coo_j"][idx_pde, :coo_len]
        self.coo_vals = hdf5_group["coo_vals"][idx_pde, :coo_len]

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_list: List[PDENode] = None) -> List[List[PDENode]]:
        if u_list is None:
            raise ValueError("Argument 'u_list' must be specified.")
        node_lists = [[] for _ in u_list]
        for (i, j, value) in zip(self.coo_i, self.coo_j, self.coo_vals):
            if self.keep_all_coef:
                value = pde.new_coef(value)
            node_lists[i].append(value * u_list[j])
        return node_lists

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum_list: List[LaTeXSum],
                  symbol: Union[str, List[str], None] = None,
                  c_symbol: str = "c",
                  *,
                  l: Union[int, str] = "") -> None:
        for (i, j, value) in zip(self.coo_i, self.coo_j, self.coo_vals):
            coef_latex = f"{c_symbol}_{{{l}{i}{j}}}"
            latex_sum_list[i].add_term_with_coef(
                value, coef_latex, self._u_symbol(symbol, j))

    @staticmethod
    def _u_symbol(symbol: Union[str, List[str], None], j: int) -> str:
        r"""Get the symbol corresponding to the j-th component."""
        if symbol is None:
            return rf"u_{j}"
        if len(symbol) == 1:
            return rf"{symbol[0]}_{j}"
        return symbol[j]


class MultiComponentDegree2Term(MultiComponentLinearTerm):
    r"""
    Degree-two term (in vector form) for multi-component PDEs:
        $$f(u)_i = \sum_{j,k}b_{ijk}u_ju_k,$$
    where $0 \le i,j,k \le d_u-1$, $j \le k$.
    Coefficient tensor $b_{ijk}$ stored in the COOrdinate format.
    """

    def __init__(self,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        coo_len = hdf5_group["coo_len"][idx_pde]
        self.coo_k = hdf5_group["coo_k"][idx_pde, :coo_len]

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_list: List[PDENode] = None) -> List[List[PDENode]]:
        if u_list is None:
            raise ValueError("Argument 'u_list' must be specified.")
        node_lists = [[] for _ in u_list]
        for (i, j, k, value) in zip(self.coo_i, self.coo_j, self.coo_k, self.coo_vals):
            if self.keep_all_coef:
                value = pde.new_coef(value)
            if j == k:
                node_lists[i].append(value * u_list[j].square)
            else:
                node_lists[i].append(pde.prod(value, u_list[j], u_list[k]))
        return node_lists

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum_list: List[LaTeXSum],
                  symbol: Union[str, List[str], None] = None,
                  c_symbol: str = "b",
                  *,
                  l: Union[int, str] = "") -> None:
        for (i, j, k, value) in zip(self.coo_i, self.coo_j, self.coo_k, self.coo_vals):
            coef_latex = f"{c_symbol}_{{{l}{i}{j}{k}}}"
            if j == k:
                term_latex = self._u_symbol(symbol, j) + r"^2"
            else:
                term_latex = self._u_symbol(symbol, j) + self._u_symbol(symbol, k)
            latex_sum_list[i].add_term_with_coef(value, coef_latex, term_latex)


class MultiComponentMixedTerm(PDETermBase):
    r"""
    Polynomial term $f(u)$ (in vector form) with degree up to two for
    multi-component PDEs:
        $$f(u)_i = \sum_ja_{ij}u_j + \sum_{j,k}b_{ijk}u_ju_k,$$
    where $0 \le i,j,k \le d_u-1$, $j \le k$. The coefficients $a,b$ are sparse
    arrays, with a total of at most $3d$ non-zero entries.
    """

    def __init__(self,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.lin_term = MultiComponentLinearTerm(
            hdf5_group["lin"], idx_pde, keep_all_coef)
        self.deg2_term = MultiComponentDegree2Term(
            hdf5_group["deg2"], idx_pde, keep_all_coef)

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_list: List[PDENode] = None) -> List[List[PDENode]]:
        node_lists = self.lin_term.nodes(pde, u_list)
        deg2_lists = self.deg2_term.nodes(pde, u_list)
        for i, node_list_i in enumerate(node_lists):
            node_list_i.extend(deg2_lists[i])
        return node_lists

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum_list: List[LaTeXSum],
                  symbol: Union[str, List[str], None] = None,
                  *,
                  l: Union[int, str] = "") -> None:
        self.lin_term.add_latex(latex_sum_list, symbol, "c", l=l)
        self.deg2_term.add_latex(latex_sum_list, symbol, "b", l=l)

r"""Load terms from python dictionary instead of HDF5 file."""
from abc import ABC, abstractmethod
from typing import Union, Dict

import numpy as np
from numpy.typing import NDArray

from src.data.multi_pde import terms
from src.data.multi_pde.terms import LaTeXSum
from ..pde_dag import PDENodesCollector, PDENode


class PDETermBase(ABC):
    r""" Abstract basic class of a PDE term. """

    @abstractmethod
    def __init__(self,
                 term: Union[Dict, NDArray[float], float],
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


class ScalarCoef(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.ScalarCoef.__doc__
    value: float

    def __init__(self,
                 term: float,
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, float):
            raise ValueError("Input 'term' of ScalarCoef should be a float scalar.")
        self.value = term

    nodes = terms.ScalarCoef.nodes
    add_latex = terms.ScalarCoef.add_latex


class LgNonNegScalarCoef(ScalarCoef):
    __doc__ = terms.LgNonNegScalarCoef.__doc__

    nodes = terms.LgNonNegScalarCoef.nodes


class CoefArray(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.CoefArray.__doc__
    value: NDArray[float]

    def __init__(self,
                 term: NDArray[float],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, np.ndarray):
            raise ValueError("Input 'term' of CoefArray should be a numpy array.")
        self.value = term

    nodes = terms.CoefArray.nodes
    add_latex = terms.CoefArray.add_latex
    _get_value = terms.CoefArray._get_value  # pylint: disable=protected-access


class FieldCoef(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.FieldCoef.__doc__

    def __init__(self,
                 term: NDArray[float],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, np.ndarray):
            raise ValueError("Input 'term' of FieldCoef should be a numpy array.")
        self.field = term

    nodes = terms.FieldCoef.nodes
    add_latex = terms.FieldCoef.add_latex


class LgNonNegFieldCoef(FieldCoef):  # pylint: disable=missing-docstring
    __doc__ = terms.LgNonNegFieldCoef.__doc__

    nodes = terms.LgNonNegFieldCoef.nodes


class ConstOrField(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.ConstOrField.__doc__
    ZERO_COEF = 0
    UNIT_COEF = 1
    SCALAR_COEF = 2
    FIELD_COEF = 3

    def __init__(self,
                 term: Dict[str, Union[int, NDArray[float]]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict):
            raise ValueError("Input 'term' of ConstOrField should be a dictionary.")
        self.coef_type = term["coef_type"]
        self.field = term["field"]

    type_char = terms.ConstOrField.type_char
    nodes = terms.ConstOrField.nodes
    add_latex = terms.ConstOrField.add_latex
    _get_value = terms.ConstOrField._get_value  # pylint: disable=protected-access
    _from_field_and_type = terms.ConstOrField._from_field_and_type  # pylint: disable=protected-access
    __add__ = terms.ConstOrField.__add__
    __radd__ = terms.ConstOrField.__radd__
    __mul__ = terms.ConstOrField.__mul__
    __rmul__ = terms.ConstOrField.__rmul__
    __sub__ = terms.ConstOrField.__sub__
    __rsub__ = terms.ConstOrField.__rsub__
    __neg__ = terms.ConstOrField.__neg__
    inverse = terms.ConstOrField.inverse
    __truediv__ = terms.ConstOrField.__truediv__
    __rtruediv__ = terms.ConstOrField.__rtruediv__


# class ReNamedConstOrField(ConstOrField):
#     __doc__ = terms.ReNamedConstOrField.__doc__

#     def __init__(self,
#                  term: Dict[str, Union[int, NDArray[float]]],
#                  keep_all_coef: bool,
#                  coef_type_name: str="coef_type",
#                  field_name: str="field") -> None:
#         self.coef_type = term[coef_type_name]
#         self.field = term[field_name]
#         super().__init__(term, keep_all_coef)


class LgNonNegConstOrField(ConstOrField):  # pylint: disable=missing-docstring
    __doc__ = terms.LgNonNegConstOrField.__doc__

    nodes = terms.LgNonNegConstOrField.nodes


class TimeIndepForce(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.TimeIndepForce.__doc__

    def __init__(self,
                 term: Dict[str, Union[int, NDArray[float]]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict):
            raise ValueError("Input 'term' of TimeIndepForce should be a dictionary.")
        field = term["field"]  # Shape: (*, n_vars)
        self.n_vars = field.shape[-1]
        self.fields = [field[..., i] for i in range(self.n_vars)]

    nodes = terms.TimeIndepForce.nodes
    add_latex = terms.TimeIndepForce.add_latex


class TimeDepForce(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.TimeDepForce.__doc__

    def __init__(self,
                 term: Dict[str, Dict[str, Union[int, NDArray[float]]]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict) or not isinstance(term["fr"], dict) or \
                not isinstance(term["ft"], dict):
            raise ValueError("Input 'term' of TimeDepForce should be a nested dictionary.")
        fr_field = term["fr"]["field"]  # Shape: (*, n_vars)
        self.n_vars = fr_field.shape[-1]
        self.fr_fields = [fr_field[..., i] for i in range(self.n_vars)]
        self.ft_field = term["ft"]["field"]

    nodes = terms.TimeDepForce.nodes
    add_latex = terms.TimeDepForce.add_latex


class StiffnessMatrix2D2Comp(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.StiffnessMatrix2D2Comp.__doc__
    ISOTROPY = 0
    SUPPORTED_TYPES = [ISOTROPY]
    n_vars: int = 2
    value_cls: type = ConstOrField

    def __init__(self,
                 term: Dict[str, Union[int, NDArray[float]]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict):
            raise ValueError("Input 'term' of StiffnessMatrix2D2Comp should be a dictionary.")
        self.values = []
        stiff_type = term.get("stiff_type", self.ISOTROPY)
        if stiff_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported 'stiff_type' {stiff_type}.")
        for key in term:
            self.values.append(self.value_cls(
                term[key], keep_all_coef))
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

    nodes = terms.StiffnessMatrix2D2Comp.nodes
    add_latex = terms.StiffnessMatrix2D2Comp.add_latex


class PolySinusNonlinTerm(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.PolySinusNonlinTerm.__doc__
    value: NDArray[float]

    def __init__(self,
                 term: NDArray[float],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, np.ndarray):
            raise ValueError("Input 'term' of PolySinusNonlinTerm should be a numpy array.")
        self.value = term

    nodes = terms.PolySinusNonlinTerm.nodes
    add_latex = terms.PolySinusNonlinTerm.add_latex


class HomSpatialOrder2Term(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.HomSpatialOrder2Term.__doc__
    NON_DIV_FORM = 0
    FACTORED_FORM = 1
    DIV_FORM = 2
    LG_COEF: bool = False

    def __init__(self,
                 term: Dict[str, Union[int, NDArray[float]]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict):
            raise ValueError("Input 'term' of HomSpatialOrder2Term should be a dictionary.")
        self.diff_type = term["diff_type"]
        self.value = term["value"]

    nodes = terms.HomSpatialOrder2Term.nodes
    add_latex = terms.HomSpatialOrder2Term.add_latex


class LgCoefHomSpatialOrder2Term(HomSpatialOrder2Term):  # pylint: disable=missing-docstring
    __doc__ = terms.LgCoefHomSpatialOrder2Term.__doc__
    LG_COEF: bool = True


class InhomSpatialOrder2Term(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.InhomSpatialOrder2Term.__doc__
    NON_DIV_FORM = 0
    FACTORED_FORM = 1
    DIV_FORM = 2
    LG_COEF: bool = False

    def __init__(self,
                 term: Dict[str, Union[int, NDArray[float]]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict):
            raise ValueError("Input 'term' of InhomSpatialOrder2Term should be a dictionary.")
        self.diff_type = term["diff_type"]
        self.field = term["field"]

    nodes = terms.InhomSpatialOrder2Term.nodes
    add_latex = terms.InhomSpatialOrder2Term.add_latex


class LgCoefInhomSpatialOrder2Term(InhomSpatialOrder2Term):  # pylint: disable=missing-docstring
    __doc__ = terms.LgCoefInhomSpatialOrder2Term.__doc__
    LG_COEF: bool = True


class MultiComponentLinearTerm(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.MultiComponentLinearTerm.__doc__

    def __init__(self,
                 term: Dict[str, Union[int, NDArray[float]]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        coo_len = term["coo_len"]
        self.coo_i = term["coo_i"][:coo_len]
        self.coo_j = term["coo_j"][:coo_len]
        self.coo_vals = term["coo_vals"][:coo_len]

    nodes = terms.MultiComponentLinearTerm.nodes
    add_latex = terms.MultiComponentLinearTerm.add_latex
    _u_symbol = terms.MultiComponentLinearTerm._u_symbol  # pylint: disable=protected-access


class MultiComponentDegree2Term(MultiComponentLinearTerm):  # pylint: disable=missing-docstring
    __doc__ = terms.MultiComponentDegree2Term.__doc__

    def __init__(self,
                 term: Dict[str, Union[int, NDArray[float]]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict):
            raise ValueError("Input 'term' of MultiComponentDegree2Term should be a dictionary.")
        coo_len = term["coo_len"]
        self.coo_k = term["coo_k"][:coo_len]

    nodes = terms.MultiComponentDegree2Term.nodes
    add_latex = terms.MultiComponentDegree2Term.add_latex


class MultiComponentMixedTerm(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = terms.MultiComponentMixedTerm.__doc__

    def __init__(self,
                 term: Dict[str, Dict[str, Union[int, NDArray[float]]]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict) or not isinstance(term["lin"], dict) or \
                not isinstance(term["deg2"], dict):
            raise ValueError("Input 'term' of MultiComponentMixedTerm "
                             "should be a nested dictionary.")
        self.lin_term = MultiComponentLinearTerm(term["lin"], keep_all_coef)
        self.deg2_term = MultiComponentDegree2Term(term["deg2"], keep_all_coef)

    nodes = terms.MultiComponentMixedTerm.nodes
    add_latex = terms.MultiComponentMixedTerm.add_latex

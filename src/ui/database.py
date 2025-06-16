r"""Database classes for coefficients and boundary conditions in the PDE."""
from typing import Dict, List
import numpy as np
from numpy.typing import NDArray
from src.ui.basic import PDEDatabase
from src.ui.utils import eval_expression_to_field, term_coef2latex, term_coef_str2latex


class PolySinusNonlinDatabase(PDEDatabase):
    r"""
    Database for the non-linear term $f(u)$ in the PDE, in the form
    $f(u) = \sum_{k=1}^3c_{0k}u^k + \sum_{j=1}^Jc_{j0}h_j(c_{j1}u+c_{j2}u^2)$,
    where $h_j\in\{\sin,\cos\}$.
    Each element of the term (a (J+1) * 4 numpy array) is stored as a separate data entry.
    """

    def __init__(self, name, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        term = kwargs.get("term", np.zeros((2, 4)))
        if not isinstance(term, np.ndarray) or term.ndim != 2 or term.shape[1] != 4:
            raise ValueError("Input 'term' should be a (J+1) * 4 numpy array.")
        self.u_symbol = kwargs.get("u_symbol", "u")
        self.field_str = kwargs.get("field_str", "f")
        self._is_initialized = False
        self.set_value("rows", term.shape[0])
        self.set_value("cols", term.shape[1])
        self.dependencies = self._register_dependencies()
        self._initialize_entries(term)
        self._is_initialized = True

    def _initialize_entries(self, term: NDArray[float]) -> None:
        """Initialize each element of the term as a separate data entry."""
        rows = self.get_value("rows")
        cols = self.get_value("cols")
        for j in range(rows):
            for k in range(cols):
                key = f"term.{j}.{k}"
                self.set_value(key, term[j, k])
        self.set_value("latex", self.latex)

    @property
    def term(self) -> NDArray[float]:
        rows = self.get_value("rows")
        cols = self.get_value("cols")
        term = np.zeros((rows, cols))
        for j in range(rows):
            for k in range(cols):
                key = f"term.{j}.{k}"
                term[j, k] = self.get_value(key)
        return term

    @property
    def latex(self) -> str:
        r"""Get the LaTeX representation of the term."""
        rows = self.get_value("rows")
        cols = self.get_value("cols")
        u_symbol = self.u_symbol

        # polynomial part
        poly_terms = []
        for k in range(cols):
            coef_value = self.get_value(f"term.0.{k}")
            if k == 0:
                term_latex = ""
            elif k == 1:
                term_latex = u_symbol
            else:
                term_latex = f"{u_symbol}^{k}"
            term_coef_latex = term_coef2latex(coef_value, term_latex)
            if term_coef_latex != "":
                poly_terms.append(term_coef_latex)
        poly_sum = " + ".join(poly_terms)

        # sinusoidal part
        sinus_terms = []
        for j in range(1, rows):
            if self.get_value(f"term.{j}.0") == 0:
                continue
            h_j = r"\sin" if self.get_value(f"term.{j}.3") > 0 else r"\cos"
            inner_poly_terms = []
            for k in range(1, 3):
                coef_value = self.get_value(f"term.{j}.{k}")
                if coef_value == 0:
                    continue
                if k == 1:
                    term_latex = f"{u_symbol}"
                else:
                    term_latex = f"{u_symbol}^{k}"
                term_coef_latex = term_coef2latex(coef_value, term_latex)
                if term_coef_latex != "":
                    inner_poly_terms.append(term_coef_latex)
            if inner_poly_terms == []:
                inner_poly_terms = ["0"]
            inner_poly_sum = " + ".join(inner_poly_terms)
            sinus_terms.append(term_coef2latex(
                self.get_value(f"term.{j}.0"), f"{h_j}({inner_poly_sum})"))
        sinus_sum = " + ".join(sinus_terms)

        if not poly_sum:
            latex_sum = sinus_sum
        elif not sinus_sum:
            latex_sum = poly_sum
        else:
            latex_sum = f"{poly_sum} + {sinus_sum}"
        # replace "+ -" with "-"
        latex_sum = latex_sum.replace("+ -", "- ")
        if not latex_sum:
            latex_sum = "0"
        return latex_sum

    def auto_update(self, key):
        """
        Automatically update the value of a data entry based on its dependencies.
        """
        if key == "latex":
            self.set_value(key, self.latex)

    def get_rep(self) -> NDArray[float]:
        """
        Get a nested dictionary or value representation of the database.
        """
        return self.term.copy()

    def _register_dependencies(self) -> Dict[str, str]:
        """
        Register the dependencies for the database.
        """
        dependencies = {}
        rows = self.get_value("rows")
        cols = self.get_value("cols")
        for j in range(rows):
            for k in range(cols):
                key = f"term.{j}.{k}"
                dependencies[key] = ["latex"]
        return dependencies


class ConstOrFieldDatabase(PDEDatabase):
    r"""
    Database for a constant or field value in the PDE. The value is represented by
    a single float, a expression, or a 1D or 2D numpy array.
    """
    # value types for user interface
    FLOAT_VALUE = 0
    STR_VALUE = 1
    ARRAY_VALUE = 2

    # coefficient types provided to the PDE solver
    ZERO_COEF = 0
    UNIT_COEF = 1
    SCALAR_COEF = 2
    FIELD_COEF = 3

    def __init__(self, name, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        self.coord_dict = kwargs.get(
            "coord_dict", {"x": np.linspace(0, 1, 129)[:-1], "y": np.linspace(0, 1, 129)[:-1]})
        self.ax_names = kwargs.get("ax_names", ["x", "y"])
        if "shape" not in kwargs:
            shape = [len(self.coord_dict[ax]) for ax in self.ax_names]
        else:
            shape = kwargs["shape"]
        self.shape = shape
        float_value = kwargs.get("float_value", 0.)
        str_value = kwargs.get("str_value", "0")
        array_value = kwargs.get("array_value", np.zeros(self.shape, dtype=float))
        is_enabled = kwargs.get("is_enabled", True)
        value_type = kwargs.get("value_type", self.FLOAT_VALUE)
        self.field_str = kwargs.get("field_str", "f")
        self._is_initialized = False
        self.dependencies = self._register_dependencies()
        self._initialize_entries(float_value, str_value, array_value, is_enabled, value_type)
        self._is_initialized = True

    def _initialize_entries(self,
                            float_value: float,
                            str_value: str,
                            array_value: NDArray[float],
                            is_enabled: bool,
                            value_type: int) -> None:
        """Initialize the value as a data entry."""
        if not isinstance(float_value, float):
            raise ValueError("Input 'float_value' should be a float.")
        if not isinstance(str_value, str):
            raise ValueError("Input 'str_value' should be a string.")
        if not isinstance(array_value, np.ndarray):
            raise ValueError("Input 'array_value' should be a numpy array.")
        if array_value.shape != tuple(self.shape):
            raise ValueError("Input 'array_value' should have shape equal to 'shape'.")
        if not isinstance(is_enabled, bool):
            raise ValueError("Input 'is_enabled' should be a boolean.")
        self.set_value("float_value", float_value)
        self.set_value("str_value", str_value)
        self.set_value("array_value", array_value)
        self.set_value("value_type", value_type)
        self.set_value("is_enabled", is_enabled)
        self.set_value("field", self.field)
        self.set_value("latex", self.latex)

    def _get_coef_type(self) -> int:
        r"""Get the coefficient type for the PDE solver."""
        value_type = self.get_value("value_type")
        if value_type == self.FLOAT_VALUE:
            float_value = self.get_value("float_value")
            if float_value == 0.:
                return self.ZERO_COEF
            if float_value == 1.:
                return self.UNIT_COEF
            return self.SCALAR_COEF
        if value_type == self.STR_VALUE:
            return self.FIELD_COEF
        if value_type == self.ARRAY_VALUE:
            return self.FIELD_COEF
        raise ValueError("Unexpected 'value_type'.")

    @property
    def field(self) -> NDArray[float]:
        r"""Get the field value."""
        value_type = self.get_value("value_type")
        if value_type == self.FLOAT_VALUE:
            float_value = self.get_value("float_value")
            return np.full(self.shape, float_value)
        if value_type == self.STR_VALUE:
            str_value = self.get_value("str_value")
            return eval_expression_to_field(str_value, self.ax_names, self.coord_dict)
        if value_type == self.ARRAY_VALUE:
            return self.get_value("array_value")
        raise ValueError("Unexpected 'value_type'.")

    @property
    def latex(self) -> str:
        r"""Get the LaTeX representation of the value."""
        value_type = self.get_value("value_type")
        if value_type == self.FLOAT_VALUE:
            float_value = self.get_value("float_value")
            if float_value == 0.:
                return "0"
            if float_value == 1.:
                return "1"
            return f"{float_value:.2f}"
        if value_type == self.STR_VALUE:
            str_value = self.get_value("str_value")
            return str_value
        if value_type == self.ARRAY_VALUE:
            var_str = ", ".join(self.ax_names)
            return rf"{self.field_str}({var_str})"
        raise ValueError("Unexpected 'value_type'.")

    def set_value(self, key, value):
        super().set_value(key, value)
        if self._is_initialized and key in ["float_value", "str_value", "array_value"]:
            if key == "float_value":
                self.set_value("value_type", self.FLOAT_VALUE)
            elif key == "str_value":
                self.set_value("value_type", self.STR_VALUE)
            elif key == "array_value":
                self.set_value("value_type", self.ARRAY_VALUE)

    def auto_update(self, key):
        """
        Automatically update the value of a data entry based on its dependencies.
        """
        if key == "latex":
            self.set_value(key, self.latex)
        elif key == "field":
            self.set_value(key, self.field)

    def get_rep(self) -> float:
        """
        Get a nested dictionary or value representation of the database.
        """
        return {"field": self.field.copy(), "coef_type": self._get_coef_type()}

    def _register_dependencies(self) -> Dict[str, List[str]]:
        """
        Register the dependencies for the database.
        """
        return {
            "float_value": ["field", "latex"],
            "str_value": ["field", "latex"],
            "array_value": ["field", "latex"],
            "value_type": ["field", "latex"],
        }


class EdgeBoundaryConditionV2Database(PDEDatabase):
    r"""
    Database for the non-periodic robin boundary condition.
    """
    ROBIN_D = 0
    ROBIN_N = 1

    def __init__(self, name, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        bc_type = kwargs.get("bc_type", self.ROBIN_D)
        self.ax_names = kwargs.get("ax_names", ["x"])
        self.coord_dict = kwargs.get("coord_dict", {"x": np.linspace(0, 1, 129)[:-1],
                                                    "y": np.linspace(0, 1, 129)[:-1]})
        if "alpha" not in kwargs:
            alpha = ConstOrFieldDatabase("alpha", ax_names=self.ax_names,
                                         field_str=r"\alpha", coord_dict=self.coord_dict)
        else:
            alpha = kwargs["alpha"]
        if "beta" not in kwargs:
            beta = ConstOrFieldDatabase("beta", ax_names=self.ax_names, field_str=r"\beta",
                                        coord_dict=self.coord_dict)
        else:
            beta = kwargs["beta"]
        self.u_symbol = kwargs.get("u_symbol", "u")
        is_enabled = kwargs.get("is_enabled", True)
        if "key_name" not in kwargs:
            raise ValueError("Input 'key_name' should be specified.")
        key_name = kwargs["key_name"]
        self._is_initialized = False
        self.dependencies = self._register_dependencies()
        self._initialize_entries(bc_type, alpha, beta, key_name, is_enabled)
        self._is_initialized = True

    def _initialize_entries(self,
                            bc_type: int,
                            alpha: ConstOrFieldDatabase,
                            beta: ConstOrFieldDatabase,
                            key_name: str,
                            is_enabled: bool,
                            ) -> None:
        """Initialize the entries."""
        if not isinstance(bc_type, int):
            raise ValueError("Input 'bc_type' should be an integer.")
        if not isinstance(alpha, ConstOrFieldDatabase):
            raise ValueError("Input 'alpha' should be a ConstOrFieldDatabase.")
        if not isinstance(beta, ConstOrFieldDatabase):
            raise ValueError("Input 'beta' should be a ConstOrFieldDatabase.")
        if not isinstance(key_name, str):
            raise ValueError("Input 'key_name' should be a string.")
        if not isinstance(is_enabled, bool):
            raise ValueError("Input 'is_enabled' should be a boolean.")
        self.set_value("bc_type", bc_type)
        self.set_subdatabase("alpha", alpha)
        self.set_subdatabase("beta", beta)
        self.set_value("key_name", key_name)
        self.set_value("is_enabled", is_enabled)
        self.set_value("latex", self.latex)

    def auto_update(self, key):
        """
        Automatically update the value of a data entry based on its dependencies.
        """
        if key == "latex":
            self.set_value(key, self.latex)
        elif key == "alpha.is_enabled":
            self.set_value(key, self.get_value("is_enabled"))
        elif key == "beta.is_enabled":
            self.set_value(key, self.get_value("is_enabled"))

    @property
    def latex(self):
        r"""Get the LaTeX representation of the boundary condition."""
        bc_type = self.get_value("bc_type")
        var_str = self.u_symbol
        alpha_latex = rf"{self.get_value('alpha.latex')}"
        beta_latex = rf"{self.get_value('beta.latex')}"

        full_latex = ""
        if bc_type == self.ROBIN_D:
            term1 = rf"{var_str}"
            full_latex += term1
            term2 = term_coef_str2latex(alpha_latex, rf"\partial_n {var_str}")
            if term2:
                if term2[0] == "-":
                    term2_sign = "-"
                    term2_body = term2[1:]
                else:
                    term2_sign = "+"
                    term2_body = term2
                full_latex += rf" {term2_sign} {term2_body}"
            term3 = term_coef_str2latex(beta_latex, "")
            if term3:
                if term3[0] == "-":
                    term3_sign = "-"
                    term3_body = term3[1:]
                else:
                    term3_sign = "+"
                    term3_body = term3
                full_latex += rf" {term3_sign} {term3_body}"
            full_latex += rf" = 0"
            return full_latex
        if bc_type == self.ROBIN_N:
            term1 = term_coef_str2latex(alpha_latex, var_str)
            term2 = rf"\partial_n {var_str}"
            terms = [term1, term2]
            terms = [term for term in terms if term != ""]
            full_latex = " + ".join(terms)
            term3 = term_coef_str2latex(beta_latex, "")
            if term3:
                if term3[0] == "-":
                    term3_sign = "-"
                    term3_body = term3[1:]
                else:
                    term3_sign = "+"
                    term3_body = term3
                full_latex += rf" {term3_sign} {term3_body}"
            full_latex += rf" = 0"
            return full_latex
        raise ValueError("Unexpected 'bc_type'.")

    @property
    def latex_simplified(self) -> str:
        r"""Get the simplified LaTeX representation of the boundary condition."""
        bc_type = self.get_value("bc_type")
        var_str = self.u_symbol
        coord_str = ", ".join(self.ax_names)
        alpha_latex = rf"\alpha({coord_str})"
        beta_latex = rf"\beta({coord_str})"
        if bc_type == self.ROBIN_D:
            return rf"{var_str} + {alpha_latex} \partial_n {var_str} + {beta_latex}= 0"
        if bc_type == self.ROBIN_N:
            return rf"{alpha_latex} {var_str} + \partial_n {var_str} + {beta_latex} = 0"
        raise ValueError("Unexpected 'bc_type'.")

    def get_rep(self) -> Dict:
        """
        Get a nested dictionary or value representation of the database.
        """
        return {
            "bc_type": self.get_value("bc_type"),
            "alpha": self.get_subdatabase("alpha").get_rep(),
            "beta": self.get_subdatabase("beta").get_rep(),
            "is_enabled": self.get_value("is_enabled"),
        }

    def _register_dependencies(self) -> Dict[str, List[str]]:
        """
        Register the dependencies for the database.
        """
        dependencies = {
            "bc_type": ["latex"],
            "alpha.latex": ["latex"],
            "beta.latex": ["latex"],
            "is_enabled": ["alpha.is_enabled", "beta.is_enabled"],
        }
        return dependencies


class FullBoundaryConditionV2Database(PDEDatabase):
    r"""
    Database for the full boundary condition of a PDE.
    """
    def __init__(self, name, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        self.edge_keynames = kwargs.get("edge_keynames", ["x_low", "x_high", "y_low", "y_high"])
        self.ax_names = kwargs.get("ax_names", ["x", "y"])
        self.u_symbol = kwargs.get("u_symbol", "u")
        self.coord_dict = kwargs.get("coord_dict", {"x": np.linspace(0, 1, 129)[:-1],
                                                    "y": np.linspace(0, 1, 129)[:-1]})
        if "edge_bc_dict" not in kwargs:
            edge_bc_dict = {}
            for key_name in self.edge_keynames:
                ax_names = [ax for ax in self.ax_names if ax != key_name[0]]
                edge_bc = EdgeBoundaryConditionV2Database(
                    f"{key_name}",
                    key_name=key_name,
                    ax_names=ax_names,
                    u_symbol=self.u_symbol,
                    coord_dict=self.coord_dict,
                )
                edge_bc_dict[key_name] = edge_bc
        else:
            edge_bc_dict = kwargs["edge_bc_dict"]
        is_enabled_x = kwargs.get("is_enabled_x", False)
        is_enabled_y = kwargs.get("is_enabled_y", False)
        self._is_initialized = False
        self.dependencies = self._register_dependencies()
        self._initialize_entries(edge_bc_dict, is_enabled_x, is_enabled_y)
        self._is_initialized = True

    def _initialize_entries(self,
                            edge_bc_dict: Dict[str, EdgeBoundaryConditionV2Database],
                            is_enabled_x: bool,
                            is_enabled_y: bool,
                            ) -> None:
        """Initialize the entries."""
        if not isinstance(edge_bc_dict, dict):
            raise ValueError("Input 'edge_bc_dict' should be a dictionary.")
        for key_name, edge_bc in edge_bc_dict.items():
            if not isinstance(edge_bc, EdgeBoundaryConditionV2Database):
                raise ValueError("Values of 'edge_bc_dict' should be EdgeBoundaryConditionV2Database.")
            self.set_subdatabase(key_name, edge_bc)
        if not isinstance(is_enabled_x, bool):
            raise ValueError("Input 'is_enabled_x' should be a boolean.")
        if not isinstance(is_enabled_y, bool):
            raise ValueError("Input 'is_enabled_y' should be a boolean.")
        self.set_value("is_enabled_x", is_enabled_x)
        self.set_value("is_enabled_y", is_enabled_y)
        self.set_value("latex", self.latex)

    def auto_update(self, key):
        """
        Automatically update the value of a data entry based on its dependencies.
        """
        enabled_keys = [f"{key_name}.is_enabled" for key_name in self.edge_keynames]
        if key == "latex":
            self.set_value(key, self.latex)
        elif key in enabled_keys:
            if key in self._register_dependencies().get("is_enabled_x", []):
                self.set_value(key, self.get_value("is_enabled_x"))
            if key in self._register_dependencies().get("is_enabled_y", []):
                self.set_value(key, self.get_value("is_enabled_y"))

    def get_rep(self) -> Dict:
        """
        Get a nested dictionary or value representation of the database.
        """
        rep_dict = {}
        for key_name in self.edge_keynames:
            ax_name = key_name[0]
            if ax_name == "x":
                is_enabled = self.get_value("is_enabled_x")
            elif ax_name == "y":
                is_enabled = self.get_value("is_enabled_y")
            if is_enabled:
                rep_dict[key_name] = self.get_subdatabase(key_name).get_rep()
        return rep_dict

    @property
    def latex(self) -> str:
        r"""Get the LaTeX representation of the full boundary condition."""
        # x-direction
        is_enabled_x = self.get_value("is_enabled_x")
        if is_enabled_x:
            x_latex_list = []
            for key_name in self.edge_keynames:
                ax_name = key_name[0]
                if ax_name == "x":
                    rest_str = key_name.split("_")[1]
                    latex_key_name = rf"{ax_name}_{{{rest_str}}}"
                    x_latex_list.append(rf"{latex_key_name}: {self.get_value(f'{key_name}.latex')}")
            x_latex = "$\n$".join(x_latex_list)
        else:
            x_latex = "x-direction: perodic"
        # y-direction
        is_enabled_y = self.get_value("is_enabled_y")
        if is_enabled_y:
            y_latex_list = []
            for key_name in self.edge_keynames:
                ax_name = key_name[0]
                if ax_name == "y":
                    rest_str = key_name.split("_")[1]
                    latex_key_name = rf"{ax_name}_{{{rest_str}}}"
                    y_latex_list.append(f"{latex_key_name}: {self.get_value(f'{key_name}.latex')}")
            y_latex = "$\n$".join(y_latex_list)
        else:
            y_latex = "y-direction: perodic"
        full_latex = "$\n$".join([x_latex, y_latex])
        return full_latex

    @property
    def latex_simplified(self) -> str:
        r"""Get the simplified LaTeX representation of the full boundary condition."""
        # x-direction
        is_enabled_x = self.get_value("is_enabled_x")
        if is_enabled_x:
            x_latex_list = []
            for key_name in self.edge_keynames:
                ax_name = key_name[0]
                if ax_name == "x":
                    rest_str = key_name.split("_")[1]
                    latex_key_name = rf"{ax_name}_{{{rest_str}}}"
                    latex_value = self.get_subdatabase(key_name).latex_simplified
                    x_latex_list.append(rf"{latex_key_name}: {latex_value}")
            x_latex = "$\n$".join(x_latex_list)
        else:
            x_latex = "x-direction: perodic"
        # y-direction
        is_enabled_y = self.get_value("is_enabled_y")
        if is_enabled_y:
            y_latex_list = []
            for key_name in self.edge_keynames:
                ax_name = key_name[0]
                if ax_name == "y":
                    rest_str = key_name.split("_")[1]
                    latex_key_name = rf"{ax_name}_{{{rest_str}}}"
                    latex_value = self.get_subdatabase(key_name).latex_simplified
                    y_latex_list.append(f"{latex_key_name}: {latex_value}")
            y_latex = "$\n$".join(y_latex_list)
        else:
            y_latex = "y-direction: perodic"
        full_latex = "$\n$".join([x_latex, y_latex])
        return full_latex

    def _register_dependencies(self) -> Dict[str, str]:
        """
        Register the dependencies for the database.
        """
        dependencies = {"is_enabled_x": [], "is_enabled_y": []}
        for key_name in self.edge_keynames:
            ax_name = key_name[0]
            if ax_name == "x":
                dependencies[f"is_enabled_x"].append(f"{key_name}.is_enabled")
            elif ax_name == "y":
                dependencies[f"is_enabled_y"].append(f"{key_name}.is_enabled")
        dependencies["is_enabled_x"].append("latex")
        dependencies["is_enabled_y"].append("latex")
        for key_name in self.edge_keynames:
            dependencies[f"{key_name}.latex"] = ["latex"]
        return dependencies

r"""Database and ui widget for diffusion-convection-reaction(dcr) equation."""
import argparse
import sys
import re
from typing import Dict, List, Optional
import numpy as np
from numpy.typing import NDArray
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget, QApplication
from PyQt5.QtGui import QFont
from src.ui.database import ConstOrFieldDatabase, FullBoundaryConditionV2Database, PolySinusNonlinDatabase
from src.ui.widgets import ConstOrFieldWidget, FullBoundaryConditionV2Widget, PolySinusNonlinWidget
from src.ui.basic import PDEDatabase, UIManager
from src.ui.elements import SliderInput, LatexText, InteractiveChart, CheckboxInput, ClickButton
from src.ui.utils import term_coef2latex, Plotter2DVideo, add_parentheses
from src.ui.pde_types import PDEformerSolverDCR
from src.utils import load_config
from src.utils.tools import sample_grf


class DCRDataBase(PDEDatabase):
    r"""
    Database for diffusion-convection-reaction equation.
    """
    IS_WAVE: bool = False
    NON_DIV_FORM = 0
    FACTORED_FORM = 1
    DIV_FORM = 2

    def __init__(self, name, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
        self.ax_names = kwargs.get("ax_names", ["x", "y"])
        self.coord_dict = kwargs.get(
            "coord_dict", {"x": np.linspace(0, 1, 129)[:-1], "y": np.linspace(0, 1, 129)[:-1]})
        self.sol_coord_dict = kwargs.get("sol_coord_dict", self.coord_dict)
        self.solver = kwargs.get("solver", None)
        self.video_dt = kwargs.get("video_dt", 0.1)
        self.coef_min = kwargs.get("coef_min", -1.0)
        self.coef_max = kwargs.get("coef_max", 1.0)
        detailed_latex_repr = kwargs.get("detailed_latex_repr", False)
        self.dynamic_update_sol = kwargs.get("dynamic_update_sol", False)

        # initialize sub-databases, do not get from kwargs
        # second-order term Lu
        lu_field = ConstOrFieldDatabase("a", ax_names=self.ax_names, field_str=r"a")
        diff_type = self.NON_DIV_FORM

        f0 = PolySinusNonlinDatabase("f0", ax_names=self.ax_names, field_str=r"f_0")
        f1 = PolySinusNonlinDatabase("f1", ax_names=self.ax_names, field_str=r"f_1")
        f2 = PolySinusNonlinDatabase("f2", ax_names=self.ax_names, field_str=r"f_2")
        s = ConstOrFieldDatabase("s", ax_names=self.ax_names, field_str=r"s")

        random_ic = sample_grf(batch_size=1, imshow=False)
        # normalize random_ic to [self.coef_min, self.coef_max]
        random_ic = (random_ic - random_ic.min()) / (random_ic.max() - random_ic.min()) * \
                    (self.coef_max - self.coef_min) + self.coef_min
        u_ic = ConstOrFieldDatabase("u_ic", ax_names=self.ax_names, field_str=r"u_0",
                                    value_type=ConstOrFieldDatabase.ARRAY_VALUE,
                                    array_value=random_ic)

        bc = FullBoundaryConditionV2Database(
            "bc",
            edge_keynames=["x_low", "x_high", "y_low", "y_high"],
            ax_names=self.ax_names,
            coord_dict=self.coord_dict,
        )

        self._is_initialized = False
        self.dependencies = self._register_dependencies()
        self._initialize_entries(lu_field, diff_type, f0, f1, f2, s, u_ic, bc, detailed_latex_repr)
        self._is_initialized = True

    def _initialize_entries(self,
                            lu_field: ConstOrFieldDatabase,
                            diff_type: int,
                            f0: PolySinusNonlinDatabase,
                            f1: PolySinusNonlinDatabase,
                            f2: PolySinusNonlinDatabase,
                            s: ConstOrFieldDatabase,
                            u_ic: ConstOrFieldDatabase,
                            bc: FullBoundaryConditionV2Database,
                            detailed_latex_repr: bool,
                            ) -> None:
        """Initialize the entries."""
        if not isinstance(detailed_latex_repr, bool):
            raise ValueError("Input 'detailed_latex_repr' should be a boolean.")
        self.set_subdatabase("Lu", lu_field)
        self.set_value("diff_type", diff_type)
        self.set_subdatabase("f0", f0)
        self.set_subdatabase("f1", f1)
        self.set_subdatabase("f2", f2)
        self.set_subdatabase("s", s)
        self.set_subdatabase("u_ic", u_ic)
        self.set_subdatabase("bc", bc)
        self.set_value("detailed_latex_repr", detailed_latex_repr)
        diff_str = self.diff_str
        if diff_str == "":
            diff_str = "0"
        self.set_value("diff_str", diff_str)
        self.set_value("latex", self.latex)
        self.set_value("video_dt", self.video_dt)
        self.set_value("show_dynamic", False)
        self.set_value("t_snapshot", 0.0)
        self.set_value("t_coord", np.array([0.0]))
        self.set_value("solution", self.solution)

    def auto_update(self, key):
        """
        Automatically update the value of a data entry based on its dependencies.
        """
        if key == "latex":
            self.set_value(key, self.latex)
        elif key == "diff_str":
            diff_str = self.diff_str
            if diff_str == "":
                diff_str = "0"
            self.set_value(key, diff_str)
        elif key == "solution":
            if not self._is_initialized:
                return
            self.set_value(key, self.solution)
        elif key == "t_coord":
            if self.get_value("show_dynamic"):
                self.set_value(key, np.linspace(0, 1, 11))
            else:
                self.set_value(key, np.array([self.get_value("t_snapshot")]))

    @property
    def diff_str(self):
        r"""Return the diffusion term in string format."""
        diff_type = self.get_value("diff_type")
        # if self.get_value("detailed_latex_repr"):
        if diff_type == self.NON_DIV_FORM:
            if self.get_value("Lu.value_type") == ConstOrFieldDatabase.FLOAT_VALUE:
                return term_coef2latex(-self.get_value("Lu.float_value"), rf"\Delta u")
            return rf"-{add_parentheses(self.get_value('Lu.latex'))}\Delta u"
        if diff_type == self.FACTORED_FORM:
            if self.get_value("Lu.value_type") == ConstOrFieldDatabase.FLOAT_VALUE:
                inner_str = term_coef2latex(np.sqrt(self.get_value("Lu.float_value")), rf"\nabla u")
                return term_coef2latex(-np.sqrt(self.get_value("Lu.float_value")), rf"\nabla\cdot({inner_str})")
            sqrt_a = rf"\sqrt{{ {self.get_value('Lu.latex')} }}"
            return rf"-{sqrt_a}\nabla\cdot({sqrt_a}\nabla u)"
        if diff_type == self.DIV_FORM:
            if self.get_value("Lu.value_type") == ConstOrFieldDatabase.FLOAT_VALUE:
                return term_coef2latex(-self.get_value("Lu.float_value"), rf"\nabla\cdot\nabla u")
            return rf"-\nabla\cdot({add_parentheses(self.get_value('Lu.latex'))}\nabla u)"
        raise ValueError("Unexpected 'diff_type'.")

    @property
    def latex(self):
        r"""Return the latex representation of the PDE."""
        if self.get_value("detailed_latex_repr"):
            diff_str = self.get_value("diff_str")
        else:
            diff_str = "Lu"

        if self.get_value("detailed_latex_repr"):
            f0_str = self.get_value("f0.latex")
            f1_str = self.get_value("f1.latex")
            f2_str = self.get_value("f2.latex")
            s_str = self.get_value("s.latex")
            ic_str = rf"u(x, y, 0) = {self.get_value('u_ic.latex')}"
            bc_str = self.get_value("bc.latex")
        else:
            f0_str = r"f_0(u)"
            f1_str = r"f_1(u)"
            f2_str = r"f_2(u)"
            s_str = r"s(x, y)"
            ic_str = r"u(x, y, 0) = u_0(x, y)"
            bc_str = self.get_subdatabase("bc").latex_simplified

        # pde_str = rf"u_t {diff_sign} {diff_str} + {f0_str} + ({f1_str})_x + ({f2_str})_y + {s_str} = 0"
        pde_str = rf"u_t"
        if diff_str and diff_str != "0":
            if diff_str[0] == "-":
                diff_sign = "-"
                diff_str = diff_str[1:]
            else:
                diff_sign = "+"
            pde_str += rf" {diff_sign} {diff_str}"
        if f0_str and f0_str != "0":
            pde_str += rf" + {add_parentheses(f0_str)}"
        if f1_str and f1_str != "0":
            pde_str += rf" + {add_parentheses(f1_str)}_x"
        if f2_str and f2_str != "0":
            pde_str += rf" + {add_parentheses(f2_str)}_y"
        if s_str and s_str != "0":
            if s_str[0] == "-":
                s_sign = "-"
                s_str = s_str[1:]
            else:
                s_sign = "+"
            pde_str += rf" {s_sign} {s_str}"
        pde_str += " = 0"
        # separate pde_str into multiple lines
        max_summands_per_line = 6
        pde_body = pde_str.rstrip(" = 0")
        eq_str = "= 0"

        # Split the pde_body into terms, including the signs
        terms = re.split(r'(\s[+-]\s)', pde_body)

        # Assemble summands with their signs
        summands = []
        i = 0
        if terms[0].strip() not in ('+', '-'):
            # First term has no sign
            summands.append(terms[0].strip())
            i = 1

        while i < len(terms):
            sign = terms[i].strip()
            term = terms[i + 1].strip()
            summands.append(f"{sign} {term}")
            i += 2

        # Group summands into lines with at most 6 summands per line
        max_summands_per_line = 6
        lines = []
        for i in range(0, len(summands), max_summands_per_line):
            line_summands = summands[i:i + max_summands_per_line]
            line_str = ' '.join(line_summands)
            lines.append(line_str)

        # Add '= 0' to the last line
        lines[-1] += f" {eq_str}"

        # Join the lines with LaTeX line breaks
        pde_str = "$\n$".join(lines)
        full_str = "$\n$".join([pde_str, ic_str, bc_str])

        return full_str

    def _register_dependencies(self):
        """
        Register the dependencies for the database.
        """
        # do not update the solution automatically
        dependencies = {
            "Lu.latex": ["diff_str"],
            "diff_type": ["diff_str"],
            "diff_str": ["latex"],
            "f0.latex": ["latex"],
            "f1.latex": ["latex"],
            "f2.latex": ["latex"],
            "s.latex": ["latex"],
            "u_ic.latex": ["latex"],
            "bc.latex": ["latex"],
            "detailed_latex_repr": ["latex"],
            "show_dynamic": ["t_coord", "solution"],
            "t_snapshot": ["t_coord", "solution"],
        }
        if self.dynamic_update_sol:
            dependencies["latex"] = ["solution"]
        return dependencies

    def get_rep(self):
        """
        Get a nested dictionary or value representation of the database.
        """
        term_dict = {}
        lu_dict = {}
        lu_dict["diff_type"] = self.get_value("diff_type")
        if self.get_value("Lu.value_type") == ConstOrFieldDatabase.FLOAT_VALUE:
            lu_dict["value"] = self.get_value("Lu.float_value")
        else:
            lu_dict["field"] = self.get_value("Lu.field")

        term_dict["Lu"] = lu_dict
        if self.get_value("bc.is_enabled_x") or self.get_value("bc.is_enabled_y"):
            term_dict["bc"] = self.get_subdatabase("bc").get_rep()
        term_dict["f0"] = self.get_subdatabase("f0").get_rep()
        term_dict["f1"] = self.get_subdatabase("f1").get_rep()
        term_dict["f2"] = self.get_subdatabase("f2").get_rep()
        term_dict["s"] = self.get_subdatabase("s").get_rep()
        term_dict["u_ic"] = self.get_value("u_ic.field")

        arg_dict = {}
        arg_dict["t_coord"] = self.get_value("t_coord")
        for ax in self.ax_names:
            arg_dict[f"{ax}_coord"] = self.coord_dict[ax]
        enabled = np.array([self.get_value("bc.is_enabled_x"), self.get_value("bc.is_enabled_y")])
        periodic = [~enabled[0], ~enabled[1]]
        arg_dict["periodic"] = periodic
        if self.get_value("Lu.value_type") == ConstOrFieldDatabase.FLOAT_VALUE:
            arg_dict["inhom_diff_u"] = False
        else:
            arg_dict["inhom_diff_u"] = True

        return {"term_dict": term_dict, "arg_dict": arg_dict}

    @property
    def solution(self):
        r"""Return the solution of the PDE."""
        t_coord = self.get_value("t_coord")
        # sol_shape = (len(t_coord), self.coord_dict["x"].shape[0], self.coord_dict["y"].shape[0])
        sol_shape = (len(t_coord), len(self.sol_coord_dict["x"]), len(self.sol_coord_dict["y"]))
        if self.solver is not None:
            rep_dict = self.get_rep()
            term_dict = rep_dict["term_dict"]
            arg_dict = rep_dict["arg_dict"]
            for ax in self.ax_names:
                arg_dict[f"{ax}_coord"] = self.sol_coord_dict[ax]
            sol = self.solver.solve(term_dict, arg_dict)  # shape: (n_t, n_x, n_y, n_z=1, n_vars=1)
            sol = sol.reshape(sol.shape[0], sol.shape[1], -1)  # shape: (n_t, n_x, n_y)
            if sol.shape != sol_shape:
                raise ValueError(f"Unexpected solution shape: {sol.shape}.")
            return sol
        # random generate a solution for testing
        sol = np.random.rand(*sol_shape)
        return sol


class DCRWidget(QWidget):
    """
    Layout for the diffusion-convection-reaction equation.
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 sub_db_key: str,
                 ax_names: List[str],
                 coord_dict: Dict[str, NDArray[float]],
                 sol_coord_dict: Optional[Dict[str, NDArray[float]]] = None,
                 coef_min: float = -1.0,
                 coef_max: float = 1.0) -> None:
        super().__init__()
        self.name = name
        self.ui_manager = ui_manager
        self.sub_db_key = sub_db_key
        self.ax_names = ax_names
        self.coord_dict = coord_dict
        self.sol_coord_dict = sol_coord_dict if sol_coord_dict is not None else coord_dict
        self.coef_min = coef_min
        self.coef_max = coef_max

        # Create the main horizontal layout
        main_layout = QHBoxLayout()

        # Checkbox for detailed latex representation
        detailed_latex_repr = CheckboxInput(
            name=f"{self.name}.detailed_latex_repr",
            ui_manager=self.ui_manager,
            depends_on={"bool": f"{self.sub_db_key}.detailed_latex_repr"},
            updates={"bool": f"{self.sub_db_key}.detailed_latex_repr"},
            label="Show detail",
            data_identifier="bool",
        )

        # LatexText for the PDE
        pde_latex = LatexText(
            name=f"{self.name}.pde",
            ui_manager=self.ui_manager,
            depends_on={"str": f"{self.sub_db_key}.latex"},
            updates={},
            data_identifier="str",
            default_text="",
            prefix="",
        )

        # Left side: QTabWidget with two tabs (Settings and Boundary Conditions)
        self.tab_widget = QTabWidget()

        # Tab for Settings (merged first four tabs)
        settings_tab = self.create_settings_tab()
        self.tab_widget.addTab(settings_tab, "Settings")

        # Tab for Boundary Conditions
        bc_tab = self.create_boundary_conditions_tab()
        self.tab_widget.addTab(bc_tab, "Boundary Conditions")

        # Left side layout containing the tab widget
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.tab_widget)

        # Right side: Solution Display
        solution_widget = self.create_solution_widget()

        left_container = QWidget()
        left_container.setLayout(left_layout)
        left_container.setFixedWidth(1550)
        right_layout = QVBoxLayout()
        right_layout.addWidget(detailed_latex_repr)
        right_layout.addWidget(pde_latex)
        right_layout.addWidget(solution_widget)
        right_container = QWidget()
        right_container.setLayout(right_layout)
        # splitter.setSizes([300, 400])  # Initial sizes for left and right panes

        # main layout
        main_layout.addWidget(left_container)
        main_layout.addWidget(right_container)
        main_layout.addStretch()
        self.setLayout(main_layout)

    def create_nonlinear_group_box(self) -> QGroupBox:
        """
        Creates a group box for a nonlinear function.
        """
        nonlinear_group_box = QGroupBox("Nonlinear Functions")
        nonlinear_group_box.setFont(QFont("Arial", 12))
        nonlinear_layout = QVBoxLayout()

        # PolySinusNonlinWidget for f0
        f0_widget = PolySinusNonlinWidget(
            name=f"{self.name}.f0",
            ui_manager=self.ui_manager,
            sub_db_key=f"{self.sub_db_key}.f0",
            latex_prefix=r"f_0(u) = ",
        )
        f0_group = QGroupBox("f₀(u)")
        f0_group.setFont(QFont("Arial", 12))
        f0_group.setLayout(f0_widget.layout())
        nonlinear_layout.addWidget(f0_group)

        # PolySinusNonlinWidget for f1
        f1_widget = PolySinusNonlinWidget(
            name=f"{self.name}.f1",
            ui_manager=self.ui_manager,
            sub_db_key=f"{self.sub_db_key}.f1",
            latex_prefix=r"f_1(u) = ",
        )
        f1_group = QGroupBox("f₁(u)")
        f1_group.setFont(QFont("Arial", 12))
        f1_group.setLayout(f1_widget.layout())
        nonlinear_layout.addWidget(f1_group)

        # PolySinusNonlinWidget for f2
        f2_widget = PolySinusNonlinWidget(
            name=f"{self.name}.f2",
            ui_manager=self.ui_manager,
            sub_db_key=f"{self.sub_db_key}.f2",
            latex_prefix=r"f_2(u) = ",
        )
        f2_group = QGroupBox("f₂(u)")
        f2_group.setFont(QFont("Arial", 12))
        f2_group.setLayout(f2_widget.layout())
        nonlinear_layout.addWidget(f2_group)

        nonlinear_group_box.setLayout(nonlinear_layout)
        return nonlinear_group_box


    def create_settings_tab(self) -> QWidget:
        """
        Creates the Settings tab containing group boxes for various settings.
        """
        settings_layout = QHBoxLayout()

        # Diffusion Coefficient Group
        diffusion_group_box = QGroupBox("Diffusion Coefficient")
        diffusion_group_box.setFont(QFont("Arial", 12))
        diffusion_layout = QVBoxLayout()

        # LatexText for the diffusion term
        diffusion_latex = LatexText(
            name=f"{self.name}.Lu_latex",
            ui_manager=self.ui_manager,
            depends_on={"str": f"{self.sub_db_key}.diff_str"},
            updates={},
            data_identifier="str",
            default_text="",
            prefix="Lu = ",
        )
        diffusion_layout.addWidget(diffusion_latex)

        # SliderInput for diffusion term type
        diff_type_input = SliderInput(
            name=f"{self.name}.diff_type",
            ui_manager=self.ui_manager,
            depends_on={"data": f"{self.sub_db_key}.diff_type"},
            updates={"data": f"{self.sub_db_key}.diff_type"},
            min_value=0,
            max_value=2,
            tick_interval=1,
            slider_label="Diffusion Type",
            min_label="Non-divergence",
            mid_label="Factored",
            max_label="Divergence",
            show_value_box=False,
            real_min_value=0,
            real_max_value=2,
            conversion="linear",
        )
        diffusion_layout.addWidget(diff_type_input)

        # ConstOrFieldWidget for Lu (Diffusion Coefficient)
        lu_widget = ConstOrFieldWidget(
            name=f"{self.name}.Lu",
            ui_manager=self.ui_manager,
            sub_db_key=f"{self.sub_db_key}.Lu",
            ax_names=self.ax_names,
            coord_dict=self.coord_dict,
            coef_min=max(0., self.coef_min),
            coef_max=self.coef_max,
            field_str="a",
            conversion="linear",
        )
        diffusion_layout.addWidget(lu_widget)
        diffusion_group_box.setLayout(diffusion_layout)

        nonlinear_group_box = self.create_nonlinear_group_box()

        # Source Term Group
        source_group_box = QGroupBox("s(x, y)")
        source_group_box.setFont(QFont("Arial", 12))
        source_layout = QVBoxLayout()

        s_widget = ConstOrFieldWidget(
            name=f"{self.name}.s",
            ui_manager=self.ui_manager,
            sub_db_key=f"{self.sub_db_key}.s",
            ax_names=self.ax_names,
            coord_dict=self.coord_dict,
            coef_min=self.coef_min,
            coef_max=self.coef_max,
            field_str="s",
        )
        source_layout.addWidget(s_widget)
        source_group_box.setLayout(source_layout)

        # Initial Condition Group
        ic_group_box = QGroupBox("Initial Condition")
        ic_group_box.setFont(QFont("Arial", 12))
        ic_layout = QVBoxLayout()

        u_ic_widget = ConstOrFieldWidget(
            name=f"{self.name}.u_ic",
            ui_manager=self.ui_manager,
            sub_db_key=f"{self.sub_db_key}.u_ic",
            ax_names=self.ax_names,
            coord_dict=self.coord_dict,
            coef_min=self.coef_min,
            coef_max=self.coef_max,
            field_str="u₀",
        )
        ic_layout.addWidget(u_ic_widget)
        ic_group_box.setLayout(ic_layout)

        # Add stretch to push all widgets to the top
        settings_layout.addWidget(nonlinear_group_box)
        right_layout = QVBoxLayout()
        right_layout.addWidget(diffusion_group_box)
        right_layout.addWidget(source_group_box)
        right_layout.addWidget(ic_group_box)
        settings_layout.addLayout(right_layout)
        # settings_layout.addWidget(ic_group_box)
        settings_layout.addStretch()

        settings_tab = QWidget()
        settings_tab.setLayout(settings_layout)
        return settings_tab

    def create_boundary_conditions_tab(self) -> QWidget:
        """
        Creates the Boundary Conditions tab.
        """
        bc_layout = QVBoxLayout()

        bc_widget = FullBoundaryConditionV2Widget(
            name=f"{self.name}.bc",
            ui_manager=self.ui_manager,
            sub_db_key=f"{self.sub_db_key}.bc",
            edge_keynames=["x_low", "x_high", "y_low", "y_high"],
            ax_names=self.ax_names,
            coord_dict=self.coord_dict,
            u_symbol="u",
            coef_min=self.coef_min,
            coef_max=self.coef_max,
        )
        bc_layout.addWidget(bc_widget)

        # Add stretch to push content to the top
        bc_layout.addStretch()

        bc_tab = QWidget()
        bc_tab.setLayout(bc_layout)
        return bc_tab

    def create_solution_widget(self) -> QWidget:
        """
        Creates the widget to display the solution.
        """
        # slider for controlling video_dt
        video_dt_slider = SliderInput(
            name=f"{self.name}.video_dt",
            ui_manager=self.ui_manager,
            depends_on={"data": f"{self.sub_db_key}.video_dt"},
            updates={"data": f"{self.sub_db_key}.video_dt"},
            min_value=0,
            max_value=100,
            tick_interval=10,
            slider_label="Time between frames (s)",
            show_value_box=True,
            real_min_value=0.01,
            real_max_value=1.0,
            conversion="log",
        )

        # Button for updating the solution
        update_button = ClickButton(
            name=f"{self.name}.update_solution",
            ui_manager=self.ui_manager,
            depends_on={},
            updates={"auto_update": f"{self.sub_db_key}.solution"},
            label="Update Solution",
        )

        # Checkbox for showing dynamic solution
        show_dynamic_checkbox = CheckboxInput(
            name=f"{self.name}.show_dynamic",
            ui_manager=self.ui_manager,
            depends_on={"bool": f"{self.sub_db_key}.show_dynamic"},
            updates={"bool": f"{self.sub_db_key}.show_dynamic"},
            label="Show dynamic solution",
            data_identifier="bool",
        )


        # Slider for selecting the time step if not showing dynamic solution
        t_slider = SliderInput(
            name=f"{self.name}.t_slider",
            ui_manager=self.ui_manager,
            depends_on={"data": f"{self.sub_db_key}.t_snapshot", "disabled": f"{self.sub_db_key}.show_dynamic"},
            updates={"data": f"{self.sub_db_key}.t_snapshot"},
            min_value=0,
            max_value=100,
            tick_interval=10,
            slider_label="Time",
            show_value_box=True,
            real_min_value=0.,
            real_max_value=1.,
            conversion="linear",
        )

        sol_coord_dict = self.sol_coord_dict
        sol_coord_dict["t"] = self.ui_manager.get_db_value(f"{self.sub_db_key}.t_coord")

        video_plotter = Plotter2DVideo(
            coord_dict=sol_coord_dict,
            title="Solution",
            cmap="jet",
            show_colorbar=True,
            vmin=None,
            vmax=None,
            title_fontsize=16,
            video_dt=100,
        )

        solution_widget = InteractiveChart(
            name=f"{self.name}.solution",
            ui_manager=self.ui_manager,
            depends_on={f"data": f"{self.sub_db_key}.solution",
                        "video_dt": f"{self.sub_db_key}.video_dt",
                        "t_coord": f"{self.sub_db_key}.t_coord"},
            updates={},
            plot_func=video_plotter,
            data_identifier="data",
            signal=update_button.updated,  # update the plot when the button is clicked
        )
        solution_layout = QVBoxLayout()
        slider_and_button_layout = QHBoxLayout()
        slider_and_button_layout.addWidget(video_dt_slider)
        slider_and_button_layout.addWidget(update_button)
        t_controls_layout = QHBoxLayout()
        t_controls_layout.addWidget(show_dynamic_checkbox)
        t_controls_layout.addWidget(t_slider)
        solution_layout.addLayout(slider_and_button_layout)
        solution_layout.addLayout(t_controls_layout)
        solution_layout.addWidget(solution_widget)
        full_solution_widget = QWidget()
        full_solution_widget.setLayout(solution_layout)
        return full_solution_widget

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UI for Diffusion-Convection-Reaction Equation")
    parser.add_argument("--config_file_path", "-c", type=str, default="configs/config_base2.yaml")
    parser.add_argument("--dynamic_update_sol", "-d", action="store_true")
    parser.add_argument("--nx_sol", "-nx", type=int, default=32)
    parser.add_argument("--ny_sol", "-ny", type=int, default=32)
    input_args = parser.parse_args()
    config_file_path = input_args.config_file_path
    config = load_config(config_file_path)

    solver = PDEformerSolverDCR(config)

    app = QApplication(sys.argv)
    coord_dict_outer = {"x": np.linspace(0, 1, 129)[:-1], "y": np.linspace(0, 1, 129)[:-1]}
    sol_coord_dict_outer = {"x": np.linspace(0, 1, input_args.nx_sol + 1)[:-1],
                            "y": np.linspace(0, 1, input_args.ny_sol + 1)[:-1]}
    ax_names_outer = ["x", "y"]
    root_db = PDEDatabase(name="root")
    coef_min = -1.0
    coef_max = 1.0
    db = DCRDataBase(
        name="dcr",
        ax_names=ax_names_outer,
        coord_dict=coord_dict_outer,
        sol_coord_dict=sol_coord_dict_outer,
        solver=solver,
        detailed_latex_repr=True,
        dynamic_update_sol=input_args.dynamic_update_sol,
        coef_min=coef_min,
        coef_max=coef_max,
    )
    root_db.set_subdatabase("dcr", db)
    main_ui_manager = UIManager(root_db)
    dcr_widget = DCRWidget(
        name="dcr",
        ui_manager=main_ui_manager,
        sub_db_key="dcr",
        ax_names=ax_names_outer,
        coord_dict=coord_dict_outer,
        sol_coord_dict=sol_coord_dict_outer,
        coef_min=coef_min,
        coef_max=coef_max,
    )
    main_ui_manager.update_all_dependencies()
    dcr_widget.setWindowTitle("Diffusion-Convection-Reaction Equation")
    dcr_widget.show()
    sys.exit(app.exec_())

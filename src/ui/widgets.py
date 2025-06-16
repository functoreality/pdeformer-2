r"""UI widgets for coefficients and boundary conditions in the PDE."""
from typing import Optional, Dict, List
import numpy as np
from numpy.typing import NDArray
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox
from PyQt5.QtGui import QFont
from src.ui.basic import UIManager
from src.ui.utils import Plotter1D, Plotter2D
from src.ui.elements import SliderInput, LatexText, InteractiveChart, InteractiveTextInput, FileLoadButton, CheckboxInput, ValueGeneratorButton
from src.utils.tools import sample_grf


class PolySinusNonlinWidget(QWidget):
    r"""
    Layout for the non-linear term $f(u)$ in the PDE, in the form
    $f(u) = \sum_{k=1}^3c_{0k}u^k + \sum_{j=1}^Jc_{j0}h_j(c_{j1}u+c_{j2}u^2)$,
    where $h_j\in\{\sin,\cos\}$.
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 sub_db_key: str,
                 latex_prefix: str = r"f(u) = "
                 ) -> None:
        super().__init__()
        self.name = name
        self.ui_manager = ui_manager
        self.sub_db_key = sub_db_key
        self.latex_prefix = latex_prefix

        if self.ui_manager.get_db_value(f"{sub_db_key}.cols") != 4:
            raise ValueError("The number of columns in the PolySinusNonlin term should be 4.")

        layout = QVBoxLayout()
        # Text box for the LaTeX representation of the term
        all_depends_on = {f"term.{j}.{k}": f"{sub_db_key}.term.{j}.{k}"
                          for j in range(4) for k in range(4)}
        all_depends_on["str"] = f"{sub_db_key}.latex"
        self.latex_text = LatexText(
            name=f"{name}.latex",
            ui_manager=ui_manager,
            depends_on=all_depends_on,
            updates={},
            data_identifier="str",
            default_text="0",
            prefix=self.latex_prefix,
        )
        layout.addWidget(self.latex_text)

        # Slider inputs for the coefficients
        slider_layout = QHBoxLayout()
        self.slider_groups = [[] for _ in range(4)]
        # polynomial part
        poly_group_box = QGroupBox("c\u2081u + c\u2082u\u00b2 + c\u2083u\u00b3")
        poly_group_box.setFont(QFont("Arial", 12))
        slider_sub_layout = QVBoxLayout()
        unicode_map = {0: "c\u2080", 1: "c\u2081", 2: "c\u2082", 3: "c\u2083"}
        for k in range(1, self.ui_manager.get_db_value(f"{sub_db_key}.cols")):
            slider_label = f"{unicode_map[k]}"
            slider = SliderInput(
                name=f"{name}.term.0.{k}",
                ui_manager=ui_manager,
                depends_on={"data": f"{sub_db_key}.term.0.{k}"},
                updates={"data": f"{sub_db_key}.term.0.{k}"},
                min_value=-100,
                max_value=100,
                tick_interval=10,
                slider_label=slider_label,
                real_min_value=-1,
                real_max_value=1,
                conversion="linear",
            )
            slider_sub_layout.addWidget(slider)
            self.slider_groups[0].append(slider)
        poly_group_box.setLayout(slider_sub_layout)
        slider_layout.addWidget(poly_group_box)
        # sinusoidal part
        for j in range(1, self.ui_manager.get_db_value(f"{sub_db_key}.rows")):
            sinus_group_box = QGroupBox(f"c\u2080h(c\u2081u + c\u2082u\u00b2)")
            sinus_group_box.setFont(QFont("Arial", 12))
            slider_sub_layout = QVBoxLayout()
            for k in range(3):
                slider_label = unicode_map[k]
                slider = SliderInput(
                    name=f"{name}.term.{j}.{k}",
                    ui_manager=ui_manager,
                    depends_on={"data": f"{sub_db_key}.term.{j}.{k}"},
                    updates={"data": f"{sub_db_key}.term.{j}.{k}"},
                    min_value=-100,
                    max_value=100,
                    tick_interval=10,
                    slider_label=slider_label,
                    real_min_value=-1,
                    real_max_value=1,
                    conversion="linear",
                )
                slider_sub_layout.addWidget(slider)
                self.slider_groups[k].append(slider)
            # The last slider is for sin/cos
            slider = SliderInput(
                name=f"{name}.term.{j}.3",
                ui_manager=ui_manager,
                depends_on={"data": f"{sub_db_key}.term.{j}.3"},
                updates={"data": f"{sub_db_key}.term.{j}.3"},
                min_value=0,
                max_value=1,
                min_label=r"cos",
                max_label=r"sin",
                show_value_box=False,
                tick_interval=1,
                slider_label="h",
                real_min_value=-1,
                real_max_value=1,
                conversion="linear",
            )
            slider_sub_layout.addWidget(slider)
            self.slider_groups[3].append(slider)
            sinus_group_box.setLayout(slider_sub_layout)
            slider_layout.addWidget(sinus_group_box)
        layout.addLayout(slider_layout)

        self.setLayout(layout)


class ConstOrFieldWidget(QWidget):
    r"""
    Layout for a constant or field value in the PDE.
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 sub_db_key: str,
                 ax_names: Optional[List[str]] = None,
                 coord_dict: Optional[Dict[str, NDArray[float]]] = None,
                 coef_min: float = -1.,
                 coef_max: float = 1.,
                 field_str: str = "f",
                 conversion: str = "linear",
                 ) -> None:
        super().__init__()
        self.name = name
        self.ui_manager = ui_manager
        self.sub_db_key = sub_db_key
        if ax_names is None:
            ax_names = ["x", "y"]
        if coord_dict is None:
            coord_dict = {"x": np.linspace(0, 1, 129)[:-1], "y": np.linspace(0, 1, 129)[:-1]}
        coord_dict = {ax: coord_dict[ax] for ax in ax_names}
        self.ax_names = ax_names
        self.coord_dict = coord_dict
        self.func_str = rf"{field_str}({', '.join(ax_names)})"

        # Interactive text box for the value
        prefix = f"{self.func_str} = "
        self.text_box = InteractiveTextInput(
            name=f"{name}.str_value",
            ui_manager=ui_manager,
            depends_on={f"str": f"{sub_db_key}.str_value", "enabled": f"{sub_db_key}.is_enabled"},
            updates={f"str": f"{sub_db_key}.str_value"},
            data_identifier="str",
            default_text="0",
            prefix=prefix,
        )

        # Slider input for the float value
        self.slider = SliderInput(
            name=f"{name}.float_value",
            ui_manager=ui_manager,
            depends_on={"data": f"{sub_db_key}.float_value", "enabled": f"{sub_db_key}.is_enabled"},
            updates={"data": f"{sub_db_key}.float_value"},
            min_value=-100,
            max_value=100,
            tick_interval=10,
            slider_label="Value",
            real_min_value=coef_min,
            real_max_value=coef_max,
            conversion=conversion,
        )

        # Field load button for the array value
        self.load_button = FileLoadButton(
            name=f"{name}.array_value",
            ui_manager=ui_manager,
            depends_on={f"npy": f"{sub_db_key}.array_value", "enabled": f"{sub_db_key}.is_enabled"},
            updates={f"npy": f"{sub_db_key}.array_value"},
            label="load npy",
            file_type="npy",
        )

        # Interactive chart for the field value
        if len(coord_dict) == 1:
            plot_func = Plotter1D(
                coord_dict=coord_dict,
                ax_names=ax_names,
                title=rf"${self.func_str}$",
                vmin=coef_min,
                vmax=coef_max,
            )
        elif len(coord_dict) == 2:
            plot_func = Plotter2D(
                coord_dict=coord_dict,
                ax_names=ax_names,
                title=rf"${self.func_str}$",
                cmap="jet",
                show_colorbar=True,
                vmin=coef_min,
                vmax=coef_max,
            )
        else:
            raise ValueError("Only 1D and 2D fields are supported in 'ConstOrFieldWidget'.")

        self.chart = InteractiveChart(
            name=f"{name}.chart",
            ui_manager=ui_manager,
            depends_on={f"data": f"{sub_db_key}.field"},
            updates={},
            plot_func=plot_func,
        )

        if len(coord_dict) == 2:
            # random field button
            def generator():
                field = sample_grf(batch_size=1, imshow=False)
                # linearly scale the field to the range [coef_min, coef_max]
                field_min = np.min(field)
                field_max = np.max(field)
                field = (field - field_min) / (field_max - field_min) * (coef_max - coef_min) + coef_min
                return field

            self.random_field_button = ValueGeneratorButton(
                name=f"{name}.random_gen",
                ui_manager=ui_manager,
                depends_on={"enabled": f"{sub_db_key}.is_enabled"},
                updates={"data": f"{sub_db_key}.array_value"},
                label="random field",
                generator=generator,
            )

        if len(coord_dict) == 1:
            layout = QVBoxLayout()
            text_box_and_button_layout = QHBoxLayout()
            text_box_and_button_layout.addWidget(self.text_box)
            text_box_and_button_layout.addWidget(self.load_button)
            layout.addWidget(self.slider)
            layout.addLayout(text_box_and_button_layout)
            layout.addWidget(self.chart)
        elif len(coord_dict) == 2:
            layout = QVBoxLayout()
            text_box_and_button_layout = QVBoxLayout()
            text_box_and_button_layout.addWidget(self.text_box)
            text_box_and_button_layout.addWidget(self.random_field_button)
            text_box_and_button_layout.addWidget(self.load_button)
            text_box_button_and_figure_layout = QHBoxLayout()
            text_box_button_and_figure_layout.addLayout(text_box_and_button_layout)
            text_box_button_and_figure_layout.addWidget(self.chart)
            layout.addWidget(self.slider)
            layout.addLayout(text_box_button_and_figure_layout)
        else:
            raise ValueError("Only 1D and 2D fields are supported in 'ConstOrFieldWidget'.")

        self.setLayout(layout)


class EdgeBoundaryConditionV2Widget(QWidget):
    r"""
    Layout for the non-periodic robin boundary condition for a single edge.
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 sub_db_key: str,
                 key_name: str,
                 coord_dict: Dict[str, NDArray[float]],
                 ax_names: List[str],
                 coef_min: float = -1.,
                 coef_max: float = 1.,
                 ) -> None:
        super().__init__()
        self.name = name
        self.ui_manager = ui_manager
        self.sub_db_key = sub_db_key
        self.key_name = key_name
        self.coord_dict = coord_dict
        self.ax_names = ax_names

        # SliderInput for the BC type
        bc_type = SliderInput(
            name=f"{name}.bc_type",
            ui_manager=ui_manager,
            depends_on={"data": f"{sub_db_key}.bc_type", "enabled": f"{sub_db_key}.is_enabled"},
            updates={"data": f"{sub_db_key}.bc_type"},
            min_value=0,
            max_value=1,
            min_label="D",
            max_label="N",
            show_value_box=False,
            tick_interval=1,
            slider_label="BC Type",
            real_min_value=0,
            real_max_value=1,
            conversion="linear",
        )

        # Sub-layout for the alpha coefficient
        alpha_layout = ConstOrFieldWidget(
            name=f"{name}.alpha",
            ui_manager=ui_manager,
            sub_db_key=f"{sub_db_key}.alpha",
            ax_names=ax_names,
            coord_dict=coord_dict,
            coef_min=coef_min,
            coef_max=coef_max,
            field_str=r"\alpha",
        )

        # Sub-layout for the beta coefficient
        beta_layout = ConstOrFieldWidget(
            name=f"{name}.beta",
            ui_manager=ui_manager,
            sub_db_key=f"{sub_db_key}.beta",
            ax_names=ax_names,
            coord_dict=coord_dict,
            coef_min=coef_min,
            coef_max=coef_max,
            field_str=r"\beta",
        )

        # Text box for the LaTeX representation of the BC
        latex_text = LatexText(
            name=f"{name}.latex",
            ui_manager=ui_manager,
            depends_on={f"str": f"{sub_db_key}.latex"},
            updates={},
            data_identifier="str",
            default_text="",
            prefix="BC: ",
        )

        layout = QVBoxLayout()
        latex_enabled_layout = QHBoxLayout()
        latex_enabled_layout.addWidget(latex_text)
        # latex_enabled_layout.addWidget(is_enabled)
        layout.addLayout(latex_enabled_layout)
        layout.addWidget(bc_type)
        alpha_group_box = QGroupBox("Alpha")
        alpha_group_box.setFont(QFont("Arial", 12))
        alpha_group_box.setLayout(alpha_layout.layout())
        beta_group_box = QGroupBox("Beta")
        beta_group_box.setFont(QFont("Arial", 12))
        beta_group_box.setLayout(beta_layout.layout())
        alpha_beta_layout = QHBoxLayout()
        alpha_beta_layout.addWidget(alpha_group_box)
        alpha_beta_layout.addWidget(beta_group_box)
        layout.addLayout(alpha_beta_layout)
        self.setLayout(layout)


class FullBoundaryConditionV2Widget(QWidget):
    r"""
    Layout for the full boundary condition of a PDE.
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 sub_db_key: str,
                 edge_keynames: List[str],
                 ax_names: List[str],
                 coord_dict: Dict[str, NDArray[float]],
                 u_symbol: str = "u",
                 coef_min: float = -1.,
                 coef_max: float = 1.,
                 ) -> None:
        super().__init__()
        self.name = name
        self.ui_manager = ui_manager
        self.sub_db_key = sub_db_key
        self.edge_keynames = edge_keynames
        self.ax_names = ax_names
        self.coord_dict = coord_dict
        self.u_symbol = u_symbol

        # Toggle button for enabling/disabling the BC in x-direction
        is_enabled_x = CheckboxInput(
            name=f"{name}.is_enabled_x",
            ui_manager=ui_manager,
            depends_on={f"bool": f"{sub_db_key}.is_enabled_x"},
            updates={f"bool": f"{sub_db_key}.is_enabled_x"},
            data_identifier="bool",
            label="Enabled in x-direction",
        )

        # Toggle button for enabling/disabling the BC in y-direction
        is_enabled_y = CheckboxInput(
            name=f"{name}.is_enabled_y",
            ui_manager=ui_manager,
            depends_on={f"bool": f"{sub_db_key}.is_enabled_y"},
            updates={f"bool": f"{sub_db_key}.is_enabled_y"},
            data_identifier="bool",
            label="Enabled in y-direction",
        )

        # Sub-layout for the BCs of each edge
        edge_group_boxes = []
        for key_name in edge_keynames:
            ax_names = [ax for ax in self.ax_names if ax != key_name[0]]
            edge_layout = EdgeBoundaryConditionV2Widget(
                name=f"{name}.{key_name}",
                ui_manager=ui_manager,
                sub_db_key=f"{sub_db_key}.{key_name}",
                key_name=key_name,
                coord_dict={ax: self.coord_dict[ax] for ax in ax_names},
                ax_names=ax_names,
                coef_min=coef_min,
                coef_max=coef_max,
            )
            edge_group_box = QGroupBox(key_name)
            edge_group_box.setFont(QFont("Arial", 12))
            edge_group_box.setLayout(edge_layout.layout())
            edge_group_boxes.append(edge_group_box)

        layout = QVBoxLayout()
        layout.addWidget(is_enabled_x)
        x_layout = QHBoxLayout()
        for edge_group_box in edge_group_boxes:
            if edge_group_box.title()[0] == "x":
                x_layout.addWidget(edge_group_box)
        layout.addLayout(x_layout)
        layout.addWidget(is_enabled_y)
        y_layout = QHBoxLayout()
        for edge_group_box in edge_group_boxes:
            if edge_group_box.title()[0] == "y":
                y_layout.addWidget(edge_group_box)
        layout.addLayout(y_layout)
        self.setLayout(layout)

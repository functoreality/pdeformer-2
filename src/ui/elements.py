r"""User interface elements."""
from typing import Optional, Dict, Union
import io
import time
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QLabel, QLineEdit, QHBoxLayout, QSlider, QVBoxLayout, QCheckBox, QPushButton, QFileDialog, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from src.ui.basic import UIElement, UIManager
Value = Union[int, float, NDArray[float]]


class BaseText(UIElement):
    r"""Base text UI element.

    Args:
        kwargs:
            data_identifier (str): The identifier of data in depends_on and updates.
            Default: "data".
            label (str): Text to be displayed by the QLabel. Default: "".
            font_label (QFont): The font of the QLabel. Default: QFont("Arial", 14, weight=QFont.Bold).
            default_text (str): Default text for QLabel. Default: "".
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 depends_on: Optional[Dict[str, str]] = None,
                 updates: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        super().__init__(name, ui_manager, depends_on, updates)
        self.data_identifier = kwargs.get("data_identifier", "data")
        # Create and configure the label
        self.label = QLabel(kwargs.get("label", ""))
        self.label.setFont(kwargs.get("font_label", QFont("Arial", 14, weight=QFont.Bold)))
        self.label.setText(kwargs.get("default_text", ""))

        # Layout configuration for the main widget
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label)
        self.setLayout(main_layout)

    def update(self) -> None:
        super().update()
        if self.data_identifier in self.depends_on:
            key = self.depends_on[self.data_identifier]
            data = self.ui_manager.get_db_value(key)
            self.label.setText(self._data2str(self.data_identifier, data))

    def _data2str(self, data_identifier: str, data: Value) -> str:
        r"""Convert the data to a string. Override this method for custom conversion."""
        if data_identifier == "data":
            return str(data)
        if data_identifier == "str":
            return data
        raise ValueError(f"Unsupported data_identifier: {data_identifier}")


class InteractiveTextInput(UIElement):
    r"""Interactive text input UI element, allowing users to input text.

    Args:
        kwargs:
            data_identifier (str): The identifier of data in depends_on and updates.
            input_width (int): The designated width of the QLineEdit. Default: 200.
            font_input (QFont): The font of the QLineEdit. Default: QFont("Arial", 14).
            default_text (str): Default text in the QLineEdit. Default: "".
            prefix (str): The prefix to be added to the input text. Default: "".
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 depends_on: Optional[Dict[str, str]] = None,
                 updates: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        # Create and configure the input field
        super().__init__(name, ui_manager, depends_on, updates, **kwargs)
        self.data_identifier = kwargs.get("data_identifier", "data")
        self.prefix = kwargs.get("prefix", "")
        self.layout = QHBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        self.input = QLineEdit()
        self.input.setFont(kwargs.get("font_input", QFont("Arial", 14)))
        self.input.setFixedWidth(kwargs.get("input_width", 200))
        # # Create a prefix for the input field
        # prefix_action = QAction(self.prefix, self.input)
        # self.input.addAction(prefix_action, QLineEdit.LeadingPosition)
        # self.input.setStyleSheet("""
        #     QLineEdit {
        #         padding-left: 50px;
        #     }
        # """)
        # Update when enter is pressed
        self.input.returnPressed.connect(lambda: self.ui_manager.update_ui(self.name))
        self.input.setText(kwargs.get("default_text", ""))

        # Add the input field to the layout
        self.layout.addWidget(self.input)
        self.setLayout(self.layout)

    def update(self) -> None:
        super().update()
        if self.data_identifier in self.depends_on:
            key = self.depends_on[self.data_identifier]
            data = self.ui_manager.get_db_value(key)
            self.input.setText(self._data2str(self.data_identifier, data))
        if "enabled" in self.depends_on:
            key = self.depends_on["enabled"]
            enabled = self.ui_manager.get_db_value(key)
            self.input.setEnabled(enabled)

    def update_database(self) -> None:
        if self.data_identifier in self.updates:
            key = self.updates[self.data_identifier]
            data = self._str2data(self.data_identifier, self.input.text())
            self.ui_manager.set_db_value(key, data)

    def _str2data(self, data_identifier: str, text: str) -> Value:
        r"""Convert the string to data. Override this method for custom conversion."""
        if data_identifier in ["data", "str"]:
            return text
        raise ValueError(f"Unsupported data_identifier: {data_identifier}")

    _data2str = BaseText._data2str  # pylint: disable=protected-access


class SliderInput(UIElement):
    r"""
    Slider input UI element, allowing users to select a value between a defined range.

    Args:
        kwargs:
            min_value (int): Minimum value of the slider. Default: 0.
            max_value (int): Maximum value of the slider. Default: 100.
            real_min_value (float): Minimum value of the real range. Default: 0.0.
            real_max_value (float): Maximum value of the real range. Default: 1.0.
            tick_interval (int): The interval between slider ticks. Default: 10.
            slider_width (int): The designated width of the slider. Default: 350.
            conversion (str): Conversion type between slider value and real value. Default: "linear".
            show_value_box (bool): Whether to show the value box. Default: True.
            min_label (str): Label for the minimum value. Default: "<b>{real_min_value}</b>".
            max_label (str): Label for the maximum value. Default: "<b>{real_max_value}</b>".
            slider_label (str): Text label for the slider. Default: "".
            track (bool): Whether to track the slider value. Default: False.
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 depends_on: Optional[Dict[str, str]] = None,
                 updates: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        super().__init__(name, ui_manager, depends_on, updates)
        self.data_identifier = kwargs.get("data_identifier", "data")

        # Get parameters from kwargs
        self.min_value = kwargs.get("min_value", 0)
        self.max_value = kwargs.get("max_value", 100)
        self.real_min_value = kwargs.get("real_min_value", 0.0)
        self.real_max_value = kwargs.get("real_max_value", 1.0)
        self.tick_interval = kwargs.get("tick_interval", 10)
        self.slider_width = kwargs.get("slider_width", 350)
        self.conversion = kwargs.get("conversion", "linear")
        self.show_value_box = kwargs.get("show_value_box", True)
        min_label = kwargs.get("min_label", f"<b>{self.real_min_value}</b>")
        max_label = kwargs.get("max_label", f"<b>{self.real_max_value}</b>")
        slider_label = kwargs.get("slider_label", "")
        track = kwargs.get("track", False)
        font = QFont("Arial", 12)

        # Setup the layout for the main widget
        main_layout = QHBoxLayout()

        # Optional label for the slider
        self.slider_label = QLabel(slider_label)
        self.slider_label.setFont(font)
        if not slider_label:
            self.slider_label.hide()
        main_layout.addWidget(self.slider_label)

        # Setup the layout for slider, value box, and labels
        slider_layout = QHBoxLayout()

        # Setup labels for the min and max slider values
        self.min_label = QLabel(min_label)
        self.max_label = QLabel(max_label)
        # self.min_label.setFixedWidth(50)
        # self.max_label.setFixedWidth(50)
        self.min_label.setAlignment(Qt.AlignCenter)
        self.max_label.setAlignment(Qt.AlignCenter)
        self.min_label.setFont(font)
        self.max_label.setFont(font)

        # Setup the slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.min_value)
        self.slider.setMaximum(self.max_value)
        self.slider.setValue(self.min_value)  # Initialize slider value to minimum
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(self.tick_interval)
        # self.slider.setFixedWidth(self.slider_width)

        # Setup the value box
        self.value_box = QLineEdit(str(self._slider_to_real(self.slider.value())))
        self.value_box.setFixedWidth(70)
        self.value_box.setFont(font)
        if not self.show_value_box:
            # If value box is not shown, use a QLabel to occupy the space
            self.value_box.hide()
            self.occupy_label = QLabel()
            self.occupy_label.setFixedWidth(70)

        # Add widgets to the slider layout
        slider_layout.addWidget(self.min_label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.max_label)
        if self.show_value_box:
            slider_layout.addWidget(self.value_box)
        else:
            slider_layout.addWidget(self.occupy_label)

        # Connect slider and value box changes
        self._previous_update_time = time.time()
        self.slider.valueChanged.connect(self.update_from_slider)
        self.slider.setTracking(track)
        self.value_box.returnPressed.connect(self.update_from_value_box)

        # Add slider layout to the main layout
        main_layout.addLayout(slider_layout)
        self.setLayout(main_layout)

    def _slider_to_real(self, slider_value: int) -> float:
        r"""Convert slider value to real value based on the conversion type."""
        if self.conversion == "linear":
            return self.real_min_value + (self.real_max_value - self.real_min_value) * \
                   (slider_value - self.min_value) / (self.max_value - self.min_value)
        if self.conversion == "log":
            min_log = np.log10(self.real_min_value)
            max_log = np.log10(self.real_max_value)
            return float(10 ** (min_log + (max_log - min_log) * \
                                (slider_value - self.min_value) / (self.max_value - self.min_value)))
        raise NotImplementedError(f"Conversion type '{self.conversion}' is not implemented.")

    def _real_to_slider(self, real_value: float) -> int:
        r"""Convert real value to slider value based on the conversion type."""
        if self.conversion == "linear":
            return int(self.min_value + (self.max_value - self.min_value) * \
                       (real_value - self.real_min_value) / (self.real_max_value - self.real_min_value))
        if self.conversion == "log":
            real_value = max(self.real_min_value, min(real_value, self.real_max_value))
            min_log = np.log10(self.real_min_value)
            max_log = np.log10(self.real_max_value)
            return int(self.min_value + (self.max_value - self.min_value) * \
                       (np.log10(real_value) - min_log) / (max_log - min_log))
        raise NotImplementedError(f"Conversion type '{self.conversion}' is not implemented.")

    def update_from_slider(self, value: int) -> None:
        r"""Updates the value box and database when the slider is moved."""
        # Convert slider value to real value
        real_value = self._slider_to_real(value)

        # Update the value box
        if str(real_value) != self.value_box.text():
            self.value_box.setText(f"{real_value:.3f}")

        # Update the database
        if not self._updating:
            self.ui_manager.update_ui(self.name)

    def update_from_value_box(self) -> None:
        r"""Updates the slider and database when the value box is edited."""
        try:
            real_value = float(self.value_box.text())
            # Ensure value is within the real range
            real_value = max(self.real_min_value, min(real_value, self.real_max_value))
            slider_value = self._real_to_slider(real_value)
            self.slider.setValue(slider_value)

            # Update the database
            if not self._updating:
                self.ui_manager.update_ui(self.name)
        except ValueError:
            # Handle invalid input in the value box
            self.value_box.setText(f"{self._slider_to_real(self.slider.value()):.3f}")

    def update(self) -> None:
        r"""Updates the slider and value box based on the value from the database."""
        self._updating = True
        super().update()
        if self.data_identifier in self.depends_on:
            key = self.depends_on[self.data_identifier]
            real_value = self.ui_manager.get_db_value(key)
            slider_value = self._real_to_slider(real_value)
            self.slider.setValue(slider_value)
            self.value_box.setText(f"{real_value:.3f}")
        if "enabled" in self.depends_on:
            key = self.depends_on["enabled"]
            enabled = self.ui_manager.get_db_value(key)
            self.slider.setEnabled(enabled)
            self.value_box.setEnabled(enabled)
        if "disabled" in self.depends_on:
            key = self.depends_on["disabled"]
            disabled = self.ui_manager.get_db_value(key)
            self.slider.setEnabled(not disabled)
            self.value_box.setEnabled(not disabled)
        self._updating = False

    def update_database(self) -> None:
        r"""Updates the database based on the current value of the slider."""
        if self.data_identifier in self.updates:
            key = self.updates[self.data_identifier]
            self.ui_manager.set_db_value(key, self._slider_to_real(self.slider.value()))


class CheckboxInput(UIElement):
    r"""Checkbox input UI element, allowing users to toggle a boolean value.

    Args:
        kwargs:
            label (str): The text label for the checkbox. Default: "".
            font (QFont): The font of the QCheckBox. Default: QFont("Arial", 14, weight=QFont.Bold).
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 depends_on: Optional[Dict[str, str]] = None,
                 updates: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        super().__init__(name, ui_manager, depends_on, updates)

        # Create and configure the checkbox
        label = kwargs.get("label", "")
        font = kwargs.get("font", QFont("Arial", 14, weight=QFont.Bold))

        self.checkbox = QCheckBox(label, self)
        self.checkbox.setFont(font)
        # self.checkbox.setFixedWidth(width)

        # Connect the checkbox state change signal to the UIManager
        self.checkbox.stateChanged.connect(lambda: self.ui_manager.update_ui(self.name))

        # Layout configuration
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.checkbox)
        self.setLayout(main_layout)

    def update(self) -> None:
        r"""Updates the checkbox based on the value from the database."""
        super().update()
        if "bool" in self.depends_on:
            key = self.depends_on["bool"]
            value = self.ui_manager.get_db_value(key)
            self.checkbox.setChecked(value)

    def update_database(self) -> None:
        r"""Updates the database based on the current state of the checkbox."""
        if "bool" in self.updates:
            key = self.updates["bool"]
            self.ui_manager.set_db_value(key, self.checkbox.isChecked())


class AutoUpdate(QThread):
    r"""Asynchronous thread for auto-updating the database."""
    updated = pyqtSignal()

    def __init__(self, ui_manager: UIManager, key: str) -> None:
        super().__init__()
        self.ui_manager = ui_manager
        self.key = key

    def run(self) -> None:
        self.ui_manager.update_db_value(self.key)
        self.updated.emit()


class ClickButton(UIElement):
    r"""Click Button UI element, allowing users to trigger an action by clicking the button.

    Args:
        kwargs:
            label (str): The text label for the QPushButton. Default: "Click".
            font (QFont): The font of the QPushButton. Default: QFont("Arial", 14, weight=QFont.Bold).
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 depends_on: Optional[Dict[str, str]] = None,
                 updates: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        super().__init__(name, ui_manager, depends_on, updates)

        # Create and configure the button
        label = kwargs.get("label", "Click")
        font = kwargs.get("font", QFont("Arial", 14, weight=QFont.Bold))

        self.button = QPushButton(label)
        self.button.setFont(font)
        if "auto_update" in self.updates:
            self.auto_update = AutoUpdate(ui_manager, self.updates["auto_update"])
            self.updated = self.auto_update.updated

        self.button.clicked.connect(lambda: self.ui_manager.update_ui(self.name))
        self.updated.connect(lambda: self.button.setEnabled(True))

        # Layout configuration
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.button)
        self.setLayout(main_layout)

    def update_database(self) -> None:
        r"""Update the database based on the button click."""
        if "auto_update" in self.updates:
            # set the clicked button to be disabled
            self.button.setEnabled(False)
            self.auto_update.start()


class ValueGeneratorButton(UIElement):
    r"""Value Generator Button UI element, generate a random value or set default
    value by a function after clicking the button.

    Args:
        kwargs:
            label (str): The text label for the QPushButton. Default: "Generate".
            font (QFont): The font of the QPushButton. Default: QFont("Arial", 14, weight=QFont.Bold).
            generator (callable): The function that generates the random or default value.
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 depends_on: Optional[Dict[str, str]] = None,
                 updates: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        super().__init__(name, ui_manager, depends_on, updates)
        label = kwargs.get("label", "Generate")
        font = kwargs.get("font", QFont("Arial", 14, weight=QFont.Bold))
        self.generator = kwargs.get("generator", lambda: 0)

        self.button = QPushButton(label)
        self.button.setFont(font)
        self.button.clicked.connect(lambda: self.ui_manager.update_ui(self.name))

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.button)
        self.setLayout(main_layout)

    def update_database(self) -> None:
        if "data" in self.updates:
            key = self.updates["data"]
            self.ui_manager.set_db_value(key, self.generator())


class FileLoadButton(UIElement):
    r"""File Load Button UI element, allowing users to load files from their system.

    Args:
        kwargs:
            label (str): The text label for the file load button. Default: "Load File".
            file_type (str): The file type that the button should load. Default: "npy".
            font (QFont): The font of the QPushButton. Default: QFont("Arial", 14, weight=QFont.Bold).
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 depends_on: Optional[Dict[str, str]] = None,
                 updates: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        super().__init__(name, ui_manager, depends_on, updates)

        # Create and configure the file load button
        label = kwargs.get("label", "Load File")
        file_type = kwargs.get("file_type", "npy")
        font = kwargs.get("font", QFont("Arial", 14, weight=QFont.Bold))

        self.file_button = QPushButton(label)
        self.file_button.setFont(font)
        self.file_button.clicked.connect(lambda: self.ui_manager.update_ui(self.name))

        # Store file type
        self.file_type = file_type

        # Layout configuration
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.file_button)
        self.setLayout(main_layout)

    def update(self) -> None:
        r"""Update the UI element."""
        super().update()
        if "enabled" in self.depends_on:
            key = self.depends_on["enabled"]
            enabled = self.ui_manager.get_db_value(key)
            self.file_button.setEnabled(enabled)

    def update_database(self) -> None:
        r"""Open a file dialog to load a file and update the database."""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_filter = f"{self.file_type.upper()} files (*.{self.file_type})"
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter, options=options)

        if file_name:
            try:
                # Load the data from the selected file
                if self.file_type == "npy":
                    data = np.load(file_name)
                    if "npy" in self.updates:
                        key = self.updates["npy"]
                        self.ui_manager.set_db_value(key, data)

                else:
                    raise NotImplementedError(f"File type '{self.file_type}' is not supported yet.")

            except Exception as e:  # pylint: disable=broad-except
                print(f"Failed to load file: {e}")


class InteractiveChart(UIElement):
    r"""Interactive Chart UI element using matplotlib's FigureCanvasQTAgg.

    Args:
        plot_func (callable): The function that performs the plotting on the provided figure and canvas.
                             Should accept `figure`, `canvas`, `data`, `*args`, and `**kwargs` as parameters.
        kwargs:
            fig_size (Tuple[int, int]): The size of the figure. Default: (5, 4).
            signal (pyqtSignal): The signal to trigger the update. Default: None.
    """
    def __init__(self, name: str, ui_manager: UIManager, plot_func: callable,
                 depends_on: Optional[Dict[str, str]] = None,
                 updates: Optional[Dict[str, str]] = None, **kwargs) -> None:
        super().__init__(name, ui_manager, depends_on, updates)

        self.plot_func = plot_func
        self.signal = kwargs.get("signal", None)

        # Create a matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # Set size policy to expanding
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        # Layout configuration
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)
        if self.signal:
            self.signal.connect(self.update)

    def update(self) -> None:
        r"""Update the chart by re-running the plot function."""
        super().update()
        if "data" in self.depends_on:
            key = self.depends_on["data"]
            data = self.ui_manager.get_db_value(key)
            self.plot_func(self.figure, self.canvas, data)
        if "video_dt" in self.depends_on:
            key = self.depends_on["video_dt"]
            video_dt = self.ui_manager.get_db_value(key)
            self.plot_func.set_video_dt_real(video_dt)
        if "t_coord" in self.depends_on:
            key = self.depends_on["t_coord"]
            t_coord = self.ui_manager.get_db_value(key)
            self.plot_func.set_t_coord(t_coord)

    def update_database(self) -> None:
        r"""No direct database update from the chart itself."""

    def resizeEvent(self, event):  # pylint: disable=invalid-name
        super().resizeEvent(event)
        # Adjust figure size
        dpi = self.figure.get_dpi()
        width = self.canvas.width() / dpi
        height = self.canvas.height() / dpi
        self.figure.set_size_inches(width, height, forward=True)
        self.canvas.draw()


class LatexText(BaseText):
    r"""A UI element that displays LaTeX-rendered text.

    This class inherits from BaseText and overrides the update method to render LaTeX expressions.

    Args:
        kwargs:
            dpi (int): The resolution of the rendered image. Default: 120.
            fontsize (int): The font size for rendering LaTeX. Default: 14.
            prefix (str): The prefix to be added to the LaTeX expression. Default: "".
    """

    def __init__(self,
                 name: str,
                 ui_manager: UIManager,
                 depends_on: Optional[Dict[str, str]] = None,
                 updates: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        # Get additional parameters for LaTeX rendering
        self.dpi = kwargs.get("dpi", 120)
        self.fontsize = kwargs.get("fontsize", 16)
        self.prefix = kwargs.get("prefix", "")
        super().__init__(name, ui_manager, depends_on, updates, **kwargs)

        # Adjust the QLabel settings
        self.label.setAlignment(Qt.AlignCenter)
        # self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def update(self) -> None:
        super().update()
        if self.data_identifier in self.depends_on:
            key = self.depends_on[self.data_identifier]
            data = self.ui_manager.get_db_value(key)
            data = self.prefix + data
            # Render LaTeX and set it to the label
            pixmap = self._render_latex_to_pixmap(data)
            if pixmap:
                self.label.setPixmap(pixmap)
            else:
                # If rendering fails, display error text
                self.label.setText("Invalid LaTeX expression")

    def _render_latex_to_pixmap(self, latex_str: str) -> Optional[QPixmap]:
        r"""Render LaTeX string to QPixmap."""
        try:
            # Create a figure and render the LaTeX expression
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, f"${latex_str}$", fontsize=self.fontsize)
            plt.axis('off')

            # Save the figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight', pad_inches=0.0)
            plt.close(fig)
            buf.seek(0)

            # Load the image from the buffer into QPixmap
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            return pixmap
        except Exception as e:  # pylint: disable=broad-except
            print(f"Failed to render LaTeX: {e}")
            return None

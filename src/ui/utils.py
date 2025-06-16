r"""Utility functions for symbolic expressions and plotting."""
from typing import Dict, List
import numpy as np
from numpy.typing import NDArray
from PyQt5.QtCore import QTimer
import sympy as sp
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def term_coef2latex(coef_value: float, term_latex: str) -> str:
    r"""
        Get the LaTeX expression for the multiplication of a coefficient and a
        specific term.
    """
    if coef_value == 0:
        return ""
    if coef_value == 1:
        if term_latex == "":
            return "1"
        return term_latex
    if coef_value == -1:
        if term_latex == "":
            return "-1"
        return "-" + term_latex
    return f"{coef_value:.2f}{term_latex}"


def term_coef_str2latex(coef_str: str, term_latex: str) -> str:
    r"""
    Get the LaTeX expression for the multiplication of a coefficient string and a
    specific term.
    """
    try:
        coef_value = float(coef_str)
        return term_coef2latex(coef_value, term_latex)
    except ValueError:
        if not coef_str.strip():
            # coef_str is empty or whitespace
            return ""
        if not term_latex.strip():
            # term_latex is empty or whitespace
            return coef_str
        return f"{add_parentheses(coef_str)}{term_latex}"


def eval_expression_to_field(expression: str,
                             ax_names: List[str],
                             coord_dict: Dict[str, NDArray[float]]) -> NDArray[float]:
    r"""
    Evaluate a given symbolic expression involving x and y to produce a field as a numpy array.

    Args:
        expression (str): The symbolic expression to evaluate.
        ax_names (List[str]): The names of the axes in the expression
        coord_dict (Dict[str, NDArray[float]]): A dictionary containing the coordinates of the field.

    Returns:
        NDArray[float]: The evaluated field as a 2D numpy array of shape (len(y_coord), len(x_coord)).
    """
    n_dims = len(ax_names)
    # Convert the string expression to a SymPy expression
    try:
        expr = sp.sympify(expression)
    except Exception as e:
        raise ValueError(f"Error parsing expression '{expression}': {e}")

    # Check if the expression is constant by verifying if there are no free symbols
    if len(expr.free_symbols) == 0:  # pylint: disable=len-as-condition
        # The expression is a constant
        constant_value = float(expr.evalf())  # Evaluate the constant to a float

        if n_dims == 1:
            axis = ax_names[0]
            if axis not in coord_dict:
                raise ValueError(f"Coordinate '{axis}' not found in 'coord_dict'.")
            coord = coord_dict[axis]
            # Create a 1D array filled with the constant value
            return np.full_like(coord, constant_value, dtype=float)

        if n_dims == 2:
            axis_x, axis_y = ax_names
            if axis_x not in coord_dict or axis_y not in coord_dict:
                raise ValueError(f"Coordinates '{axis_x}' and/or '{axis_y}' not found in 'coord_dict'.")
            len_x = len(coord_dict[axis_x])
            len_y = len(coord_dict[axis_y])
            # Create a 2D array filled with the constant value
            return np.full((len_y, len_x), constant_value, dtype=float)

        raise ValueError("Only 1D and 2D fields are supported.")

    if n_dims == 1:
        if ax_names[0] not in coord_dict:
            raise ValueError(f"Coordinate '{ax_names[0]}' not found in 'coord_dict'.")
        coord = coord_dict[ax_names[0]]
        var = sp.symbols(ax_names[0])
        f_lambdified = sp.lambdify(var, expr, 'numpy')
        try:
            field = f_lambdified(coord)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {e}")
        return field

    if n_dims == 2:
        if ax_names[0] not in coord_dict or ax_names[1] not in coord_dict:
            raise ValueError(f"Coordinates '{ax_names[0]}' and '{ax_names[1]}' not found in 'coord_dict'.")
        coord1 = coord_dict[ax_names[0]]
        coord2 = coord_dict[ax_names[1]]
        var1, var2 = sp.symbols(f"{ax_names[0]} {ax_names[1]}")
        x_grid, y_grid = np.meshgrid(coord1, coord2, indexing='ij')
        f_lambdified = sp.lambdify((var1, var2), expr, 'numpy')
        try:
            field = f_lambdified(x_grid, y_grid)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {e}")
        return field

    raise ValueError("Only 1D and 2D fields are supported.")


def add_parentheses(latex_str: str) -> str:
    r"""
    Add parentheses to a LaTeX string if it contains more than one term,
    ignoring split chars (+-*/) inside existing parentheses.
    """
    nesting = 0
    split_chars = ['+', '-', '*', '/']
    for char in latex_str:
        if char == '(':
            nesting += 1
        elif char == ')':
            nesting -= 1
        elif char in split_chars and nesting == 0:
            # Found a '+' or '-' not inside parentheses
            return rf"({latex_str})"
    return latex_str


class Plotter2D:
    r"""
    Plotter for 2D fields.
    """

    def __init__(self,
                 coord_dict: Dict[str, NDArray[float]],
                 **kwargs) -> None:
        if "x" not in coord_dict or "y" not in coord_dict:
            raise ValueError("Input 'coord_dict' should contain 'x' and 'y' coordinates.")
        self.x_coord = coord_dict["x"]
        self.y_coord = coord_dict["y"]
        self.title = kwargs.get("title", "")
        self.cmap = kwargs.get("cmap", "jet")
        self.show_colorbar = kwargs.get("show_colorbar", True)
        self.vmin = kwargs.get("vmin", 0)
        self.vmax = kwargs.get("vmax", 1)
        self.title_fontsize = kwargs.get("title_fontsize", 16)

    def __call__(self,
                 figure: Figure,
                 canvas: FigureCanvas,
                 field: NDArray[float]) -> None:
        r"""
        Plot the 2D field on the given figure and canvas.
        """
        figure.clear()
        ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])

        # Plot the data
        vmin = min(np.min(field), self.vmin)
        vmax = max(np.max(field), self.vmax)
        im = ax.imshow(
            field,
            extent=[self.x_coord[0], self.x_coord[-1], self.y_coord[0], self.y_coord[-1]],
            origin='lower',
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Add the title with increased font size
        ax.set_title(self.title, fontsize=self.title_fontsize, pad=10)

        # Remove axis decorations if desired
        ax.axis('off')
        if self.show_colorbar:
            figure.colorbar(im, ax=ax)

        canvas.draw()


class Plotter2DVideo:
    """
    Plotter for dynamic 2D fields (video).
    """

    def __init__(self,
                 coord_dict: Dict[str, np.ndarray],
                 **kwargs) -> None:
        """
        Initializes the Plotter2DVideo.

        Args:
            coord_dict (Dict[str, np.ndarray]): Dictionary containing 'x', 'y', and 't' coordinates.
            **kwargs: Additional keyword arguments for customization.
        """
        if "x" not in coord_dict or "y" not in coord_dict or "t" not in coord_dict:
            raise ValueError("Input 'coord_dict' should contain 'x', 'y', and 't' coordinates.")
        self.x_coord = coord_dict["x"]
        self.y_coord = coord_dict["y"]
        self.t_coord = coord_dict["t"]
        self.title = kwargs.get("title", "Dynamic 2D Field")
        self.cmap = kwargs.get("cmap", "jet")
        self.show_colorbar = kwargs.get("show_colorbar", True)
        self.vmin = kwargs.get("vmin", None)  # Automatically set based on data if None
        self.vmax = kwargs.get("vmax", None)
        self.title_fontsize = kwargs.get("title_fontsize", 16)
        self.video_dt = kwargs.get("video_dt", 100)  # Frame interval in milliseconds
        self.loop = kwargs.get("loop", True)  # Whether to loop the video

        # Animation-related attributes
        self.anim = None
        self.current_frame = 0
        self.timer = None

    def __call__(self,
                 figure: Figure,
                 canvas: FigureCanvas,
                 field: np.ndarray) -> None:
        """
        Plots the dynamic 2D field on the given figure and canvas.

        Args:
            figure (Figure): Matplotlib Figure object.
            canvas (FigureCanvas): Matplotlib FigureCanvas object.
            field (np.ndarray): 3D NumPy array with shape (time_steps, y, x).
        """
        if field.ndim != 3:
            raise ValueError("Input 'field' should be a 3D array with shape (time_steps, y, x).")

        self.field = field
        self.num_frames = field.shape[0]
        self.current_frame = 0
        vmin = np.min(field)
        vmax = np.max(field)
        if self.vmin is not None:
            vmin = min(vmin, self.vmin)
        if self.vmax is not None:
            vmax = max(vmax, self.vmax)

        figure.clear()
        ax = figure.add_subplot(111)

        # Initialize image with the first frame
        initial_field = self.field[self.current_frame]
        self.im = ax.imshow(
            initial_field,
            extent=[self.x_coord[0], self.x_coord[-1], self.y_coord[0], self.y_coord[-1]],
            origin='lower',
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Add title to the plot
        title_str = (f"{self.title} - Frame {self.current_frame+1}/{self.num_frames} - "
                     f"Time {self.t_coord[self.current_frame]:.2f}")
        ax.set_title(title_str, fontsize=self.title_fontsize, pad=10)

        # Remove axes for a cleaner look
        ax.axis('off')

        # Add colorbar if enabled
        if self.show_colorbar:
            self.cbar = figure.colorbar(self.im, ax=ax)

        canvas.draw()

        # Setup QTimer for animation
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.update_frame(figure, canvas))
        self.timer.start(self.video_dt)

    def update_frame(self, figure: Figure, canvas: FigureCanvas):  # pylint: disable=unused-argument
        """
        Updates the frame in the animation.

        Args:
            figure (Figure): Matplotlib Figure object.
            canvas (FigureCanvas): Matplotlib FigureCanvas object.
        """
        self.current_frame += 1
        if self.current_frame >= self.num_frames:
            if self.loop:
                self.current_frame = 0  # Reset to the first frame for looping
            else:
                self.timer.stop()
                return

        # Update image data with the current frame
        self.im.set_data(self.field[self.current_frame])

        # Update the title to reflect the current frame
        title_str = (f"{self.title} - Frame {self.current_frame+1}/{self.num_frames}"
                     f" - Time {self.t_coord[self.current_frame]:.2f}")
        self.im.axes.set_title(title_str, fontsize=self.title_fontsize, pad=10)

        # Redraw the canvas to display the updated frame
        canvas.draw()

    def set_video_dt_real(self, video_dt_real: float) -> None:
        """
        Sets the frame interval for the video.

        Args:
            video_dt_real (float): Frame interval in seconds.
        """
        self.video_dt = int(video_dt_real * 1000)  # Convert to milliseconds
        self.timer.setInterval(self.video_dt)

    def set_t_coord(self, t_coord: np.ndarray) -> None:
        """
        Sets the time coordinates for the field.

        Args:
            t_coord (np.ndarray): 1D NumPy array containing the time coordinates.
        """
        if self.timer is not None and t_coord.shape[0] != self.num_frames:
            self.timer.stop()
            self.timer = None
        self.t_coord = t_coord


class Plotter2DSnapshots:
    r"""
    Plotter for multiple snapshots of 2D fields.
    """

    def __init__(self,
                 coord_dict: Dict[str, NDArray[float]],
                 **kwargs) -> None:
        if "x" not in coord_dict or "y" not in coord_dict:
            raise ValueError("Input 'coord_dict' should contain 'x' and 'y' coordinates.")
        self.x_coord = coord_dict["x"]
        self.y_coord = coord_dict["y"]
        self.title = kwargs.get("title", "2D Field Snapshots")
        self.cmap = kwargs.get("cmap", "jet")
        self.show_colorbar = kwargs.get("show_colorbar", True)
        self.vmin = kwargs.get("vmin", None)  # Auto set based on data
        self.vmax = kwargs.get("vmax", None)
        self.title_fontsize = kwargs.get("title_fontsize", 12)
        self.rows = kwargs.get("rows", 2)
        self.cols = kwargs.get("cols", 2)
        self.spacing = kwargs.get("spacing", 0.4)

    def __call__(self,
                 figure: Figure,
                 canvas: FigureCanvas,
                 field: NDArray[float]) -> None:
        r"""
        Plot multiple snapshots of 2D fields on the given figure and canvas.
        """
        if field.ndim != 3:
            raise ValueError("Input 'field' should be a 3D array with shape (snapshots, y, x).")

        self.field = field
        self.num_snapshots = field.shape[0]
        if self.num_snapshots > self.rows * self.cols:
            raise ValueError(f"Number of snapshots {self.num_snapshots} exceeds grid capacity {self.rows * self.cols}.")

        figure.clear()
        axes = figure.subplots(self.rows, self.cols, squeeze=False)
        axes = axes.flatten()

        for i in range(self.num_snapshots):
            ax = axes[i]
            im = ax.imshow(
                self.field[i],
                extent=[self.x_coord[0], self.x_coord[-1], self.y_coord[0], self.y_coord[-1]],
                origin='lower',
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax
            )
            ax.set_title(f"Snapshot {i+1}", fontsize=self.title_fontsize)
            ax.axis('off')

        # Remove unused subplots
        for j in range(self.num_snapshots, self.rows * self.cols):
            axes[j].axis('off')

        # Add colorbar
        if self.show_colorbar:
            # Place colorbar beside the last subplot
            cbar_ax = figure.add_axes([0.92, 0.1, 0.02, 0.8])
            figure.colorbar(im, cax=cbar_ax)

        # Add overall title
        figure.suptitle(self.title, fontsize=self.title_fontsize + 2)

        # Adjust spacing
        figure.subplots_adjust(wspace=self.spacing, hspace=self.spacing)

        canvas.draw()


class Plotter1D:
    r"""
    Plotter for 1D fields.
    """

    def __init__(self,
                 coord_dict: Dict[str, NDArray[float]],
                 **kwargs) -> None:
        ax_names = kwargs.get("ax_names", ["x"])
        if len(ax_names) != 1:
            raise ValueError("Input 'ax_names' should contain only one axis name.")
        if ax_names[0] not in coord_dict:
            raise ValueError(f"Coordinate '{ax_names[0]}' not found in 'coord_dict'.")
        self.ax_name = ax_names[0]
        self.coord = coord_dict[ax_names[0]]
        self.title = kwargs.get("title", "")
        self.vmin = kwargs.get("vmin", 0)
        self.vmax = kwargs.get("vmax", 1)

    def __call__(self,
                 figure: Figure,
                 canvas: FigureCanvas,
                 field: NDArray[float]) -> None:
        r"""
        Plot the 1D field on the given figure and canvas.
        """
        figure.clear()
        ax = figure.add_subplot(111)
        ax.set_title(self.title)
        ax.set_xlabel(self.ax_name)
        ax.set_ylabel("Field Value")
        ax.plot(self.coord, field)
        canvas.draw()

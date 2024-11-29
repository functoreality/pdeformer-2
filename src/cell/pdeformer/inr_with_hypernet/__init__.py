r"""Common Implicit Neural Representations (INRs)"""
from .siren import Siren, SirenWithHypernet
from .mfn import MFNNet, MFNNetWithHypernet
from .poly_inr import PolyINR, PolyINRWithHypernet
from .wrapper import get_inr_with_hypernet

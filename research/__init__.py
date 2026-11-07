"""Root namespace bridge to world_models_fpga/research."""
import os

_pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "world_models_fpga", "research"))
if os.path.isdir(_pkg_dir) and _pkg_dir not in __path__:
    __path__.append(_pkg_dir)

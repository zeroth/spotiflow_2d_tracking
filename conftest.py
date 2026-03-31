"""Root conftest: insert lightweight stubs for heavy C-extension packages
so that pytest collection does not trigger torch DLL loading on Windows."""
import sys
from unittest.mock import MagicMock


def _stub_if_absent(name: str) -> None:
    if name not in sys.modules:
        sys.modules[name] = MagicMock()


# Stub the torch family before anything tries to import it.
for _mod in [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch.utils.data",
]:
    _stub_if_absent(_mod)

# Stub spotiflow and its submodules so that module-level
# `from spotiflow.model import Spotiflow` in _segmentation.py
# resolves without loading the real C-extension chain.
for _mod in [
    "spotiflow",
    "spotiflow.model",
    "spotiflow.model.backbones",
    "spotiflow.model.backbones.resnet",
]:
    _stub_if_absent(_mod)

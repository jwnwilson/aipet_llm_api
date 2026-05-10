"""Unit-test fixtures and sys.modules stubs for optional train-time packages.

kaggle and google-api-python-client are declared in [train] optional deps and
are not installed in the base dev environment.  We stub them here so that unit
tests can import the adapter modules and monkeypatch individual symbols without
needing a full `uv sync --extra train`.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _stub(dotted: str) -> MagicMock:
    """Add a MagicMock for *dotted* to sys.modules and link it onto its parent.

    Linking the parent is required so that monkeypatch can resolve dotted paths
    like ``"googleapiclient.http.MediaIoBaseDownload"`` via successive getattr
    calls starting from the root module.
    """
    if dotted in sys.modules:
        return sys.modules[dotted]  # type: ignore[return-value]
    mock = MagicMock()
    sys.modules[dotted] = mock
    if "." in dotted:
        parent_name, _, attr = dotted.rpartition(".")
        parent = _stub(parent_name)
        setattr(parent, attr, mock)
    return mock


# ---------------------------------------------------------------------------
# kaggle (used by KaggleTrainingAdapter._wait_for_dataset and _kaggle_bin)
# ---------------------------------------------------------------------------
if "kaggle" not in sys.modules:
    _stub("kaggle")
    _stub("kaggle.api")
    _stub("kaggle.api.kaggle_api_extended")

# ---------------------------------------------------------------------------
# google-api-python-client (used by ColabTrainingAdapter)
#
# IMPORTANT: do NOT stub "google" itself — it is a namespace package shared
# with google-protobuf which temporalio depends on.  Only register the specific
# sub-packages that our adapters need.
# ---------------------------------------------------------------------------
if "googleapiclient" not in sys.modules:
    _stub("googleapiclient")
    _stub("googleapiclient.http")
    _stub("googleapiclient.discovery")

if "google.auth" not in sys.modules:
    # Register sub-packages directly without touching sys.modules["google"]
    sys.modules["google.auth"] = MagicMock()
    sys.modules["google.oauth2"] = MagicMock()
    sys.modules["google.oauth2.service_account"] = MagicMock()

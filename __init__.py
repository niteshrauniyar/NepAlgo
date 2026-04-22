import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fetchers.api import fetch_from_api
from fetchers.sharesansar import fetch_from_sharesansar
from fetchers.nepsealpha import fetch_from_nepsealpha

__all__ = ["fetch_from_api", "fetch_from_sharesansar", "fetch_from_nepsealpha"]

"""QAOA Hybrid Routing. Select it with routing_method='qaoa_hybrid'."""

from pkgutil import extend_path

# Let `python plugin_smoke.py` find the installed Rust extension from a checkout.
__path__ = extend_path(__path__, __name__)

__version__ = "2.0.1"

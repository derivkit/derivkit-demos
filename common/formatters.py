"""Formatting helper functions."""

from __future__ import annotations

import numpy as np

__all__ = ["format_value_with_uncertainty"]


def format_value_with_uncertainty(value: float, uncertainty: float) -> str:
    """Format 'value ± uncertainty' with sensible sig figs (PDG-like).

    Rules:
      - Uncertainty to 1 sig fig (2 if leading digit is 1).
      - Value rounded to the same decimals as the uncertainty.

    Args:
        value: The value to format.
        uncertainty: The uncertainty to format.

    Returns:
        The formatted value.
    """
    u = abs(float(uncertainty))
    if u == 0 or not np.isfinite(u):
        return f"{value:.6g} ± 0"

    leading = int(f"{u:.1e}".split("e")[0].replace(".", "").lstrip("0")[:1] or "1")
    sig_u = 2 if leading == 1 else 1
    exp = int(np.floor(np.log10(u)))
    decimals = max(0, -exp + (sig_u - 1))

    return f"{value:.{decimals}f} ± {u:.{decimals}f}"

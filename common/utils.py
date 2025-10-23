"""General-purpose utility functions."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.figure as mpl_figure

__all__ = ["random_generator",
           "add_gaussian_noise",
           "save_fig",
           "build_filepath",
           "ensure_dir",
           "sanitize_stem",
           "find_anchor",
           "resolve_outdir",]

_SAFE_STEM = re.compile(r"[^A-Za-z0-9._\-]+")


def random_generator(seed: int | None = None) -> np.random.Generator:
    """Create a reproducible NumPy random number generator.

    This function wraps ``np.random.default_rng`` and ensures that the input
    seed (if given) is cast to an integer. It is mainly used to provide a
    consistent interface for generating independent random streams across
    Monte Carlo draws.

    Args:
        seed: Optional integer seed for reproducibility. If ``None``, a random
            seed will be drawn from system entropy.

    Returns:
        A NumPy ``Generator`` instance that can be used to produce random draws.

    Example:
        >>> rng = random_generator(42)
        >>> rng.normal(size=3).shape
        (3,)
    """
    return np.random.default_rng(None if seed is None else int(seed))


def add_gaussian_noise(
    y: ArrayLike,
    sigma: float | ArrayLike,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Add element-wise Gaussian noise to an array-like input.

    The function treats the input ``y`` as the clean data and adds random noise
    drawn from a normal distribution with mean zero and standard deviation
    ``sigma``. The output keeps the same shape as the input and is always
    returned as ``float64``.

    Args:
        y: Array-like clean data to which noise will be added.
        sigma: Standard deviation of the Gaussian noise. Can be a scalar or an
            array broadcastable to the shape of ``y``.
        rng: NumPy random number generator used to draw noise samples.

    Returns:
        A NumPy array with Gaussian noise added element-wise to ``y``.

    Example:
        >>> rng = random_generator(0)
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> noisy = add_gaussian_noise(data, sigma=0.1, rng=rng)
        >>> noisy.shape
        (3,)
    """
    y_arr = np.asarray(y, float)
    sig_arr = np.asarray(sigma, float)
    noise = rng.normal(0.0, sig_arr, size=y_arr.shape)
    return (y_arr + noise).astype(np.float64, copy=False)


def ensure_dir(directory: str | Path) -> Path:
    """Create directory if it doesn't exist and return it as Path."""
    d = Path(directory).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def sanitize_stem(stem: str) -> str:
    """Make a filename stem filesystem-friendly (keeps letters, digits, ._-)."""
    s = stem.strip()
    s = _SAFE_STEM.sub("_", s)
    # Avoid empty or dot-only names
    return s or "figure"


def build_filepath(
    directory: str | Path,
    stem: str,
    ext: Optional[str] = "png",
) -> Path:
    """
    Build a full path like '<directory>/<stem>.<ext>'.
    - If `stem` already ends with an extension and `ext` is None, keep it.
    - If both `stem` has an extension and `ext` is provided, `ext` wins.
    """
    d = ensure_dir(directory)
    p = Path(sanitize_stem(stem))

    # If user passed 'name.png' as stem and no ext override: keep it
    if p.suffix and ext is None:
        return d / p

    # Normalize extension (no leading dot)
    if ext is None:
        # No ext in stem and no ext override -> default to png
        ext = "png"
    ext = ext.lstrip(".")

    # Replace suffix if any, else add one
    if p.suffix:
        p = p.with_suffix(f".{ext}")
    else:
        p = p.with_name(p.name + f".{ext}")

    return d / p


def find_anchor(start: str | Path, anchor_name: str) -> Optional[Path]:
    """Walk upward from 'start' to find a directory named 'anchor_name'."""
    p = Path(start).resolve()
    for parent in [p] + list(p.parents):
        if parent.name == anchor_name:
            return parent
    return None


def resolve_outdir(
    directory: str | Path | None,
    *,
    file: str | Path,
    anchor: str | None = None,
    default_rel: str = "plots",
) -> Path:
    """
    Decide a save directory:

    - directory is None           -> '<file_parent>/<default_rel>'
    - directory is absolute       -> that exact path
    - directory is relative + anchor given
                                   -> '<found_anchor>/<directory>' (fallback: '<file_parent>/<directory>')
    - directory is relative, no anchor
                                   -> '<file_parent>/<directory>'
    """
    file_parent = Path(file).resolve().parent
    if directory is None:
        target = file_parent / default_rel
    else:
        directory = Path(directory)
        if directory.is_absolute():
            target = directory
        elif anchor:
            root = find_anchor(file_parent, anchor)
            target = (root / directory) if root else (file_parent / directory)
        else:
            target = file_parent / directory
    return ensure_dir(target)



def save_fig(
    fig: mpl_figure.Figure,
    directory: str | Path,
    stem: str,
    ext: Optional[str] = "png",
    *,
    dpi: int = 300,
    bbox_inches: str | None = "tight",
    pad_inches: float | None = 0.02,
    transparent: bool = False,
    facecolor: Optional[str] = None,
    edgecolor: Optional[str] = None,
    metadata: Optional[Mapping[str, str]] = None,
    overwrite: bool = True,
) -> Path:
    """Save a Matplotlib figure to '<directory>/<stem>.<ext>'.

    Args:
        fig: matplotlib.figure.Figure
            The figure to save.
        directory: str | Path
            Output directory (created if needed).
        stem: str
            Filename stem (with or without extension).
        ext: Optional[str]
            File extension (e.g., 'png', 'pdf', 'svg'). If None, use stem’s extension or 'png'.
        dpi: int
            Dots per inch for raster formats.
        bbox_inches: str | None
            Bounding box option (e.g., 'tight'). Set to None to disable.
        pad_inches: float | None
            Padding when using bbox_inches='tight'.
        transparent: bool
            Save with transparent background.
        facecolor, edgecolor: Optional[str]
            Override face/edge colors (falls back to figure’s colors).
        metadata: Optional[Mapping[str, str]]
            Optional metadata for some backends (e.g., PDF).
        overwrite: bool
            If False and file exists, append a numeric suffix.

    Returns:
        Path
            The final path used for saving.
    """
    outfile = build_filepath(directory, stem, ext)

    if not overwrite and outfile.exists():
        base = outfile.with_suffix("")
        suffix = 1
        while True:
            candidate = base.with_name(f"{base.name}_{suffix}").with_suffix(outfile.suffix)
            if not candidate.exists():
                outfile = candidate
                break
            suffix += 1

    save_kwargs = {
        "dpi": dpi,
        "transparent": transparent,
        "metadata": metadata or {},
    }
    # Only pass bbox/pad if specified
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        save_kwargs["pad_inches"] = pad_inches
    if facecolor is not None:
        save_kwargs["facecolor"] = facecolor
    if edgecolor is not None:
        save_kwargs["edgecolor"] = edgecolor

    fig.savefig(outfile, **save_kwargs)
    return outfile

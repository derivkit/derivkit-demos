from __future__ import annotations
from pathlib import Path
from typing import Mapping, Optional
import re
import matplotlib.figure as mpl_figure

__all__ = [
    "ensure_dir", "sanitize_stem", "build_filepath",
    "find_anchor", "resolve_outdir", "save_fig",
]

_SAFE_STEM = re.compile(r"[^A-Za-z0-9._\\-]+")


def ensure_dir(directory: str | Path) -> Path:
    """Creates directory if it doesn't exist and return it as Path.

    Args:
        directory:
            Target directory path.

    Returns:
        Path
    """
    d = Path(directory).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def sanitize_stem(stem: str) -> str:
    """Make a filename stem filesystem-friendly (keeps letters, digits, ._-).

    Args:
        stem:
            Input filename stem.

    Returns:
        Sanitized filename stem.
    """
    s = stem.strip()
    s = _SAFE_STEM.sub("_", s)
    # Avoid empty or dot-only names
    return s or "figure"


def build_filepath(
    directory: str | Path,
    stem: str,
    ext: Optional[str] = "png",
) -> Path:
    """Builds a full file path from directory, stem, and extension.
    Args:
        directory:
            Target directory (created if needed).
        stem:
            Filename stem (with or without extension).
        ext:
            File extension (e.g., 'png', 'pdf', 'svg'). If None,
            use stem’s extension or default to 'png'.
    Returns:
        Path
            Full file path.
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
    """Function that resolves an output directory based on various inputs.

    Decide a save directory:
        - directory is None: '<file_parent>/<default_rel>'
        - directory is absolute: that exact path
        - directory is relative + anchor given: '<found_anchor>/<directory>' (fallback: '<file_parent>/<directory>')
        - directory is relative, no anchor:'<file_parent>/<directory>'

    Args:
        directory:
            Desired output directory (absolute or relative to anchor/file).
        file:
            Reference file path (used to find parent or anchor).
        anchor:
            Optional anchor directory name to search for upward from file.
        default_rel:
            Default relative path from file parent if directory is None.

    Returns:
        Path
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
        fig:
            The figure to save.
        directory:
            Output directory (created if needed).
        stem:
            Filename stem (with or without extension).
        ext:
            File extension (e.g., 'png', 'pdf', 'svg'). If None, use stem’s extension or 'png'.
            Default is 'png'.
        dpi:
            Dots per inch for raster formats. Default is 300.
        bbox_inches:
            Bounding box option (e.g., 'tight'). Set to None to disable. Default is 'tight'.
        pad_inches:
            Padding when using bbox_inches='tight'. Default is 0.02 inch. Set to None to disable.
        transparent:
            Save with transparent background.
        facecolor:
            Override face colors (falls back to figure’s colors).
        edgecolor:
            Override edge colors (falls back to figure’s colors).
        metadata:
            Optional metadata for some backends (e.g., PDF).
        overwrite:
            If False and file exists, append a numeric suffix. Default is True.

    Returns:
        Path
            The final path used for saving.
            :param edgecolor:
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

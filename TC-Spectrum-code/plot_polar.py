"""
plot_polar.py

Plot variables from the storm-centered polar-grid file read by
polar_grid_io.py:

  * polar slice (azimuth x radius) at the vertical level nearest z_target
  * radius-height cross-section (azimuthal mean) over the whole column

Run from a shell:

    PYTHONPATH=./TC-Spectrum-code \      
    python plot_polar.py vars_polar_8099_LES.dat \ 
      --grid les --var theta --dr 100 --dz 125 --z-target 500 \
      --save LES_theta_500m.png

    python PATH/TO/TC-Spectrum-code/polar_grid_io.py vars_polar_8099_LES.dat --grid les

If your file is big-endian, add --bswap (same as the reader).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from polar_grid_io import read_co, fields, VAR_NAMES, NVARS

# Sensible colormap + label per variable
_VAR_META = {
    "rho":   ("viridis", r"$\rho$ (kg m$^{-3}$)"),
    "radl":  ("RdBu_r",  "radial wind (m s$^{-1}$)"),
    "tang":  ("RdBu_r",  "tangential wind (m s$^{-1}$)"),
    "vert":  ("RdBu_r",  "vertical velocity (m s$^{-1}$)"),
    "theta": ("plasma",  r"$\theta$ (K)"),
}


def _var_index(var):
    if isinstance(var, int):
        return var
    if var in VAR_NAMES:
        return VAR_NAMES.index(var)
    raise ValueError(f"unknown var {var!r}; expected one of {VAR_NAMES} or an int")


def _level_index(nz, z_target, dz):
    """Cell-center heights z_k = (k + 0.5) * dz; return nearest k in [0, nz-1]."""
    k = int(round(z_target / dz - 0.5))
    return int(np.clip(k, 0, nz - 1))


def _color_limits(arr, vmin=None, vmax=None, diverging=False):
    fin = arr[np.isfinite(arr)]
    if fin.size == 0:
        raise ValueError("slice has no finite values -- try bswap=True in the reader")
    lo = np.quantile(fin, 0.02) if vmin is None else float(vmin)
    hi = np.quantile(fin, 0.98) if vmax is None else float(vmax)
    if lo == hi:
        hi = lo + 1e-12
    if diverging:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    return lo, hi


def plot_slice(co, var="theta", *, dr=1.0, r0=0.0, dz=125.0, z_target=500.0,
               theta_zero="N", clockwise=True, cmap=None, vmin=None, vmax=None,
               ax=None, title=None):
    """Polar pcolormesh of one variable at the level nearest z_target."""
    v = _var_index(var)
    np_, nl, nz, _ = co.shape
    k = _level_index(nz, z_target, dz)
    slice2d = np.asarray(co[:, :, k, v], dtype=np.float64)   # (np, nl)

    # cell edges so shading="flat" closes the circle cleanly
    r_edges = r0 + np.arange(np_ + 1) * float(dr)
    az_edges = np.linspace(0.0, 2 * np.pi, nl + 1)
    AZ, R = np.meshgrid(az_edges, r_edges)                   # (np+1, nl+1)

    name = VAR_NAMES[v] if v < len(VAR_NAMES) else f"var{v+1}"
    meta_cmap, label = _VAR_META.get(name, ("viridis", name))
    if cmap is None:
        cmap = meta_cmap
    diverging = name in ("radl", "tang", "vert")
    lo, hi = _color_limits(slice2d, vmin, vmax, diverging=diverging)

    created = ax is None
    if created:
        fig = plt.figure(figsize=(7.5, 6.5))
        ax = fig.add_subplot(projection="polar")
    else:
        fig = ax.figure

    pc = ax.pcolormesh(AZ, R, slice2d, shading="flat",
                       cmap=cmap, vmin=lo, vmax=hi)
    ax.set_theta_zero_location(theta_zero)
    ax.set_theta_direction(-1 if clockwise else 1)
    ax.set_rlabel_position(135)

    cb = fig.colorbar(pc, ax=ax, pad=0.10, shrink=0.85)
    cb.set_label(label)
    z_here = (k + 0.5) * dz
    ax.set_title(title or f"{name} at z ~ {z_here:.0f} m  (k={k+1})")
    return fig, ax, k


def plot_rz(co, var="tang", *, dr=1.0, r0=0.0, dz=125.0,
            cmap=None, vmin=None, vmax=None, ax=None, title=None):
    """Radius-height cross section of the azimuthal mean of one variable."""
    v = _var_index(var)
    np_, nl, nz, _ = co.shape
    rz = np.nanmean(np.asarray(co[:, :, :, v], dtype=np.float64), axis=1)  # (np, nz)

    r_centers = r0 + (np.arange(np_) + 0.5) * float(dr)
    z_centers = (np.arange(nz) + 0.5) * float(dz)
    # transpose so y=height, x=radius
    Z, R = np.meshgrid(z_centers, r_centers)

    name = VAR_NAMES[v] if v < len(VAR_NAMES) else f"var{v+1}"
    meta_cmap, label = _VAR_META.get(name, ("viridis", name))
    if cmap is None:
        cmap = meta_cmap
    diverging = name in ("radl", "tang", "vert")
    lo, hi = _color_limits(rz, vmin, vmax, diverging=diverging)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    pc = ax.pcolormesh(R, Z, rz, shading="auto", cmap=cmap, vmin=lo, vmax=hi)
    ax.set_xlabel("radius (m)")
    ax.set_ylabel("height (m)")
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label(label)
    ax.set_title(title or f"azimuthal mean of {name}")
    return fig, ax


def plot_panel(co, *, dr=1.0, r0=0.0, dz=125.0, z_target=500.0,
               theta_zero="N", clockwise=True):
    """All 5 variables as polar slices in a 2x3 grid at one level."""
    n = min(co.shape[3], NVARS)
    fig = plt.figure(figsize=(15, 9))
    last_k = None
    for i in range(n):
        ax = fig.add_subplot(2, 3, i + 1, projection="polar")
        _, _, last_k = plot_slice(co, var=i, dr=dr, r0=r0, dz=dz,
                                  z_target=z_target, theta_zero=theta_zero,
                                  clockwise=clockwise, ax=ax)
    z_here = (last_k + 0.5) * dz if last_k is not None else z_target
    fig.suptitle(f"polar grid at z ~ {z_here:.0f} m  (k={ (last_k or 0)+1 })",
                 fontsize=14)
    fig.tight_layout()
    return fig


# --- CLI ---------------------------------------------------------------------

def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("filepath", help="path to test.dat")
    p.add_argument("--grid", choices=("les", "meso"),
                   help="preset dims: les=(1000,1800,160), meso=(45,360,160)")
    p.add_argument("--np", dest="np_", type=int)
    p.add_argument("--nl", type=int)
    p.add_argument("--nz", type=int)
    p.add_argument("--nvars", type=int, default=NVARS)
    p.add_argument("--bswap", action="store_true",
                   help="byte-swap (file was written big-endian)")

    p.add_argument("--kind", choices=("polar", "rz", "panel"), default="polar")
    p.add_argument("--var", default="theta",
                   help="variable name (rho|radl|tang|vert|theta) or 1-based index")
    p.add_argument("--dr", type=float, default=1.0, help="radial spacing (m)")
    p.add_argument("--r0", type=float, default=0.0, help="innermost radius (m)")
    p.add_argument("--dz", type=float, default=125.0, help="vertical spacing (m)")
    p.add_argument("--z-target", type=float, default=500.0,
                   help="target height (m) for polar slice")
    p.add_argument("--vmin", type=float)
    p.add_argument("--vmax", type=float)
    p.add_argument("--cmap")
    p.add_argument("--theta-zero", default="N", choices=("N", "E", "S", "W"))
    p.add_argument("--ccw", action="store_true",
                   help="azimuth increases counter-clockwise (default: clockwise)")
    p.add_argument("--save", help="save PNG to this path instead of showing")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    if args.grid == "les":
        dims = dict(np_=1000, nl=1800, nz=160)
    elif args.grid == "meso":
        dims = dict(np_=45, nl=360, nz=160)
    else:
        missing = [n for n, v in (("--np", args.np_), ("--nl", args.nl),
                                  ("--nz", args.nz)) if v is None]
        if missing:
            p.error("supply --grid, or all of --np --nl --nz; missing "
                    + ", ".join(missing))
        dims = dict(np_=args.np_, nl=args.nl, nz=args.nz)

    co = read_co(args.filepath, **dims, nvars=args.nvars, bswap=args.bswap)

    # accept either name or 1-based index from the CLI
    var = args.var
    try:
        var = int(var) - 1
    except ValueError:
        pass

    if args.kind == "polar":
        fig, _, _ = plot_slice(co, var=var, dr=args.dr, r0=args.r0, dz=args.dz,
                               z_target=args.z_target,
                               theta_zero=args.theta_zero,
                               clockwise=not args.ccw,
                               cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)
    elif args.kind == "rz":
        fig, _ = plot_rz(co, var=var, dr=args.dr, r0=args.r0, dz=args.dz,
                         cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)
    elif args.kind == "panel":
        fig = plot_panel(co, dr=args.dr, r0=args.r0, dz=args.dz,
                         z_target=args.z_target,
                         theta_zero=args.theta_zero,
                         clockwise=not args.ccw)

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    _main()

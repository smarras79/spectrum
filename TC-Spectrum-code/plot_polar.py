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

    # tangential wind r-z cross section (azimuthal mean), LES grid
    python plot_polar.py test.dat --grid les --var tang --kind rz \
        --dr 100 --dz 125 --save tang_rz.png

    # multi-panel: pick the variables you want, one panel each
    python plot_polar.py test.dat --grid les \
        --var theta tang vert --kind polar \
        --z-target 500 --dr 100 --dz 125 --save LES_panel_500m.png

    # multi-panel: all 5 variables at one level
    python plot_polar.py test.dat --grid meso --kind panel \
        --z-target 500 --dr 2000 --dz 125 --save vars_500m.png

#
# SPECTRA
#
1) Default: single level nearest --z-target
# spectrum at ~500 m only (default)
PYTHONPATH=./TC-Spectrum-code python plot_polar.py vars_polar_8099_LES.dat \
  --grid les --kind spec-both --dr 100 --dz 125 --z-target 500 \
  --save LES_spec_500m.png

2)Pass --vertical to average per-level spectra across the column:
2a)
# average per-level spectra over all 160 vertical levels
PYTHONPATH=./TC-Spectrum-code python plot_polar.py vars_polar_8099_LES.dat \
  --grid les --kind spec-both --dr 100 --dz 125 --vertical \
  --save LES_spec_vertmean.png

2b)
# average only between 0 and 2000 m
PYTHONPATH=./TC-Spectrum-code python plot_polar.py vars_polar_8099_LES.dat \
  --grid les --kind spec-both --dr 100 --dz 125 --vertical \
  --z-min 0 --z-max 2000 \
  --save LES_spec_vertmean_0_2km.png

3) Radial FFT averaged over azimuth (sampled every 1 degree) and over
   heights from z = 0 to 5 km (every level):
PYTHONPATH=./TC-Spectrum-code python plot_polar.py vars_polar_8099_LES.dat \
  --grid les --kind spec-r --dr 100 --dz 125 \
  --vertical --z-min 0 --z-max 5000 --az-step 1.0 \
  --save LES_spec_r_1deg_0_5km.png

   Or use the equivalent preset flag (sets --vertical, --z-min 0,
   --z-max 5000, --az-step 1.0 in one shot):
PYTHONPATH=./TC-Spectrum-code python plot_polar.py vars_polar_8099_LES.dat \
  --grid les --kind spec-r --dr 100 --dz 125 --avg-r-1deg-0to5km \
  --save LES_spec_r_1deg_0_5km.png


4) Overlay LES and MESO spectra on the same plot:
# vertical-mean spectrum for both LES and MESO between 0 and 2 km
PYTHONPATH=./TC-Spectrum-code python plot_polar.py vars_polar_8099_LES.dat \
  --grid les --kind spec-both --dr 100 --dz 125 --vertical \
  --z-min 0 --z-max 2000 \
  --overlay-file vars_polar_8099_MESO.dat --overlay-grid meso \
  --overlay-dr 2000 \
  --save LES_vs_MESO_spec_0_2km.png

If your file is big-endian, add --bswap (same as the reader).
"""

from __future__ import annotations

import argparse
import math
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


def _grid_shape(n):
    """Pick a near-square (nrows, ncols) layout that fits n panels."""
    ncols = max(1, math.ceil(math.sqrt(n)))
    nrows = max(1, math.ceil(n / ncols))
    return nrows, ncols


def _resolve_var_list(co, var_list):
    """None/empty -> all available; otherwise resolve names/ints to 0-based indices."""
    if not var_list:
        return list(range(min(co.shape[3], NVARS)))
    return [_var_index(v) for v in var_list]


def plot_panel(co, var_list=None, *, dr=1.0, r0=0.0, dz=125.0, z_target=500.0,
               theta_zero="N", clockwise=True):
    """Polar slices of one or more variables in a near-square grid at one level.

    var_list : iterable of names ("rho","radl","tang","vert","theta") or 0-based
               indices; None / empty -> all 5.
    """
    indices = _resolve_var_list(co, var_list)
    n = len(indices)
    nrows, ncols = _grid_shape(n)
    fig = plt.figure(figsize=(5.0 * ncols, 4.5 * nrows))
    last_k = None
    for i, vi in enumerate(indices):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="polar")
        _, _, last_k = plot_slice(co, var=vi, dr=dr, r0=r0, dz=dz,
                                  z_target=z_target, theta_zero=theta_zero,
                                  clockwise=clockwise, ax=ax)
    z_here = (last_k + 0.5) * dz if last_k is not None else z_target
    fig.suptitle(f"polar grid at z ~ {z_here:.0f} m  (k={(last_k or 0) + 1})",
                 fontsize=14)
    fig.tight_layout()
    return fig


def plot_rz_panel(co, var_list=None, *, dr=1.0, r0=0.0, dz=125.0):
    """Radius-height (azimuthal mean) panels for one or more variables."""
    indices = _resolve_var_list(co, var_list)
    n = len(indices)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows),
                             squeeze=False)
    for i, vi in enumerate(indices):
        r, c = divmod(i, ncols)
        plot_rz(co, var=vi, dr=dr, r0=r0, dz=dz, ax=axes[r][c])
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle("azimuthal-mean radius-height cross sections", fontsize=14)
    fig.tight_layout()
    return fig


# --- TKE spectra -------------------------------------------------------------
#
# TKE per unit mass uses the three wind components on the polar grid:
#     E = 0.5 * (u_r'^2 + u_theta'^2 + w'^2)
# where primes are fluctuations about the mean along the FFT axis.
# Spectra are formed by FFT'ing each 1-D profile and averaging the resulting
# power spectra over the orthogonal direction (so we get a smoothed E(k),
# not the FFT of an averaged profile).

_VEL_INDICES = (1, 2, 3)   # radl, tang, vert


def _vel_slice(co, k_lev):
    """Return (u_r, u_t, u_w) at one vertical level as (np, nl) float64 arrays."""
    return tuple(np.asarray(co[:, :, k_lev, i], dtype=np.float64)
                 for i in _VEL_INDICES)


def _level_range(nz, dz, z_min=None, z_max=None):
    """Return (k_lo, k_hi_exclusive) for cell-center levels inside [z_min, z_max]."""
    z_centers = (np.arange(nz) + 0.5) * dz
    lo = 0 if z_min is None else int(np.searchsorted(z_centers, float(z_min), side="left"))
    hi = nz if z_max is None else int(np.searchsorted(z_centers, float(z_max), side="right"))
    lo = max(0, min(lo, nz - 1))
    hi = max(lo + 1, min(hi, nz))
    return lo, hi


def _azimuth_indices(nl, az_step_deg=None):
    """Indices into the azimuth axis that sample the circle every az_step_deg.

    nl cells span 360 degrees, so the native spacing is 360/nl deg per cell.
    None or <=0 -> use all azimuth cells.
    """
    if az_step_deg is None or az_step_deg <= 0:
        return np.arange(nl)
    native = 360.0 / nl
    stride = max(1, int(round(az_step_deg / native)))
    return np.arange(0, nl, stride)


def tke_spectrum_radial(co, *, z_target=500.0, dz=125.0, dr=1.0, window="hann",
                        vertical=False, z_min=None, z_max=None,
                        az_step_deg=None):
    """1D radial TKE spectrum, averaging per-azimuth spectra over azimuth.

    For each azimuthal line at the chosen vertical level(s) we subtract the
    radial mean (DC removal), apply a Hann window, FFT in r, sum the three
    component power spectra (* 1/2), then average the spectra across azimuth.

    Parameters
    ----------
    vertical : bool
        False (default): use the single level nearest z_target.
        True           : compute the spectrum at every level inside the
                         [z_min, z_max] window (defaults to all levels)
                         and average the resulting spectra.
    z_min, z_max : float or None
        Inclusive height range (m) when vertical=True. None -> open end.
    az_step_deg : float or None
        If given, subsample the azimuth axis to use one line every
        az_step_deg degrees (rounded to the nearest cell stride). None or
        <=0 uses every azimuthal cell.

    Returns
    -------
    k_r : ndarray, rad m^-1, length np//2 + 1
    E_r : ndarray, m^3 s^-2 (PSD of TKE per radial wavenumber)
    """
    np_, nl, nz, _ = co.shape
    if vertical:
        k_lo, k_hi = _level_range(nz, dz, z_min, z_max)
        k_levels = range(k_lo, k_hi)
    else:
        k_levels = (_level_index(nz, z_target, dz),)

    az_idx = _azimuth_indices(nl, az_step_deg)

    N = np_
    if window == "hann":
        win = np.hanning(N).reshape(N, 1)
    elif window in ("none", "boxcar", None):
        win = np.ones((N, 1))
    else:
        raise ValueError(f"unknown window {window!r}")
    win_factor = float((win ** 2).sum()) / N    # window energy correction

    acc = np.zeros(N // 2 + 1, dtype=np.float64)
    for k_lev in k_levels:
        u_r, u_t, u_w = _vel_slice(co, k_lev)
        u_r = u_r[:, az_idx]
        u_t = u_t[:, az_idx]
        u_w = u_w[:, az_idx]
        # remove mean along radial axis -> turbulent fluctuations
        u_r -= u_r.mean(axis=0, keepdims=True)
        u_t -= u_t.mean(axis=0, keepdims=True)
        u_w -= u_w.mean(axis=0, keepdims=True)

        Ur = np.fft.rfft(u_r * win, axis=0)
        Ut = np.fft.rfft(u_t * win, axis=0)
        Uw = np.fft.rfft(u_w * win, axis=0)

        psd = 0.5 * (np.abs(Ur) ** 2 + np.abs(Ut) ** 2 + np.abs(Uw) ** 2)
        psd /= (N * N * win_factor)
        psd[1:-1, :] *= 2.0                      # one-sided
        psd *= (N * dr) / (2.0 * np.pi)          # convert to E(k_r) (int E dk = <TKE>)
        acc += psd.mean(axis=1)

    E_r = acc / len(k_levels)
    k_r = 2.0 * np.pi * np.fft.rfftfreq(N, d=dr)
    return k_r, E_r


def tke_spectrum_azimuthal(co, *, z_target=500.0, dz=125.0,
                           vertical=False, z_min=None, z_max=None):
    """1D azimuthal TKE spectrum, averaging per-radius spectra over radius.

    The azimuthal direction is periodic, so no window is needed. For each
    radius at the chosen vertical level(s) we subtract the azimuthal mean,
    FFT in azimuth, sum the three component power spectra (* 1/2), then
    average across radius.

    Parameters
    ----------
    vertical : bool
        False (default): use the single level nearest z_target.
        True           : average per-level spectra over levels inside
                         [z_min, z_max] (defaults to all levels).
    """
    np_, nl, nz, _ = co.shape
    if vertical:
        k_lo, k_hi = _level_range(nz, dz, z_min, z_max)
        k_levels = range(k_lo, k_hi)
    else:
        k_levels = (_level_index(nz, z_target, dz),)

    N = nl
    acc = np.zeros(N // 2 + 1, dtype=np.float64)
    for k_lev in k_levels:
        u_r, u_t, u_w = _vel_slice(co, k_lev)
        u_r -= u_r.mean(axis=1, keepdims=True)
        u_t -= u_t.mean(axis=1, keepdims=True)
        u_w -= u_w.mean(axis=1, keepdims=True)

        Ur = np.fft.rfft(u_r, axis=1)
        Ut = np.fft.rfft(u_t, axis=1)
        Uw = np.fft.rfft(u_w, axis=1)

        psd = 0.5 * (np.abs(Ur) ** 2 + np.abs(Ut) ** 2 + np.abs(Uw) ** 2)
        psd /= (N * N)
        psd[:, 1:-1] *= 2.0                      # one-sided
        acc += psd.mean(axis=0)

    E_m = acc / len(k_levels)
    m = np.arange(N // 2 + 1)
    return m, E_m


def _add_kolmogorov(ax, x, y, exponent=-5.0 / 3.0, label=None):
    """Overlay a slope-reference line on a log-log spectrum, anchored to data."""
    mask = (x > 0) & np.isfinite(y) & (y > 0)
    if mask.sum() < 4:
        return
    xs, ys = x[mask], y[mask]
    anchor = max(1, len(xs) // 8)
    amp = ys[anchor] / xs[anchor] ** exponent
    ax.loglog(xs, amp * xs ** exponent, "k--", lw=1, alpha=0.6,
              label=label or fr"$\propto k^{{{exponent:.3g}}}$")


def _z_label(co, dz, vertical, z_target, z_min, z_max):
    """Title fragment describing the vertical aggregation in use."""
    if not vertical:
        z_here = (_level_index(co.shape[2], z_target, dz) + 0.5) * dz
        return f"z ~ {z_here:.0f} m"
    nz = co.shape[2]
    k_lo, k_hi = _level_range(nz, dz, z_min, z_max)
    z_lo = (k_lo + 0.5) * dz
    z_hi = (k_hi - 0.5) * dz
    return f"vertical mean, z = {z_lo:.0f}-{z_hi:.0f} m ({k_hi - k_lo} levels)"


def plot_spectrum_radial(co, *, z_target=500.0, dz=125.0, dr=1.0,
                         window="hann", kolmogorov=True,
                         vertical=False, z_min=None, z_max=None,
                         az_step_deg=None, ax=None, label=None, title=None):
    k_r, E_r = tke_spectrum_radial(co, z_target=z_target, dz=dz, dr=dr,
                                   window=window, vertical=vertical,
                                   z_min=z_min, z_max=z_max,
                                   az_step_deg=az_step_deg)
    created = ax is None
    fig = plt.figure(figsize=(7, 5)) if created else ax.figure
    if created:
        ax = fig.add_subplot(111)

    mask = (k_r > 0) & (E_r > 0)
    ax.loglog(k_r[mask], E_r[mask], "-o", ms=3,
              label=label or "radial TKE spectrum")
    if kolmogorov:
        _add_kolmogorov(ax, k_r, E_r, exponent=-5.0 / 3.0,
                        label=r"$k_r^{-5/3}$")
    ax.set_xlabel(r"$k_r$ (rad m$^{-1}$)")
    ax.set_ylabel(r"$E(k_r)$ (m$^{3}$ s$^{-2}$)")
    nl = co.shape[1]
    n_az = len(_azimuth_indices(nl, az_step_deg))
    az_note = (f"every {az_step_deg:g} deg, {n_az} lines"
               if az_step_deg else f"all {nl} lines")
    default_title = (f"radial TKE spectrum @ "
                     f"{_z_label(co, dz, vertical, z_target, z_min, z_max)}\n"
                     f"(azimuthal mean of per-line spectra: {az_note})")
    ax.set_title(title if title is not None else default_title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_spectrum_azimuthal(co, *, z_target=500.0, dz=125.0,
                            kolmogorov=True, vertical=False,
                            z_min=None, z_max=None, ax=None,
                            label=None, title=None):
    m, E_m = tke_spectrum_azimuthal(co, z_target=z_target, dz=dz,
                                    vertical=vertical, z_min=z_min, z_max=z_max)
    created = ax is None
    fig = plt.figure(figsize=(7, 5)) if created else ax.figure
    if created:
        ax = fig.add_subplot(111)

    mask = (m > 0) & (E_m > 0)
    ax.loglog(m[mask], E_m[mask], "-o", ms=3,
              label=label or "azimuthal TKE spectrum")
    if kolmogorov:
        _add_kolmogorov(ax, m.astype(float), E_m, exponent=-5.0 / 3.0,
                        label=r"$m^{-5/3}$")
    ax.set_xlabel(r"azimuthal mode number $m$")
    ax.set_ylabel(r"$E(m)$ (m$^{2}$ s$^{-2}$)")
    default_title = (f"azimuthal TKE spectrum @ "
                     f"{_z_label(co, dz, vertical, z_target, z_min, z_max)}\n"
                     "(radial mean of per-radius spectra)")
    ax.set_title(title if title is not None else default_title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_spectrum_both(co, *, z_target=500.0, dz=125.0, dr=1.0,
                       window="hann", kolmogorov=True,
                       vertical=False, z_min=None, z_max=None,
                       az_step_deg=None, ax_r=None, ax_az=None,
                       label=None, title_r=None, title_az=None):
    if ax_r is None or ax_az is None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax_r, ax_az = axes[0], axes[1]
    else:
        fig = ax_r.figure
    plot_spectrum_radial(co, z_target=z_target, dz=dz, dr=dr,
                         window=window, kolmogorov=kolmogorov,
                         vertical=vertical, z_min=z_min, z_max=z_max,
                         az_step_deg=az_step_deg,
                         ax=ax_r, label=label, title=title_r)
    plot_spectrum_azimuthal(co, z_target=z_target, dz=dz,
                            kolmogorov=kolmogorov,
                            vertical=vertical, z_min=z_min, z_max=z_max,
                            ax=ax_az, label=label, title=title_az)
    fig.tight_layout()
    return fig, ax_r, ax_az


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

    p.add_argument("--kind",
                   choices=("polar", "rz", "panel",
                            "spec-r", "spec-az", "spec-both"),
                   default="polar",
                   help="polar slice, r-z cross section, multi-var panel, "
                        "radial TKE spectrum, azimuthal TKE spectrum, or both.")
    p.add_argument("--window", choices=("hann", "none"), default="hann",
                   help="window for the radial FFT (non-periodic axis); "
                        "azimuthal FFT is always unwindowed (periodic).")
    p.add_argument("--no-kolmogorov", action="store_true",
                   help="omit the -5/3 reference line on spectrum plots.")
    p.add_argument("--vertical", action="store_true",
                   help="spectrum plots only: instead of the single level "
                        "nearest --z-target (default), compute the spectrum "
                        "at every level (or every level inside [--z-min, "
                        "--z-max]) and average them.")
    p.add_argument("--z-min", type=float, default=None,
                   help="lower height bound (m) for --vertical averaging.")
    p.add_argument("--z-max", type=float, default=None,
                   help="upper height bound (m) for --vertical averaging.")
    p.add_argument("--az-step", type=float, default=None,
                   help="radial spectrum only: subsample azimuth, using one "
                        "FFT line every AZ_STEP degrees (e.g. 1.0 means one "
                        "spectrum per degree, averaged over those lines). "
                        "Default: use every azimuthal cell.")
    p.add_argument("--avg-r-1deg-0to5km", action="store_true",
                   help="preset for --kind spec-r/spec-both: take the radial "
                        "FFT, sample every 1 deg in azimuth and every level "
                        "from z=0 to z=5 km, and average those spectra. "
                        "Equivalent to --vertical --z-min 0 --z-max 5000 "
                        "--az-step 1.0.")
    p.add_argument("--var", nargs="+", default=None,
                   help="one or more variable names (rho|radl|tang|vert|theta) "
                        "and/or 1-based indices. Pass several to get a panel "
                        "figure with one subplot per variable. "
                        "If omitted: 'theta' for --kind polar, 'tang' for "
                        "--kind rz, all 5 for --kind panel.")
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

    # --- overlay (second dataset, typically LES vs MESO) -------------------
    p.add_argument("--overlay-file",
                   help="second data file whose spectrum will be overlaid "
                        "on the same axes as the primary spectrum. Use this "
                        "to plot LES and MESO spectra together. Only used "
                        "with --kind spec-r/spec-az/spec-both.")
    p.add_argument("--overlay-grid", choices=("les", "meso"),
                   help="grid preset for --overlay-file. Defaults to the "
                        "opposite of --grid (les<->meso).")
    p.add_argument("--overlay-np", dest="overlay_np_", type=int)
    p.add_argument("--overlay-nl", type=int)
    p.add_argument("--overlay-nz", type=int)
    p.add_argument("--overlay-nvars", type=int, default=NVARS)
    p.add_argument("--overlay-bswap", action="store_true",
                   help="byte-swap the overlay file (big-endian).")
    p.add_argument("--overlay-dr", type=float, default=None,
                   help="radial spacing (m) for --overlay-file. Defaults: "
                        "100 for overlay-grid=les, 2000 for overlay-grid=meso, "
                        "else falls back to --dr.")
    p.add_argument("--overlay-dz", type=float, default=None,
                   help="vertical spacing (m) for --overlay-file. "
                        "Default: --dz.")
    p.add_argument("--primary-label",
                   help="legend label for the primary spectrum. "
                        "Default: uppercased --grid (e.g. 'LES').")
    p.add_argument("--overlay-label",
                   help="legend label for the overlay spectrum. "
                        "Default: uppercased --overlay-grid (e.g. 'MESO').")
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

    if args.avg_r_1deg_0to5km:
        args.vertical = True
        if args.z_min is None:
            args.z_min = 0.0
        if args.z_max is None:
            args.z_max = 5000.0
        if args.az_step is None:
            args.az_step = 1.0

    # accept names ("theta") or 1-based indices ("5") in any combination
    def _parse(v):
        try:
            return int(v) - 1
        except ValueError:
            return v
    user_vars = [_parse(v) for v in args.var] if args.var else None

    if args.kind == "polar":
        var_list = user_vars if user_vars is not None else ["theta"]
        if len(var_list) > 1:
            fig = plot_panel(co, var_list, dr=args.dr, r0=args.r0, dz=args.dz,
                             z_target=args.z_target,
                             theta_zero=args.theta_zero,
                             clockwise=not args.ccw)
        else:
            fig, _, _ = plot_slice(co, var=var_list[0],
                                   dr=args.dr, r0=args.r0, dz=args.dz,
                                   z_target=args.z_target,
                                   theta_zero=args.theta_zero,
                                   clockwise=not args.ccw,
                                   cmap=args.cmap,
                                   vmin=args.vmin, vmax=args.vmax)
    elif args.kind == "rz":
        var_list = user_vars if user_vars is not None else ["tang"]
        if len(var_list) > 1:
            fig = plot_rz_panel(co, var_list,
                                dr=args.dr, r0=args.r0, dz=args.dz)
        else:
            fig, _ = plot_rz(co, var=var_list[0],
                             dr=args.dr, r0=args.r0, dz=args.dz,
                             cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)
    elif args.kind == "panel":
        # user_vars=None -> all 5; otherwise just the ones they listed
        fig = plot_panel(co, user_vars, dr=args.dr, r0=args.r0, dz=args.dz,
                         z_target=args.z_target,
                         theta_zero=args.theta_zero,
                         clockwise=not args.ccw)
    elif args.kind in ("spec-r", "spec-az", "spec-both"):
        # ---- resolve overlay (LES vs MESO) dataset, if any ----------------
        co_ov = None
        ov_dr = None
        ov_dz = None
        primary_label = (args.primary_label
                         or (args.grid.upper() if args.grid else "primary"))
        overlay_label = None
        if args.overlay_file:
            if args.overlay_grid:
                ov_grid = args.overlay_grid
            elif args.grid == "les":
                ov_grid = "meso"
            elif args.grid == "meso":
                ov_grid = "les"
            else:
                ov_grid = None

            if ov_grid == "les":
                ov_dims = dict(np_=1000, nl=1800, nz=160)
            elif ov_grid == "meso":
                ov_dims = dict(np_=45, nl=360, nz=160)
            else:
                ov_missing = [n for n, v in
                              (("--overlay-np", args.overlay_np_),
                               ("--overlay-nl", args.overlay_nl),
                               ("--overlay-nz", args.overlay_nz)) if v is None]
                if ov_missing:
                    p.error("supply --overlay-grid, or all of --overlay-np "
                            "--overlay-nl --overlay-nz; missing "
                            + ", ".join(ov_missing))
                ov_dims = dict(np_=args.overlay_np_, nl=args.overlay_nl,
                               nz=args.overlay_nz)

            co_ov = read_co(args.overlay_file, **ov_dims,
                            nvars=args.overlay_nvars, bswap=args.overlay_bswap)

            if args.overlay_dr is not None:
                ov_dr = args.overlay_dr
            elif ov_grid == "les":
                ov_dr = 100.0
            elif ov_grid == "meso":
                ov_dr = 2000.0
            else:
                ov_dr = args.dr
            ov_dz = args.overlay_dz if args.overlay_dz is not None else args.dz

            overlay_label = (args.overlay_label
                             or (ov_grid.upper() if ov_grid else "overlay"))

        overlay_title_r = ("radial TKE spectrum: "
                           f"{primary_label} vs {overlay_label}"
                           if co_ov is not None else None)
        overlay_title_az = ("azimuthal TKE spectrum: "
                            f"{primary_label} vs {overlay_label}"
                            if co_ov is not None else None)

        if args.kind == "spec-r":
            fig, ax = plot_spectrum_radial(co, z_target=args.z_target,
                                           dz=args.dz, dr=args.dr,
                                           window=args.window,
                                           kolmogorov=not args.no_kolmogorov,
                                           vertical=args.vertical,
                                           z_min=args.z_min, z_max=args.z_max,
                                           az_step_deg=args.az_step,
                                           label=primary_label,
                                           title=overlay_title_r)
            if co_ov is not None:
                plot_spectrum_radial(co_ov, z_target=args.z_target,
                                     dz=ov_dz, dr=ov_dr,
                                     window=args.window,
                                     kolmogorov=False,
                                     vertical=args.vertical,
                                     z_min=args.z_min, z_max=args.z_max,
                                     az_step_deg=args.az_step,
                                     ax=ax, label=overlay_label,
                                     title=overlay_title_r)
        elif args.kind == "spec-az":
            fig, ax = plot_spectrum_azimuthal(co, z_target=args.z_target,
                                              dz=args.dz,
                                              kolmogorov=not args.no_kolmogorov,
                                              vertical=args.vertical,
                                              z_min=args.z_min,
                                              z_max=args.z_max,
                                              label=primary_label,
                                              title=overlay_title_az)
            if co_ov is not None:
                plot_spectrum_azimuthal(co_ov, z_target=args.z_target,
                                        dz=ov_dz,
                                        kolmogorov=False,
                                        vertical=args.vertical,
                                        z_min=args.z_min, z_max=args.z_max,
                                        ax=ax, label=overlay_label,
                                        title=overlay_title_az)
        else:  # spec-both
            fig, ax_r, ax_az = plot_spectrum_both(
                co, z_target=args.z_target,
                dz=args.dz, dr=args.dr,
                window=args.window,
                kolmogorov=not args.no_kolmogorov,
                vertical=args.vertical,
                z_min=args.z_min, z_max=args.z_max,
                az_step_deg=args.az_step,
                label=primary_label,
                title_r=overlay_title_r,
                title_az=overlay_title_az)
            if co_ov is not None:
                plot_spectrum_both(co_ov, z_target=args.z_target,
                                   dz=ov_dz, dr=ov_dr,
                                   window=args.window,
                                   kolmogorov=False,
                                   vertical=args.vertical,
                                   z_min=args.z_min, z_max=args.z_max,
                                   az_step_deg=args.az_step,
                                   ax_r=ax_r, ax_az=ax_az,
                                   label=overlay_label,
                                   title_r=overlay_title_r,
                                   title_az=overlay_title_az)

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    _main()

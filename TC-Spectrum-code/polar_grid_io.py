"""
polar_grid_io.py

Read/inspect the storm-centered polar-grid binary file produced by the
Fortran code:

    open(15, file=TRIM(directory)//'test.dat', form='unformatted', &
         access='stream', status='unknown')
    ! co(np, nl, nz, 5), single precision
    ! order: rho, radl, tang, vert, theta
    write(15) co

Because access='stream', the file is a raw byte stream of Float32 values
with NO record-length markers. Fortran is column-major, so we read with
numpy and reshape with order='F' to get axes (np, nl, nz, nvars) that
match the Fortran declaration exactly.

Run from a shell:

    # MESO (2 km) grid
    python polar_grid_io.py /path/to/test.dat --grid meso

    # LES grid
    python polar_grid_io.py /path/to/test.dat --grid les

    # Custom dims, with byte-swap if the data was written big-endian
    python polar_grid_io.py test.dat --np 45 --nl 360 --nz 160 --nvars 5 --bswap

Or import from another script:

    from polar_grid_io import read_co, fields, inspect_co
    co = read_co("test.dat", np_=45, nl=360, nz=160)         # shape (45, 360, 160, 5)
    f  = fields(co)
    f["theta"][:, :, k]                                       # 2D slice at level k
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

NVARS = 5
VAR_NAMES = ("rho", "radl", "tang", "vert", "theta")


def read_co(filepath, *, np_, nl, nz, nvars=NVARS, bswap=False):
    """Read co(np, nl, nz, nvars) Float32 from a Fortran stream-access file.

    Parameters
    ----------
    filepath : str
    np_, nl, nz : int          grid dimensions
    nvars       : int          trailing variable count (default 5)
    bswap       : bool         flip endianness if the file was written
                               on the opposite byte order

    Returns
    -------
    np.ndarray, shape (np_, nl, nz, nvars), dtype float32
    """
    expected = np_ * nl * nz * nvars * 4
    actual = os.path.getsize(filepath)
    if actual != expected:
        raise ValueError(
            f"File size mismatch for {filepath!r}:\n"
            f"  expected {expected} bytes (np={np_}, nl={nl}, nz={nz}, "
            f"{nvars} vars, Float32)\n"
            f"  found    {actual} bytes\n"
            "Check grid dims, variable count, precision, or whether the file\n"
            "really is stream access (a record-marked file would be 8 bytes larger)."
        )

    dtype = np.dtype("<f4") if not bswap else np.dtype(">f4")
    flat = np.fromfile(filepath, dtype=dtype)
    # Fortran column-major: leftmost index varies fastest. order='F' on reshape
    # interprets the flat buffer with that layout, matching `co(np,nl,nz,5)`.
    co = flat.reshape((np_, nl, nz, nvars), order="F")
    return co


def write_co(filepath, co, *, bswap=False):
    """Write co back out in the same raw Fortran stream-access format."""
    if co.dtype != np.float32:
        co = co.astype(np.float32)
    dtype = np.dtype("<f4") if not bswap else np.dtype(">f4")
    # ravel in Fortran order so disk layout matches `write(15) co`
    co.astype(dtype, copy=False).ravel(order="F").tofile(filepath)


def fields(co):
    """Return a dict of zero-copy views: rho, radl, tang, vert, theta."""
    return {name: co[:, :, :, i] for i, name in enumerate(VAR_NAMES[: co.shape[3]])}


# --- Convenience presets ------------------------------------------------------

def read_co_LES(filepath, *, nvars=NVARS, bswap=False):
    """LES grid: np=1000, nl=1800, nz=160."""
    return read_co(filepath, np_=1000, nl=1800, nz=160, nvars=nvars, bswap=bswap)


def read_co_MESO(filepath, *, nvars=NVARS, bswap=False):
    """MESO (2 km) grid: np=45, nl=360, nz=160."""
    return read_co(filepath, np_=45, nl=360, nz=160, nvars=nvars, bswap=bswap)


# --- Diagnostic --------------------------------------------------------------

def inspect_co(filepath, *, np_, nl, nz, nvars=NVARS, bswap=False,
               z_target=500.0, dz=125.0):
    """Print per-variable stats so you can tell what is in the file.

    If every variable's min/max/mean looks like 0 or ~1e-38..1e-41, the file
    is the opposite byte order — re-run with bswap=True.
    """
    co = read_co(filepath, np_=np_, nl=nl, nz=nz, nvars=nvars, bswap=bswap)
    print(f"Read OK. shape(co) = {co.shape}   (np, nl, nz, nvars)\n")

    header = f"{'var':<6}{'name':<8}{'min':>14}{'max':>14}{'mean':>14}{'std':>14}{'n_nonfin':>12}"
    print(header)
    for v in range(nvars):
        b = co[:, :, :, v].ravel()
        finite_mask = np.isfinite(b)
        nbad = int((~finite_mask).sum())
        fin = b[finite_mask]
        name = VAR_NAMES[v] if v < len(VAR_NAMES) else f"var{v+1}"
        if fin.size == 0:
            print(f"{v+1:<6}{name:<8}{'-':>14}{'-':>14}{'-':>14}{'-':>14}{nbad:>12}")
        else:
            print(f"{v+1:<6}{name:<8}{fin.min():>14.4g}{fin.max():>14.4g}"
                  f"{fin.mean():>14.4g}{fin.std():>14.4g}{nbad:>12}")

    # Slice at the level closest to z_target (assuming cell-center heights)
    k = int(np.clip(round(z_target / dz - 0.5), 0, nz - 1))
    z_here = (k + 0.5) * dz
    print(f"\nAt vertical level k={k+1} (~{z_here:.1f} m, assuming dz={dz}):")
    print(f"{'var':<6}{'name':<8}{'min':>14}{'max':>14}{'range':>14}")
    for v in range(nvars):
        s = co[:, :, k, v].ravel()
        fin = s[np.isfinite(s)]
        name = VAR_NAMES[v] if v < len(VAR_NAMES) else f"var{v+1}"
        if fin.size == 0:
            print(f"{v+1:<6}{name:<8}{'-':>14}{'-':>14}{'-':>14}")
        else:
            lo, hi = float(fin.min()), float(fin.max())
            print(f"{v+1:<6}{name:<8}{lo:>14.4g}{hi:>14.4g}{hi-lo:>14.4g}")

    print("""
How to read this:
  * rho   ~ 0.5 - 1.3 kg/m^3
  * radl, tang ~ -60 .. 60 m/s
  * vert  ~ -20 .. 20 m/s
  * theta ~ 250 - 330 K
  * If EVERY variable looks like ~0 / ~1e-38 / ~1e-41 / many non-finites,
    the byte order is wrong: re-run with bswap=True.
""")
    return co


# --- CLI ---------------------------------------------------------------------

def _main(argv=None):
    p = argparse.ArgumentParser(
        description="Read & inspect a Fortran stream-access polar-grid .dat file."
    )
    p.add_argument("filepath", help="path to test.dat")
    p.add_argument("--grid", choices=("les", "meso"),
                   help="preset dims: les=(1000,1800,160), meso=(45,360,160)")
    p.add_argument("--np", dest="np_", type=int, help="radial points")
    p.add_argument("--nl", type=int, help="azimuthal lines")
    p.add_argument("--nz", type=int, help="vertical levels")
    p.add_argument("--nvars", type=int, default=NVARS)
    p.add_argument("--bswap", action="store_true",
                   help="byte-swap (file was written big-endian)")
    p.add_argument("--dz", type=float, default=125.0,
                   help="uniform vertical spacing in m (for the level report)")
    p.add_argument("--z-target", type=float, default=500.0,
                   help="height in m for the per-level diagnostic slice")
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

    inspect_co(args.filepath, **dims, nvars=args.nvars, bswap=args.bswap,
               z_target=args.z_target, dz=args.dz)


if __name__ == "__main__":
    _main()

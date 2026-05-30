# polar_grid_io.jl
#
# Read/write the storm-centered polar-grid binary file produced by the Fortran code:
#
#     open(15, file=TRIM(directory)//'test.dat', form='unformatted', &
#          access='stream', status='unknown')
#     ! co(np,nl,nz,5), single precision
#     ! order: rho, radl, tang, vert, theta
#     write(15) co
#
# Because access='stream', the file is a raw byte stream of Float32 values with
# NO record-length markers. Fortran and Julia are both column-major, so the
# bytes drop straight into an Array{Float32} with identical dimension order.

const NVARS = 5   # default variable count (rho, radl, tang, vert, theta)

"""
    read_co(filepath; np, nl, nz, nvars=5, bswap=false) -> Array{Float32,4}

Read the polar-grid array `co(np, nl, nz, nvars)` (single precision) from a
Fortran stream-access binary file.

Keyword arguments:
- `np`, `nl`, `nz` : grid dimensions (radial points, azimuthal lines, vertical levels).
- `nvars`          : number of variables in the trailing dimension (default 5).
- `bswap`          : set `true` if the file was written on a machine with the
                     opposite byte order (e.g. big-endian data read on little-endian).

The file size is validated against the expected size; a mismatch usually means
the grid dimensions, the variable count, the precision, or the stream-vs-record
assumption is wrong.
"""
function read_co(filepath::AbstractString; np::Int, nl::Int, nz::Int, nvars::Int=NVARS, bswap::Bool=false)
    expected_bytes = np * nl * nz * nvars * sizeof(Float32)
    actual_bytes   = filesize(filepath)
    if actual_bytes != expected_bytes
        error("""
              File size mismatch for "$filepath":
                expected $(expected_bytes) bytes  (np=$np, nl=$nl, nz=$nz, $nvars vars, Float32)
                found    $(actual_bytes) bytes
              Check the grid dimensions, the variable count (nvars), the precision,
              or whether the file really is stream access (a record-marked file
              would be 8 bytes larger).
              """)
    end

    co = Array{Float32}(undef, np, nl, nz, nvars)
    open(filepath, "r") do io
        read!(io, co)            # column-major fill, matching Fortran
    end

    if bswap
        co .= Base.bswap.(co)    # flip endianness in place if requested
    end
    return co
end

"""
    write_co(filepath, co; bswap=false)

Write a `co(np, nl, nz, 5)` Float32 array back out in the same raw Fortran
stream-access format (no record markers). Equivalent to `write(15) co`.
"""
function write_co(filepath::AbstractString, co::AbstractArray{Float32,4}; bswap::Bool=false)
    open(filepath, "w") do io
        if bswap
            write(io, Base.bswap.(co))
        else
            write(io, co)
        end
    end
    return nothing
end

# --- Convenience presets for the two grids -----------------------------------

"Read the LES grid: np=1000, nl=1800, nz=160."
read_co_LES(filepath; nvars::Int=NVARS, bswap::Bool=false)  = read_co(filepath; np=1000, nl=1800, nz=160, nvars=nvars, bswap=bswap)

"Read the MESO (2 km) grid: np=45, nl=360, nz=160."
read_co_MESO(filepath; nvars::Int=NVARS, bswap::Bool=false) = read_co(filepath; np=45,   nl=360,  nz=160, nvars=nvars, bswap=bswap)

# --- Named views into the 5 variables ----------------------------------------

"""
    fields(co) -> NamedTuple

Return zero-copy views of the five variables along the last dimension:
`rho`, `radl` (radial wind), `tang` (tangential wind), `vert` (vertical
velocity), `theta` (potential temperature).
"""
function fields(co::AbstractArray{<:Real,4})
    return (
        rho   = @view(co[:, :, :, 1]),
        radl  = @view(co[:, :, :, 2]),
        tang  = @view(co[:, :, :, 3]),
        vert  = @view(co[:, :, :, 4]),
        theta = @view(co[:, :, :, 5]),
    )
end

# --- Example usage ------------------------------------------------------------
#
#   include("polar_grid_io.jl")
#
#   directory = "/path/to/data/"
#   co = read_co_LES(joinpath(directory, "test.dat"))     # or read_co_MESO(...)
#
#   f = fields(co)
#   f.rho, f.radl, f.tang, f.vert, f.theta                # views, no copy
#
#   # Or read with explicit dimensions:
#   co = read_co(joinpath(directory, "test.dat"); np=1000, nl=1800, nz=160)

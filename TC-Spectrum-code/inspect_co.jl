# inspect_co.jl
#
# Print per-variable statistics so we can see what is actually in the file:
#   - which variable slot holds theta (potential temperature ~ 250-320 K)
#   - whether the data was read correctly at all
#   - whether there are fill/sentinel values or an endianness problem
#
#   include("inspect_co.jl")
#   co = inspect_co("./vars_polar_8100_2km.dat"; np=45, nl=360, nz=160, nvars=9)
#
# If the printed numbers look like nonsense (e.g. ~1e-38, ~1e38, lots of
# non-finite values), the file is the opposite byte order: re-run with bswap=true.

include("polar_grid_io.jl")
using Statistics
using Printf

function inspect_co(filepath::AbstractString; np::Int, nl::Int, nz::Int,
                    nvars::Int, bswap::Bool=false, z_target::Real=500.0, dz::Real=125.0)
    co = read_co(filepath; np=np, nl=nl, nz=nz, nvars=nvars, bswap=bswap)
    println("Read OK. size(co) = ", size(co), "   (np, nl, nz, nvars)\n")

    @printf("%-5s %14s %14s %14s %14s %10s\n", "var", "min", "max", "mean", "std", "n_nonfin")
    for v in 1:nvars
        b = vec(co[:, :, :, v])
        fin = filter(isfinite, b)
        nbad = length(b) - length(fin)
        if isempty(fin)
            @printf("%-5d %14s %14s %14s %14s %10d\n", v, "—", "—", "—", "—", nbad)
        else
            @printf("%-5d %14.4g %14.4g %14.4g %14.4g %10d\n",
                    v, minimum(fin), maximum(fin), mean(fin), std(fin), nbad)
        end
    end

    # Look at the ~500 m level of each variable to find which one varies there
    k = clamp(round(Int, z_target / dz + 0.5), 1, nz)
    println("\nAt vertical level k=$k (≈ $(round((k-0.5)*dz; digits=1)) m if dz=$dz):")
    @printf("%-5s %14s %14s %14s\n", "var", "min", "max", "range")
    for v in 1:nvars
        s = vec(co[:, :, k, v])
        fin = filter(isfinite, s)
        if isempty(fin)
            @printf("%-5d %14s %14s %14s\n", v, "—", "—", "—")
        else
            lo, hi = minimum(fin), maximum(fin)
            @printf("%-5d %14.4g %14.4g %14.4g\n", v, lo, hi, hi - lo)
        end
    end

    println("""

    How to read this:
      * theta should look like potential temperature: roughly 250-330 K, with a
        nonzero range at the 500 m level. Whichever var matches is your var_index.
      * rho ~ 0.5-1.3 kg/m^3; winds (radl/tang/vert) span roughly -60..60 m/s.
      * If a variable has range = 0 at the level, it is constant -> that is what
        produced the uniform circle.
      * If EVERY variable looks like junk (~1e-38 / ~1e38 / many non-finite),
        the byte order is wrong: re-run inspect_co(...; bswap=true).
    """)
    return co
end

# plot_theta_slice.jl
#
# Plot a 2D slice of potential temperature (theta) on the storm-centered polar
# grid at the vertical level nearest a target height (default ~500 m).
#
# Depends on:
#   - polar_grid_io.jl   (the reader from before)
#   - PythonPlot.jl      (matplotlib; the standard tool for polar pcolormesh)
#
# Install the plotting backend once:
#   import Pkg; Pkg.add("PythonPlot")
#
# matplotlib's polar projection is used because it tiles a (radius × azimuth)
# field directly with the correct geometry and closes the circle cleanly.

include("polar_grid_io.jl")
using PythonPlot
using Statistics

"""
    plot_theta_slice(co; var_index=5, z=nothing, dz=nothing, z_target=500.0,
                     dr=1.0, r0=0.0, theta_zero="N", clockwise=true,
                     cmap="viridis", savepath=nothing)

Plot the variable at `var_index` (theta) on the polar grid at the vertical level
closest to `z_target` metres. Set `var_index` to whichever slot theta occupies
in your file's variable ordering.

You MUST supply the vertical grid one of two ways:
  - `z`  : a length-`nz` vector of level heights in metres, or
  - `dz` : a uniform spacing in metres (cell-center heights are assumed:
           z = (0.5, 1.5, ...) * dz; change if your levels sit at faces).

Radial geometry (only affects the radial axis scaling, not the data):
  - `dr` : radial spacing in metres (e.g. 2000.0 for the 2 km MESO grid).
  - `r0` : radius of the innermost grid point in metres (default 0).

Display options:
  - `theta_zero` : compass location of 0° ("N", "E", "S", "W").
  - `clockwise`  : if true, azimuth increases clockwise (meteorological convention).
  - `cmap`       : colormap (use "viridis"/"plasma" for absolute theta,
                   "RdBu_r" for anomalies).
  - `savepath`   : if given, save a PNG there.

Returns `(fig, ax, k)` where `k` is the chosen vertical index.
"""
function plot_theta_slice(co::AbstractArray{<:Real,4};
                          var_index::Int=5,
                          z::Union{Nothing,AbstractVector}=nothing,
                          dz::Union{Nothing,Real}=nothing,
                          z_target::Real=500.0,
                          dr::Real=1.0,
                          r0::Real=0.0,
                          theta_zero::AbstractString="N",
                          clockwise::Bool=true,
                          cmap::AbstractString="viridis",
                          vmin::Union{Nothing,Real}=nothing,
                          vmax::Union{Nothing,Real}=nothing,
                          savepath::Union{Nothing,AbstractString}=nothing)

    np, nl, nz, nv = size(co)
    1 <= var_index <= nv || error("var_index=$var_index out of range (file has $nv variables).")

    # --- vertical level nearest z_target -------------------------------------
    zlev = if z !== nothing
        collect(float.(z))
    elseif dz !== nothing
        collect((0.5:1.0:nz - 0.5) .* float(dz))   # cell-center heights
    else
        error("Provide either `z` (length-$nz vector of heights in m) or `dz` (uniform spacing in m).")
    end
    length(zlev) == nz || error("length(z)=$(length(zlev)) but nz=$nz.")
    k = argmin(abs.(zlev .- z_target))
    @info "Vertical level k=$k at z=$(round(zlev[k]; digits=2)) m (requested ≈ $z_target m)"

    # --- extract the slice: (np, nl) = radius × azimuth ----------------------
    theta = @view co[:, :, :, var_index]
    slice = Array{Float64}(theta[:, :, k])         # promote for plotting

    # --- polar mesh, using cell EDGES so shading="flat" closes the circle ----
    r_edges  = r0 .+ (0:np) .* float(dr)           # length np+1 (m)
    az_edges = range(0, 2π; length = nl + 1)       # length nl+1, wraps to 2π
    R  = repeat(reshape(collect(r_edges),  np + 1, 1), 1, nl + 1)   # (np+1, nl+1)
    AZ = repeat(reshape(collect(az_edges), 1, nl + 1), np + 1, 1)   # (np+1, nl+1)

    # --- robust color limits from finite data (ignores fills/outliers) -------
    fin = filter(isfinite, vec(slice))
    if isempty(fin)
        error("Slice has no finite values — likely an endianness or read problem; try bswap=true in read_co.")
    end
    if minimum(fin) == maximum(fin)
        @warn "Slice is constant (value = $(first(fin))); the plot will be a single color. Wrong var_index or level?"
    end
    vlo = vmin === nothing ? quantile(fin, 0.02) : float(vmin)
    vhi = vmax === nothing ? quantile(fin, 0.98) : float(vmax)
    vlo == vhi && (vhi = vlo + eps(Float64))   # avoid degenerate scale

    # --- plot -----------------------------------------------------------------
    fig = figure(figsize = (7.5, 6.5))
    ax  = fig.add_subplot(projection = "polar")
    pc  = ax.pcolormesh(AZ, R, slice; shading = "flat", cmap = cmap, vmin = vlo, vmax = vhi)

    ax.set_theta_zero_location(theta_zero)
    ax.set_theta_direction(clockwise ? -1 : 1)
    ax.set_rlabel_position(135)

    cb = fig.colorbar(pc, ax = ax, pad = 0.10, shrink = 0.85)
    cb.set_label("Potential temperature θ (K)")
    ax.set_title("θ at z ≈ $(round(zlev[k]; digits=1)) m")

    fig.tight_layout()
    savepath !== nothing && fig.savefig(savepath; dpi = 150, bbox_inches = "tight")
    return fig, ax, k
end

# --- Example -----------------------------------------------------------------
#
#   include("plot_theta_slice.jl")
#
#   # LES grid, 2 km example uses MESO; set heights + radial spacing for yours:
#   co = read_co_LES("/path/to/data/test.dat")
#
#   # Option A: you know the level heights (best — pass the real model levels):
#   zheights = [...]                       # length 160, in metres
#   fig, ax, k = plot_theta_slice(co; z=zheights, z_target=500.0, dr=100.0)
#
#   # Option B: assume uniform spacing (e.g. dz = 125 m) and 2 km radial cells:
#   fig, ax, k = plot_theta_slice(co; dz=125.0, z_target=500.0,
#                                 dr=2000.0, savepath="theta_500m.png")
#
#   PythonPlot.show()   # if running interactively

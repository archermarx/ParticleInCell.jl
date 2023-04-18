using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_010 = mkpath("test/results/010")
const VERTICAL_RES = 1080
const FONT_SIZE = round(Int, VERTICAL_RES/600 * 12)
const PLOT_SCALING_OPTIONS = (;
    titlefontsize=FONT_SIZE*3÷2,
    legendfontsize=FONT_SIZE÷1.2,
    xtickfontsize=FONT_SIZE,
    ytickfontsize=FONT_SIZE,
    xguidefontsize=FONT_SIZE,
    yguidefontsize=FONT_SIZE,
    framestyle=:box,
)


function ion_plasma_wave(;path, mi, N=101, N_ppc=100, xmax=π, Δt=0.1, tmax=2π, amplitude=0.01, wavenumber=1, travelling = false)

    N_p = N_ppc * N
    
    ions, electrons, fields, grid = ParticleInCell.initialize(
        N_p, N, xmax;
        mi = mi * ParticleInCell.m_p
    )

    ωpi = 1/sqrt(ions.mass)

    if travelling
        wavespeed = ωpi / wavenumber 
    else
        wavespeed = 0.0
    end

    ParticleInCell.perturb!(ions, amplitude, wavenumber, wavespeed, xmax)

    results = ParticleInCell.simulate(ions, electrons, fields, grid; Δt, tmax, solve_electrons = false)
    
    (; t, x, E, ρ) = results

    contour_options = (;
        ylims = (0, 1), yticks = LinRange(0, 1, 6), linewidth = 0, right_margin = 30mm, 
        xlabel = "tωₚᵢ / 2π", ylabel = "xωₚ/2πc", cmap = :balance,
        PLOT_SCALING_OPTIONS ...
    )

    contour_E = contourf(t ./ 2π * ωpi, x ./ 2π, E; title = "δE / mcωp", contour_options...)
    contour_ρ = contourf(t ./ 2π * ωpi, x ./ 2π, ρ; title = "δn / n₀", contour_options...)

    p = plot(contour_ρ, contour_E, layout = (2, 1), size = (VERTICAL_RES, VERTICAL_RES))

    display(p)

    suffix = (travelling ? "travelling" : "standing") * "_k=$(wavenumber)" * "_mi=$(mi)"

    savefig(p, joinpath(path, "$(suffix).png"))
    return results.x, results.t, ρ, E
end

begin
    for mi in [0.005, 0.01, 0.1, 1.0, 10.0, 100.0]
        #===============================
        Test case 010: Ion plasma oscillations
        ===============================#
        ωpi = sqrt(1 / ParticleInCell.m_p / mi)
        common_options = (;
            N = 64,
            N_ppc = 64,
            xmax = 2π,
            tmax = 4π / ωpi,
            amplitude = 0.01,
            Δt = 0.01 / ωpi,
            path = RESULT_PATH_010,
            mi, 
        )    
        ion_plasma_wave(;wavenumber = 1, travelling = true, common_options...)
        ion_plasma_wave(;wavenumber = 2, travelling = true, common_options...)
        ion_plasma_wave(;wavenumber = 1, travelling = false, common_options...)
        ion_plasma_wave(;wavenumber = 2, travelling = false, common_options...)
    end
end
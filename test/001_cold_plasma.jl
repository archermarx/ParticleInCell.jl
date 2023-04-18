using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_001 = mkpath("test/results/001")
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

function cold_plasma_wave(;path, N=101, N_ppc=100, xmax=π, Δt=0.1, tmax=2π, amplitude=0.01, wavenumber=1, wave_speed=1.0, suffix)

    N_p = N_ppc * N
    
    ions, electrons, fields, grid = ParticleInCell.initialize(
        N_p, N, xmax;
        perturbation_amplitude = amplitude,
        perturbation_wavenumber = wavenumber,
        perturbation_speed = wave_speed / wavenumber,
        mi = Inf
    )

    results = ParticleInCell.simulate(ions, electrons, fields, grid; Δt, tmax)
    
    (; t, x, E, ne) = results

    contour_options = (;
        ylims = (0, 1), yticks = LinRange(0, 1, 6), linewidth = 0, right_margin = 30mm, 
        xlabel = "tωₚ / 2π", ylabel = "xωₚ/2πc", cmap = :balance,
        PLOT_SCALING_OPTIONS ...
    )

    E = results.E
    n = results.ne

    contour_E = contourf(t ./ 2π, x ./ 2π, E; title = "δE / mcωp", contour_options...)
    contour_ρ = contourf(t ./ 2π, x ./ 2π, ne .+ 1.0; title = "δn / n₀", contour_options...)

    p = plot(contour_ρ, contour_E, layout = (2, 1), size = (VERTICAL_RES, VERTICAL_RES))

    display(p)

    savefig(p, joinpath(path, "$(suffix).png"))
    return results.x, results.t, n, E
end

begin
    #===============================
    Test case 001: Simple plasma oscillations
    ===============================#
    common_options = (;
        N = 128,
        N_ppc = 64,
        xmax = 2π,
        tmax = 4π,
        amplitude = 0.01,
        Δt = 0.01,
        path = RESULT_PATH_001
    )    
    x, t, n, E = cold_plasma_wave(;suffix = "travelling_k=1", wavenumber = 1, wave_speed = 1, common_options...)
    cold_plasma_wave(;suffix = "standing_k=1", wavenumber = 1, wave_speed = 0, common_options...)
    cold_plasma_wave(;suffix = "travelling_k=2", wavenumber = 2, wave_speed = 1, common_options...)
    cold_plasma_wave(;suffix = "standing_k=2",  wavenumber = 2, wave_speed = 0, common_options...)
end
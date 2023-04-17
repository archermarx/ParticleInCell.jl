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

    particles, fields, grid = ParticleInCell.initialize(
        N_p, N, xmax;
        perturbation_amplitude = amplitude,
        perturbation_wavenumber = wavenumber,
        perturbation_speed = wave_speed / wavenumber,
    )

    num_timesteps = ceil(Int, tmax / Δt)

    E_cache = zeros(N, num_timesteps+1)
    δρ_cache = zeros(N, num_timesteps+1)
    v_cache = zeros(N_p, num_timesteps+1)
    x_cache = zeros(N_p, num_timesteps+1)

    E_cache[:, 1] = fields.Ex
    δρ_cache[:, 1] = fields.ρ .- 1
    v_cache[:, 1] = particles.vx
    x_cache[:, 1] = particles.x

    for i in 2:num_timesteps+1
        ParticleInCell.update!(particles, fields, grid, Δt)
        E_cache[:, i] = copy(fields.Ex)
        δρ_cache[:, i] = copy(fields.ρ) .- 1.0
        v_cache[:, i] = copy(particles.vx)
        x_cache[:, i] = copy(particles.x)
    end

    t = LinRange(0, tmax, num_timesteps+1) ./ 2π
    
    contour_options = (;
        ylims = (0, 1), yticks = LinRange(0, 1, 6), linewidth = 0, right_margin = 30mm, 
        xlabel = "tωₚ / 2π", ylabel = "xωₚ/2πc", cmap = :balance,
        PLOT_SCALING_OPTIONS ...
    )

    contour_E = contourf(t, grid.x ./ 2π, E_cache; title = "δE / mcωp", contour_options...)
    contour_ρ = contourf(t, grid.x ./ 2π, δρ_cache; title = "δn / n₀", contour_options...)

    p = plot(contour_ρ, contour_E, layout = (2, 1), size = (VERTICAL_RES, VERTICAL_RES))

    display(p)

    savefig(p, joinpath(path, "$(suffix).png"))
    return δρ_cache, E_cache
end

let
    #===============================
    Test case 002: Nonlinear plasma waves
    ===============================#
    # Generate plots for test case 002
    common_options = (;
        N = 128,
        N_ppc = 64,
        xmax = 2π,
        tmax = 4π,
        amplitude = 0.01,
        Δt = 0.01,
        path = RESULT_PATH_001
    )    
    cold_plasma_wave(;suffix = "travelling_k=1", wavenumber = 1, wave_speed = 1, common_options...)
    cold_plasma_wave(;suffix = "standing_k=1", wavenumber = 1, wave_speed = 0, common_options...)
    cold_plasma_wave(;suffix = "travelling_k=2", wavenumber = 2, wave_speed = 1, common_options...)
    cold_plasma_wave(;suffix = "standing_k=2",  wavenumber = 2, wave_speed = 0, common_options...)
end
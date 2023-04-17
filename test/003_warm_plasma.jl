using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_003 = mkpath("test/results/003")
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

function warm_plasma_waves(;path, quiet = false, N=256, N_ppc=256, xmax=2π, Δt=0.1, tmax=20π, amplitude=0.05, wavenumber=20, wavespeed=0, thermal_velocity=0.0125, suffix)
    # Initialize particles with maxwellian Vdf and sinusoidal perturbation
    N_p = N * N_ppc

    particles, fields, grid = ParticleInCell.initialize(N_p, N, xmax)

    ParticleInCell.maxwellian_vdf!(particles, thermal_velocity; quiet)
    ParticleInCell.perturb!(particles, amplitude, wavenumber, wavespeed, xmax)

    num_timesteps = ceil(Int, tmax / Δt)

    E_cache = zeros(N, num_timesteps+1)
    δρ_cache = zeros(N, num_timesteps+1)
    v_cache = zeros(N_p, num_timesteps+1)
    x_cache = zeros(N_p, num_timesteps+1)
    T_cache = zeros(num_timesteps+1)

    E_cache[:, 1] = fields.Ex
    δρ_cache[:, 1] = fields.ρ .- 1
    v_cache[:, 1] = particles.vx
    x_cache[:, 1] = particles.x
   # T_cache[1] = kinetic_energy(particles, xmax)

    for i in 2:num_timesteps+1
        ParticleInCell.update!(particles, fields, grid, Δt)
        E_cache[:, i] = copy(fields.Ex)
        δρ_cache[:, i] = copy(fields.ρ) .- 1.0
        v_cache[:, i] = copy(particles.vx)
        x_cache[:, i] = copy(particles.x)
        #T_cache[i] = kinetic_energy(particles, xmax)
    end

    t = LinRange(0, tmax, num_timesteps+1)

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

    return (t = t, x = grid.x, n = δρ_cache, E = E_cache, xs = x_cache, vs = v_cache)
end

begin
    #===============================
    Problem 3: Warm plasma waves
    ===============================#

    N = 256
    N_ppc = 256
    xmax = 2π
    Δx = xmax / N
    Δt = 0.1
    thermal_velocity = 0.0125
    wavenumber = 20
    wavespeed = 1/wavenumber
    amplitude = 0.1
    wave_speed = 0
    tmax = 20π

    common_options = (;
        N, N_ppc, xmax, Δt, thermal_velocity, wavenumber, amplitude, tmax, 
        path = RESULT_PATH_003
    )

    results_standing = warm_plasma_waves(;quiet = false, suffix = "standing", wavespeed = 0.0, common_options...)
    results_standing_quiet = warm_plasma_waves(;quiet = true, suffix = "standing_quiet", wavespeed = 0.0, common_options...)
    results_travelling = warm_plasma_waves(;quiet = false, suffix = "travelling", wavespeed, common_options...)
    results_travelling_quiet = warm_plasma_waves(;quiet = true, suffix = "travelling_quiet", wavespeed, common_options...)
end

begin
    Nt = size(results_standing.E, 2)

    kmax = 60
    ωmax = π

    results = [results_standing, results_standing_quiet, results_travelling, results_travelling_quiet]
    suffixes = ["standing", "standing_quiet", "travelling", "travelling_quiet"]

    for (res, suffix) in zip(results, suffixes)
        plots = []
        for (q, name) in zip([res.n, res.E], ("n", "E"))

            ks, ωs, q̃ = ParticleInCell.fft_field(res.t, res.x, res.n; kmax, ωmax)

            p = plot(;
                xlabel = "kc/ωₚₑ", ylabel = "ω/ωₚₑ",
                xlims = (1, kmax), ylims = (0,ωmax),
            )
            heatmap!(p, ks, ωs, log2.(q̃), title = "log₂|ℱ($(name))|² ($(replace(suffix, "_" => ", ")))")
            plot!(
                p, ks, sqrt.(1.0 .+ 3*thermal_velocity^2*ks.^2);
                lw = 4, lc = :blue, ls = :dash, label = "Bohm-Gross dispersion",
                PLOT_SCALING_OPTIONS...
            )

            push!(plots, p)
        end

        p = plot(plots[1], plots[2], layout = (2, 1), size = (VERTICAL_RES, 1.5 * VERTICAL_RES))
        display(p)
        savefig(p, joinpath(RESULT_PATH_003, "fft_$(suffix).png"))
    end
end

begin
    vs = LinRange(-4 * thermal_velocity, 4 * thermal_velocity, 100)
    fs = @. 1/sqrt(2π)/thermal_velocity * exp(-vs^2/2/thermal_velocity^2)

    p = plot(;
        xlabel = "v/c", ylabel = "f(v)",
        margin = 20Plots.mm,
        size = (1.5 * VERTICAL_RES, VERTICAL_RES),
        PLOT_SCALING_OPTIONS...
    )
    vs_noisy = results_standing.vs[:, 1]
    vs_quiet = results_standing_quiet.vs[:, 1]
    histogram!(p, vs_noisy, lw = 0.0, fillalpha = 0.5, normalize=true, label = "Random")
    histogram!(p, vs_quiet, lw = 0.0, fillalpha = 0.5, normalize=true, label = "Quiet start")
    plot!(p, vs, fs, lw = 4, lc = :black, ls = :dash, label = "Maxwellian", la = 0.5)
    savefig(p, joinpath(RESULT_PATH_003, "warm_initialization.png"))
    display(p)
end
    
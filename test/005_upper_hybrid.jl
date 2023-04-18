using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_005 = mkpath("test/results/005")
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


function upper_hybrid_oscillation(;path = RESULT_PATH_005, N=101, N_ppc=100, B0 = √(3), xmax=2π, Δt=0.01, periods=10, perturb = :vx, wavenumber=1, suffix)

    ωᵤₕ = sqrt(1 + B0^2)
    Tᵤₕ = 2π / ωᵤₕ
    tmax = periods * Tᵤₕ

    N_p = N_ppc * N

    particles, fields, grid = ParticleInCell.initialize(
        N_p, N, xmax;
    )

    for i in 1:N_p
        if perturb == :vx
            amplitude=0.01 * (1 + B0^2)
            particles.vx[i] += amplitude * sin(2π * wavenumber * particles.x[i] / xmax)
        elseif perturb == :x || perturb == :vy
            amplitude=0.01
            δx = amplitude * sin(2π * wavenumber * particles.x[i] / xmax)
            particles.x[i] += δx
            if perturb == :vy
                δvy = δx / B0
                particles.vy[i] += δvy
            end
        end
    end

    num_timesteps = ceil(Int, tmax / Δt)

    E_cache = zeros(N, num_timesteps+1)
    δρ_cache = zeros(N, num_timesteps+1)
    vx_cache = zeros(N_p, num_timesteps+1)
    vy_cache = zeros(N_p, num_timesteps+1)
    x_cache = zeros(N_p, num_timesteps+1)

    E_cache[:, 1] = fields.Ex
    δρ_cache[:, 1] = fields.ρ .- 1
    x_cache[:, 1] = particles.x
    vx_cache[:, 1] = particles.vx
    vy_cache[:, 1] = particles.vy

    fields.Bz .= B0
    particles.Bz .= B0

    for i in 2:num_timesteps+1
        ParticleInCell.update!(particles, fields, grid, Δt)
        E_cache[:, i] = copy(fields.Ex)
        δρ_cache[:, i] = copy(fields.ρ) .- 1.0
        vx_cache[:, i] = copy(particles.vx)
        vy_cache[:, i] = copy(particles.vy)
        x_cache[:, i] = copy(particles.x)
    end

    t = LinRange(0, tmax, num_timesteps+1)

    contour_options = (;
        ylims = (0, 1), yticks = LinRange(0, 1, 6), linewidth = 0, right_margin = 30mm, 
        xlabel = "tωₚ / 2π", ylabel = "xωₚ/2πc", cmap = :balance,
        legend = :topright,
        PLOT_SCALING_OPTIONS ...
    )

    contour_E = contourf(t ./ 2π, grid.x ./ 2π, E_cache; title = "δE / mcωp", contour_options...)
    contour_ρ = contourf(t ./ 2π, grid.x ./ 2π, δρ_cache; title = "δn / n₀", contour_options...)

    vline!(contour_E, Tᵤₕ .* (1:periods) ./ 2π, label = "Expected period", lw = 6, ls = :dash, lc = :black)
    vline!(contour_ρ, Tᵤₕ .* (1:periods) ./ 2π, label = "Expected period", lw = 6, ls = :dash, lc = :black)


    p = plot(contour_ρ, contour_E, layout = (2, 1), size = (VERTICAL_RES, VERTICAL_RES))

    display(p)

    savefig(p, joinpath(path, "$(suffix).png"))

    return t, grid.x, x_cache, vx_cache, vy_cache, δρ_cache, E_cache
end

begin
    # Launch upper hybrid oscillations
    t, x, xs, vxs, vys, ρs, Es = upper_hybrid_oscillation(B0 = √(3), perturb = :vx, suffix="hybrid_perturb_vx")
    t, x, xs, vxs, vys, ρs, Es = upper_hybrid_oscillation(B0 = √(3), perturb = :x, suffix="hybrid_perturb_x")
    t, x, xs, vxs, vys, ρs, Es = upper_hybrid_oscillation(B0 = √(3), perturb = :vy, suffix="hybrid_perturb_vy")
end

begin
    # Trajectory of single particle in the vy perturbatuion case
    particle_id = 1
    px = plot(t[2:end], vxs[particle_id, 2:end], label = "", xlabel = "tωp", ylabel = "vx/c", title = "x-velocity")
    py = plot(t[2:end], vys[particle_id, 2:end], label = "", xlabel = "tωp", ylabel = "vy/c", title = "y-velocity")
    p = plot(px, py, layout = (2, 1), framestyle = :box)
end
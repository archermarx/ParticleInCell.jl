using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_006 = mkpath("test/results/006")
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

function magnetized_ring(B0 = 1/√(5))
    N = 128
    N_ppc = 128
    N_p = N * N_ppc
    xmax = 2π/10
    tmax = 100.0
    Δt = 0.2
    k = 10
    δn = 0.001
    v0 = 4.5 * B0 / k

    particles, fields, grid = ParticleInCell.initialize(
        N_p, N, xmax;
    )

    # Perturb particles
    for i = 1:N_p
        δx = δn * sin(k * particles.x[i]) / k
        θᵢ = 73 * (2π * i / N_p)
        vx = v0 * cos(θᵢ)
        vy = v0 * sin(θᵢ)

        particles.x[i] += δx
        particles.vx[i] = vx
        particles.vy[i] = vy
    end

    return ParticleInCell.simulate(particles, fields, grid; Δt, tmax, B0)
end

function plot_magnetized_ring(t, x, xs, vxs, vys, n, E; suffix = "" , animate = false)
    plot_size = (1660, 1080)
    fx_initial = ParticleInCell.plot_vdf(xs[:, 1], vys[:, 1], type="1D", vlims = (-0.25, 0.25), t = t[1])
    fx_final = ParticleInCell.plot_vdf(xs[:, end], vys[:, end], type="1D", vlims = (-0.25, 0.25), t = t[end])
    fv_initial = ParticleInCell.plot_vdf(vxs[:, 1], vys[:, 1], type="2D", vlims = (-0.25, 0.25), t = t[1])
    fv_final = ParticleInCell.plot_vdf(vxs[:, end], vys[:, end], type="2D", vlims = (-0.25, 0.25), t = t[end])

    savefig(fx_initial, joinpath(RESULT_PATH_006, "vdf_initial_1D_$(suffix).png"))
    savefig(fx_final, joinpath(RESULT_PATH_006, "vdf_final_1D_$(suffix).png"))
    savefig(fv_initial, joinpath(RESULT_PATH_006, "vdf_initial_2D_$(suffix).png"))
    savefig(fv_final, joinpath(RESULT_PATH_006, "vdf_final_2D_$(suffix).png"))

    if animate
        ParticleInCell.animate_vdf(xs, vys; suffix, frameskip=2, type="1D", vlims = (-0.25, 0.25), ts = t, dir = RESULT_PATH_006)
        ParticleInCell.animate_vdf(vxs, vys; suffix, frameskip=2, type="2D", vlims = (-0.25, 0.25), ts = t, dir = RESULT_PATH_006)
    end

    n_amplitude = zeros(length(t))
    E_amplitude = zeros(length(t))
    for i in 1:length(t)
        n_amplitude[i] = maximum(n[:, i]) - 1.0
        E_amplitude[i] = sqrt(sum(E[:, i].^2 ./ 2))
    end

    p_growth = plot(;
        xlabel = "tωp", ylabel = "Amplitude (arb.)", yaxis = :log,
        size = (1080, 1080),
        titlefontsize=FONT_SIZE*3÷2,
        legendfontsize=FONT_SIZE,
        xtickfontsize=FONT_SIZE,
        ytickfontsize=FONT_SIZE,
        xguidefontsize=FONT_SIZE,
        yguidefontsize=FONT_SIZE,
        framestyle=:box,
        legend = :bottomright,
        margin = 10Plots.mm,
    )

    plot!(t, n_amplitude, label = "n", lw = 2, lc = :red)
    plot!(t, E_amplitude, label = "E", lw = 2, lc = :blue)

    if suffix == "ring_unstable"

        if suffix == "ring_stable"
            γ_expected = 0.0
        elseif suffix == "ring_unstable"
            γ_expected = 0.265 / √(10)
        end
        #E_ind = 1
        #n_ind = 1
        expected_growth_n = @. 1e-3/sqrt(2) * exp(γ_expected * (t))
        expected_growth_E = @. 1e-4/sqrt(2) * exp(γ_expected * (t))
        
        plot!(t, expected_growth_n, label = "Expected (n)", ls = :dash, lw = 2, lc = :red)
        plot!(t, expected_growth_E, label = "Expected (E)", ls = :dash, lw = 2, lc = :blue)
    end

    savefig(p_growth, joinpath(RESULT_PATH_006, "growth_$(suffix).png"))
    
    display(p_growth)
    
    contour_options = (;
        ylims = (0, 0.1), yticks = LinRange(0, 0.1, 6), linewidth = 0, right_margin = 30mm, 
        xlabel = "tωₚ / 2π", ylabel = "xωₚ/2πc", cmap = :balance,
        legend = :topright,
        PLOT_SCALING_OPTIONS ...
    )

    contour_E = contourf(t ./ 2π, x ./ 2π, E; title = "δE / mcωp", contour_options...)
    contour_ρ = contourf(t ./ 2π, x ./ 2π, n .- 1.0; title = "δn / n₀", contour_options...)

    p = plot(contour_ρ, contour_E, layout = (2, 1), size = (VERTICAL_RES, VERTICAL_RES))

    display(p)

    savefig(p, joinpath(RESULT_PATH_006, "contour_$(suffix).png"))
end

let animate = true
    t, x_stable, xs_stable, vxs_stable, vys_stable, n_stable, E_stable = magnetized_ring(1 / √(5))
    plot_magnetized_ring(t, x_stable, xs_stable, vxs_stable, vys_stable, n_stable, E_stable; animate, suffix = "ring_stable")
end

let animate = true
    t, x_unstable, xs_unstable, vxs_unstable, vys_unstable, n_unstable, E_unstable = magnetized_ring(1 / √(10))
    plot_magnetized_ring(t, x_unstable, xs_unstable, vxs_unstable, vys_unstable, n_unstable, E_unstable; animate, suffix = "ring_unstable")
end
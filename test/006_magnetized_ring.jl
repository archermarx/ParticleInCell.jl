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

    ions, electrons, fields, grid = ParticleInCell.initialize(
        N_p, N, xmax;
    )

    # Perturb particles
    for i = 1:N_p
        δx = δn * sin(k * electrons.x[i]) / k
        θᵢ = 73 * (2π * i / N_p)
        vx = v0 * cos(θᵢ)
        vy = v0 * sin(θᵢ)

        electrons.x[i] += δx
        electrons.vx[i] = vx
        electrons.vy[i] = vy
    end

    return ParticleInCell.simulate(ions, electrons, fields, grid; Δt, tmax, B0)
end

function plot_magnetized_ring(results; suffix = "" , animate = false)

    (;t, x, ρ, E) = results
    xs = results.electrons.x
    vxs = results.electrons.vx
    vys = results.electrons.vy

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
        ParticleInCell.animate_vdf(xs, vys; suffix, frameskip=2, type="1D", vlims = (-0.25, 0.25), ts = t, dir = RESULT_PATH_006, PLOT_SCALING_OPTIONS...)
        ParticleInCell.animate_vdf(vxs, vys; suffix, frameskip=2, type="2D", vlims = (-0.25, 0.25), ts = t, dir = RESULT_PATH_006, PLOT_SCALING_OPTIONS...)
    end

    n_amplitude = zeros(length(t))
    E_amplitude = zeros(length(t))
    for i in 1:length(t)
        min_ρ, max_ρ = extrema(ρ[:, i])
        n_amplitude[i] = max_ρ - min_ρ
        E_amplitude[i] = sqrt(sum(E[:, i].^2 ./ 2))
    end

    p_growth = plot(;
        xlabel = "tωp", ylabel = "Amplitude (arb.)", yaxis = :log,
        size = (1080, 1080),
        margin = 10Plots.mm,
        PLOT_SCALING_OPTIONS...
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
    contour_ρ = contourf(t ./ 2π, x ./ 2π, ρ .- 1.0; title = "δn / n₀", contour_options...)

    p = plot(contour_ρ, contour_E, layout = (2, 1), size = (VERTICAL_RES, VERTICAL_RES))

    display(p)

    savefig(p, joinpath(RESULT_PATH_006, "contour_$(suffix).png"))
end

let animate = false
    results = magnetized_ring(1 / √(5))
    plot_magnetized_ring(results; animate, suffix = "ring_stable")
end

let animate = false
    results = magnetized_ring(1 / √(10))
    plot_magnetized_ring(results; animate, suffix = "ring_unstable")
end
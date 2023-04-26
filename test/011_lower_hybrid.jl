using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_011 = mkpath("test/results/011")
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


function lower_hybrid_wave(N_ppc, N; B0 = 1.0, Lx = 2π, tmax = 20π, Δt = 0.2, mi_ratio = 1.0, perturb_k = 1.0, perturb_amp = 0.0, travelling = false)

    Ly = 1.0
    Lz = 1.0

    ions, electrons, fields, grid = ParticleInCell.initialize(
        N_ppc * N, N, Lx, Ly, Lz;
        mi = mi_ratio,
    )

    # Perturb electron velocities
    for i in eachindex(electrons.x)
        δvx = perturb_amp * (1 + B0^2) * sin(perturb_k * electrons.x[i])
        electrons.vx[i] += δvx
    end

    #for i in eachindex(ions.x)
    #    δx = perturb_amp / perturb_k * cos(2π * perturb_k * ions.x[i] / Lx)
    #    ions.vx[i] -= δx/2
    #end

    results = ParticleInCell.simulate(
        ions, electrons, fields, grid;
        Δt, tmax, particle_save_interval = 100,
        B0
    )

    return results
end

begin
    N_ppc = 64
    N = 64
    mi = 10.0
    perturb_amp = 0.01
    B0 = √(3)
    ωci = B0 / mi
    ωce = B0
    ωpi = 1 / sqrt(mi)
    ωᵤₕ = sqrt(1 + B0^2)
    ωₗₕ = inv(sqrt(inv(ωci^2 + ωpi^2) + inv(ωce * ωci)))

    # Fit an integer number of perturbation wavelengths in the domain
    num_wavelengths = 2
    perturb_k = num_wavelengths
    Lx = 2π

    periods = 4
    tmax = periods * 2π / ωₗₕ
end

begin
    @time (t, x, ρi, ρe, ρ, E, ions, electrons) = lower_hybrid_wave(
        N_ppc, N; Δt = 0.025, tmax, Lx, B0,
        mi_ratio = mi, perturb_amp, perturb_k,
    )
end

begin
    contour_options = (;
        ylims = (0, Lx / 2π), linewidth = 0, left_margin = 20Plots.mm, right_margin = 10Plots.mm, 
        xlabel = "tωₚ / 2π", ylabel = "xωₚ/2πc", cmap = :balance,
        legend = :outertop,
        PLOT_SCALING_OPTIONS ...
    )

    max_dim = 200
    N = length(t)
    if N > max_dim
        subsample_interval = N ÷ max_dim
    else
        subsample_interval = 1
    end

    inds = 1:subsample_interval:N
    @show inds

    Tᵤₕ = 2π / ωᵤₕ
    Tₗₕ = 2π / ωₗₕ
    num_periods_LH = t[end] / Tₗₕ
    num_periods_UH = t[end] / Tᵤₕ

    amp_ρi = max(abs.(extrema(ρi .- 1.0))...)
    amp_ρe = max(abs.(extrema(ρe .+ 1.0))...)
    amp_ρ = max(abs.(extrema(ρ))...)
    clims_ρi = (-amp_ρi, amp_ρi)
    clims_ρe = (-amp_ρe, amp_ρe)
    clims_ρ = (-amp_ρ, amp_ρ)
    
    contour_ρe = heatmap(t[inds], x ./ 2π, ρe[:, inds] .+ 1.0; clims = clims_ρe, title = "δρe / n₀", contour_options...)
    contour_ρi = heatmap(t[inds], x ./ 2π, ρi[:, inds] .- 1.0; clims = clims_ρi, title = "δρi / n₀", contour_options...)
    contour_ρ = heatmap(t[inds], x ./ 2π, ρ[:, inds]; clims = clims_ρ, title = "δρ / n₀", contour_options...)
    contour_E = heatmap(t[inds], x ./ 2π, E[:, inds]; title = "δE / mcωp", contour_options...)

    vline!(contour_ρ, Tₗₕ .* (1:num_periods_LH), label = "Expected period", lw = 6, ls = :dash, lc = :black)
    vline!(contour_ρi, Tₗₕ .* (1:num_periods_LH), label = "Expected period", lw = 6, ls = :dash, lc = :black)
    vline!(contour_E, Tₗₕ .* (1:num_periods_LH), label = "Expected period", lw = 6, ls = :dash, lc = :black)

    p = plot(contour_ρe, contour_ρi, contour_ρ, contour_E, layout = (4, 1), size = (VERTICAL_RES, 2 * VERTICAL_RES))

    display(p)
    savefig(p, joinpath(RESULT_PATH_011, "contour_mi=$mi.png"))
end

begin
    ind = 4
    max_ind = min(size(E, 2), 10000)
    E_ind = E[ind, 1:max_ind]
    t_ind = t[1:max_ind]
    p = plot(; framestyle = :box)
    plot!(t_ind, E_ind, label = "Electric field")
    Tᵤₕ = 2π / ωᵤₕ
    num_periods_UH = t[max_ind] / Tᵤₕ
    vline!(Tᵤₕ .* (1:num_periods_UH), label = "Upper hybrid periods", legend = :outertop)

    Tₗₕ = 2π / ωₗₕ
    num_periods_LH = t[max_ind] / Tₗₕ
    vline!(Tₗₕ * [1:num_periods_LH], lw = 2, lc = :black, ls = :dash, label = "Lower hybrid periods")
    display(p)
    savefig(p, joinpath(RESULT_PATH_011, "oscillations_mi=$mi.png"))
end
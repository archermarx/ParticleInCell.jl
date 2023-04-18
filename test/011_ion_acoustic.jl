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

function ion_acoustic_wave(N_ppc, N, v_th = 0.0125; electron_drift_speed = 0.0, Lx = 2π, tmax = 20π, Δt = 0.2, mi_ratio = 1.0, quiet = false, perturb_k = 1.0, perturb_amp = 0.0, travelling = false)

    Ly = 1.0
    Lz = 1.0

    ions, electrons, fields, grid = ParticleInCell.initialize(
        N_ppc * N, N, Lx, Ly, Lz;
        mi = mi_ratio * ParticleInCell.m_p,
    )
    
    ParticleInCell.maxwellian_vdf!(electrons, v_th; quiet)

    # Set electrons drifting and perturb ion and electron densities
    for i in eachindex(electrons.x)
        electrons.vx[i] += electron_drift_speed
        δx = perturb_amp / perturb_k * cos(perturb_k * ions.x[i])
        electrons.x[i] += δx/2
    end

    #for i in eachindex(ions.x)
    #    δx = perturb_amp / perturb_k * cos(2π * perturb_k * ions.x[i] / Lx)
    #    ions.x[i] -= δx/2
    #end

    ParticleInCell.apply_periodic_boundaries!(ions, Lx, Ly, Lz)
    ParticleInCell.apply_periodic_boundaries!(electrons, Lx, Ly, Lz)

    results = ParticleInCell.simulate(ions, electrons, fields, grid; Δt, tmax)

    return results
end

begin
    N_ppc = 1024
    N = 1024
    v_th = 0.025
    mi = 0.1 #1.008
    perturb_amp = 0.1
    ωpi = 1 / sqrt(mi * ParticleInCell.m_p)
    cs = v_th * ωpi

    λd = v_th
    
    # Perturb at wavenumber of maximum growth
    perturb_k = 1 / √(2) / λd
    
    # Fit an integer number of perturbation wavelengths in the domain
    num_wavelengths = 20
    Lx = 2π * num_wavelengths / perturb_k

    tmax = 40π / ωpi
    electron_drift_speed = 10 * cs
end

begin
    @time (t, x, ρi, ρe, ρ, E, ions, electrons) = ion_acoustic_wave(
        N_ppc, N, v_th; Δt = 0.025, tmax, Lx,
        mi_ratio = mi, quiet = true, perturb_amp, perturb_k,
        electron_drift_speed,
    )
end

begin
    contour_options = (;
        ylims = (0, Lx / 2π), linewidth = 0, left_margin = 20Plots.mm, right_margin = 10Plots.mm, 
        xlabel = "tωₚ / 2π", ylabel = "xωₚ/2πc", cmap = :balance,
        legend = :topright,
        PLOT_SCALING_OPTIONS ...
    )
    
    clims = (-0.1, 0.1)

    contour_ρe = heatmap(t ./ 2π, x ./ 2π, ρe .+ 1.0; clims, title = "δρe / n₀", contour_options...)
    contour_ρi = heatmap(t ./ 2π, x ./ 2π, ρi .- 1.0; clims, title = "δρi / n₀", contour_options...)
    contour_ρ = heatmap(t ./ 2π, x ./ 2π, ρ; clims, title = "δρ / n₀", contour_options...)
    contour_E = heatmap(t ./ 2π, x ./ 2π, E; title = "δE / mcωp", contour_options...)

    p = plot(contour_ρe, contour_ρi, contour_ρ, contour_E, layout = (4, 1), size = (VERTICAL_RES, 2 * VERTICAL_RES))

    display(p)
    savefig(p, joinpath(RESULT_PATH_011, "contour.png"))
end

begin
    kmax = 2 / λd
    ωmax = 2 * ωpi
    k, ω, Ẽ = ParticleInCell.fft_field(t, x, E; kmax, ωmax)


    kλd = @. k * λd
    ω_ωpi = @. ω / ωpi

    ω_analytic = @. k * cs * sqrt(1 / (1 + kλd^2)) / ωpi

    p = plot(;
        xlims = (0, kmax * λd),
        ylims = (0, ωmax / ωpi),
        xlabel = "k λd",
        ylabel = "ω / ωpi",
        PLOT_SCALING_OPTIONS...,
        size = (VERTICAL_RES, 2/3 * VERTICAL_RES),
        margin = 10Plots.mm,
        patchsize = (40, 20)
    )

    heatmap!(p, kλd, ω_ωpi, log2.(Ẽ), colormap = :turbo)
    vline!(p, [perturb_k * λd], lc = :black, lw = 3, label = "Initial perturbation")
    plot!(p, kλd, k .* cs ./ ωpi, lw = 4, lc = :blue, ls = :dash, label = "Ion acoustic dispersion relation")
    plot!(p, kλd, ω_analytic, lw = 4, lc = :blue, label = "ω/k = c_s")

    savefig(p, joinpath(RESULT_PATH_011, "spectrum_ion_acoustic.png"))
    display(p)
end

begin
    E_amplitude = mapslices(ParticleInCell.peak_to_peak_amplitude, E, dims=1)'
    ρe_amplitude = mapslices(ParticleInCell.peak_to_peak_amplitude, ρe, dims=1)'
    ρi_amplitude = mapslices(ParticleInCell.peak_to_peak_amplitude, ρi, dims=1)'

    ω_r = perturb_k * cs * sqrt(1 / (1 + perturb_k^2 * λd^2))
    linear_growth_rate = -ω_r * sqrt(π/8) * (1 - electron_drift_speed / cs) * ωpi
    linear_growth_func(c, t) = c * exp(linear_growth_rate * t)

    linear_growth = linear_growth_func.(1e-3, t)

    p = plot(; yaxis = :log, 
        ylabel = "Amplitude (arb.)",
        xlabel = "tωpi",
        size = (VERTICAL_RES, VERTICAL_RES),
        ylims = (1e-4, 100),
        margin = 10Plots.mm,
        legend = :outertop,
        PLOT_SCALING_OPTIONS...
    )

    lw = 4
    plot!(p, t * ωpi, E_amplitude; lw, label = "|E|")
    plot!(p, t * ωpi, ρe_amplitude; lw, label = "ne")
    plot!(p, t * ωpi, ρi_amplitude; lw, label = "ni")
    plot!(p, t * ωpi, linear_growth; lw, label = "Linear growth rate", lc = :black, ls = :dash)
    savefig(p, joinpath(RESULT_PATH_011, "growth_ion_acoustic.png"))
end
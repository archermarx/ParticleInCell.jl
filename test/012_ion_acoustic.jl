using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_012 = mkpath("test/results/012")
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

function ion_acoustic_wave(N_ppc, N, v_th = 0.0125; electron_drift_speed = 0.0, Lx = 2π, tmax = 20π, Δt = 0.2, mi = 1.0, quiet = false, perturb_k = 1.0, perturb_amp = 0.0, travelling = false)

    Ly = 1.0
    Lz = 1.0

    ions, electrons, fields, grid = ParticleInCell.initialize(
        N_ppc * N, N, Lx, Ly, Lz; mi
    )
    
    ParticleInCell.maxwellian_vdf!(electrons, v_th; quiet)

    # Set electrons drifting and perturb ion and electron densities
    for i in eachindex(electrons.x)
        electrons.vx[i] += electron_drift_speed
        δx = perturb_amp / perturb_k * cos(perturb_k * ions.x[i])
        electrons.x[i] += δx
    end

    ParticleInCell.apply_periodic_boundaries!(ions, Lx, Ly, Lz)
    ParticleInCell.apply_periodic_boundaries!(electrons, Lx, Ly, Lz)

    results = ParticleInCell.simulate(
        ions, electrons, fields, grid;
        Δt, tmax, particle_save_interval = 100
    )

    return results
end

begin
    N_ppc = 256
    mi = 1836.0
    perturb_amp = 0.05
    ωpi = 1 / sqrt(mi)
    v_th = 0.001

    cs = v_th * ωpi
    λd = v_th
    
    # Perturb at wavenumber of maximum growth
    perturb_k = 1 / √(2) / λd
    
    # Fit an integer number of perturbation wavelengths in the domain
    num_wavelengths = 40
    Lx = 2π * num_wavelengths / perturb_k
    Δx = λd / 2
    N = ceil(Int, Lx / Δx)

    Δt = 0.25
    periods = 40
    tmax = 2π / ωpi * periods
    Me = 0.5
    electron_drift_speed = Me * v_th

    @show N
end

begin
    @time (t, x, ρi, ρe, ρ, E, ions, electrons) = ion_acoustic_wave(
        N_ppc, N, v_th; Δt, tmax, Lx, mi,
        quiet = true, perturb_amp, perturb_k,
        electron_drift_speed,
    )
end

begin
    kmax = 2 / λd
    ωmax = 2 * ωpi
    k, ω, Ẽ = ParticleInCell.fft_field(t, x, E; kmax, ωmax)


    kλd = @. k * λd
    ω_ωpi = @. ω / ωpi

    ω_an(k) = k * v_th * inv(sqrt(mi * (1 + k^2 * v_th^2)))

    p = plot(;
        xlims = (0, kmax * λd),
        ylims = (0, ωmax / ωpi),
        xlabel = "kλd",
        ylabel = "ω/ωpi",
        PLOT_SCALING_OPTIONS...,
        size = (VERTICAL_RES, VERTICAL_RES),
        margin = 10Plots.mm,
        legend = :outertop
    )

    heatmap!(p, kλd, ω_ωpi, log2.(Ẽ))
    vline!(p, [perturb_k * λd], lc = :black, lw = 4, label = "Initial perturbation")
    plot!(p, kλd, k .* cs ./ ωpi, lw = 4, lc = :blue, ls = :dash, label = "ω/k = c_s")
    plot!(p, kλd, ω_an.(k) ./ ωpi, lw = 4, lc = :blue, label = "Ion acoustic dispersion relation")

    savefig(p, joinpath(RESULT_PATH_012, "spectrum_mi=$(mi)_Me=$(Me).png"))
    display(p)
end

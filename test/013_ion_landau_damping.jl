using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_013 = mkpath("test/results/013")
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

function ion_landau_damping(N_ppc, N, v_the = 0.0125, v_thi = 0.0; electron_drift_speed = 0.0, Lx = 2π, tmax = 20π, Δt = 0.2, mi = 1.0, quiet = false, perturb_k = 1.0, perturb_amp = 0.0)

    Ly = 1.0
    Lz = 1.0

    ions, electrons, fields, grid = ParticleInCell.initialize(
        N_ppc * N, N, Lx, Ly, Lz; mi
    )
    
    ParticleInCell.maxwellian_vdf!(electrons, v_the; quiet)
    ParticleInCell.maxwellian_vdf!(electrons, v_thi; quiet)

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
    N_ppc = 512
    mi = 50.0
    perturb_amp = 0.1
    ωpi = 1 / sqrt(mi)
    v_the = 0.001

    Te_Ti = 1.0

    v_thi = sqrt(1/Te_Ti/mi) * v_the

    cs = v_th * ωpi
    λd = v_th
    
    # Perturb at wavenumber of maximum growth
    perturb_k = 1 / √(2) / λd
    
    # Fit an integer number of perturbation wavelengths in the domain
    num_wavelengths = 5
    Lx = 2π * num_wavelengths / perturb_k
    Δx = λd / 4
    N = ceil(Int, Lx / Δx)

    Δt = 0.1
    periods = 5
    tmax = 2π / ωpi * periods
    Me = 0.0
    electron_drift_speed = Me * v_th

    @show N
end

begin
    @time (t, x, ρi, ρe, ρ, E, ions, electrons) = ion_landau_damping(
        N_ppc, N, v_the, v_thi; Δt, tmax, Lx, mi,
        quiet = true, perturb_amp, perturb_k,
        electron_drift_speed,
    )
end

begin

    ω_an(k) = k * v_th * inv(sqrt(mi * (1 + k^2 * v_th^2)))

    ωr = ω_an(perturb_k)

    Te = v_the^2
    Ti = mi * v_thi^2

    denom = 1 + perturb_k^2 * v_th^2

    damping_rate = √(π/8) * ωr / denom^1.5 * (
        1/√(mi) + 
        (Te / Ti)^1.5 * exp(-Te / 2 / Ti / denom)
    )
    
    field_energy = 0.5 * mapslices(x -> sum(x_i^2 for x_i in x), E, dims = 1)'[:]
    E₀ = field_energy[1]
    field_energy /= E₀

    damping_func = @. exp(-damping_rate * t) 
  
    p = plot(;
        title = "Ion Landau damping with Tₑ/Tᵢ = $Te_Ti,\n Nppc = $N_ppc, mᵢ / mₑ = $mi",
        xlabel = "tωpi",
        ylabel = "E / E₀",
        yaxis = :log, ylims = (minimum(field_energy), 2),
        framestyle = :box, 
        size = (VERTICAL_RES, VERTICAL_RES),
        legend = :outertop,
        topmargin = 10Plots.mm,
        PLOT_SCALING_OPTIONS...
    )

    lw = 4
    plot!(p, t .* ωpi, field_energy; lw, label = "Electrostatic energy (simulation)")
    plot!(p, t .* ωpi, damping_func; lw, label = "Analytic damping rate at kλd = 1/√2")

    savefig(p, joinpath(RESULT_PATH_013, "landau_damping_Te_Ti=$(Te_Ti)_mi=$(mi)_Nppc=$(Nppc).png"))
    p

end
using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit
using KernelDensity

const RESULT_PATH_009 = mkpath("test/results/009")
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

# Problem 3: Landau damping
function landau_damping(N, N_ppc; wavenumber, wave_speed, amplitude, v_th = 0.04, xmax, tmax, quiet)
    N_p = N * N_ppc
    Δt = 0.2

    ions, electrons, fields, grid = ParticleInCell.initialize(N_p, N, xmax;)

    ParticleInCell.maxwellian_vdf!(electrons, v_th; quiet)

    # Perturb particles to establish travelling wave
    for i in 1:N_p
        δx = amplitude / wavenumber * cos(wavenumber * electrons.x[i])
        δv = wave_speed * amplitude * sin(wavenumber * electrons.x[i])
        electrons.x[i] += δx
        electrons.vx[i] += δv
    end

    return ParticleInCell.simulate(ions, electrons, fields, grid; Δt, tmax)
end

for quiet in [true, false]
    wavenumber = 10
    wave_speed = 1 / wavenumber
    amplitude = 0.05
    xmax = 2 * 2π/wavenumber
    tmax = 10 * π
    v_th = 0.04
    N = 1024
    N_ppc = 1024
    @time results = landau_damping(N, N_ppc; wavenumber, v_th, wave_speed, amplitude, xmax, tmax, quiet)

    (;t, x) = results
    ns = results.ρ
    Es = results.E
    xs = results.electrons.x
    vxs = results.electrons.vx
    vys = results.electrons.vy

    plot_size = (1080, 1080)
    margin = 10Plots.mm

    heatmap_options = (;ylims = (0, xmax), c = :balance, margin, size=plot_size, xlabel = "tωp", ylabel = "xωp/c", right_margin = 2*margin,PLOT_SCALING_OPTIONS...)
    hm_n = heatmap(t, x, ns .- 1; heatmap_options...)
    hm_E = heatmap(t, x, Es; heatmap_options...)

    quiet_str = quiet ? "_quiet" : ""

    display(hm_n)
    display(hm_E)
    savefig(hm_n, joinpath(RESULT_PATH_009, "landau_n" * quiet_str * ".png"))
    savefig(hm_E, joinpath(RESULT_PATH_009, "landau_E" * quiet_str * ".png"))

    vdf_options = (;label = "", normalize = true, lw = 6, xlims = (-0.15, 0.2), size = plot_size, margin, xlabel = "v/c", ylabel = "f(v)", PLOT_SCALING_OPTIONS..., ylims = (0, 10))
    den1 = kde(vxs[:, 1])
    den2 = kde(vxs[:, end])
    
    p1 = plot(den1.x, den1.density; vdf_options...)
    vline!(p1, [wave_speed], label = "ω/k", lw = 4, lc = :red, ls = :dash)
    display(p1)
    
    p2 = plot(den2.x, den2.density; vdf_options...)
    vline!(p2, [wave_speed], label = "ω/k", lw = 4, lc = :red, ls = :dash)
    display(p2)
    savefig(p1, joinpath(RESULT_PATH_009, "landau_vdf_before" * quiet_str * ".png"))
    savefig(p2, joinpath(RESULT_PATH_009, "landau_vdf_after" * quiet_str * ".png"))

    kλd = wavenumber * v_th
    expected_damping_rate = -sqrt(π/8) / kλd^3 * exp(-0.5 / kλd^2)
    max_E = mapslices(maximum, Es.^2, dims=1)'
    Nt = length(t)

    eqn = @. (x, p) -> p[1] * exp(expected_damping_rate*x)
    fit_landau = curve_fit(eqn, t[1:Nt÷3], max_E[1:Nt÷3], [1e-5])

    p = plot(;margin = 10Plots.mm, size = (1080, 1080), xlims = extrema(t), PLOT_SCALING_OPTIONS..., xlabel = "tωp", ylabel = "E²/2", ylims = extrema(max_E))
    plot!(p, t, max_E, yaxis = :log, lw = 4, label = "")
    plot!(p, t, eqn(t, fit_landau.param), lw = 4, lc = :red, ls = :dash, yaxis = :log, label = "Linear damping rate")
    display(p)
    savefig(p, joinpath(RESULT_PATH_009, "damping_rate" * quiet_str * ".png"))
end
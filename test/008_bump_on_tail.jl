using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_008 = mkpath("test/results/008")
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

# Problem 2: beam plasma (bump on tail) instability
function bump_on_tail(N, N_ppc; v_d = 3.0, v_th = 0.01, tmax = 30, ratio, quiet)
    N_p = N * N_ppc
    xmax = 20π
    tmax = tmax * π
    Δt = 0.2

    particles, fields, grid = ParticleInCell.initialize(
        N_p, N, xmax; charge_per_particle = 1
    )

    ParticleInCell.maxwellian_vdf!(particles, v_th; quiet)

    # Assign one in n_b_ratio particles to the beam
    for i in 1:ratio:N_p
        particles.vx[i] = v_d
        particles.vy[i] = 0.0
    end

    return ParticleInCell.simulate(particles, fields, grid; Δt, tmax)
end

begin
    N = 512
    N_ppc = 512
    Np = N * N_ppc
    tmax = 15

    v_th = 0.01
    v_d = 3.0

    ratios = [2, 4, 8, 16, 32]

    for quiet in [true, false]
        growthrates = zeros(length(ratios))
        initial_amplitudes = zeros(length(ratios))

        p = plot(;
            legend = :topleft,
            yaxis = :log,
            size = (1080, 1080),
            xlabel = "tωp", ylabel = "Δv",
            title = "Bump on tail, quiet = $(quiet)",
            PLOT_SCALING_OPTIONS...
        )
        for (i, ratio) in enumerate(ratios)
            t, x, xs, vxs, vys, ns, Es = bump_on_tail(N, N_ppc; v_th, v_d, tmax, ratio, quiet)
            vlims = (-0.5, 3.5)

            if ratio ∈ [2, 4, 8, 16, 32]
                Δvs = [
                    std(vxs[1:ratio:Np, j])
                    for j in eachindex(t)
                ]

                plot!(p, t[5:end], Δvs[5:end], label = "Ratio = $ratio", lw = 4)
            end

            if ratio == 10
                frames = 0:1:15
                for f in frames
                    ind = findfirst(>=(f*π), t)
                    x = xs[:, ind]
                    v = vxs[:, ind]

                    q = ParticleInCell.plot_vdf(
                        x, v; type = "1D",
                        vlims, t = "$(f)π, N = $(N), Nppc = $(N_ppc)",
                        style = :histogram, bins = 200, normalize=true,
                        c = :viridis, clims = (0, 0.15),
                        colorbar = false
                    )
                    savefig(q, joinpath(RESULT_PATH_008, "beam_plasma_$(f)pi" * (quiet ? "_quiet" : "") * ".png"))
                end
            end

            max_n = mapslices(maximum, abs.(ns), dims=1)'
            max_E = mapslices(maximum, abs.(Es), dims=1)'

            i1 = findlast(<(5π), t)
            i2 = findfirst(>(8π), t)
            t_linear = t[i1:i2]
            E_linear = max_E[i1:i2]
            @. exponential(x, p) = p[2] * exp(p[1] * x)
            fit = curve_fit(exponential, t_linear, E_linear, [0.25, 0.01])
            growthrates[i] = fit.param[1]
            initial_amplitudes[i] = fit.param[2]
        end

        savefig(p, joinpath(RESULT_PATH_008, "beam_plasma_growth" * (quiet ? "_quiet" : "") * ".png"))

        display(p)
    end
end
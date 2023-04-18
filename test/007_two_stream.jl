using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_007 = mkpath("test/results/007")
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



function two_stream_instability(N = 64, N_ppc = 16; v0 = 3, k = √(3) / 2v0, periods = 1, tmax = 40)
    N_p = N * N_ppc
    xmax = 2π / k * periods #8π*√(3) * periods
    tmax = tmax * π
    Δt = 0.2
    δn = 0.001

    particles, fields, grid = ParticleInCell.initialize(
        N_p, N, xmax; charge_per_particle = 2
    )

    # Perturb particles
    for i = 1:N_p
        δx = δn * sin(k * particles.x[i]) / k

        particles.x[i] += δx

        if iseven(i)
            particles.vx[i] = v0
        else
            particles.vx[i] = -v0
        end
    end

    return ParticleInCell.simulate(particles, fields, grid; Δt, tmax)
end

begin
    v0 = 3
    N = 256
    N_ppc = 256
    k1 = √(3)/2/v0
    k2 = 1.1*√(2)/v0
    titles = ["√3/2", "1.1 √(2)"]
    tmax = 40
    for (i, (k, k_str)) in enumerate(zip([k1,k2], titles))
        t, x, xs, vxs, vys, ns, Es = two_stream_instability(N, N_ppc; v0, k, periods = 50, tmax)
        
        fft_ind = findfirst(>(4π), t)
        fft_range = 1:fft_ind
        vlims = (-8, 8)
        ks, Ẽ = ParticleInCell.fft_time(t[fft_range], x, Es)

        hm = heatmap(ks, t[fft_range], log2.(Ẽ'); xlabel = "k", ylabel = "tωₚ", size = (1000, 1000), PLOT_SCALING_OPTIONS...)
        vline!([k], lw = 2, lc = :black, ls = :dash, label = "k₀")
        if i == 2
            vline!([k1], lw = 2, lc = :blue, ls = :dash, label = "kv₀ = √(3)/2")
        end
        title!("log₂|ñ|², kv₀ = $(k_str)")
        display(hm)
        savefig(hm, joinpath(RESULT_PATH_007, "growth_two_stream_wavenumber_$(i).png"))

        max_n = mapslices(maximum, abs.(ns), dims=1)'
        max_E = mapslices(maximum, abs.(Es), dims=1)'

        p = plot(;
            margin=10Plots.mm,
            xlabel = "tωp / π", ylabel = "Amplitude (arb.)",
            size = (900, 900), legend = :outertop,
            yaxis = :log, ylims = (0.001, 100), xlims = (0, tmax), 
            PLOT_SCALING_OPTIONS...
        )

        plot!(
            Shape([14.0, 1000, 1000, 14.0], [1e-4, 1e-4, 1e2, 1e2]),
            lw = 3, lc = RGB(0.6, 0.6, 0.0), fc = :yellow, fillalpha = 0.3, label = "Saturation regime"
        )

        plot!(
            t./π, max_n;
            label = "Maximum density", lw = 3, lc = :blue
        )
        plot!(
            t./π, max_E;
            label = "Maximum electric field", lw = 3, lc = :red
        )
        plot!(
            t./π, 0.00001*exp.(t./2),
            lw = 4, ls = :dash, lc = :black, label = "Expected linear growth"
        )

        #display(p)
        savefig(p, joinpath(RESULT_PATH_007, "growth_two_stream_$(i).png"))

        # plot k versus time, not k versus omega
        frames = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20]
        for f in frames
            ind = findfirst(>(f*π), t)
            x = xs[:, ind]
            v = vxs[:, ind]

            q = ParticleInCell.plot_vdf(x, v; type = "1D", vlims, t = "$(f)π, N = $(N), Nppc = $(N_ppc)", style = :histogram, bins = 200)
            savefig(q, joinpath(RESULT_PATH_007, "two_stream_t=$(f)pi_$(i).png"))
        end
    end
end


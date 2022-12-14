begin
    using Revise
    using ParticleInCell
    using Plots
    using Printf
    using FFTW
    using LsqFit
    using Statistics

    const WS7_RESULTS_DIR = mkpath("results/worksheet_7")
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
end

function fft_time(t, x, n; kmax = Inf)
    N = size(n, 1)

    Δx = x[2] - x[1]
    
    ks = (2π * fftshift(fftfreq(N, 1/Δx)))[N÷2+1:end]

    last_k_ind = findfirst(>(kmax), ks)

    if isnothing(last_k_ind)
        last_k_ind = lastindex(ks)        
    end
    
    ks = ks[2:last_k_ind]

    ñ = (abs.(fftshift(fft(n, 1))).^2)[N÷2+1:end, :]'[:, 2:last_k_ind]

    ñ = zeros(length(ks), length(t))

    for i in 1:length(t)
        n_slice = n[:, i]
        ñ[:, i] = (abs.(fftshift(fft(n_slice))).^2)[N÷2+1:end][2:last_k_ind]
    end
    return ks, ñ
end

function fft_field(t, x, n; kmax = Inf, ωmax = Inf)
    N = size(n, 1)
    Nt = size(n, 2)

    Δx = x[2] - x[1]
    Δt = t[2] - t[1]
    
    ks = (2π * fftshift(fftfreq(N, 1/Δx)))[N÷2+1:end]
    ωs = (2π * fftshift(fftfreq(Nt, 1/Δt)))[Nt÷2+1:end]

    last_ω_ind = findfirst(>(ωmax), ωs)
    last_k_ind = findfirst(>(kmax), ks)

    if isnothing(last_ω_ind)
        last_ω_ind = lastindex(ωs) 
    end

    if isnothing(last_k_ind)
        last_k_ind = lastindex(ks)        
    end
    
    ks = ks[2:last_k_ind]
    ωs = ωs[1:last_ω_ind]

    ñ = (abs.(fftshift(fft(n))).^2)[N÷2+1:end, Nt÷2+1:end]'[1:last_ω_ind, 2:last_k_ind]

    return ks, ωs, ñ
end

begin
    # Problem 1: Two-stream instability

    function two_stream_instability(N = 64, N_ppc = 16; v0 = 3, k = √(3) / 2v0, periods = 1, tmax = 40)
        N_p = N * N_ppc
        xmin = 0.0
        xmax = 2π / k * periods #8π*√(3) * periods
        tmax = tmax * π
        Δt = 0.2
        δn = 0.001

        particles, fields, grid = ParticleInCell.initialize(
            N_p, N_p, N, xmin, xmax; charge_per_particle = 2
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

    v0 = 3
    N = 128
    N_ppc = 64
    k1 = √(3)/2/v0
    k2 = 1.1*√(2)/v0
    titles = ["√3/2", "1.1 √(2)"]
    for (i, (k, k_str)) in enumerate(zip([k1,k2], titles))
        t, x, xs, vxs, vys, ns, Es = two_stream_instability(N, N_ppc; v0, k, periods = 50, tmax = 40)
        
        fft_ind = findfirst(>(4π), t)
        fft_range = 1:fft_ind
        vlims = (-8, 8)
        ks, Ẽ = fft_time(t[fft_range], x, Es)

        hm = heatmap(ks, t[fft_range], log2.(Ẽ'); xlabel = "k", ylabel = "tωₚ", size = (1000, 1000), PLOT_SCALING_OPTIONS...)
        vline!([k], lw = 4, lc = :black, ls = :dash, label = "k₀")
        if i == 2
            vline!([k1], lw = 4, lc = :blue, ls = :dash, label = "kv₀ = √(3)/2")
        end
        title!("log₂|ñ|², kv₀ = $(k_str)")
        display(hm)
        savefig(hm, "$(WS7_RESULTS_DIR)/growth_two_stream_wavenumber_$(i).png")

        max_n = mapslices(maximum, abs.(ns), dims=1)'
        max_E = mapslices(maximum, abs.(Es), dims=1)'
        p = plot(;margin=10Plots.mm, xlabel = "tωp / π", ylabel = "Amplitude (arb.)", size = (900, 900), PLOT_SCALING_OPTIONS...)
        plot!(Shape([17.0, 100, 100, 17.0] ./ π, [1e-2, 1e-2, 1e2, 1e2]), lw = 3, lc = RGB(0.6, 0.6, 0.0), fc = :yellow, fillalpha = 0.3, label = "Saturation regime")
        plot!(t./π, max_n, yaxis = :log, ylims = (0.1, 20), xlims = (0, 20), lw = 3, label = "Maximum density", lc = :blue)
        plot!(t./π, max_E, label = "Maximum electric field", lw = 3, lc = :red)
        plot!(t./π, 0.001*exp.(t./2), lw = 4, ls = :dash, lc = :black, label = "Expected linear growth", legend = :bottomright)

        #display(p)
        savefig(p, "$(WS7_RESULTS_DIR)/growth_two_stream_$(i).png")

        # plot k versus time, not k versus omega
        frames = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20]
        for f in frames
            ind = findfirst(>(f*π), t)
            x = xs[:, ind]
            v = vxs[:, ind]

            q = ParticleInCell.plot_vdf(x, v; type = "1D", vlims, t = "$(f)π, N = $(N), Nppc = $(N_ppc)", style = :histogram, bins = 200)
            savefig(q, "$(WS7_RESULTS_DIR)/two_stream_t=$(f)pi_$(i).png")
        end
    end
end



begin
    # Problem 2: beam plasma (bump on tail) instability
    function bump_on_tail(N, N_ppc; v_d = 3.0, v_th = 0.01, tmax = 30, ratio)
        N_p = N * N_ppc
        v_d = 3.0
        xmin = 0.0
        xmax = 20π
        tmax = tmax * π
        Δt = 0.2

        particles, fields, grid = ParticleInCell.initialize(
            N_p, N_p, N, xmin, xmax; charge_per_particle = 1
        )

        ParticleInCell.maxwellian_vdf!(particles, v_th)

        # Assign one in n_b_ratio particles to the beam
        for i in 1:ratio:N_p
            particles.vx[i] = v_d
            particles.vy[i] = 0.0
        end

        return ParticleInCell.simulate(particles, fields, grid; Δt, tmax)
    end

    N = 512
    N_ppc = 512
    Np = N * N_ppc
    tmax = 15

    v_th = 0.01
    v_d = 3.0

    ratios = [10]#2:2:32
    growthrates = zeros(length(ratios))
    initial_amplitudes = zeros(length(ratios))

    p = plot(; legend = :topleft, yaxis = :log, size = (1080, 1080), xlabel = "tωp", ylabel = "Δv", PLOT_SCALING_OPTIONS...)
    for (i, ratio) in enumerate(ratios)
        t, x, xs, vxs, vys, ns, Es = bump_on_tail(N, N_ppc; v_th, v_d, tmax, ratio)
        vlims = (-0.5, 3.5)

        if ratio ∈ [2, 4, 8, 16, 32]
            Δvs = [
                std(vxs[1:ratio:Np, j])
                for j in eachindex(t)
            ]

            plot!(p, t[5:end], Δvs[5:end], label = "Ratio = $ratio", lw = 4)
        end

        if ratio == 10
            frames = [0, 3, 6, 9, 12, 15]
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
                savefig(q, "$(WS7_RESULTS_DIR)/beam_plasma_$(f)pi.png")
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
    #savefig(p, "$(WS7_RESULTS_DIR)/beam_thermalization.png")
    display(p)
end

begin
    @. line(x, p) = p[1] + p[2] * x^(1/3)
    fit = curve_fit(line, ratios, growthrates, [0.3, -0.01])

    @show fit.param
    p = plot(;xlabel = "Plasma to beam ratio", ylabel = "Linear growth rate", size = (900, 900), PLOT_SCALING_OPTIONS...)
    scatter!(p, ratios, growthrates, label = "", ms = 6, mc = :black)
    plot!(p, ratios, line(ratios, fit.param); lw = 4, lc = :red, ls = :dash, label = "")

    savefig(p, joinpath(WS7_RESULTS_DIR, "beam_plasma_growth_vs_ratio.png"))
    display(p)
end

begin
    # Problem 3: Landau damping
    function landau_damping(N, N_ppc; wavenumber, wave_speed, amplitude, v_th = 0.04, xmax, tmax)
        N_p = N * N_ppc
        xmin = 0.0
        Δt = 0.2

        particles, fields, grid = ParticleInCell.initialize(
            N_p, N_p, N, xmin, xmax; charge_per_particle = 1
        )

        ParticleInCell.maxwellian_vdf!(particles, v_th)

        # Perturb particles to establish travelling wave
        for i in 1:N_p
            δx = amplitude / wavenumber * cos(wavenumber * particles.x[i])
            δv = wave_speed * amplitude * sin(wavenumber * particles.x[i])
            particles.x[i] += δx
            particles.vx[i] += δv
        end

        return ParticleInCell.simulate(particles, fields, grid; Δt, tmax)
    end

    wavenumber = 10
    wave_speed = 1 / wavenumber
    amplitude = 0.05
    xmax = 2 * 2π/wavenumber
    tmax = 10 * π
    v_th = 0.04

    N = 512
    N_ppc = 1024
    @time t, x, xs, vxs, vys, ns, Es = landau_damping(N, N_ppc; wavenumber, v_th, wave_speed, amplitude, xmax, tmax)
end

begin
    plot_size = (1080, 1080)
    margin = 10Plots.mm

    heatmap_options = (;ylims = (0, xmax), c = :balance, margin, size=plot_size, xlabel = "tωp", ylabel = "xωp/c", right_margin = 2*margin,PLOT_SCALING_OPTIONS...)
    hm_n = heatmap(t, x, ns .- 1; heatmap_options...)
    hm_E = heatmap(t, x, Es; heatmap_options...)

    display(hm_n)
    display(hm_E)
    savefig(hm_n, "$(WS7_RESULTS_DIR)/landau_n")
    savefig(hm_E, "$(WS7_RESULTS_DIR)/landau_E")
end

using KernelDensity

begin
    vdf_options = (;label = "", normalize = true, lw = 6, xlims = (-0.15, 0.2), size = plot_size, margin, xlabel = "v/c", ylabel = "f(v)", PLOT_SCALING_OPTIONS..., ylims = (0, 10))
    den1 = kde(vxs[:, 1])
    den2 = kde(vxs[:, end])
    
    p1 = plot(den1.x, den1.density; vdf_options...)
    vline!(p1, [wave_speed], label = "ω/k", lw = 4, lc = :red, ls = :dash)
    display(p1)
    
    p2 = plot(den2.x, den2.density; vdf_options...)
    vline!(p2, [wave_speed], label = "ω/k", lw = 4, lc = :red, ls = :dash)
    display(p2)
    savefig(p1, "$(WS7_RESULTS_DIR)/landau_vdf_before.png")
    savefig(p2, "$(WS7_RESULTS_DIR)/landau_vdf_after.png")
end

begin
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
    savefig(p, "$(WS7_RESULTS_DIR)/damping_landau.png")
end
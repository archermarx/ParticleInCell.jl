begin
    using ParticleInCell
    using Plots
    using Statistics
    using FFTW
    using StatsBase
    using Plots.PlotMeasures
    using LsqFit
end

function cold_plasma_wave(;N=101, N_ppc=100, xmax=π, Δt=0.1, tmax=2π, amplitude=0.01, wavenumber=1, wave_speed=1.0, suffix)

    N_p = N_ppc * N

    particles, fields, grid = ParticleInCell.initialize(
        N_p, N_p, N, 0.0, xmax;
        perturbation_amplitude = amplitude,
        perturbation_wavenumber = wavenumber,
        perturbation_speed = wave_speed / wavenumber,
    )

    num_timesteps = ceil(Int, tmax / Δt)

    E_cache = zeros(N, num_timesteps+1)
    δρ_cache = zeros(N, num_timesteps+1)
    v_cache = zeros(N_p, num_timesteps+1)
    x_cache = zeros(N_p, num_timesteps+1)

    E_cache[:, 1] = fields.Ex
    δρ_cache[:, 1] = fields.ρ .- 1
    v_cache[:, 1] = particles.vx
    x_cache[:, 1] = particles.x

    for i in 2:num_timesteps+1
        ParticleInCell.update!(particles, fields, grid, Δt)
        E_cache[:, i] = copy(fields.Ex)
        δρ_cache[:, i] = copy(fields.ρ) .- 1.0
        v_cache[:, i] = copy(particles.vx)
        x_cache[:, i] = copy(particles.x)
    end

    t = LinRange(0, tmax, num_timesteps+1)

    contour_E = contourf(t, grid.x, E_cache, xlabel = "tωₚ", ylabel = "xωₚ/c", c=:balance, linewidth=0, title = "eE / mcωₚ", right_margin=10mm)
    contour_ρ = contourf(t, grid.x, δρ_cache, xlabel = "tωₚ", ylabel = "xωₚ/c", c=:balance, linewidth=0, title = "δn / n₀", right_margin=10mm)

    display(contour_E)
    display(contour_ρ)

    savefig(contour_E, "results/E_$(suffix).svg")
    savefig(contour_ρ, "results/n_$(suffix).svg")
end

let
#===============================
  Problem 1: Cold plasma waves
===============================#
    # Generate plots for problem 1
    cold_plasma_wave(suffix = "travelling_k=1", N = 50, N_ppc = 50, xmax = 2π, amplitude = 0.01, wavenumber = 1, wave_speed = 1, tmax=4π, Δt = 0.01)
    cold_plasma_wave(suffix = "standing_k=1", N = 50, N_ppc = 50, xmax = 2π, amplitude = 0.01, wavenumber = 1, wave_speed = 0, tmax=4π, Δt = 0.01)
    cold_plasma_wave(suffix = "travelling_k=2", N = 50, N_ppc = 50, xmax = 2π, amplitude = 0.01, wavenumber = 2, wave_speed = 1, tmax=4π, Δt = 0.01)
    cold_plasma_wave(suffix = "standing_k=2", N = 50, N_ppc = 50, xmax = 2π, amplitude = 0.01, wavenumber = 2, wave_speed = 0, tmax=4π, Δt = 0.01)
end

kinetic_energy(particles, L) = 0.5 / L * sum(v^2 for v in particles.vx if !isnan(v)) / particles.num_particles

let
#===============================
  Problem 2: Thermal plasmas
===============================#

    # Generate maxwellian vdf plot for problem 2
    N = 1024
    N_ppc = 4096
    N_p = N * N_ppc
    xmax = 1.0
    thermal_velocity = 0.025

    particles, fields, grid = ParticleInCell.initialize(N_p, N_p, N, 0.0, xmax)

    ParticleInCell.maxwellian_vdf!(particles, thermal_velocity)
    p = histogram2d(
        particles.x, particles.vx,
        xlabel = "xωₚ/c", ylabel = "v/c",
        title = "Initial Maxwellian, 1024 cells, 4096 particles/cell",
        normalize=true
    )
    hline!(
        p, [thermal_velocity, -thermal_velocity], lw = 2, ls = :dash, lc = :blue,
        label = "±vₜₕ", legend = :outertop
    )
    display(p)
    savefig(p, "results/maxwellian.svg")

    # Generate remaining plots for problem 2
    Ns = [8, 16, 32, 64, 128, 256]
    N_ppcs = [1, 2, 4, 8, 16, 32, 64]

    xmax = 2π
    tmax = 400π
    thermal_velocity = 0.025
    Δt = 0.1

    p_N = plot(;
        yaxis = :log, legend = :outerright,
        title = "Kinetic energy vs Δx (16 particles/cell)",
        xlabel = "tωₚ", ylabel = "Kinetic energy (norm.)",
    )

    function plasma_heating(;N=101, N_ppc=50, xmax=2π, Δt=0.1, tmax=4π, thermal_velocity=0.025)
        N_p = N_ppc * N

        particles, fields, grid = ParticleInCell.initialize(
            N_p, N_p*2, N, 0.0, xmax;
            perturbation_amplitude = 0.0
        )

        ParticleInCell.maxwellian_vdf!(particles, thermal_velocity)

        num_timesteps = ceil(Int, tmax / Δt)

        E_cache = zeros(N, num_timesteps+1)
        δρ_cache = zeros(N, num_timesteps+1)
        v_cache = zeros(N_p * 2, num_timesteps+1)
        x_cache = zeros(N_p * 2, num_timesteps+1)
        T_cache = zeros(num_timesteps+1)

        E_cache[:, 1] = fields.Ex
        δρ_cache[:, 1] = fields.ρ .- 1
        v_cache[:, 1] = particles.vx
        x_cache[:, 1] = particles.x
        T_cache[1] = kinetic_energy(particles, xmax)

        for i in 2:num_timesteps+1
            ParticleInCell.update!(particles, fields, grid, Δt)
            E_cache[:, i] = copy(fields.Ex)
            δρ_cache[:, i] = copy(fields.ρ) .- 1.0
            v_cache[:, i] = copy(particles.vx)
            x_cache[:, i] = copy(particles.x)
            T_cache[i] = kinetic_energy(particles, xmax)
        end

        t = LinRange(0, tmax, num_timesteps+1)

        # Compute the initial heating rate by fitting a line to the first 1/5th of the data
        N = length(t) ÷ 5
        t_reduced = t[1:N]
        T_reduced = T_cache[1:N]
        @. line(x, p) = p[1] * x + p[2]
        fit = curve_fit(line, t_reduced, T_reduced, [0, T_cache[1]])
        heating_rate = fit.param[1]

        return t, T_cache, heating_rate
    end

    heating_rates = zeros(length(Ns))

    for (i, N) in enumerate(Ns)
        t, T, heating_rates[i] = plasma_heating(;N, N_ppc=16, thermal_velocity, tmax, Δt, xmax)
        plot!(p_N, t, T, label = "Δx=2π/$N")
    end

    savefig(p_N, "results/heating_curve_N.svg")

    display(p_N)
    Δx = xmax ./ Ns
    λd_Δx = @. thermal_velocity / Δx

    p_rate_N = plot(
        λd_Δx, heating_rates;
        xlabel = "λd/Δx", ylabel = "Heating rate",
        xaxis = :log, yaxis = :log,
        label = "",
        lc = :black, lw = 2, ls = :dash, title = "Heating rate vs λd/Δx"
    )

    scatter!(p_rate_N, λd_Δx, heating_rates, color = :black, label = "")
    savefig(p_rate_N, "results/heating_rate.svg")

    display(p_rate_N)

    p_Nppc = plot(;
        yaxis = :log, legend = :outerright,
        title = "Kinetic energy vs particles/cell (Δx = 2π/64)",
        xlabel = "tωₚ", ylabel = "Kinetic energy (norm.)"
    )

    for N_ppc in N_ppcs
        t, T = plasma_heating(;N = 64, N_ppc, thermal_velocity, tmax, Δt, xmax)
        plot!(p_Nppc, t, T, label = "Nppc=$N_ppc")
    end

    savefig(p_Nppc, "results/heating_curve_Nppc.svg")

    display(p_Nppc)
end

let
#=========================
  Non-linear plasma waves
=========================#

    function nonlinear_plasma_wave_exact(x, L, amplitude, wavenumber, max_modes=20)
        n = 0.0
        coeff = 2π * wavenumber / L
        for m in 0:max_modes
            d_dx_D_m = zero(Complex{Float64})
            for k in 0:m
                d_dx_D_m += binomial(m, k) * (2*k - m)^m * exp(im *(2*k - m)*coeff*x)
            end
            d_dx_D_m *= im^m / 2^m
            n += (-1)^m / factorial(big(m)) * (amplitude)^m * real(d_dx_D_m)
        end
        return n
    end

    N = 128
    N_ppc = 64
    N_p = N * N_ppc
    xmax = 2π
    amplitude = 0.5
    k = 1
    Δt = 0.01
    tmax = 4π
    wavenumber = 1

    particles, fields, grid = ParticleInCell.initialize(N_p, N_p, N, 0.0, xmax;
        perturbation_amplitude = amplitude,
        perturbation_wavenumber = k,
        perturbation_speed = 1 / wavenumber,
    )

    n_expected = nonlinear_plasma_wave_exact.(grid.x, xmax, amplitude, k)

    p = plot(; xlabel = "xωₚ/c", ylabel = "δn/n₀, e E m/ ωₚc")
    plot!(p, grid.x, fields.ρ .- 1, label = "Numerical density", lw = 2)
    if amplitude ≤ 0.5
        plot!(p, grid.x, n_expected .- 1, ls = :dash, label = "Analytical density", lw = 2, lc = :red)
    end
    plot!(p, grid.x, fields.Ex, label = "Electic field", lw = 2, lc = :black)
    display(p)

    savefig(p, "results/nonlinear_wave_exact_amplitude=$amplitude.svg")

    # check both standing and travelling waves
    cold_plasma_wave(;wave_speed = 0, suffix = "nonlinear_standing_k=1", N, N_ppc, xmax, amplitude, wavenumber, tmax, Δt)
end


let
#===============================
  Problem 3: Warm plasma waves
===============================#

    function warm_plasma_waves(;N=256, N_ppc=256, xmax=2π, Δt=0.1, tmax=20π, amplitude=0.05, wavenumber=20, wave_speed=0, thermal_velocity=0.0125, suffix)
        # Initialize particles with maxwellian Vdf and sinusoidal perturbation
        N_p = N * N_ppc

        particles, fields, grid = ParticleInCell.initialize(N_p, N_p, N, 0.0, xmax)

        ParticleInCell.maxwellian_vdf!(particles, thermal_velocity)
        ParticleInCell.perturb!(particles, amplitude, wavenumber, wave_speed/wavenumber, xmax)


        num_timesteps = ceil(Int, tmax / Δt)

        E_cache = zeros(N, num_timesteps+1)
        δρ_cache = zeros(N, num_timesteps+1)
        v_cache = zeros(N_p, num_timesteps+1)
        x_cache = zeros(N_p, num_timesteps+1)
        T_cache = zeros(num_timesteps+1)

        E_cache[:, 1] = fields.Ex
        δρ_cache[:, 1] = fields.ρ .- 1
        v_cache[:, 1] = particles.vx
        x_cache[:, 1] = particles.x
        T_cache[1] = kinetic_energy(particles, xmax)

        for i in 2:num_timesteps+1
            ParticleInCell.update!(particles, fields, grid, Δt)
            E_cache[:, i] = copy(fields.Ex)
            δρ_cache[:, i] = copy(fields.ρ) .- 1.0
            v_cache[:, i] = copy(particles.vx)
            x_cache[:, i] = copy(particles.x)
            T_cache[i] = kinetic_energy(particles, xmax)
        end

        t = LinRange(0, tmax, num_timesteps+1)

        contour_E = heatmap(t, grid.x, E_cache, xlabel = "tωₚ", ylabel = "xωₚ/c", c=:balance, linewidth=0, title = "eE / mcωₚ", right_margin=10mm)
        contour_ρ = heatmap(t, grid.x, δρ_cache, xlabel = "tωₚ", ylabel = "xωₚ/c", c=:balance, linewidth=0, title = "δn / n₀", right_margin=10mm)

        savefig(contour_E, "results/E_warm_$(suffix).svg")
        savefig(contour_ρ, "results/n_warm_$(suffix).svg")

        display(contour_E)
        display(contour_ρ)

        return E_cache, δρ_cache
    end

    N = 256
    N_ppc = 256
    xmax = 2π
    Δx = xmax / N
    Δt = 0.1
    thermal_velocity = 0.0125
    wavenumber = 20
    wave_speed = 0
    tmax = 20π

    E_standing, ρ_standing = warm_plasma_waves(;N, N_ppc, xmax, Δt, thermal_velocity, wavenumber, wave_speed=0, tmax, suffix="standing")
    E_travelling, ρ_travelling = warm_plasma_waves(;N, N_ppc, xmax, Δt, thermal_velocity, wavenumber, wave_speed=20*thermal_velocity, tmax, suffix="travelling")

    Nt = size(E_travelling, 2)
    @show Nt
    ks = (2π * fftshift(fftfreq(N, 1/Δx)))[N÷2+1:end]
    ωs = (2π * fftshift(fftfreq(Nt, 1/Δt)))[Nt÷2+1:end]

    kmax = 60
    ωmax = π

    last_ω_ind = findfirst(>(ωmax), ωs)
    last_k_ind = findfirst(>(kmax), ks)

    ks = ks[2:last_k_ind]
    ωs = ωs[1:last_ω_ind]

    for (ρ, E, suffix) in zip([ρ_standing, ρ_travelling], [E_standing, E_travelling], ["standing", "travelling"])
        for (q, name) in zip([ρ, E], ["n", "E"])

            q̃ = (abs.(fftshift(fft(q))).^2)[N÷2+1:end, Nt÷2+1:end]'[1:last_ω_ind, 2:last_k_ind]

            p = plot(xlabel = "k", ylabel = "ω")
            heatmap!(p, ks, ωs, log2.(q̃), title = "log₂|ℱ($(name))|² ($suffix)", xlims = (1, kmax), ylims = (0,ωmax))
            plot!(
                p, ks, sqrt.(1.0 .+ 3*thermal_velocity^2*ks.^2);
                lw = 2, lc = :blue, ls = :dash, label = "Bohm-Gross dispersion"
            )

            display(p)

            savefig(p, "results/$(name)_$(suffix)_bohm_gross.svg")
        end
    end
end

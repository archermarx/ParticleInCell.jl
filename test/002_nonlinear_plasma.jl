using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

include("001_cold_plasma.jl")

const RESULT_PATH_002 = mkpath("test/results/002")

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

let
    #===============================
    Test case 002: Nonlinear plasma waves
    ===============================#
    # Generate plots for test case 002

    N = 128
    N_ppc = 64
    N_p = N * N_ppc
    xmax = 2π
    tmax = 4π
    amplitude = 0.5
    Δt = 0.01
    path = RESULT_PATH_002
    wavenumber = 1

    common_options = (;
        N, N_ppc, xmax, tmax, amplitude, Δt, path, wavenumber
    ) 

    # First, check to make sure initial condition is correctly computed
    particles, fields, grid = ParticleInCell.initialize(N_p, N, xmax;
        perturbation_amplitude = amplitude,
        perturbation_wavenumber = 1,
        perturbation_speed = 1,
    )

    n_expected = nonlinear_plasma_wave_exact.(grid.x, xmax, amplitude, wavenumber)

    p = plot(;
        xlabel = "xωₚ/c",
        ylabel = "δn/n₀, eE/mcωₚ",
        size = (1.5 * VERTICAL_RES, VERTICAL_RES), PLOT_SCALING_OPTIONS...,
        margin = 20Plots.mm
    )
    plot!(p, grid.x, fields.ρ .- 1, label = "Numerical density", lw = 4)
    if amplitude ≤ 0.5
        plot!(p, grid.x, n_expected .- 1, ls = :dash, label = "Analytical density", lw = 4, lc = :red)
    end
    plot!(p, grid.x, fields.Ex, label = "Electic field", lw = 4, lc = :black)
    display(p)
    
    savefig(p, joinpath(path, "nonlinear_amplitude=$amplitude.png"))

    # Then, check that standing and travelling waves work correctly
    cold_plasma_wave(;suffix = "travelling_k=1", wave_speed = 1, common_options...)
    cold_plasma_wave(;suffix = "standing_k=1", wave_speed = 0, common_options...)
end
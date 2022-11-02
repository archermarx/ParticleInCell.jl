using ParticleInCell
using Test
using Plots
using Statistics
using FFTW
using Plots.PlotMeasures

function allapprox(x::AbstractVector{T1}, y::AbstractVector{T2}, atol = sqrt(max(eps(T1), eps(T2)))) where {T1, T2}
    return all(isapprox.(x, y, atol=atol))
end

function compute_frequency(signal, Δt)
    N = length(signal)
    fs = 1 / Δt
    F = abs.(fftshift(fft(signal))[N÷2+1:end])
    freqs = fftshift(fftfreq(N, fs))[N÷2+1:end]
    peak_freq = freqs[argmax(F)]
    return freqs, F, peak_freq
end

@testset "Particle initialization" begin
    xmin, xmax = 0.0, 1.0
    num_gridpts = 100

    grid = ParticleInCell.Grid(;xmin, xmax, num_gridpts)

    # 1. When number of particles equals number of grid points, then particles are
    # initialized on cell centers (not hard-coded, but the way i distribute particles)
    # should have this property
    num_particles = num_gridpts
    max_particles = 2*num_particles
    particles = ParticleInCell.Particles(num_particles, max_particles, grid)
    (;x, vx, vy, Ex, Ey, Bz) = particles

    # Check that arrays have correct length
    @test length(x) == max_particles
    @test length(vx) == max_particles
    @test length(vy) == max_particles
    @test length(Ex) == max_particles
    @test length(Ey) == max_particles
    @test length(Bz) == max_particles
    @test particles.num_particles == num_particles

    # Check that velocities and forces are correctly initialized
    @test all(vx .== 0.0)
    @test all(vy .== 0.0)
    @test all(Ex .== 0.0)
    @test all(Ey .== 0.0)
    @test all(Bz .== 0.0)

    # Check that positions are uniformly-distributed
    @test isapprox(x[1], xmin+grid.Δx/2, atol = sqrt(eps(Float64)))
    @test isapprox(x[num_particles], xmax-grid.Δx/2, atol = sqrt(eps(Float64)))
    @test all(x[num_particles+1:end] .== 0.0)
    @test all(diff(x[1:num_particles]) .≈ grid.Δx)

    # 2. Otherwise, the distance between particles is still uniform
    num_particles = 701
    max_particles = 2 * num_particles
    particles = ParticleInCell.Particles(num_particles, max_particles, grid)

    diffs = [diff(particles.x[1:num_particles]); particles.x[1] - (particles.x[num_particles] - grid.L)]
    all_diffs_equal = true
    diff1 = diffs[1]
    for i in 2:lastindex(diffs)
        if !(diffs[i] ≈ diff1)
            all_diffs_equal = false
            break
        end
    end
    @test all_diffs_equal
end

@testset "Field initialization" begin
    num_gridpts = 101
    fields = ParticleInCell.Fields(num_gridpts)

    (;ρ, jx, jy, Ex, Ey, Bz, ϕ) = fields

    # Check that arrays are correctly initialized to zero
    @test ρ == zeros(num_gridpts)
    @test jx == zeros(num_gridpts)
    @test jy == zeros(num_gridpts)
    @test Ex == zeros(num_gridpts)
    @test Ey == zeros(num_gridpts)
    @test Bz == zeros(num_gridpts)
    @test ϕ == zeros(num_gridpts)
end

@testset "Grid initialization" begin
    xmin = 5.0
    xmax = 10.0
    num_gridpts = 5

    @test ParticleInCell.Grid(;xmin, xmax, num_gridpts).Δx == 1.0
    @test ParticleInCell.Grid(;xmin, xmax, num_gridpts).L == 5.0
end

@testset "Particle location" begin
    xmin = 2.0
    xmax = 3.0
    Δx = 0.2

    num_gridpts = 6

    @test all(ParticleInCell.locate_particle(2.0, Δx, xmin, num_gridpts) .≈ (1, 2, 2.0, 2.2))
    @test all(ParticleInCell.locate_particle(3.0 - sqrt(eps(Float64)), Δx, xmin, num_gridpts) .≈ (5, 6, 2.8, 3.0))
    @test all(ParticleInCell.locate_particle(3.0 + sqrt(eps(Float64)), Δx, xmin, num_gridpts) .≈ (6, 1, 3.0, 3.2))
    @test all(ParticleInCell.locate_particle(3.1, Δx, xmin, num_gridpts) .≈ (6, 1, 3.0, 3.2))
    @test all(ParticleInCell.locate_particle(1.9, Δx, xmin, num_gridpts) .≈ (6, 1, 1.8, 2.0))
    @test all(ParticleInCell.locate_particle(2.3, Δx, xmin, num_gridpts) .≈ (2, 3, 2.2, 2.4))
    @test all(ParticleInCell.locate_particle(2.5, Δx, xmin, num_gridpts) .≈ (3, 4, 2.4, 2.6))
    @test all(ParticleInCell.locate_particle(2.7, Δx, xmin, num_gridpts) .≈ (4, 5, 2.6, 2.8))
    @test all(ParticleInCell.locate_particle(2.9, Δx, xmin, num_gridpts) .≈ (5, 6, 2.8, 3.0))
end

@testset "Linear weighting" begin
    xmin = 7.0
    xmax = 19.0
    Δx = 1.0
    num_gridpts = length(xmin:Δx:xmax)

    particles_per_cell = 10
    num_particles = particles_per_cell * num_gridpts

    W = 1 / particles_per_cell

    for t in 0.01:0.01:0.99
        # Make sure that this weighing scheme properly locates particles off of the right boundary
        j, j_plus_1, δⱼ, δⱼ₊₁ = ParticleInCell.linear_weighting(xmin - t * Δx, Δx, xmin, num_gridpts)
        @test j == num_gridpts
        @test j_plus_1 == 1
        @test δⱼ ≈ t  / Δx
        @test δⱼ₊₁ ≈ (1 - t) / Δx

        # Make sure that this weighing scheme properly locates particles off of the left boundary
        j, j_plus_1, δⱼ, δⱼ₊₁ = ParticleInCell.linear_weighting(xmax + t * Δx, Δx, xmin, num_gridpts)
        @test j == num_gridpts
        @test j_plus_1 == 1
        @test δⱼ ≈ (1 - t) / Δx
        @test δⱼ₊₁ ≈ t / Δx

        # Check for interior points
        for i in 1:num_gridpts-1
            xi = xmin + (i - 1) * Δx
            j, j_plus_1, δⱼ, δⱼ₊₁ = ParticleInCell.linear_weighting(xi + t * Δx, Δx, xmin, num_gridpts)
            @test j == i
            @test j_plus_1 == i+1
            @test δⱼ ≈ (1 - t) / Δx
            @test δⱼ₊₁ ≈ t / Δx
        end
    end
end

# TODO: add tests for when number of particles is less than the number of cells
@testset "Charge and current density initialization" begin
    xmin, xmax = 0.0, 1.0
    num_gridpts = 100

    for particles_per_cell in 1.0:0.2:10
        num_particles = round(Int, particles_per_cell*num_gridpts)
        max_particles = 2*num_particles

        # Initialize particles, grid, fields
        particles, fields, grid = ParticleInCell.initialize(num_particles, max_particles, num_gridpts, xmin, xmax)

        # check that charge density is equal to 1 when particles are uniformly initialized
        @test all(isapprox.(fields.ρ, 1.0, rtol = 1/particles_per_cell^2))

        # check that current denstiy is zero
        @test all(isapprox.(fields.jx, 0.0, rtol = 1/particles_per_cell^2))
        @test all(isapprox.(fields.jy, 0.0, rtol = 1/particles_per_cell^2))
    end
end

# If particles are initialized uniformly, they shouldn't move at all.
@testset "No self-acceleration" begin
    xmin = 0.0
    xmax = 1.0
    Δx = 0.01
    Δt = 0.01
    particles_per_cell = 5
    num_gridpts = 101
    num_particles = particles_per_cell * num_gridpts
    max_particles = 3 * num_particles

    particles, fields, grid = ParticleInCell.initialize(num_particles, max_particles, num_gridpts, xmin, xmax)

    x0 = copy(particles.x)
    vx0 = copy(particles.vx)
    vy0 = copy(particles.vy)
    Ex0_particle = copy(particles.Ex)
    Ey0_particle = copy(particles.Ey)
    Bz0_particle = copy(particles.Bz)

    ρ0 = copy(fields.ρ)
    Ex0 = copy(fields.Ex)
    Ey0 = copy(fields.Ey)
    jx0 = copy(fields.jx)
    jy0 = copy(fields.jy)
    ϕ0 = copy(fields.ϕ)
    Bz0 = copy(fields.Bz)

    # Run for num_timesteps timesteps to make sure nothing changes
    num_timesteps = 100
    for i in 1:num_timesteps
        ParticleInCell.update!(particles, fields, particles, fields, grid, Δt)
    end

    # Check that particle positions and momenta have not changed
    @test allapprox(x0, particles.x)
    @test allapprox(vx0, particles.vx)
    @test allapprox(vy0, particles.vy)

    # Check that forces on particles have not changed
    @test allapprox(Ex0_particle, particles.Ex)
    @test allapprox(Ey0_particle, particles.Ey)
    @test allapprox(Bz0_particle, particles.Bz)

    # Check that fields have not changed
    @test allapprox(ρ0, fields.ρ)
    @test allapprox(Ex0, fields.Ex)
    @test allapprox(Ey0, fields.Ey)
    @test allapprox(jx0, fields.jx)
    @test allapprox(jy0, fields.jy)
    @test allapprox(ϕ0, fields.ϕ)
    @test allapprox(Bz0, fields.Bz)
end


@testset "Gyro orbit preservation (particle pusher)" begin

    # test neutrality on gyro-orbits (single particle)
    num_particles = 1
    num_gridpts = 10
    xmin = -2.0
    xmax = 2.0

    grid = ParticleInCell.Grid(;xmin, xmax, num_gridpts)
    particles = ParticleInCell.Particles(num_particles, num_particles, grid)

    # particle has y velocity of 1 and the background magnetic field has strength 1
    particles.vy[1] = 1.0

    Bz = 1.0
    particles.Bz[1] = 1.0

    @test particles.x[1] ≈ 0.5 * (xmin + xmax)
    @test particles.Ex[1] ≈ 0.0
    @test particles.Ey[1] ≈ 0.0

    @test length(particles.x) == 1

    tmax = 2*π
    num_timesteps = 100
    Δt = tmax / num_timesteps

    particle_cache = [deepcopy(particles) for i in 1:num_timesteps+1]
    for i in 1:num_timesteps
        # push particles
        ParticleInCell.push_particles!(particle_cache[i+1], particle_cache[i], grid, Δt)
    end

    velocity_magnitudes = [hypot(p.vx[1],p.vy[1]) for p in particle_cache]

    # check that velocity magnitude has not changed from 1 (i.e. that energy is conserved)
    @test all(velocity_magnitudes .≈ 1)

    # extract x, vx, vy, t
    xs = [p.x[1] for p in particle_cache]
    vxs = [p.vx[1] for p in particle_cache]
    vys = [p.vy[1] for p in particle_cache]
    ts = LinRange(0, tmax, num_timesteps+1)

    # check against analytic solutions
    @test all(@. isapprox(vys, cos(ts), atol = 1 / num_timesteps))
    @test all(@. isapprox(vxs, sin(ts), atol = 1 / num_timesteps))
end


function test_two_particle_oscillation(num_gridpts, xmax, perturbation, max_time)
    xmin = 0.0
    Δx = (xmax - xmin) / num_gridpts
    Δt = 0.1
    num_particles = 2
    max_particles = 2
    num_timesteps = ceil(Int, max_time/Δt)

    particles, fields, grid = ParticleInCell.initialize(num_particles, max_particles, num_gridpts, xmin, xmax)

    ts = LinRange(0, max_time / 2π, num_timesteps)
    x1s = zeros(num_timesteps)
    x2s = zeros(num_timesteps)
    v1s = zeros(num_timesteps)
    v2s = zeros(num_timesteps)

    particles.x[1] += perturbation
    particles.x[2] -= perturbation

    for i in 1:num_timesteps
        ParticleInCell.update!(particles, fields, particles, fields, grid, Δt)
        x1s[i] = particles.x[1]
        x2s[i] = particles.x[2]
        v1s[i] = particles.vx[1]
        v2s[i] = particles.vx[2]
    end

    # Get p2p amplitude of oscillation
    a1 = maximum(x1s) - minimum(x1s)
    a2 = maximum(x2s) - minimum(x2s)

    # Get peak freq of oscillation
    _, _, f1 = compute_frequency(x1s .- mean(x1s), Δt)
    _, _, f2 = compute_frequency(x2s .- mean(x2s), Δt)
    return f1, f2, a1, a2
end

@testset "Two-particle harmonic oscillation" begin
    # Check that two particles exhibit simple harmonic motion with frequency ω = 1 rad/s
    # This should be the same for all grid resolutions and domain sizes
    for xmax in [0.01, 0.05, 0.1, 0.5, 1.0, π, 2π]
        #=
        num_gridpts = 101
        perturbation = 0.1
        =#
        for perturbation in [0.01, 0.02, 0.05, 0.1]
            for num_gridpts in [11, 31, 51, 71, 101]
                f1, f2, a1, a2 = test_two_particle_oscillation(num_gridpts, xmax, perturbation * xmax, 2π)

                # Convert frequency to rad/s
                ω1 = 2π * f1
                ω2 = 2π * f2

                # Check that peak freqency is 1 rad/s
                @test isapprox(ω1, 1, atol=0.01)
                @test isapprox(ω2, 1, atol=0.01)

                # check that oscillatory amplitude is 2 * perturbation
                @test isapprox(a1, 2*perturbation*xmax, rtol=0.01)
                @test isapprox(a2, 2*perturbation*xmax, rtol=0.01)
            end
        end
    end
end

@testset "Density computation with perturbation" begin
    # This test checks that if we perturb all particle positions sinusoidally by a small value,
    # then the resulting charge density should follow an expected form
    xmin = 0.0
    Δx = 0.01
    Δt = 0.01
    particles_per_cell = 51
    num_gridpts = 101
    num_particles = particles_per_cell * num_gridpts
    max_particles = 1 * num_particles

    for xmax in [0.1, 1.0, π]
        Δx = (xmax - xmin) / num_gridpts
        L = xmax - xmin + Δx
        for perturb_amp in [0.015, 0.01, 0.005, 0.001]
            for perturb_k in [1, 2, 3, 4, 5]
                particles, fields, grid = ParticleInCell.initialize(
                    num_particles, max_particles, num_gridpts, xmin, xmax,
                    perturbation_amplitude = perturb_amp, perturbation_wavenumber = perturb_k
                )

                ρ0 = mean(fields.ρ)

                ρ_computed = fields.ρ
                ρ_expected = ρ0 .+ perturb_amp .* sin.(2π * perturb_k .* grid.x / L)

                #=
                p = plot(grid.x, ρ_expected, label = "Expected charge density")
                plot!(grid.x, fields.ρ, label = "Calculated charge density", legend = :outertop)
                display(p)
                =#

                @test sum(((ρ_computed .- ρ_expected) / num_gridpts).^2) < sqrt(eps(Float64))
            end
        end
    end
end

@testset "Periodic BCs on moving uniform charge" begin
    xmin = 0.0
    xmax = 1.0
    Δx = 1.0
    Δt = 0.01
    particles_per_cell = 2
    num_gridpts = 11
    num_particles = particles_per_cell * num_gridpts
    max_particles = 3 * num_particles

    particles, fields, grid = ParticleInCell.initialize(num_particles, max_particles, num_gridpts, xmin, xmax)

    # add rightward-moving velocity to all particles
    for i in 1:num_particles
        particles.vx[i] = 1.0
    end

    num_timesteps = 100

    E_cache = zeros(num_gridpts, num_timesteps+1)
    ρ_cache = zeros(num_gridpts, num_timesteps+1)
    x_cache = zeros(max_particles, num_timesteps+1)
    vx_cache = zeros(max_particles, num_timesteps+1)

    E_cache[:, 1] = fields.Ex
    ρ_cache[:, 1] = fields.ρ
    x_cache[:, 1] = particles.x
    vx_cache[:, 1] = particles.vx

    for i in 2:num_timesteps+1
        ParticleInCell.update!(particles, fields, grid, Δt)
        E_cache[:, i] = copy(fields.Ex)
        ρ_cache[:, i] = copy(fields.ρ)
        x_cache[:, i] = copy(particles.x)
        vx_cache[:, i] = copy(particles.vx)
    end

    @test allapprox(E_cache[:, end], E_cache[:, 1])
    @test allapprox(ρ_cache[:, end], ρ_cache[:, 1])
    @test allapprox(vx_cache[:, end], vx_cache[:, 1])

    #=p = plot(ρ_cache[:, 1], label = "t = 0", ylims = [0.0, 1.0])
    plot!(ρ_cache[:, 41], label = "t = $Δt", legend = :outertop)
    display(p)=#
end

function worksheet5(;N=101, N_ppc=100, xmax=π, Δt=0.1, tmax=2π, amplitude=0.01, wavenumber=1, wave_speed=1.0)

    N_p = N_ppc * N

    particles, fields, grid = ParticleInCell.initialize(
        N_p, N_p*2, N, 0.0, xmax;
        perturbation_amplitude = amplitude,
        perturbation_wavenumber = wavenumber,
        perturbation_speed = wave_speed,
    )

    num_timesteps = ceil(Int, tmax / Δt)

    E_cache = zeros(N, num_timesteps+1)
    δρ_cache = zeros(N, num_timesteps+1)
    v_cache = zeros(N_p * 2, num_timesteps+1)
    x_cache = zeros(N_p * 2, num_timesteps+1)

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
    dt_fac = round(Int, 0.1 / Δt) * 2
    @show dt_fac
    t2s = dt_fac:dt_fac:32*dt_fac

    p = plot(grid.x, δρ_cache[:, 1], label = "δn₀", xlabel = "xωₚ/c", ylabel = "δn/n₀, eE/mcωₚ", color = :red, legend = :outertop)
    #plot!(grid.x, E_cache[:, 1], label = "E₀", legend = :outertop, color = :blue)
    for t2 in t2s
        plot!(p, grid.x, δρ_cache[:, t2], label = "", color = :red, ls = :dash)
        #plot!(p, grid.x, E_cache[:, t2], label = "", color = :blue, ls = :dash)
    end
    display(p)

    #=p2 = scatter(x_cache[1:N_p, 1], v_cache[1:N_p, 1], xlabel = "xωₚ/c")
    scatter!(x_cache[1:N_p, t2], v_cache[1:N_p, t2])
    display(p2)=#

    contour_E = contourf(t, grid.x, E_cache, xlabel = "tωₚ", ylabel = "xωₚ/c", c=:balance, linewidth=0, title = "eE / mcωₚ", right_margin=10mm)
    contour_ρ = contourf(t, grid.x, δρ_cache, xlabel = "tωₚ", ylabel = "xωₚ/c", c=:balance, linewidth=0, title = "δn / n₀", right_margin=10mm)

    display(contour_E)
    display(contour_ρ)
end

# TODO: fix oscillation in wave envelope
# independent of xmax
# independent of perturbation amplitude
# independent of gridpoints or number of particles
# independent of wavenumber when wave_speed = 1/wavenumber
# independent of end time
# depends on timestep!

begin
    wavenumber = 1
    worksheet5(N = 50 * wavenumber, N_ppc = 50*wavenumber, xmax = 2π, amplitude = 0.01, wavenumber = wavenumber, wave_speed = 1/wavenumber, tmax=4π, Δt = 0.01)
end

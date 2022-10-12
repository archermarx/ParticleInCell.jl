using ParticleInCell
using Test

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
    @test isapprox(x[1], xmin, atol = sqrt(eps(Float64)))
    @test isapprox(x[num_particles], xmax, atol = sqrt(eps(Float64)))
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
    num_gridpts = 6

    @test ParticleInCell.Grid(;xmin, xmax, Δx = 1.0).num_gridpts == num_gridpts
    @test ParticleInCell.Grid(;xmin, xmax, num_gridpts).Δx == 1.0
    @test ParticleInCell.Grid(;xmin, xmax, num_gridpts).L == 6.0
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
        ParticleInCell.update!(particles, fields, grid, Δx)
    end

    # Check that particle positions and momenta have not changed
    @test all(x0 .≈ particles.x)
    @test all(vx0 .≈ particles.vx)
    @test all(vy0 .≈ particles.vy)

    # Check that forces on particles have not changed
    @test all(Ex0_particle .≈ particles.Ex)
    @test all(Ey0_particle .≈ particles.Ey)
    @test all(Bz0_particle .≈ particles.Bz)

    # Check that fields have not changed
    @test all(ρ0 .≈ fields.ρ)
    @test all(Ex0 .≈ fields.Ex)
    @test all(Ey0 .≈ fields.Ey)
    @test all(jx0 .≈ fields.jx)
    @test all(jy0 .≈ fields.jy)
    @test all(ϕ0 .≈ fields.ϕ)
    @test all(Bz0 .≈ fields.Bz)
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
        ParticleInCell.push_particles!(particle_cache[i+1], particle_cache[i], Δt)
    end

    velocity_magnitudes = [hypot(p.vx[1],p.vy[1]) for p in particle_cache]

    # check that velocity magnitude has not changed from 1 (i.e. that energy is conserved)
    @test all(velocity_magnitudes .≈ 1)

    xs = [p.x[1] for p in particle_cache]
    vxs = [p.vx[1] for p in particle_cache]
    vys = [p.vy[1] for p in particle_cache]
    ts = LinRange(0, tmax, num_timesteps+1)

    # check against analytic solutions
    @test all(@. isapprox(vys, cos(ts), atol = 1 / num_timesteps))
    @test all(@. isapprox(vxs, sin(ts), atol = 1 / num_timesteps))

    # TODO: verify position
end

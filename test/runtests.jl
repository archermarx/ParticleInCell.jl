using ParticleInCell
using Test

@testset "Particle initialization" begin
    xmin, xmax = 0.0, 1.0
    num_particles = 100
    max_particles = 200
    num_gridpts = 79

    x, vx, vy, Fx, Fy = ParticleInCell.initialize_particles(num_particles, max_particles, num_gridpts, xmin, xmax)

    # Check that arrays have correct length
    @test length(x) == max_particles
    @test length(vx) == max_particles
    @test length(vy) == max_particles
    @test length(Fx) == max_particles
    @test length(Fy) == max_particles

    # Check that velocities and forces are correctly initialized
    @test all(vx .== 0.0)
    @test all(vy .== 0.0)
    @test all(Fx .== 0.0)
    @test all(Fy .== 0.0)

    Δx = (xmax - xmin) / num_gridpts
    Δx_particles = (xmax - xmin + Δx) / (num_particles)

    # Check that positions are uniformly-distributed
    @test x[1] == xmin - Δx/2
    @test x[num_particles] == xmax + Δx/2 - Δx_particles
    @test all(x[num_particles+1:end] .== 0.0)
    @test all(diff(x[1:num_particles]) .≈ Δx_particles)
end

@testset "Field initialization" begin
    num_gridpts = 101
    ρ, jx, jy, Ex, Ey, B = ParticleInCell.initialize_fields(num_gridpts)

    # Check that arrays are correctly initialized
    @test ρ == zeros(num_gridpts)
    @test jx == zeros(num_gridpts)
    @test jy == zeros(num_gridpts)
    @test Ex == zeros(num_gridpts)
    @test Ey == zeros(num_gridpts)
    @test B == zeros(num_gridpts)
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
        j, j_plus_1, Xⱼ, Xⱼ₊₁, δⱼ, δⱼ₊₁ = ParticleInCell.linear_weighting(xmin - t * Δx, Δx, xmin, num_gridpts, num_particles)
        @test j == num_gridpts
        @test j_plus_1 == 1
        @test Xⱼ ≈ xmin-Δx
        @test Xⱼ₊₁ ≈ xmin
        @test δⱼ ≈ t  / Δx
        @test δⱼ₊₁ ≈ (1 - t) / Δx

        j, j_plus_1, Xⱼ, Xⱼ₊₁, δⱼ, δⱼ₊₁ = ParticleInCell.linear_weighting(xmax + t * Δx, Δx, xmin, num_gridpts, num_particles)
        @test j == num_gridpts
        @test j_plus_1 == 1
        @test Xⱼ ≈ xmax
        @test Xⱼ₊₁ ≈ xmax+Δx
        @test δⱼ ≈ (1 - t) / Δx
        @test δⱼ₊₁ ≈ t / Δx

        for i in 1:num_gridpts-1
            xi = xmin + (i - 1) * Δx
            j, j_plus_1, Xⱼ, Xⱼ₊₁, δⱼ, δⱼ₊₁ = ParticleInCell.linear_weighting(xi + t * Δx, Δx, xmin, num_gridpts, num_particles)
            @test j == i
            @test j_plus_1 == i+1
            @test Xⱼ == xi
            @test Xⱼ₊₁ == xi + Δx
            @test δⱼ ≈ (1 - t) / Δx
            @test δⱼ₊₁ ≈ t / Δx
        end
    end

end

@testset "Charge and current density initialization" begin
    xmin, xmax = 0.0, 1.0
    num_gridpts = 12
    particles_per_cell = 2
    num_particles = particles_per_cell*num_gridpts+1
    max_particles = 2*num_particles

    x, vx, vy, Fx, Fy, ρ, jx, jy, Ex, Ey, B = ParticleInCell.initialize(num_particles, max_particles, num_gridpts, xmin, xmax)

    # check that charge density is equal to 1 when particles are uniformly initialized
    @test all(isapprox.(ρ, 1.0, rtol = 1/particles_per_cell))

    # check that current denstiy is zero
    @test all(isapprox.(jx, 0.0, rtol = 1/particles_per_cell))
    @test all(isapprox.(jy, 0.0, rtol = 1/particles_per_cell))
end

@testset "No self-acceleration" begin
    xmin = 0.0
    xmax = 1.0
    Δx = 0.01
    Δt = 0.01
    particles_per_cell = 5
    num_gridpts = 101
    num_particles = particles_per_cell * num_gridpts
    max_particles = 3 * num_particles

    num_timesteps = 100

    x, vx, vy, Fx, Fy, ρ, jx, jy, Ex, Ey, Bz = ParticleInCell.initialize(num_particles, max_particles, num_gridpts, xmin, xmax)

    x0 = copy(x)
    vx0 = copy(vx)
    vy0 = copy(vy)

    for i in 1:num_timesteps
        ParticleInCell.update!(x, vx, vy, Fx, Fy, ρ, jx, jy, Ex, Ey, Bz, xmin, xmax, Δx, Δt, num_particles)
    end

    # Check that particle positions and momenta have not changed
    @test all(x0 .== x)
    @test all(vx0 .== vx)
    @test all(vy0 .== vy)
end

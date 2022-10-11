module ParticleInCell

"""
    push_particles!(x, vx, vy, Fx, Fy, num_particles)
Push particles to new positions `x` and velocities `v` as a result of forces `F`.
Uses a leapfrog scheme. Note that all quantities are normalized.
====
# Arguments
`x`: x-locations of each particle
`vx`: velocities of each particle in x direction
`vy`: velocities of each particle in y direction
`Fx`: forces on each particle in x direction
`Fy`: forces on each particle in y direction
`Δt`: time step
`num_particles`: the number of particles in the simulation
"""
function push_particles!(x, vx, vy, Fx, Fy, Δt, num_particles)

    # Loop through all particles in simulation and push them to new positions and velocities
    for i in 1:num_particles
        # 1. Update velocity to time n+1/2
        # v_i^{n+1/2} = v_i^{n-1/2} + F_i^n Δt
        vx[i] += Fx[i] * Δt
        vy[i] += Fy[i] * Δt

        # 2. Update position to time n+1
        # x_i^{n+1} = x_i^{n} + v_i^{n+1/2}Δt
        x[i] += vx[i] * Δt
    end

    return nothing
end

function initialize_particles(num_particles, max_particles, num_gridpts, xmin, xmax)
    # Initialize arrays to allow for a perscribed maximum number of particles
    x = zeros(max_particles)

    # Initial velocities and forces are zero
    vx = zeros(max_particles)
    vy = zeros(max_particles)
    Fx = zeros(max_particles)
    Fy = zeros(max_particles)

    Δx = (xmax - xmin) / num_gridpts

    # Initialize particle positions uniformly throughout the domain
    x[1:num_particles] = LinRange(xmin - Δx / 2, xmax + Δx/2, num_particles+1)[1:end-1]

    return x, vx, vy, Fx, Fy
end

function initialize_fields(num_gridpts)
    ρ = zeros(num_gridpts)
    jx = zeros(num_gridpts)
    jy = zeros(num_gridpts)
    Ex = zeros(num_gridpts)
    Ey = zeros(num_gridpts)
    B = zeros(num_gridpts)

    return ρ, jx, jy, Ex, Ey, B
end

function locate_particle(x, Δx, xmin, num_gridpts)
    j = fld(x - xmin, Δx)

    Xⱼ = Δx * j + xmin
    Xⱼ₊₁ = Xⱼ + Δx

    j = mod1(round(Int, j+1), num_gridpts)
    j_plus_1 = mod1(j+1, num_gridpts)

    return j, j_plus_1, Xⱼ, Xⱼ₊₁
end

function linear_weighting(x, Δx, xmin, num_gridpts, num_particles)
    j, j_plus_1, Xⱼ, Xⱼ₊₁ = locate_particle(x, Δx, xmin, num_gridpts)
    δⱼ = (Xⱼ₊₁ - x) / Δx
    δⱼ₊₁ = (x - Xⱼ) / Δx
    return j, j_plus_1, Xⱼ, Xⱼ₊₁, δⱼ, δⱼ₊₁
end

function interpolate_charge_to_grid!(x, vx, vy, ρ, jx, jy, xmin, xmax, num_particles)

    num_gridpts = length(ρ)

    Δx = (xmax - xmin) / num_gridpts

    W = num_gridpts / num_particles

    for i in 1:num_particles

        # Find cell center closest to but less than x[i]
        j, j_plus_1, Xⱼ, Xⱼ₊₁, δⱼ, δⱼ₊₁ = linear_weighting(x[i], Δx, xmin, num_gridpts, num_particles)

        δρⱼ = W * δⱼ
        δρⱼ₊₁ = W * δⱼ₊₁

        # assign charge using linear weighting
        ρ[j] += δρⱼ
        ρ[j_plus_1] += δρⱼ₊₁

        # assign x current density using linear weighting
        jx[j] += δρⱼ * vx[i]
        jx[j_plus_1] += δρⱼ₊₁ * vx[i]

        # assign y current density using linear weighting
        jy[j] += δρⱼ * vy[i]
        jy[j_plus_1] += δρⱼ₊₁ * vy[i]
    end

    return nothing
end

function interpolate_forces_to_particles!(x, vx, vy, Fx, Fy, ρ, Ex, Ey, Bz, xmin, xmax, num_particles)
    num_gridpts = length(ρ)
    Δx = (xmax - xmin) / num_gridpts

    for i in 1:num_particles
        # Find cell center closest to but less than x[i]
        j, j_plus_1, Xⱼ, Xⱼ₊₁, δⱼ, δⱼ₊₁ = linear_weighting(x[i], Δx, xmin, num_gridpts, num_particles)

        # Compute forces on grid points
        # x forces
        Fxⱼ   = Ex[j] + vx[j] * Bz[j]
        Fxⱼ₊₁ = Ex[j_plus_1] + vx[j_plus_1] * Bz[j_plus_1]

        # y forces
        Fyⱼ   = Ey[j] - vy[j] * Bz[j]
        Fyⱼ₊₁ = Ey[j_plus_1] - vy[j_plus_1] * Bz[j_plus_1]

        # Interpolate forces on grid points to particles
        Fx[i] = δⱼ * Fxⱼ + δⱼ₊₁ * Fxⱼ₊₁
        Fy[i] = δⱼ * Fyⱼ + δⱼ₊₁ * Fyⱼ₊₁
    end

    return nothing
end

function initialize(num_particles, max_particles, num_gridpts, xmin, xmax)
    x, vx, vy, Fx, Fy = initialize_particles(num_particles, max_particles, num_gridpts, xmin, xmax)
    ρ, jx, jy, Ex, Ey, Bz = initialize_fields(num_gridpts)

    # Compute initial charge density and current density on grid
    interpolate_charge_to_grid!(x, vx, vy, ρ, jx, jy, xmin, xmax, num_particles)

    # Solve initial electric fields

    # Compute initial forces on particles
    interpolate_forces_to_particles!(x, vx, vy, Fx, Fy, ρ, Ex, Ey, Bz, xmin, xmax, num_particles)

    return x, vx, vy, Fx, Fy, ρ, jx, jy, Ex, Ey, Bz
end

function update!(x, vx, vy, Fx, Fy, ρ, jx, jy, Ex, Ey, Bz, xmin, xmax, Δx, Δt, num_particles)

    # Push particles to new positions and velocities
    push_particles!(x, vx, vy, Fx, Fy, Δt, num_particles)

    # Interpolate charge density to grid
    interpolate_charge_to_grid!(x, vx, vy, ρ, jx, jy, xmin, xmax, num_particles)

    # Solve eletric fields on grid

    # Interpolate forces to particles
    interpolate_forces_to_particles!(x, vx, vy, Fx, Fy, ρ, Ex, Ey, Bz, xmin, xmax, num_particles)

    return nothing
end



end # module ParticleInCell

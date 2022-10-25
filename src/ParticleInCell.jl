module ParticleInCell

using FFTW

struct Grid
    xmin::Float64
    xmax::Float64
    Δx::Float64
    L::Float64
    num_gridpts::Int
    # Grid in frequency domain
    k::Vector{Float64}
    K::Vector{Float64}
    κ::Vector{Float64}
end

function Grid(; xmin, xmax, Δx=nothing, num_gridpts=nothing)
    if isnothing(Δx)
        xs = LinRange(xmin, xmax, num_gridpts)
        Δx = step(xs)
    elseif isnothing(num_gridpts)
        xs = xmin:Δx:xmax
        num_gridpts = length(xs)
    end

    # Compute domain length
    L = xmax - xmin + Δx

    # Compute grid in frequency domain
    k = 2π * fftfreq(num_gridpts)
    K = zeros(length(k))
    κ = zeros(length(k))
    for j in eachindex(k)
        K[j] = k[j] * sinc(k[j] * Δx / 2π)
        κ[j] = k[j] * sinc(k[j] * Δx / π)
    end

    return Grid(xmin, xmax, Δx, L, num_gridpts, k, K, κ)
end

struct Particles
    x::Vector{Float64}
    vx::Vector{Float64}
    vy::Vector{Float64}
    Ex::Vector{Float64}
    Ey::Vector{Float64}
    Bz::Vector{Float64}
    num_particles::Int
end

function Particles(num_particles::Int, max_particles::Int, grid::Grid)
    # Initialize arrays to allow for a perscribed maximum number of particles, in case
    # new particles are created later in the simulation.

    # Initialize particle positions uniformly throughout the domain
    x = distribute_particles(num_particles, max_particles, grid)

    # Initial velocities and forces are zero
    vx = zeros(max_particles)
    vy = zeros(max_particles)
    Ex = zeros(max_particles)
    Ey = zeros(max_particles)
    Bz = zeros(max_particles)

    return Particles(x, vx, vy, Ex, Ey, Bz, num_particles)
end

function distribute_particles(num_particles, max_particles, grid)
    x = zeros(max_particles)

    (;xmin, xmax, Δx) = grid

    x_aux = LinRange(xmin - Δx/2, xmax+Δx/2, num_particles+1)
    for i in 1:num_particles
        x[i] = 0.5*(x_aux[i]+x_aux[i+1])
    end

    return x
end

struct Fields
    ρ::Vector{Float64}
    jx::Vector{Float64}
    jy::Vector{Float64}
    Ex::Vector{Float64}
    Ey::Vector{Float64}
    ϕ::Vector{Float64}
    Bz::Vector{Float64}
    # FFT quantities
    ρ̃::Vector{Complex{Float64}}
    ϕ̃::Vector{Complex{Float64}}
    Ẽx::Vector{Complex{Float64}}
    Ẽy::Vector{Complex{Float64}}
end

function Fields(num_gridpts::Int)
    ρ = zeros(num_gridpts)
    jx = zeros(num_gridpts)
    jy = zeros(num_gridpts)
    Ex = zeros(num_gridpts)
    Ey = zeros(num_gridpts)
    ϕ = zeros(num_gridpts)
    Bz = zeros(num_gridpts)

    # Allocate arrays for fourier quantities
    ρ̃ = zeros(Complex{Float64}, num_gridpts)
    ϕ̃ = zeros(Complex{Float64}, num_gridpts)
    Ẽx = zeros(Complex{Float64}, num_gridpts)
    Ẽy = zeros(Complex{Float64}, num_gridpts)

    return Fields(ρ, jx, jy, Ex, Ey, ϕ, Bz, ρ̃, ϕ̃, Ẽx, Ẽy)
end

"""
    push_particles!(x, vx, vy, Fx, Fy, num_particles)
"""
function push_particles!(new_particles::Particles, particles::Particles, Δt)

    (;x, vx, vy, Ex, Ey, Bz, num_particles) = particles

    # Loop through all particles in simulation and push them to new positions and velocities
    for i in 1:num_particles

        t = Bz[i] * Δt / 2
        s = 2 * t / (1 + t^2)

        v_minus_x = vx[i] + 0.5 * Ex[i] * Δt
        v_minus_y = vy[i] + 0.5 * Ey[i] * Δt

        v_star_x = v_minus_x + v_minus_y * t
        v_star_y = v_minus_y - v_minus_x * t

        v_plus_x = v_minus_x + v_star_y * s
        v_plus_y = v_minus_y - v_star_x * s

        new_particles.vx[i] = v_plus_x + 0.5 * Ex[i] * Δt
        new_particles.vy[i] = v_plus_y + 0.5 * Ey[i] * Δt
        new_particles.x[i]  = x[i] + new_particles.vx[i] * Δt
    end

    return nothing
end

function locate_particle(x, Δx, xmin, num_gridpts)
    j = fld(x - xmin, Δx)

    Xⱼ = Δx * j + xmin
    Xⱼ₊₁ = Xⱼ + Δx

    j = mod1(round(Int, j + 1), num_gridpts)
    j_plus_1 = mod1(j + 1, num_gridpts)

    return j, j_plus_1, Xⱼ, Xⱼ₊₁
end

function linear_weighting(x, Δx, xmin, num_gridpts)
    j, j_plus_1, Xⱼ, Xⱼ₊₁ = locate_particle(x, Δx, xmin, num_gridpts)
    δⱼ = (Xⱼ₊₁ - x) / Δx
    δⱼ₊₁ = (x - Xⱼ) / Δx
    return j, j_plus_1, δⱼ, δⱼ₊₁
end

function interpolate_charge_to_grid!(particles::Particles, field::Fields, grid::Grid)

    (; num_gridpts, Δx, xmin) = grid
    (; num_particles, x, vx, vy) = particles
    (; ρ, jx, jy, ρ̃) = field

    W = num_gridpts / num_particles

    # zero charge and current densities
    ρ .= 0.0
    jx .= 0.0
    jy .= 0.0

    for i in 1:num_particles

        # Find cell center closest to but less than x[i]
        j, j_plus_1, δⱼ, δⱼ₊₁ = linear_weighting(x[i], Δx, xmin, num_gridpts)

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

function interpolate_fields_to_particles!(particles::Particles, fields::Fields, grid::Grid)
    (; Δx, num_gridpts, xmin) = grid
    (; num_particles, x) = particles
    (; Ex, Ey, Bz) = fields

    for i in 1:num_particles
        # Find cell center closest to but less than x[i]
        j, j_plus_1, δⱼ, δⱼ₊₁ = linear_weighting(x[i], Δx, xmin, num_gridpts)

        # Interpolate fields on grid points to particles
        particles.Ex[i] = δⱼ * Ex[j] + δⱼ₊₁ * Ex[j_plus_1]
        particles.Ey[i] = δⱼ * Ey[j] + δⱼ₊₁ * Ey[j_plus_1]
        particles.Bz[i] = δⱼ * Bz[j] + δⱼ₊₁ * Bz[j_plus_1]
    end

    return nothing
end

"""
Initialize simulation, aka allocate arrays for grid, particles, and fields
Then distribute particles and compute the initial fields
"""
function initialize(num_particles, max_particles, num_gridpts, xmin, xmax)

    grid = Grid(;num_gridpts, xmin, xmax)
    particles = Particles(num_particles, max_particles, grid)
    fields = Fields(num_gridpts)

    # Compute initial charge density and fields
    update!(particles, fields, particles, fields, grid, 0.0, push_particles=false)

    return particles, fields, grid
end


"""
Use FFT to compute potential and electric field using charge density
"""
function solve_fields_on_grid!(fields::Fields, grid::Grid)
    (;num_gridpts, K, κ) = grid
    (;ρ, ρ̃, Ex, Ey, ϕ, Ẽx, ϕ̃) = fields

    # Compute FFT of density
    ρ̃ .= ρ
    fft!(ρ̃)

    # Compute potential and electric field in frequency domain
    for j in 1:num_gridpts
        if K[j] == 0.0
            ϕ̃[j] = 0.0
        else
            ϕ̃[j]  = ρ̃[j] / K[j]^2
        end

        if κ[j] == 0.0
            Ẽx[j] = 0.0
        else
            Ẽx[j] = -im * κ[j] * ϕ̃[j]
        end
    end

    # Compute inverse fourier transforms
    ifft!(ϕ̃)
    ifft!(Ẽx)

    for j in 1:num_gridpts
        ϕ[j] = real(ϕ̃[j])
        Ex[j] = real(Ẽx[j])
        Ey[j] = 0.0     # No electric field in y direction
    end

    return nothing
end

"""
Perform one update step, which itself has four sub-steps
1. Push particles to new positions and velocities
2. Interpolate charge and current density to grid
3. Solve electric field and potential on grid
4. Interpolate electric and magnetic fields to particles
"""
function update!(new_particles::Particles, new_fields::Fields, particles::Particles, fields::Fields, grid::Grid, Δt; push_particles=true)

    if push_particles
        # Push particles to new positions and velocities
        push_particles!(new_particles, particles, Δt)
    end

    # Interpolate charge density to grid
    interpolate_charge_to_grid!(new_particles, new_fields, grid)

    # Solve eletric fields on grid
    solve_fields_on_grid!(new_fields, grid)

    # Interpolate electric and magnetic fields to particles
    interpolate_fields_to_particles!(new_particles, new_fields, grid)

    return nothing
end

end # module ParticleInCell

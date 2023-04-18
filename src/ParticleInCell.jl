module ParticleInCell

using FFTW: fft, fft!, ifft, ifft!, fftfreq, fftshift
using Statistics
using Plots
using Printf
using QuasiMonteCarlo
using Distributions: quantile, Normal, pdf

const c = 299_792_458.0
const e = 1.602176634e-19
const m_p = 1836.15267343
const m_e = 9.1093837e-31
const ϵ_0 = 8.85418782e-12
const k_B = 1.380649e-23

struct Grid
    x::AbstractVector{Float64}
    Δx::Float64
    Lx::Float64
    Ly::Float64
    Lz::Float64
    num_gridpts::Int
    # Grid in frequency domain
    k::Vector{Float64}
    K::Vector{Float64}
    κ::Vector{Float64}
end

function Grid(; Lx, Ly, Lz, num_gridpts)

    # Compute x coordinates
    x_aux = LinRange(0.0, Lx, num_gridpts+1)
    xs = [0.5 * (x_aux[i] + x_aux[i+1]) for i in 1:num_gridpts]
    Δx = xs[2] - xs[1]

    # Compute grid in frequency domain
    k = 2π * fftfreq(num_gridpts) / Δx
    K = zeros(length(k))
    κ = zeros(length(k))
    for j in eachindex(k)
        K[j] = k[j] * sinc(k[j] * Δx / 2π)
        κ[j] = k[j] * sinc(k[j] * Δx / π)
    end

    return Grid(xs, Δx, Lx, Ly, Lz, num_gridpts, k, K, κ)
end

struct Particles
    # Positions
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    # Velocities
    vx::Vector{Float64}
    vy::Vector{Float64}
    vz::Vector{Float64}
    # Fields
    Ex::Vector{Float64}
    Ey::Vector{Float64}
    Bz::Vector{Float64}
    # Other
    weights::Vector{Float64}
    mass::Float64
    charge::Float64
    num_particles::Int
end

function Particles(num_particles::Int, grid::Grid, mass = 1, charge = 1)
    # Initialize arrays to allow for a perscribed maximum number of particles, in case
    # new particles are created later in the simulation.

    # Initialize particle positions uniformly throughout the domain
    x = distribute_particles(num_particles, grid)
    y = rand(num_particles) .* grid.Ly
    z = rand(num_particles) .* grid.Lz

    # Initial velocities and forces are zero
    vx = zeros(num_particles)
    vy = zeros(num_particles)
    vz = zeros(num_particles)
    Ex = zeros(num_particles)
    Ey = zeros(num_particles)
    Bz = zeros(num_particles)

    # Particle weights
    N_ppc = num_particles / grid.num_gridpts
    weight = 1 / N_ppc
    weights = fill(weight, num_particles)
    return Particles(x, y, z, vx, vy, vz, Ex, Ey, Bz, weights, mass, charge, num_particles)
end

function distribute_particles(num_particles, grid)
    x = zeros(num_particles)

    (; Lx) = grid

    x_aux = LinRange(0, Lx, num_particles+1)
    for i in 1:num_particles
        x[i] = 0.5*(x_aux[i]+x_aux[i+1])
    end

    return x
end

function perturb!(particles, amplitude, wavenumber, wavespeed, L)

    wavenumber_aux = 2π * wavenumber / L
    wavespeed_aux = wavespeed * L / 2π

    for i in eachindex(particles.x)
        isnan(particles.x[i]) && continue # make sure we only perturb particles that exist
        δx = amplitude / wavenumber_aux * cos(wavenumber_aux * particles.x[i])
        δv = wavespeed_aux * amplitude * sin(wavenumber_aux * particles.x[i])
        particles.x[i] += δx
        particles.vx[i] += δv
    end
    return nothing
end


# This generates low-discrepancy samples from a normal distribution
function sample_normal_quiet(D, N, R = QuasiMonteCarlo.NoRand())
    # First sample from uniform distribution
    us = QuasiMonteCarlo.sample(N, zeros(D), ones(D), SobolSample(; R))

    # Then, use the inverse normal CDF to transform these samples
    # to samples from a Normal distribution
    xs = quantile.(Normal(), us)

    return xs
end

function sample_maxwellian_quiet(D, N, v_thermal, R = QuasiMonteCarlo.NoRand())
    vs = sample_normal_quiet(D, N, R)
    vs .*= v_thermal
    return vs
end

function maxwellian_vdf!(particles, thermal_velocity; quiet = false)
    N = particles.num_particles

    if quiet
        velocities = sample_maxwellian_quiet(3, N, thermal_velocity, QuasiMonteCarlo.Shift())
    else
        velocities = randn(3, particles.num_particles) .* thermal_velocity
    end

    @. @views particles.vx = velocities[1, :]
    @. @views particles.vy = velocities[2, :]
    @. @views particles.vz = velocities[3, :]
   
    return nothing
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
    # Scaling quantities
    charge_per_particle::Int
end

function Fields(num_gridpts::Int; charge_per_particle = 1)
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

    return Fields(ρ, jx, jy, Ex, Ey, ϕ, Bz, ρ̃, ϕ̃, Ẽx, Ẽy, charge_per_particle)
end


function push_particles!(new_particles::Particles, particles::Particles, grid::Grid, Δt)

    (;x, y, z, vx, vy, vz, Ex, Ey, Bz, mass, charge, num_particles) = particles
    (; Lx, Ly, Lz) = grid

    q_m = charge / mass

    # Loop through all particles in simulation and push them to new positions and velocities
    Threads.@threads for i in 1:num_particles

        t = q_m * Bz[i] * Δt / 2
        s = 2 * t / (1 + t^2)

        x_new = x[i] + 0.5 * vx[i] * Δt
        y_new = y[i] + 0.5 * vy[i] * Δt
        z_new = z[i] + 0.5 * vz[i] * Δt

        v_minus_x = vx[i] + 0.5 * q_m * Ex[i] * Δt
        v_minus_y = vy[i] + 0.5 * q_m * Ey[i] * Δt

        v_star_x = v_minus_x + v_minus_y * t
        v_star_y = v_minus_y - v_minus_x * t

        v_plus_x = v_minus_x + v_star_y * s
        v_plus_y = v_minus_y - v_star_x * s

        new_particles.vx[i] = v_plus_x + 0.5 * q_m * Ex[i] * Δt
        new_particles.vy[i] = v_plus_y + 0.5 * q_m * Ey[i] * Δt
        new_particles.vz[i] = vz[i]

        x_new = x_new + 0.5 * new_particles.vx[i] * Δt
        y_new = y_new + 0.5 * new_particles.vy[i] * Δt
        z_new = z_new + 0.5 * new_particles.vz[i] * Δt

        # Apply periodic boundary conditions for particles
        x_new = x_new - floor(x_new / Lx) * Lx
        y_new = y_new - floor(y_new / Ly) * Ly
        z_new = z_new - floor(z_new / Lz) * Lz

        new_particles.x[i] = x_new
        new_particles.y[i] = y_new
        new_particles.z[i] = z_new
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

    (; num_gridpts, Δx) = grid
    (; num_particles, x, vx, vy, charge, weights) = particles
    (; ρ, jx, jy, charge_per_particle) = field

    # zero charge and current densities
    Threads.@threads for j in 1:num_gridpts
        ρ[j] = 0.0
        jx[j] = 0.0
        jy[j] = 0.0
    end

    for i in 1:num_particles
        # Find cell center closest to but less than x[i]
        j, j_plus_1, δⱼ, δⱼ₊₁ = linear_weighting(x[i], Δx, 0.0, num_gridpts)
        
        W = charge * charge_per_particle * weights[i]

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
    (; Δx, num_gridpts) = grid
    (; num_particles, x) = particles
    (; Ex, Ey, Bz) = fields

    Threads.@threads for i in 1:num_particles
        # Find cell center closest to but less than x[i]
        j, j_plus_1, δⱼ, δⱼ₊₁ = linear_weighting(x[i], Δx, 0.0, num_gridpts)

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
function initialize(num_particles, num_gridpts, Lx, Ly = 1.0, Lz = 1.0; perturbation_amplitude = 0.0, perturbation_wavenumber = 2π, perturbation_speed = 0.0, charge_per_particle = 1)

    grid = Grid(;num_gridpts, Lx, Ly, Lz)
    particles = Particles(num_particles, grid)
    fields = Fields(num_gridpts; charge_per_particle)

    # Perturb particle positions
    perturb!(particles, perturbation_amplitude, perturbation_wavenumber, perturbation_speed, grid.Lx)

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
    Threads.@threads for j in 1:num_gridpts
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
        push_particles!(new_particles, particles, grid, Δt)
    end

    # Interpolate charge density to grid
    interpolate_charge_to_grid!(new_particles, new_fields, grid)

    # Solve eletric fields on grid
    solve_fields_on_grid!(new_fields, grid)

    # Interpolate electric and magnetic fields to particles
    interpolate_fields_to_particles!(new_particles, new_fields, grid)

    return nothing
end

function update!(particles::Particles, fields::Fields, grid::Grid, Δt; push_particles = true)
    update!(particles, fields, particles, fields, grid, Δt; push_particles)
end

function simulate(particles, fields, grid; Δt, tmax, B0 = 0.0)

    N = length(fields.ρ)
    N_p = particles.num_particles

    # Compute number of timesteps
    num_timesteps = ceil(Int, tmax / Δt)
    t = LinRange(0, tmax, num_timesteps+1)

    # Initialize fields and charge densities
    update!(particles, fields, particles, fields, grid, Δt, push_particles=false)

    # Initialize caches for variables
    E_cache  = zeros(N, num_timesteps+1)
    n_cache = zeros(N, num_timesteps+1)
    vx_cache = zeros(N_p, num_timesteps+1)
    vy_cache = zeros(N_p, num_timesteps+1)
    x_cache  = zeros(N_p, num_timesteps+1)

    E_cache[:, 1] .= fields.Ex
    n_cache[:, 1] .= fields.ρ
    x_cache[:, 1] .= particles.x
    vx_cache[:, 1] .= particles.vx
    vy_cache[:, 1] .= particles.vy

    # Assign magnetic field to fields and particles
    fields.Bz .= B0
    particles.Bz .= B0

    # Simulate
    for i in 2:num_timesteps+1
        # Push to next timestep
        ParticleInCell.update!(particles, fields, grid, Δt)
        # Save quantities
        x_cache[:, i] .= particles.x
        vx_cache[:, i] .= particles.vx
        vy_cache[:, i] .= particles.vy
        n_cache[:, i] .= fields.ρ
        E_cache[:, i] .= fields.Ex
    end

    return t, grid.x, x_cache, vx_cache, vy_cache, n_cache, E_cache
end

# Postprocessing functions

function plot_vdf(vx, vy; type="1D", vlims, t = nothing, style = :scatter, bins = nothing, kwargs...)

    pad_amount = 0.0
    
    ylims = vlims

    if isnothing(t)
        t_str = ""
    elseif t isa String
        t_str = ", tωₚ = $(t)"
    else
        t_str = @sprintf(", tωₚ = %.1f", t)
    end

    if type == "1D"
        title = "f(x, v)" * t_str
        xlabel = "x"
        ylabel = "v"
        xmin, xmax = extrema(vx)
        pad = pad_amount * (xmax - xmin)
        xlims = (xmin - pad, xmax+pad)
        aspect_ratio = :auto
    elseif type == "2D"
        title = "f(vx, vy)" * t_str
        xlabel = "vx"
        ylabel = "vy"
        xlims = ylims
        aspect_ratio = 1
    end

    #=p = plot(;
        size = (1080, 1080),
        titlefontsize=FONT_SIZE*3÷2,
        legendfontsize=FONT_SIZE÷1.5,
        xtickfontsize=FONT_SIZE,
        ytickfontsize=FONT_SIZE,
        xguidefontsize=FONT_SIZE,
        yguidefontsize=FONT_SIZE,
        framestyle=:box,
    )=#
    p = plot(;size = (1080, 1080), kwargs...)

    nbins = isnothing(bins) ? 100 : bins
    xbins = LinRange(xlims[1], xlims[2], nbins)
    ybins = LinRange(ylims[1], ylims[2], nbins)

    if style == :histogram
        histogram2d!(p, 
            vx, vy; 
            xlims, ylims, 
            bins = (xbins, ybins),
            aspect_ratio, 
            title, xlabel, ylabel,
            show_empty_bins = true,
            cbar = false,
            kwargs...
        )
    elseif style == :scatter
        scatter!(p, 
            vx, vy; 
            xlims, ylims, 
            msw = 0, mc = :black, ms = 1.0, 
            title, xlabel, ylabel,
            label = "",
            aspect_ratio,
            kwargs...
        )
    end

    return p
end

function animate_vdf(vx, vy; ts, dir = "", suffix = "", frameskip=0, type = "1D", kwargs...)
    anim = Animation()
    for i in 1:(1+frameskip):size(vx, 2)
        p = plot_vdf(vx[:, i], vy[:, i]; t = ts[i], type, kwargs...)
        frame(anim)
    end
    gif(anim, joinpath(dir, "anim_vdf_$(type)_$(suffix).mp4"))
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


end # module ParticleInCell

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

function Particles(num_particles::Int, grid::Grid; mass = 1, charge = 1)
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
    ρi::Vector{Float64}
    ρe::Vector{Float64}
    jx::Vector{Float64}
    jex::Vector{Float64}
    jix::Vector{Float64}
    jy::Vector{Float64}
    jey::Vector{Float64}
    jiy::Vector{Float64}
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
    ρi = zeros(num_gridpts)
    ρe = zeros(num_gridpts)
    jx = zeros(num_gridpts)
    jex = zeros(num_gridpts)
    jix = zeros(num_gridpts)
    jy = zeros(num_gridpts)
    jey = zeros(num_gridpts)
    jiy = zeros(num_gridpts)
    Ex = zeros(num_gridpts)
    Ey = zeros(num_gridpts)
    ϕ = zeros(num_gridpts)
    Bz = zeros(num_gridpts)

    # Allocate arrays for fourier quantities
    ρ̃ = zeros(Complex{Float64}, num_gridpts)
    ϕ̃ = zeros(Complex{Float64}, num_gridpts)
    Ẽx = zeros(Complex{Float64}, num_gridpts)
    Ẽy = zeros(Complex{Float64}, num_gridpts)

    return Fields(ρ, ρi, ρe, jx, jex, jix, jy, jey, jiy, Ex, Ey, ϕ, Bz, ρ̃, ϕ̃, Ẽx, Ẽy)
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

push_particles!(particles, grid, Δt) = push_particles!(particles, particles, grid, Δt)

periodic_boundary(x, L) = x - floor(x / L) * L

function apply_periodic_boundaries!(xs, L)
    Threads.@threads for i in eachindex(xs)
        xs[i] = periodic_boundary(xs[i], L)
    end
    return xs
end

function apply_periodic_boundaries!(particles::Particles, Lx, Ly, Lz)
    apply_periodic_boundaries!(particles.x, Lx)
    apply_periodic_boundaries!(particles.y, Ly)
    apply_periodic_boundaries!(particles.z, Lz)
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

function interpolate_charge_to_grid!(ρ, jx, jy, particles::Particles, grid::Grid)

    (; num_gridpts, Δx) = grid
    (; num_particles, x, vx, vy, charge, weights) = particles

    # zero charge and current densities
    Threads.@threads for j in 1:num_gridpts
        ρ[j] = 0.0
        jx[j] = 0.0
        jy[j] = 0.0
    end

    for i in 1:num_particles
        # Find cell center closest to but less than x[i]
        j, j_plus_1, δⱼ, δⱼ₊₁ = linear_weighting(x[i], Δx, 0.0, num_gridpts)
        
        W = charge * weights[i]

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

    return ρ, jx, jy
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
function initialize(
        num_particles, num_gridpts, Lx, Ly = 1.0, Lz = 1.0; 
        mi = Inf, ion_charge = 1.0, electron_charge = -1.0, 
        perturbation_amplitude = 0.0, perturbation_wavenumber = 2π, perturbation_speed = 0.0
    )

    grid = Grid(;num_gridpts, Lx, Ly, Lz)
    electrons = Particles(num_particles, grid; charge = electron_charge)
    ions = Particles(num_particles, grid; charge = ion_charge, mass = mi)

    fields = Fields(num_gridpts)

    # Perturb electron positions
    perturb!(electrons, perturbation_amplitude, perturbation_wavenumber, perturbation_speed, grid.Lx)

    # Compute initial charge density and fields
    update!(ions, electrons, fields, grid, 0.0, push_ions = false, push_electrons = false)

    return ions, electrons, fields, grid
end

"""
Use FFT to compute potential and electric field using charge density
"""
function solve_fields_on_grid!(fields::Fields, grid::Grid)
    (;num_gridpts, K, κ) = grid
    (;ρ, ρ̃, Ex, Ey, ϕ, Ẽx, ϕ̃) = fields

    # Compute FFT of density
    @. ρ̃ = ρ    
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
function update!(ions::Particles, electrons::Particles, fields::Fields, grid::Grid, Δt; push_ions = true, push_electrons = true)

    # Push particles to new positions and velocities
    if push_electrons
        push_particles!(electrons, grid, Δt)
    end

    if push_ions
       push_particles!(ions, grid, Δt)
    end

    # Interpolate charge density to grid
    (; ρ, ρi, ρe, jx, jix, jex, jy, jey, jiy) = fields
    (; num_gridpts) = grid

    interpolate_charge_to_grid!(ρe, jex, jey, electrons, grid)
    interpolate_charge_to_grid!(ρi, jix, jiy, ions,      grid)

    # Sum charge densities
    Threads.@threads for i in 1:num_gridpts
        ρ[i]  = ρe[i]  + ρi[i]
        jx[i] = jex[i] + jix[i]
        jy[i] = jey[i] + jiy[i]
    end

    # Solve eletric fields on grid
    solve_fields_on_grid!(fields, grid)

    # Interpolate electric and magnetic fields to particles
    interpolate_fields_to_particles!(electrons, fields, grid)
    interpolate_fields_to_particles!(ions, fields, grid)

    return nothing
end

function simulate(
        ions, electrons, fields, grid; 
        Δt, tmax, B0 = 0.0, solve_ions = true, solve_electrons = true
    )

    N = length(fields.ρ)
    N_p = ions.num_particles

    mi = ions.mass
    me = electrons.mass

    # Ions are updated every ion_subcycle_interval iterations
    ion_subcycle_interval = isinf(mi) ? typemax(Int) : 1#floor(Int, sqrt(mi / me))

    # Compute number of timesteps
    num_timesteps = ceil(Int, tmax / Δt)
    t = LinRange(0, tmax, num_timesteps+1)

    # Initialize fields and charge densities
    update!(ions, electrons, fields, grid, Δt; push_ions = false, push_electrons = false)

    # Initialize caches for variables
    ρi_cache = zeros(N, num_timesteps+1)
    ρe_cache = zeros(N, num_timesteps+1)
    E_cache  = zeros(N, num_timesteps+1)

    xi_cache  = zeros(N_p, num_timesteps+1)
    vix_cache = zeros(N_p, num_timesteps+1)
    viy_cache = zeros(N_p, num_timesteps+1)

    xe_cache  = zeros(N_p, num_timesteps+1)
    vex_cache = zeros(N_p, num_timesteps+1)
    vey_cache = zeros(N_p, num_timesteps+1)

    ρi_cache[:, 1] .= fields.ρi
    ρe_cache[:, 1] .= fields.ρe
    E_cache[:, 1]  .= fields.Ex

    xi_cache[:, 1] .= ions.x
    vix_cache[:, 1] .= ions.vx
    viy_cache[:, 1] .= ions.vy

    xe_cache[:, 1] .= electrons.x
    vex_cache[:, 1] .= electrons.vx
    vey_cache[:, 1] .= electrons.vy

    # Assign magnetic field to fields and particles
    fields.Bz .= B0
    ions.Bz .= B0
    electrons.Bz .= B0
    
    ParticleInCell.apply_periodic_boundaries!(electrons, grid.Lx, grid.Ly, grid.Lz)
    ParticleInCell.apply_periodic_boundaries!(ions, grid.Lx, grid.Ly, grid.Lz)
    ParticleInCell.update!(ions, electrons, fields, grid, Δt; push_ions = false, push_electrons = false)

    # Simulate
    for i in 2:num_timesteps+1
        # Push to next timestep
        push_ions =
            solve_ions &&
            !isinf(mi) && 
            ((i - 2) % ion_subcycle_interval == 0 || i == num_timesteps + 1)

        push_electrons = solve_electrons

        ParticleInCell.update!(ions, electrons, fields, grid, Δt; push_ions, push_electrons)

        # Save field quantities
        Threads.@threads for j in 1:N
            ρi_cache[j, i] = fields.ρi[j]
            ρe_cache[j, i] = fields.ρe[j]
            E_cache[j, i]  = fields.Ex[j]
        end

        # Save particle quantities
        Threads.@threads for j in 1:N_p
            # Ions
            xi_cache[j, i]  = ions.x[j]
            vix_cache[j, i] = ions.vx[j]
            viy_cache[j, i] = ions.vy[j]

            # Electrons
            xe_cache[j, i]  = electrons.x[j]
            vex_cache[j, i] = electrons.vx[j]
            vey_cache[j, i] = electrons.vy[j]
        end
    end

    results = (;
        t, x = grid.x,
        ρi = ρi_cache, ρe = ρe_cache, ρ = ρi_cache .+ ρe_cache,
        E = E_cache,
        electrons = (;
            x = xe_cache, vx = vex_cache, vy = vey_cache
        ),
        ions = (;
            x = xi_cache, vx = viy_cache, vy = viy_cache
        ),
    )

    return results
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

function peak_to_peak_amplitude(signal)
    min_s, max_s = extrema(signal)
    return max_s - min_s
end


end # module ParticleInCell

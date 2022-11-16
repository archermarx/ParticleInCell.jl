using ParticleInCell
using Plots
using Printf

const WS6_RESULTS_DIR = mkpath("results/worksheet_6")
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

begin
    # Problem 1: Gyro orbit preservation
    
    num_particles = 1
    num_gridpts = 2
    xmin = -2.0
    xmax = 2.0

    grid = ParticleInCell.Grid(;xmin, xmax, num_gridpts)
    particles = ParticleInCell.Particles(num_particles, num_particles, grid)

    # particle has y velocity of 1 and the background magnetic field has strength 1
    particles.vy[1] = 1.0

    Bz = 1.0
    particles.Bz[1] = 1.0

    num_periods=10
    tmax = 2π * num_periods
    num_timesteps = 20 * num_periods
    Δt = tmax / num_timesteps

    particle_cache = [deepcopy(particles) for i in 1:num_timesteps+1]
    for i in 1:num_timesteps
        # push particles
        ParticleInCell.push_particles!(particle_cache[i+1], particle_cache[i], grid, Δt)
    end

    velocity_magnitudes = [hypot(p.vx[1],p.vy[1]) for p in particle_cache]
    vxs = [p.vx[] for p in particle_cache]
    vys = [p.vy[] for p in particle_cache]

    @show vxs
    @show vys

    # check that velocity magnitude has not changed from 1 (i.e. that energy is conserved)
    @show all(velocity_magnitudes .≈ 1)

    # extract x, vx, vy, t
    xs = [p.x[1] for p in particle_cache]
    vxs = [p.vx[1] for p in particle_cache]
    vys = [p.vy[1] for p in particle_cache]
    ts = LinRange(0, tmax, num_timesteps+1)

    vx_analytic = cos.(0:0.1:3π)
    vy_analytic = sin.(0:0.1:3π)

    p = plot(;
        size = (1080, 1080),
        titlefontsize=FONT_SIZE*3÷2,
        legendfontsize=FONT_SIZE÷1.5,
        xtickfontsize=FONT_SIZE,
        ytickfontsize=FONT_SIZE,
        xguidefontsize=FONT_SIZE,
        yguidefontsize=FONT_SIZE,
        framestyle=:box,
        aspect_ratio=1, xlabel = "vx", ylabel = "vy",
        title = "Gyro orbit after 10 periods"
    )
    plot!(p, vx_analytic, vy_analytic, lw = 3, label = "Analytic")
    scatter!(p, vxs, vys, label = "Computed", msw = 0, mc = :red, ms = 6)
    display(p)

    savefig(p, joinpath(WS6_RESULTS_DIR, "problem1.png"))
end

function plot_vdf(vx, vy; type="1D", vlims, t = nothing)

    pad_amount = 0.1
    
    ylims = vlims

    if isnothing(t)
        t_str = ""
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

    p = plot(;
        size = (1080, 1080),
        titlefontsize=FONT_SIZE*3÷2,
        legendfontsize=FONT_SIZE÷1.5,
        xtickfontsize=FONT_SIZE,
        ytickfontsize=FONT_SIZE,
        xguidefontsize=FONT_SIZE,
        yguidefontsize=FONT_SIZE,
        framestyle=:box,
    )
    scatter!(p, 
        vx, vy; 
        xlims, ylims, 
        msw = 0, mc = :black, ms = 1.0, 
        title, xlabel, ylabel,
        label = "",
        aspect_ratio
    )

    return p
end

function animate_vdf(vx, vy; ts, suffix = "", frameskip=0, type = "1D", kwargs...)

    anim = Animation()
    for i in 1:(1+frameskip):size(vx, 2)
        p = plot_vdf(vx[:, i], vy[:, i]; t = ts[i], type, kwargs...)
        frame(anim)
    end
    gif(anim, joinpath(WS6_RESULTS_DIR, "anim_vdf_$(type)_$(suffix).mp4"))
end


begin
    # Problem 2: Hybrid waves

        
    function hybrid_wave(;N=101, N_ppc=100, B0 = √(3), xmax=2π, Δt=0.01, periods=10, perturb = :vx, wavenumber=1, suffix)

        ωᵤₕ = sqrt(1 + B0^2)
        Tᵤₕ = 2π / ωᵤₕ
        tmax = periods * Tᵤₕ

        N_p = N_ppc * N

        particles, fields, grid = ParticleInCell.initialize(
            N_p, N_p, N, 0.0, xmax;
        )

        for i in 1:N_p
            if perturb == :vx
                amplitude=0.01 * (1 + B0^2)
                particles.vx[i] += amplitude * sin(2π * wavenumber * particles.x[i] / xmax)
            elseif perturb == :x || perturb == :vy
                amplitude=0.01
                δx = amplitude * sin(2π * wavenumber * particles.x[i] / xmax)
                particles.x[i] += δx
                if perturb == :vy
                    δvy = δx / B0
                    particles.vy[i] += δvy
                end
            end
        end

        num_timesteps = ceil(Int, tmax / Δt)

        E_cache = zeros(N, num_timesteps+1)
        δρ_cache = zeros(N, num_timesteps+1)
        vx_cache = zeros(N_p, num_timesteps+1)
        vy_cache = zeros(N_p, num_timesteps+1)
        x_cache = zeros(N_p, num_timesteps+1)

        E_cache[:, 1] = fields.Ex
        δρ_cache[:, 1] = fields.ρ .- 1
        x_cache[:, 1] = particles.x
        vx_cache[:, 1] = particles.vx
        vy_cache[:, 1] = particles.vy

        fields.Bz .= B0
        particles.Bz .= B0

        for i in 2:num_timesteps+1
            ParticleInCell.update!(particles, fields, grid, Δt)
            E_cache[:, i] = copy(fields.Ex)
            δρ_cache[:, i] = copy(fields.ρ) .- 1.0
            vx_cache[:, i] = copy(particles.vx)
            vy_cache[:, i] = copy(particles.vy)
            x_cache[:, i] = copy(particles.x)
        end

        t = LinRange(0, tmax, num_timesteps+1)

        size = (1660, 1080)
        contour_E = contourf(t, grid.x, E_cache; xlabel = "tωₚ", ylabel = "xωₚ/c", c=:balance, linewidth=0, title = "eE / mcωₚ", size, margin=20Plots.mm, PLOT_SCALING_OPTIONS...)        
        contour_ρ = contourf(t, grid.x, δρ_cache; xlabel = "tωₚ", ylabel = "xωₚ/c", c=:balance, linewidth=0, title = "δn / n₀", size, margin=20Plots.mm, PLOT_SCALING_OPTIONS...)

        vline!(contour_E, Tᵤₕ .* (1:periods), label = "Expected period", lw = 6, ls = :dash, lc = :black)
        vline!(contour_ρ, Tᵤₕ .* (1:periods), label = "Expected period", lw = 6, ls = :dash, lc = :black)

        display(contour_E)
        display(contour_ρ)

        savefig(contour_E, "$(WS6_RESULTS_DIR)/E_$(suffix).png")
        savefig(contour_ρ, "$(WS6_RESULTS_DIR)/n_$(suffix).png")

        # make vdf, density, electric field animations

        # 2D vdf animation

        #vdf_anim_2D = Animation()
        #for i in 1:num_timesteps+1
        #    p = histogram2d(vx_cache[:, i], vy_cache[:, i], xlabel = "vx", ylabel = "vy")
        #    push!(p, vdf_anim_2D)
        #end
        
        #@show vx_cache[:, 1]
        #@show vy_cache[:, 1]
        #histogram2d(vx_cache[:, 1], vy_cache[:, 1])

        #g = gif(anim, "anim_2D_vdf_$(suffix).mp4")
        # vdf x-|v| animation
        # Density and electric field

        return t, grid.x, x_cache, vx_cache, vy_cache, δρ_cache, E_cache
    end

    t, x, xs, vxs, vys, ρs, Es = hybrid_wave(B0 = √(3), perturb = :vx, suffix="hybrid_perturb_vx")
    #t, x, xs, vxs, vys, ρs, Es = hybrid_wave(B0 = √(3), perturb = :x, suffix="hybrid_perturb_x")
    #t, x, xs, vxs, vys, ρs, Es = hybrid_wave(B0 = √(3), perturb = :vy, suffix="hybrid_perturb_vy")
end

begin
    # Problem 3: magnetized ring
    function magnetized_ring(B0 = 1/√(5))
        N = 128
        N_ppc = 128
        N_p = N * N_ppc
        xmax = π/5
        tmax = 200.0
        Δt = 0.2
        k = 10
        δn = 0.001
        v0 = 4.5 * B0 / k

        num_timesteps = ceil(Int, tmax / Δt)

        particles, fields, grid = ParticleInCell.initialize(
            N_p, N_p, N, 0.0, xmax;
        )

        # Perturb particles
        for i = 1:N_p
            δx = δn * sin(2π * k * particles.x[i] / xmax)
            θᵢ = 71 * 2π * i / N_p
            vx = v0 * cos(θᵢ)
            vy = v0 * sin(θᵢ)

            particles.x[i] += δx
            particles.vx[i] = vx
            particles.vy[i] = vy
        end

        ParticleInCell.update!(particles, fields, particles, fields, grid, Δt, push_particles=false)

        E_cache  = zeros(N, num_timesteps+1)
        δρ_cache = zeros(N, num_timesteps+1)
        vx_cache = zeros(N_p, num_timesteps+1)
        vy_cache = zeros(N_p, num_timesteps+1)
        x_cache  = zeros(N_p, num_timesteps+1)

        E_cache[:, 1] = fields.Ex
        δρ_cache[:, 1] = fields.ρ .- 1
        x_cache[:, 1] = particles.x
        vx_cache[:, 1] = particles.vx
        vy_cache[:, 1] = particles.vy

        fields.Bz .= B0
        particles.Bz .= B0

        for i in 2:num_timesteps+1
            ParticleInCell.update!(particles, fields, grid, Δt)
            E_cache[:, i] = copy(fields.Ex)
            δρ_cache[:, i] = copy(fields.ρ) .- 1.0
            vx_cache[:, i] = copy(particles.vx)
            vy_cache[:, i] = copy(particles.vy)
            x_cache[:, i] = copy(particles.x)
        end

        t = LinRange(0, tmax, num_timesteps+1)

        return t, grid.x, x_cache, vx_cache, vy_cache, δρ_cache, E_cache
    end

    function plot_magnetized_ring(t, x, xs, vxs, vys, n, E; suffix = "" , animate = false)
        plot_size = (1660, 1080)
        fx_initial = plot_vdf(xs[:, 1], vys[:, 1], type="1D", vlims = (-0.25, 0.25), t = t[1])
        fx_final = plot_vdf(xs[:, end], vys[:, end], type="1D", vlims = (-0.25, 0.25), t = t[end])
        fv_initial = plot_vdf(vxs[:, 1], vys[:, 1], type="2D", vlims = (-0.25, 0.25), t = t[1])
        fv_final = plot_vdf(vxs[:, end], vys[:, end], type="2D", vlims = (-0.25, 0.25), t = t[end])

        savefig(fx_initial, "$(WS6_RESULTS_DIR)/vdf_initial_1D_$(suffix).png")
        savefig(fx_final, "$(WS6_RESULTS_DIR)/vdf_final_1D_$(suffix).png")
        savefig(fv_initial, "$(WS6_RESULTS_DIR)/vdf_initial_2D_$(suffix).png")
        savefig(fv_final, "$(WS6_RESULTS_DIR)/vdf_final_2D_$(suffix).png")

        if animate
            animate_vdf(xs, vys; suffix, frameskip=2, type="1D", vlims = (-0.25, 0.25), ts = t)
            animate_vdf(vxs, vys; suffix, frameskip=2, type="2D", vlims = (-0.25, 0.25), ts = t)
        end

        n_amplitude = zeros(length(t))
        E_amplitude = zeros(length(t))
        for i in 1:length(t)
            n_amplitude[i] = max(abs.(extrema(n[:, i]))...)
            E_amplitude[i] = max(abs.(extrema(E[:, i]))...)
        end

        p_growth = plot(;
            xlabel = "tωp", ylabel = "Perturbation amplitude (arb.)", yaxis = :log,
            size = (1080, 1080),
            titlefontsize=FONT_SIZE*3÷2,
            legendfontsize=FONT_SIZE,
            xtickfontsize=FONT_SIZE,
            ytickfontsize=FONT_SIZE,
            xguidefontsize=FONT_SIZE,
            yguidefontsize=FONT_SIZE,
            framestyle=:box,
            legend = :bottomright,
            margin = 10Plots.mm,
            ylims = (3e-6, 3e0)
        )


        plot!(t, n_amplitude, label = "n", lw = 2, lc = :red)
        plot!(t, E_amplitude, label = "E", lw = 2, lc = :blue)

        if suffix == "ring_stable" || suffix == "ring_unstable"

            if suffix == "ring_stable"
                γ_expected = 0.0
            elseif suffix == "ring_unstable"
                γ_expected = 0.265 / √(10)
            end
            E_ind = 580
            n_ind = 677
            expected_growth_n = @. n_amplitude[n_ind] * exp(γ_expected * (t - t[n_ind]))
            expected_growth_E = @. E_amplitude[E_ind] * exp(γ_expected * (t - t[E_ind]))
            
            plot!(t, expected_growth_n, label = "Expected (n)", ls = :dash, lw = 2, lc = :red)
            plot!(t, expected_growth_E, label = "Expected (E)", ls = :dash, lw = 2, lc = :blue)
        end
        
        display(p_growth)

        contour_E = heatmap(t, x, E; xlabel = "tωₚ", ylabel = "xωₚ/c", c=:balance, linewidth=0, title = "eE / mcωₚ", size=plot_size, margin=20Plots.mm, PLOT_SCALING_OPTIONS...)        
        contour_ρ = heatmap(t, x, n; xlabel = "tωₚ", ylabel = "xωₚ/c", c=:plasma, linewidth=0, title = "δn / n₀", size=plot_size, margin=20Plots.mm, PLOT_SCALING_OPTIONS...)

        savefig(contour_E, "$(WS6_RESULTS_DIR)/E_$(suffix).png")
        savefig(contour_ρ, "$(WS6_RESULTS_DIR)/n_$(suffix).png")
    end

    animate = false

    t, x_stable, xs_stable, vxs_stable, vys_stable, n_stable, E_stable = magnetized_ring(1 / √(5))
    plot_magnetized_ring(t, x_stable, xs_stable, vxs_stable, vys_stable, n_stable, E_stable; animate, suffix = "ring_stable")

    t, x_unstable, xs_unstable, vxs_unstable, vys_unstable, n_unstable, E_unstable = magnetized_ring(1 / √(10))
    plot_magnetized_ring(t, x_unstable, xs_unstable, vxs_unstable, vys_unstable, n_unstable, E_unstable; animate, suffix = "ring_unstable")
end
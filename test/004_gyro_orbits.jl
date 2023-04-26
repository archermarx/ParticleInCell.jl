using ParticleInCell
using Plots
using Statistics
using FFTW
using StatsBase
using Plots.PlotMeasures
using LsqFit

const RESULT_PATH_004 = mkpath("test/results/004")
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
    Lx = 4.0
    Ly = 4.0
    Lz = 4.0

    grid = ParticleInCell.Grid(;Lx, Ly, Lz, num_gridpts)
    particles = ParticleInCell.Particles(num_particles, grid)

    # particle has y velocity of 1 and the background magnetic field has strength 1
    particles.x[1] = Lx / 2 - 1.0
    particles.y[1] = Ly / 2
    particles.vx[1] = 0.0
    particles.vy[1] = 1.0

    Bz = √(3)
    particles.Bz[1] = Bz

    num_periods=10
    tmax = 2π * num_periods / Bz
    num_timesteps = round(Int, 20 * num_periods)
    Δt = tmax / num_timesteps

    particle_cache = [deepcopy(particles) for i in 1:num_timesteps+1]
    for i in 1:num_timesteps
        # push particles
        ParticleInCell.push_particles!(particle_cache[i+1], particle_cache[i], grid, Δt)
    end

    vxs = [p.vx[] for p in particle_cache]
    vys = [p.vy[] for p in particle_cache]
    velocity_magnitudes = hypot.(vxs, vys)
    
    # check that velocity magnitude has not changed from 1 (i.e. that energy is conserved)
    @show all(velocity_magnitudes .≈ 1)

    # extract x, vx, vy, t
    xs = [p.x[1] for p in particle_cache]
    ys = [p.y[1] for p in particle_cache]
    vxs = [p.vx[1] for p in particle_cache]
    vys = [p.vy[1] for p in particle_cache]
    ts = LinRange(0, tmax, num_timesteps+1)

    ts_analytic = 0:0.1:tmax
    tωc = B0 * ts_analytic
    x_analytic = @. 1/B0 + xs[1] - cos(tωc) / B0
    y_analytic = @. ys[1] + sin(tωc) / B0 .- particles.vy[1] * Δt/2
    vx_analytic = @. sin.(tωc)
    vy_analytic = @. cos.(tωc)

    p = plot(;
        size = (1080, 1080),
        PLOT_SCALING_OPTIONS...,
        aspect_ratio=1, xlabel = "vx", ylabel = "vy",
        title = "Gyro orbit after 10 periods",
        legend = :outertop
    )
    plot!(p, vx_analytic, vy_analytic, lw = 3, label = "Analytic velocity")
    scatter!(p, vxs, vys, label = "Computed velocity", msw = 0, mc = :red, ms = 3)
    display(p)

    savefig(p, joinpath(RESULT_PATH_004, "orbit_phase_space.png"))

    p = plot(;
        size = (1080, 1080),
        PLOT_SCALING_OPTIONS...,
        aspect_ratio=1, xlabel = "x", ylabel = "y",
        title = "Gyro orbit after 10 periods",
        xlims = (0.2, 0.8) .* Lx,
        ylims = (0.2, 0.8) .* Ly,
        legend = :outertop,
    )
    plot!(p, x_analytic, y_analytic, lw = 3, label = "Analytic position")
    scatter!(p, xs, ys, label = "Computed position", msw = 0, mc = :red, ms = 3)
    display(p)

    savefig(p, joinpath(RESULT_PATH_004, "orbit_space.png"))

    px = plot(;
        PLOT_SCALING_OPTIONS...
    )
    plot!(px, ts_analytic, vx_analytic, label = "Analytic", lw = 4, xlabel = "tωₚ", ylabel = "vx/c")
    scatter!(px, ts, vxs, label = "Computed", mc = :red, ms = 6, msw = 0, title = "x velocity")

    py = plot(;
        PLOT_SCALING_OPTIONS...
    )
    plot!(py, ts_analytic, vy_analytic, label = "", lw = 4, xlabel = "tωₚ", ylabel = "vy/c")
    scatter!(py, ts, vys, label = "", mc = :red, msw = 0, ms = 6,title = "y velocity")

    p2 = plot(px, py, layout = (2, 1), size = (1980, 1060), margin = 10Plots.mm)
    display(p2)
    savefig(p2, joinpath(RESULT_PATH_004, "orbit_time.png"))
end


using Revise
using ParticleInCell
using Plots
using Printf

const WS7_RESULTS_DIR = mkpath("results/worksheet_7")
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
    # Problem 1: Two-stream instability

    function two_stream_instability(N = 64, N_ppc = 16)
        N_p = N * N_ppc
        xmin = 0.0
        xmax = 8π*√(3)
        tmax = 40π
        Δt = 0.2
        k = 1/(2√(3))
        δn = 0.001
        v0 = 3

        particles, fields, grid = ParticleInCell.initialize(
            N_p, N_p, N, xmin, xmax; charge_per_particle = 2
        )

        # Perturb particles
        for i = 1:N_p
            δx = δn * sin(k * particles.x[i]) / k

            particles.x[i] += δx

            if iseven(i)
                particles.vx[i] = v0
            else
                particles.vx[i] = -v0
            end
        end

        return ParticleInCell.simulate(particles, fields, grid; Δt, tmax)
    end

    N = 128
    N_ppc = 16 
    t, x, xs, vxs, vys, ns, Es = two_stream_instability(N, N_ppc)
    vlims = (-8, 8)
    #ParticleInCell.animate_vdf(xs, vxs; ts = t, dir = WS7_RESULTS_DIR, type = "1D", suffix = "twostream_N=$(N)_Nppc=$(N_ppc)", vlims)
    ParticleInCell.plot_vdf(xs[:, 1], vxs[:, 1]; type = "1D", vlims) |> display
    ParticleInCell.plot_vdf(xs[:, end], vxs[:, end]; type = "1D", vlims) |> display
end

begin
    # Problem 2: beam plasma (bump on tail) instability
    function bump_on_tail(N, N_ppc)
        N_p = N * N_ppc
        v_th = 0.05
        v_d = 3.0
        n_b_ratio = 10
        xmin = 0.0
        xmax = π
        tmin = 0.0
        tmax = 30π
        Δt = 0.2

        particles, fields, grid = ParticleInCell.initialize(
            N_p, N_p, N, xmin, xmax; charge_per_particle = 1
        )

        ParticleInCell.maxwellian_vdf!(particles, v_th)

        # Assign one in n_b_ratio particles to the beam
        for i in 1:n_b_ratio:N_p
            particles.vx[i] = v_d
            particles.vy[i] = 0.0
        end

        return ParticleInCell.simulate(particles, fields, grid; Δt, tmax)
    end

    N = 128
    N_ppc = 128
    t, x, xs, vxs, vys, ns, Es = bump_on_tail(N, N_ppc)
    vlims = (-0.5, 3.5)
    ParticleInCell.animate_vdf(xs, vxs; ts = t, dir = WS7_RESULTS_DIR, frameskip = 1, type = "1D", suffix = "beamplasma_N=$(N)_Nppc=$(N_ppc)", vlims)
    ParticleInCell.plot_vdf(xs[:, 1], vxs[:, 1]; type = "1D", vlims) |> display
    ParticleInCell.plot_vdf(xs[:, end], vxs[:, end]; type = "1D", vlims) |> display
end

begin
    # Problem 3: Landau damping
end
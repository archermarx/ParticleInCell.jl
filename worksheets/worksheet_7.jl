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

    function two_stream_instability(N = 512, N_ppc = 256)
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

    t, x, xs, vxs, vys, ns, Es = two_stream_instability()
    vlims = (-8, 8)
    ParticleInCell.animate_vdf(xs, vxs; ts = t, dir = WS7_RESULTS_DIR, type = "1D", suffix = "twostream", vlims)
    ParticleInCell.plot_vdf(xs[:, 1], vxs[:, 1]; type = "1D", vlims) |> display
    ParticleInCell.plot_vdf(xs[:, end], vxs[:, end]; type = "1D", vlims) |> display
end
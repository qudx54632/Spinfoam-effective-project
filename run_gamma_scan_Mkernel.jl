# Scan gamma values and save the hinge-space correction kernel M.

using Pkg

const SOLVER_PROJECT = normpath(joinpath(@__DIR__, "..", "LorentzianSimplexSolver"))
isdir(SOLVER_PROJECT) || error("LorentzianSimplexSolver project not found at $SOLVER_PROJECT")
Pkg.activate(SOLVER_PROJECT; io=devnull)

using LinearAlgebra
using GenericLinearAlgebra
using LorentzianSimplexSolver
using JLD2

include(joinpath(@__DIR__, "src", "EffectiveSpinfoamWorkflow.jl"))

const ScalarT = Float64
const tol = ScalarT(1e-8)

LorentzianSimplexSolver.configure_precision!(ScalarT; precision=64, tolerance=tol)

gamma_list = ScalarT(10.0) .^ range(ScalarT(-6), ScalarT(0), length=50)
outfile = joinpath(@__DIR__, "results", "gamma_scan_15move.jld2")
mkpath(dirname(outfile))

simplices = [[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6],[1,3,4,5,6],[2,3,4,5,6]]

coords_lines = [
    "0, 0, 0, 0",
    "0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
    "0, 0, 0, -3.398088489694245",
    "-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
    "0, 0, -2.942830956382712, -1.6990442448471226",
    "-0.068,-0.27,-0.5,-1.3",
]

vertex_coords = EffectiveSpinfoamWorkflow.build_vertex_coordinates(
    simplices,
    coords_lines,
    ScalarT,
)

println("[1/2] Building gamma-independent effective setup.")
data = EffectiveSpinfoamWorkflow.build_effective_data(
    simplices,
    vertex_coords,
)

println("[2/2] Scanning $(length(gamma_list)) gamma values.")
M_kernel_matrix_list = Any[nothing for _ in gamma_list]
M_kernel_norm_list = Union{Nothing, ScalarT}[nothing for _ in gamma_list]
scaled_M_kernel_norm_list = Union{Nothing, ScalarT}[nothing for _ in gamma_list]

for (i, gamma_value) in enumerate(gamma_list)
    println("[$i/$(length(gamma_list))] gamma = $gamma_value")
    terms = EffectiveSpinfoamWorkflow.compute_effective_terms(
        data,
        gamma_value;
        tolerance=tol,
    )

    M_kernel_matrix_list[i] = terms.M_kernel
    M_kernel_norm_list[i] = opnorm(terms.M_kernel)
    scaled_M_kernel_norm_list[i] = gamma_value^2 * M_kernel_norm_list[i]

    println("    ||M|| = $(M_kernel_norm_list[i])")
    println("    gamma^2 ||M|| = $(scaled_M_kernel_norm_list[i])")

    tmpfile = outfile * ".tmp"
    @save tmpfile gamma_list M_kernel_matrix_list M_kernel_norm_list scaled_M_kernel_norm_list
    mv(tmpfile, outfile; force=true)
end

println("Saved M kernels to $outfile.")

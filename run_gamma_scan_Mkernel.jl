# run_gamma_scan_Mkernel.jl
# Scan over gamma values and compute/store the M_kernel matrix at each step.
using Pkg
# Activate the local project environment used by the solver package
Pkg.activate("src/LorentzianSimplexSolver")
Pkg.instantiate()

using LorentzianSimplexSolver

using SymEngine
using JLD2
using LinearAlgebra
using Symbolics
# Load helper scripts for geometry, action, derivatives, and perturbations
include("scripts/run_geometry.jl")
include("scripts/run_action.jl")
include("scripts/run_dlogEh_dX.jl")
include("scripts/run_deta_dl.jl")
include("src/perturbations/TransverseBasis.jl")
include("src/perturbations/Soln_dY_dX.jl")
include("src/perturbations/DθDl.jl")

using .RunGeometry
using .RunAction
using .RunDlogEhDX
using .DηDLUtils
using .TransverseBasis
using .Soln_dY_dX


# ------------------------------------------------------------
# Numerical precision / tolerance
# ------------------------------------------------------------
# using DoubleFloats
# const ScalarT = Double64
const ScalarT = Float64
#const ScalarT = BigFloat
const tol = ScalarT(1e-8)

if ScalarT === BigFloat
    setprecision(BigFloat, 100)
    LorentzianSimplexSolver.PrecisionUtils.set_big_precision!(100)
    # LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(sqrt(eps(BigFloat)))
    LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(tol)
else
    LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(tol)
end

# ============================================================
# User input
# ============================================================
@variables γ

# Gamma values to scan
gamma_list = ScalarT(10.0) .^ range(ScalarT(2), ScalarT(-6), length=50)
# Output file for saved matrices
outfile = "data/gamma_scan_15move.jld2"

# ------------------------------------------------------------
# Geometry input: same simplices and coordinates as single-gamma run
# ------------------------------------------------------------
simplices = [[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6],[1,3,4,5,6],[2,3,4,5,6]]
all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)
coords_lines = [
    "0, 0, 0, 0",
    "0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
    "0, 0, 0, -3.398088489694245",
    "-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
    "0, 0, -2.942830956382712, -1.6990442448471226",
    "-0.068,-0.27,-0.5,-1.3",
]

vertex_coords = Dict{Int, Vector{ScalarT}}()
for (v, line) in zip(all_vertices, coords_lines)
    vertex_coords[v] = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(line, ScalarT)
end
# ============================================================
# Build geometry + symbolic action once (gamma-independent setup)
# ============================================================
println("[1/5] Building geometry...")
geom = RunGeometry.run_geometry_pipeline(simplices, coords_lines, ScalarT, tol)

println("[2/5] Building action and Regge data...")
γ = LorentzianSimplexSolver.DefineAction.γsym()
_, dihedral_angles, _, _, iRegge = LorentzianSimplexSolver.ReggeAction.run_Regge_action(geom, simplices, vertex_coords);
@show iRegge;
sd, S_symbols, phase_soln = RunAction.run_action(geom, dihedral_angles, γ);
S_no_phase = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_symbols, phase_soln);

println("[3/5] Preparing symbolic derivatives...")
dlogEh_dX_sym = RunDlogEhDX.run_dlogEh_dX(geom)
dlogEb_dX_sym, dlogEb_dY_sym, Y_vars = RunDlogEhDX.run_dlogEb_dXY(geom);

println("[4/5] Constructing Hessian block and perturbation basis...")
g_vars = geom.varias[:g_var]
z_vars = geom.varias[:z_var]
η_vars = geom.varias[:η_var]
X_vars = vcat(g_vars, z_vars)
vars = vcat(g_vars, z_vars, η_vars)
H_symbols = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block_half(S_no_phase, vars)
η_h_vertices = DηDLUtils.get_bulk_faces_vertices(geom)
eta_h = [LorentzianSimplexSolver.DefineSymbols.make_symbol("η_$(faces[1][1])$(faces[1][2])$(faces[1][3])") for faces in geom.connectivity[1]["OrderBulkFaces"]]
bulk_edges, bdry_edges = DηDLUtils.get_bulk_edges(geom, η_h_vertices)

# Here we perturb only one boundary edge
bdry_edges_perturb = [bdry_edges[1]]
perturb_edges = vcat(bulk_edges, bdry_edges_perturb)

nh = length(geom.connectivity[1]["OrderBulkFaces"])
nb = length(geom.connectivity[1]["OrderBDryFaces"])
nl = length(perturb_edges)
nt = nh - nl
nX = length(X_vars)

M_kernel_matrix_list = Vector{Any}(undef, length(gamma_list))

# ============================================================
# Main scan loop
# ============================================================
println("[5/5] Starting gamma scan over $(length(gamma_list)) values...")
for (ig, gamma_val) in enumerate(gamma_list)

    println("\n --------------------------------------------------")
    println("[$ig/$(length(gamma_list))] Running gamma = $gamma_val")
    # Substitute gamma into the symbolic action
    vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd, γ; γval=gamma_val);
    S_val = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_no_phase, vals);
    SF_action = SymEngine.expand(S_val)
    println("Evaluating action for $gamma_val is $SF_action")
    
    # Evaluate Hessian at this gamma
    println("  - Evaluating Hessian...")
    H_eval = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_block(H_symbols, sd; γ=gamma_val);

    # Build perturbation matrix dη/dl and the transverse basis
    println("  - Building dη/dl and transverse basis...")
    dηdl_matrix = DηDLUtils.build_dηdl_matrix(η_h_vertices, perturb_edges, vertex_coords, ScalarT, gamma_val)
    eListHT = TransverseBasis.compute_transverse_basis(dηdl_matrix, tol)

    # Evaluate dlogEh/dX at this gamma
    println("  - Evaluating dlogEh/dX...")
    dlogEh_dX_vals = RunDlogEhDX.evaluate_dlogEh_dX(dlogEh_dX_sym, geom, sd; γval=gamma_val);
    
    # Compute kernel matrix M_kernel
    println("  - Computing M_kernel...")
    eta_h_vals = [ScalarT(subs(eta_h[i], vals)) for i in 1:nh]
    Ahh = Matrix(Diagonal(eta_h_vals))
    hαβ = H_eval[1:nX, 1:nX] + transpose(dlogEh_dX_vals) * Ahh * dlogEh_dX_vals
    invHessianXX = inv(H_eval[1:nX, 1:nX]);

    ω = - dlogEh_dX_vals * invHessianXX * transpose(dlogEh_dX_vals)    
    ρ = inv(inv(Ahh) - ω);
    Sij = -transpose(eListHT) * dlogEh_dX_vals * inv(hαβ) * transpose(dlogEh_dX_vals) * eListHT
    κ = eListHT * inv(Sij) * transpose(eListHT)
    M_kernel =  ρ - ρ * inv(Ahh) * κ * inv(Ahh) * ρ

    M_kernel_matrix_list[ig]        = M_kernel
    tmpfile = outfile * ".tmp"
    @save tmpfile gamma_list M_kernel_matrix_list
    mv(tmpfile, outfile; force=true)
    println("Checkpoint saved after gamma = $gamma_val")
end
println("\nDone. All gamma values processed.")
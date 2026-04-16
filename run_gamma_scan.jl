using Pkg

Pkg.activate("src/LorentzianSimplexSolver")
Pkg.instantiate()

using LorentzianSimplexSolver
# using DoubleFloats
using SymEngine
using JLD2
using LinearAlgebra
using Symbolics

include("scripts/run_geometry.jl")
include("scripts/run_action.jl")
include("scripts/run_dlogEh_dg.jl")
include("scripts/run_djdl.jl")

include("src/perturbations/TransverseBasis.jl")
include("src/perturbations/LinearizedEOMs.jl")
include("src/perturbations/QuadraticRegge.jl")

using .RunGeometry
using .RunAction
using .RunDlogEhDG
using .DJDLUtils
using .TransverseBasis
using .QuadraticRegge


# ------------------------------------------------------------
# Define the precision and tolerance for numerical operations
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
# User input section
# ============================================================
@variables γ

# Example gamma list
gamma_list = ScalarT(10.0) .^ range(ScalarT(0), ScalarT(-6), length=40)
#gamma_list = [ScalarT(1.0)]
# Output file
outfile = "data/gamma_scan_15move_refinement.jld2"

# ------------------------------------------------------------
# Put here the same geometry input used in your single-gamma code
# ------------------------------------------------------------
simplices = [[2,3,4,6,7],[1,3,4,6,7],[1,2,4,6,7],[1,2,3,6,7],[1,2,3,4,7],[2,3,5,6,8],[1,3,5,6,8],[1,2,5,6,8],[1,2,3,6,8],[1,2,3,5,8],[2,4,5,6,9],[1,4,5,6,9],[1,2,5,6,9],[1,2,4,6,9],[1,2,4,5,9],[3,4,5,6,10],[1,4,5,6,10],[1,3,5,6,10],[1,3,4,6,10],[1,3,4,5,10],[3,4,5,6,11],[2,4,5,6,11],[2,3,5,6,11],[2,3,4,6,11],[2,3,4,5,11]]

all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)

coords_lines = [
"0, 0, 0, 0",
"0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
"0, 0, 0, -3.398088489694245",
"-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
"0, 0, -2.942830956382712, -1.6990442448471226",
"-0.068, -0.27, -0.5, -1.3",
"-0.06165622828269508, -0.7476319083813029, -0.49237746085102824, -1.6192353958776982",
"-0.013600000000000001, -0.6089055267050423, -0.8847549217020566, -1.6192353958776982",
"-0.05471634704156723, -0.6634799624836852, -1.2905133626305172, -1.3266576479993824",
"-0.05996263164968173, -0.18743250119993968, -0.8604521717798855, -1.5747576871100133",
"-0.0423034122998675, -0.5046526423930288, -1.0289336294868097, -2.25080216236223"
]

vertex_coords = Dict{Int, Vector{ScalarT}}()
for (v, line) in zip(all_vertices, coords_lines)
    vertex_coords[v] = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(line, ScalarT)
end

# ============================================================
# Geometry and symbolic setup
# ============================================================
println("Building geometry...")
geom_base = RunGeometry.run_geometry_pipeline(simplices, coords_lines, ScalarT, tol)

println("Building action...")
γ = LorentzianSimplexSolver.DefineAction.γsym()
sd_base, S_base = RunAction.run_action(geom_base, γ)

# ------------------------------------------------------------
# gamma-independent symbolic derivatives
# ------------------------------------------------------------
println("Preparing dlogEh/dg symbolic functions...")
dlogEh_dg_sym = RunDlogEhDG.run_dlogEh_dg(geom_base)

# ========================================================
# Variables and ordered Hessian block
# ========================================================
g_vars = geom_base.varias[:g_var]
z_vars = geom_base.varias[:z_var]
j_vars = geom_base.varias[:j_var]

vars = vcat(g_vars, z_vars, j_vars)

ng = length(g_vars)
nz = length(z_vars)
nx = ng + nz
nh = length(geom_base.connectivity[1]["OrderBulkFaces"])

println("Constructing symbolic Hessian block...")
H_base_block = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block_half(S_base, vars)
#println("Precomputing first derivatives...")
#dS_precomp = [SymEngine.diff(S_base, v) for v in vars]

println("Building gamma-independent matrices ...")
djdl_temp, bulk_edges, j_h_vertices = DJDLUtils.build_djdl_matrix(
    geom_base, vertex_coords, ScalarT, first(gamma_list)
)

nl = length(bulk_edges)

dthetadl = QuadraticRegge.compute_dθDl(
    simplices,
    j_h_vertices,
    bulk_edges,
    vertex_coords,
    geom_base.connectivity[1]["Tets"],
    LorentzianSimplexSolver.Dihedral.minkowski_norm2, ScalarT
)

@variables dl[1:nl]
dl_vec = collect(dl)

Spinfoam_Quadratic1_list = Vector{Any}(undef, length(gamma_list))
ReggeQuadratic_list = Vector{Any}(undef, length(gamma_list))
kappa_matrix_list = Vector{Any}(undef, length(gamma_list))
correction_list = Vector{Any}(undef, length(gamma_list))

# ============================================================
# Main scan loop
# ============================================================

for (ig, gamma_val) in enumerate(gamma_list)

    println("--------------------------------------------------")
    println("Running gamma = $gamma_val")
    vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd_base, γ; γval=gamma_val)
    S_val = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_base, vals);
    S_simpl = SymEngine.expand(S_val) 
    println("Evaluating action for $gamma_val is $S_simpl")
    # ========================================================
    # Evaluate action/Hessian at this gamma
    # ========================================================
    # evaluate Hessian
    println("Evaluating Hessian for gamma = $gamma_val")
    HessianOld = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_block(H_base_block, sd_base; γ=gamma_val)
    #HessianOld = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_from_dS(dS_precomp, vars, sd_base; γ=gamma_val)

    djdl_matrix, _, _ = DJDLUtils.build_djdl_matrix(
        geom_base,
        vertex_coords,
        ScalarT,
        gamma_val
    )

    eListHT = TransverseBasis.compute_transverse_basis(djdl_matrix, tol)

    # ========================================================
    # Linearized EOM solution
    # ========================================================

    _, HYY = LinearizedEOMs.solve_linearized_eoms(
        HessianOld,
        eListHT,
        djdl_matrix,
        nx,
        ScalarT,
        dl_vec
    )

    # ========================================================
    # dlogEh/dg evaluated at this gamma
    # ========================================================
    dlogEh_dg_vals = RunDlogEhDG.evaluate_dlogEh_dg(dlogEh_dg_sym, geom_base, sd_base; γval=gamma_val)

    # ========================================================
    # W matrix and reduced matrix S
    # ========================================================
    Hxx = HessianOld[1:nx, 1:nx]
    E = Matrix{Complex{ScalarT}}(I, nx, ng)
    Y = Hxx \ E
    invHessianXX = Y[1:ng, :]
    Wmatrix = -dlogEh_dg_vals * invHessianXX * transpose(dlogEh_dg_vals)

    Smatrix = eListHT' * Wmatrix * eListHT
    Smatrix_num = Complex{ScalarT}.(Symbolics.value.(Smatrix))
    kappa_matrix = eListHT * (Smatrix_num \ eListHT')

    # ========================================================
    # Spinfoam quadratic term
    # ========================================================
    ny = size(HYY, 1)
    Eyy = Matrix{Complex{ScalarT}}(I, ny, ng)
    Yyy = HYY \ Eyy
    invHYY_block = Yyy[1:ng, :]
    W_all_matrix = -dlogEh_dg_vals * invHYY_block * transpose(dlogEh_dg_vals)
    Spinfoam_Quadratic1 = ScalarT(0.5) * transpose(djdl_matrix) * W_all_matrix * djdl_matrix

    # ========================================================
    # Regge quadratic term
    # ========================================================
    dadl = gamma_val .* djdl_matrix
    ReggeQuadratic = im * ScalarT(0.5) * dadl' * dthetadl

    correction = dthetadl' * kappa_matrix * dthetadl

    # ========================================================
    # Store
    # ========================================================
    Spinfoam_Quadratic1_list[ig] = Spinfoam_Quadratic1
    ReggeQuadratic_list[ig]      = ReggeQuadratic
    kappa_matrix_list[ig]        = kappa_matrix
    correction_list[ig]          = correction
    tmpfile = outfile * ".tmp"

    @save tmpfile gamma_list Spinfoam_Quadratic1_list ReggeQuadratic_list kappa_matrix_list correction_list
    mv(tmpfile, outfile; force=true)

    println("Checkpoint saved after gamma = $gamma_val")
end

# ============================================================
# Save
# ============================================================

println("Done")
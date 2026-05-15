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
# gamma_list = ScalarT(10.0) .^ range(ScalarT(0), ScalarT(-6), length=40)
gamma_list = [ScalarT(1.0)]
# Output file
outfile = "data/gamma_scan_2Delta3.jld2"

# ------------------------------------------------------------
# Put here the same geometry input used in your single-gamma code
# ------------------------------------------------------------
simplices = [[1,2,3,4,6], [1,2,3,5,6], [1,2,4,5,6],[1,2,3,4,7], [1,2,3,5,7], [1,2,4,5,7]]

all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)

coords_lines = [
    "0, 0, 0, 0",
    "-0.068000000000000005, -0.21988127663727278, -0.5316227766016838, -1.3316227766016839",
    "0, 0, 0, -3.398088489694245",
    "-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
    "0, 0, -2.942830956382712, -1.6990442448471226",
    "0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
    "-2.4696884592430974, -3.893218630529324, -1.3565336794679874, -1.9090667752920147",
]

vertex_coords = Dict{Int, Vector{ScalarT}}()
for (v, line) in zip(all_vertices, coords_lines)
    vertex_coords[v] = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(line, ScalarT)
end

# ============================================================
# Geometry and symbolic setup
# ============================================================
println("Building geometry...")
geom = RunGeometry.run_geometry_pipeline(simplices, coords_lines, ScalarT, tol)

println("Building action...")
γ = LorentzianSimplexSolver.DefineAction.γsym()
_, dihedral_angles, _, _, iRegge = LorentzianSimplexSolver.ReggeAction.run_Regge_action(geom, simplices, vertex_coords);
@show iRegge;
sd, S_symbols, phase_soln = RunAction.run_action(geom, dihedral_angles, γ);
S_no_phase = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_symbols, phase_soln);

# ------------------------------------------------------------
# gamma-independent symbolic derivatives
# ------------------------------------------------------------
println("Preparing dlogEh/dX, dlogEb/dX, dlogEb/dY symbolic functions...")
dlogEh_dX_sym = RunDlogEhDX.run_dlogEh_dX(geom)
dlogEb_dX_sym, dlogEb_dY_sym, Y_vars = RunDlogEhDX.run_dlogEb_dXY(geom);

println("Preparing dkbEb/dX, dkbEb/dY, d2kbEb/dXdY, d2kbEb/dYdY symbolic functions...")
dkbEb_dX_sym, dkbEb_dY_sym, d2kbEb_dXdY_sym, d2kbEb_dYdY_sym = RunDlogEhDX.run_kblogEb_dXY(geom, Y_vars);

# ========================================================
# Variables and ordered Hessian block
# ========================================================
println("Constructing symbolic Hessian block...")
g_vars = geom.varias[:g_var]
z_vars = geom.varias[:z_var]
η_vars = geom.varias[:η_var]
X_vars = vcat(g_vars, z_vars)
vars = vcat(g_vars, z_vars, η_vars)

H_symbols = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block_half(S_no_phase, vars)

println("Computing δϵ and δΘ ...")
η_h_vertices = DηDLUtils.get_bulk_faces_vertices(geom)
eta_h = [LorentzianSimplexSolver.DefineSymbols.make_symbol("η_$(faces[1][1])$(faces[1][2])$(faces[1][3])") for faces in geom.connectivity[1]["OrderBulkFaces"]]
bulk_edges, bdry_edges = DηDLUtils.get_bulk_edges(geom, η_h_vertices)
bdry_edges_perturb = [bdry_edges[1]] # here we only perturb one boundary edge
perturb_edges = vcat(bulk_edges, bdry_edges_perturb)


nh = length(geom.connectivity[1]["OrderBulkFaces"])
nb = length(geom.connectivity[1]["OrderBDryFaces"])
nl = length(perturb_edges)
nt = nh - nl
nX = length(X_vars)

kb_vertices = [geom.connectivity[1]["TetFaces"][f[1][1]][f[1][2]][f[1][3]] for f in geom.connectivity[1]["OrderBDryFaces"]];
DϵDl = DθDl_module.compute_dθDl(simplices, η_h_vertices, perturb_edges, vertex_coords, geom.connectivity[1]["Tets"], ScalarT);
DΘDl = DθDl_module.compute_dθDl(simplices, kb_vertices, perturb_edges, vertex_coords, geom.connectivity[1]["Tets"], ScalarT);

Spinfoam_Quadratic_list = Vector{Any}(undef, length(gamma_list))
Spinfoam_linear_list = Vector{Any}(undef, length(gamma_list))
iReggeLinear_list = Vector{Any}(undef, length(gamma_list))
iReggeQuadratic_list = Vector{Any}(undef, length(gamma_list))
M_kernel_matrix_list = Vector{Any}(undef, length(gamma_list))
correction_list = Vector{Any}(undef, length(gamma_list))

# ============================================================
# Main scan loop
# ============================================================

for (ig, gamma_val) in enumerate(gamma_list)

    println("--------------------------------------------------")
    println("Running gamma = $gamma_val")
    vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd, γ; γval=gamma_val);
    S_val = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_no_phase, vals);
    SF_action = SymEngine.expand(S_val)
    println("Evaluating action for $gamma_val is $SF_action")
    
    # ========================================================
    # Evaluate action/Hessian at this gamma
    # ========================================================
    # evaluate Hessian
    println("Evaluating Hessian for gamma = $gamma_val")
    H_eval = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_block(H_symbols, sd; γ=gamma_val);

    dηdl_matrix = DηDLUtils.build_dηdl_matrix(η_h_vertices, perturb_edges, vertex_coords, ScalarT, gamma_val)
    eListHT = TransverseBasis.compute_transverse_basis(dηdl_matrix, tol)

    dkbdl, d2kb_dldl = Soln_dY_dX.dkb_dl(geom, nl, bdry_edges_perturb, vertex_coords ;γ=gamma_val);

    # ========================================================
    # dlogEh/dX,  evaluated at this gamma
    # ========================================================
    println("Evaluating dlogEh and dlogEb matrices for gamma = $gamma_val")
    dlogEh_dX_vals = RunDlogEhDX.evaluate_dlogEh_dX(dlogEh_dX_sym, geom, sd; γval=gamma_val);
    dlogEb_dX_vals, dlogEb_dY_vals = RunDlogEhDX.evaluate_dlogEb_dXY(dlogEb_dX_sym, dlogEb_dY_sym, Y_vars, geom, sd, phase_soln; γval=gamma_val)
    println("Evaluating d^2logEh and d^2logEb matrices for gamma = $gamma_val")
    dkbEb_dX_vals, dkbEb_dY_vals, d2kbEb_dXdY_vals, d2kbEb_dYdY_vals = RunDlogEhDX.evaluate_kblogEb_all(dkbEb_dX_sym, dkbEb_dY_sym, d2kbEb_dXdY_sym, d2kbEb_dYdY_sym, geom, sd, Y_vars, phase_soln; γval=gamma_val)

    d2kbEb_dXdY_vals_sumb = sum(d2kbEb_dXdY_vals[i, :, :] for i in 1:nb);
    d2kbEb_dYdY_vals_sumb = sum(d2kbEb_dYdY_vals[i, :, :] for i in 1:nb);
    dkbEb_dY_vals_sumb = sum(dkbEb_dY_vals[i, :] for i in 1:nb);

    # ========================================================
    # M_kernel matrix and reduced matrix S
    # ========================================================
    println("Evaluating Kernel matrix for gamma = $gamma_val")
    eta_h_vals = [ScalarT(subs(eta_h[i], vals)) for i in 1:nh]
    Ahh = Matrix(Diagonal(eta_h_vals))
    hαβ = H_eval[1:nX, 1:nX] + transpose(dlogEh_dX_vals) * Ahh * dlogEh_dX_vals
    invHessianXX = inv(H_eval[1:nX, 1:nX]);

    ω = - dlogEh_dX_vals * invHessianXX * transpose(dlogEh_dX_vals)    
    ρ = inv(inv(Ahh) - ω);
    Sij = -transpose(eListHT) * dlogEh_dX_vals * inv(hαβ) * transpose(dlogEh_dX_vals) * eListHT
    κ = eListHT * inv(Sij) * transpose(eListHT)
    M_kernel =  ρ - ρ * inv(Ahh) * κ * inv(Ahh) * ρ

    # ========================================================
    # Spinfoam quadratic term
    # ========================================================
    println("Evaluating Spinfoam Quadratic and linear terms for gamma = $gamma_val")
    Bα = transpose(transpose(dηdl_matrix) * dlogEh_dX_vals);
    M_matrix = vcat(dlogEb_dY_vals[:, nb+1:end] - dlogEb_dX_vals * invHessianXX * d2kbEb_dXdY_vals_sumb[:, nb+1:end], -dlogEh_dX_vals * invHessianXX * d2kbEb_dXdY_vals_sumb[:, nb+1:end]);
    dmatrix = vcat(im * gamma_val/2 * DΘDl + dlogEb_dX_vals * invHessianXX * (Bα + d2kbEb_dXdY_vals_sumb[:, 1:nb] * dkbdl), im * gamma_val/2 * DϵDl + dlogEh_dX_vals * invHessianXX * (Bα + d2kbEb_dXdY_vals_sumb[:, 1:nb] * dkbdl));
    DξDl = pinv(M_matrix, atol=tol, rtol=tol) * dmatrix;
    DYDl = vcat(dkbdl, DξDl);
    Iboundary_linear = transpose(dkbEb_dY_vals_sumb) * DYDl

    SF_linear_bdry = Iboundary_linear

    Cα = d2kbEb_dXdY_vals_sumb * DYDl;
    BA = vcat(Bα + Cα, zeros(nt, 2));

    Hαi = transpose(dlogEh_dX_vals) *  eListHT;
    HIJ = [hαβ Hαi; transpose(Hαi) zeros(nt, nt)];
    invHIJ = inv(HIJ);

    DtXDl = -invHIJ * BA;

    kb_vals = [ScalarT(subs(Y_vars[i], vals)) for i in 1:nb];
    DXDl = -invHessianXX * (Bα + d2kbEb_dXdY_vals_sumb[:, 1:nb] * dkbdl + d2kbEb_dXdY_vals_sumb[:, nb+1:end] * DξDl);
    Mm = [kb_vals[b] * dlogEb_dY_vals[b, j] for b in 1:nb, j in nb+1:length(Y_vars)];
    R = [dkbdl[i, j] *  transpose(dlogEb_dY_vals[i, nb+1:end]) * DξDl[:, j] + transpose(DXDl[:, j]) * d2kbEb_dXdY_vals[i, :, nb+1:end] * DξDl[:, j] + transpose(DξDl[:, j]) * d2kbEb_dYdY_vals[i, nb+1:end, nb+1:end] * DξDl[:, j] for i in 1:nb, j in 1:nl]

    D2ξDl2 = -pinv(Mm, atol=tol, rtol=tol) * R[:, 2];
    nxi = length(Y_vars) - nb
    D2ξDl2_matrix = [j==k==nl ? D2ξDl2[b] : 0 for b in 1:nxi, j in 1:nl, k in 1:nl]
    D2YDl2_matrix = vcat(d2kb_dldl, D2ξDl2_matrix);
    Iboundary_qadratic = 1/2 * transpose(DYDl) * d2kbEb_dYdY_vals_sumb * DYDl + 1/2 * sum(dkbEb_dY_vals_sumb[i] * D2YDl2_matrix[i, :, :] for i in 1:length(Y_vars))
    SF_quadratic_form = -1/2 * transpose(BA[1:nX, :]) * invHIJ[1:nX, 1:nX] * BA[1:nX, :] + Iboundary_qadratic

    # ========================================================
    # Regge linear, quadratic and correction terms
    # ========================================================
    iSRegge_linear = im/2 * gamma_val * transpose(dkbdl) * dihedral_angles
    iSRegge_quadratic = im*(gamma_val/4 * transpose(dηdl_matrix) * DϵDl + gamma_val/4 * (transpose(dkbdl) * DΘDl + sum(dihedral_angles[i] * d2kb_dldl[i, :, :] for i in 1:nb)))
    correction = -gamma_val^2/8 * transpose(DϵDl) * M_kernel * DϵDl

    # ========================================================
    # Store
    # ========================================================
    Spinfoam_Quadratic_list[ig] = SF_quadratic_form
    Spinfoam_linear_list[ig] = SF_linear_bdry
    iReggeLinear_list[ig]      = iSRegge_linear
    iReggeQuadratic_list[ig]      = iSRegge_quadratic
    M_kernel_matrix_list[ig]        = M_kernel
    correction_list[ig]          = correction
    tmpfile = outfile * ".tmp"

    @save tmpfile gamma_list Spinfoam_linear_list Spinfoam_Quadratic_list iReggeLinear_list iReggeQuadratic_list M_kernel_matrix_list correction_list
    mv(tmpfile, outfile; force=true)

    println("Checkpoint saved after gamma = $gamma_val")
end

# ============================================================
# Save
# ============================================================

println("Done")
# main_interactive_driver.jl
# Interactive single-gamma driver for spinfoam / Regge computation

using Pkg
Pkg.activate("src/LorentzianSimplexSolver")
Pkg.instantiate()

using LorentzianSimplexSolver
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
# Helper input functions
# ------------------------------------------------------------
function choose_scalartype()
    println("\nChoose scalar type:")
    println("  1 = Float64")
    println("  2 = Double64")
    println("  3 = BigFloat")
    print("Enter choice [1/2/3]: ")
    choice = readline()

    if choice == "1"
        return Float64, 1e-8
    elseif choice == "2"
        @eval using DoubleFloats
        return Double64, Double64(1e-8)
    elseif choice == "3"
        return BigFloat, BigFloat(1e-30)
    else
        error("Invalid choice.")
    end
end

function parse_int_list(line::String)
    return parse.(Int, split(strip(line), r"[,\s]+"))
end

function read_simplices()
    println("\nEnter simplices one per line.")
    println("Example: 1 2 3 4 6")
    println("Type DONE when finished.\n")

    simplices = Vector{Vector{Int}}()
    while true
        print("simplex> ")
        line = readline()
        if uppercase(strip(line)) == "DONE"
            break
        end
        push!(simplices, parse_int_list(line))
    end

    isempty(simplices) && error("No simplices entered.")
    return simplices
end

function read_coord_lines(nv::Int)
    println("\nEnter $nv coordinate lines, one per vertex.")
    println("Example: 0, 0, 0, 0")
    coord_lines = String[]
    for i in 1:nv
        print("coord[$i]> ")
        push!(coord_lines, readline())
    end
    return coord_lines
end

# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
println("\n==================================================")
println(" Interactive Spinfoam / Regge Driver (single γ)")
println("==================================================")

ScalarT, tol = choose_scalartype()
println("\nUsing scalar type = $(ScalarT), tolerance = $tol")

if ScalarT === BigFloat
    setprecision(BigFloat, 100)
    LorentzianSimplexSolver.PrecisionUtils.set_big_precision!(100)
end
LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(tol)

@variables γ

# Input geometry
simplices = read_simplices()

all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)

coord_lines = read_coord_lines(length(all_vertices))

vertex_coords = Dict{Int, Vector{ScalarT}}()
for (v, line) in zip(all_vertices, coord_lines)
    vertex_coords[v] = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(line, ScalarT)
end

# Input gamma
print("\nEnter one gamma value: ")
gamma_val = parse(ScalarT, readline())
println("Using gamma = $gamma_val")

# ------------------------------------------------------------
# Geometry + symbolic setup
# ------------------------------------------------------------
println("\n[1/6] Building geometry...")
geom = RunGeometry.run_geometry_pipeline(simplices, coord_lines, ScalarT, tol)

println("[2/6] Building action and Regge data...")
γsym = LorentzianSimplexSolver.DefineAction.γsym()
_, dihedral_angles, _, _, iRegge = LorentzianSimplexSolver.ReggeAction.run_Regge_action(geom, simplices, vertex_coords)
println("\niRegge:")
display(iRegge)

sd, S_symbols, phase_soln = RunAction.run_action(geom, dihedral_angles, γsym)
S_no_phase = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_symbols, phase_soln)

# ------------------------------------------------------------
# Gamma-independent symbolic derivatives
# ------------------------------------------------------------
println("[3/6] Preparing symbolic derivatives...")
dlogEh_dX_sym = RunDlogEhDX.run_dlogEh_dX(geom)
dlogEb_dX_sym, dlogEb_dY_sym, Y_vars = RunDlogEhDX.run_dlogEb_dXY(geom)
dkbEb_dX_sym, dkbEb_dY_sym, d2kbEb_dXdY_sym, d2kbEb_dYdY_sym = RunDlogEhDX.run_kblogEb_dXY(geom, Y_vars)

# ------------------------------------------------------------
# Hessian and perturbation setup
# ------------------------------------------------------------
println("[4/6] Constructing Hessian block and perturbation data...")
g_vars = geom.varias[:g_var]
z_vars = geom.varias[:z_var]
η_vars = geom.varias[:η_var]
X_vars = vcat(g_vars, z_vars)
vars = vcat(g_vars, z_vars, η_vars)

H_symbols = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block_half(S_no_phase, vars)

η_h_vertices = DηDLUtils.get_bulk_faces_vertices(geom)
eta_h = [LorentzianSimplexSolver.DefineSymbols.make_symbol("η_$(faces[1][1])$(faces[1][2])$(faces[1][3])") for faces in geom.connectivity[1]["OrderBulkFaces"]]
bulk_edges, bdry_edges = DηDLUtils.get_bulk_edges(geom, η_h_vertices)
bdry_edges_perturb = [bdry_edges[1]]   # perturb only one boundary edge
perturb_edges = vcat(bulk_edges, bdry_edges_perturb)

nh = length(geom.connectivity[1]["OrderBulkFaces"])
nb = length(geom.connectivity[1]["OrderBDryFaces"])
nl = length(perturb_edges)
nt = nh - nl
nX = length(X_vars)

kb_vertices = [geom.connectivity[1]["TetFaces"][f[1][1]][f[1][2]][f[1][3]] for f in geom.connectivity[1]["OrderBDryFaces"]]
DϵDl = DθDl_module.compute_dθDl(simplices, η_h_vertices, perturb_edges, vertex_coords, geom.connectivity[1]["Tets"], ScalarT)
DΘDl = DθDl_module.compute_dθDl(simplices, kb_vertices, perturb_edges, vertex_coords, geom.connectivity[1]["Tets"], ScalarT)

# ------------------------------------------------------------
# Evaluate everything at the chosen gamma
# ------------------------------------------------------------
println("[5/6] Evaluating all quantities at gamma = $gamma_val...")

vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd, γsym; γval=gamma_val)

# Spinfoam action
S_val = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_no_phase, vals)
SF_action = SymEngine.expand(S_val)
println("\nSpinfoam action:")
display(SF_action)

# Hessian
H_eval = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_block(H_symbols, sd; γ=gamma_val)
Hxx = H_eval[1:nX, 1:nX]
println("\nDeterminant of Hessian X-X block:")
println(det(Hxx))

# Derivative matrices
dηdl_matrix = DηDLUtils.build_dηdl_matrix(η_h_vertices, perturb_edges, vertex_coords, ScalarT, gamma_val)
eListHT = TransverseBasis.compute_transverse_basis(dηdl_matrix, tol)

dlogEh_dX_vals = RunDlogEhDX.evaluate_dlogEh_dX(dlogEh_dX_sym, geom, sd; γval=gamma_val)
dlogEb_dX_vals, dlogEb_dY_vals = RunDlogEhDX.evaluate_dlogEb_dXY(dlogEb_dX_sym, dlogEb_dY_sym, Y_vars, geom, sd, phase_soln; γval=gamma_val)
dkbEb_dX_vals, dkbEb_dY_vals, d2kbEb_dXdY_vals, d2kbEb_dYdY_vals = RunDlogEhDX.evaluate_kblogEb_all(dkbEb_dX_sym, dkbEb_dY_sym, d2kbEb_dXdY_sym, d2kbEb_dYdY_sym, geom, sd, Y_vars, phase_soln; γval=gamma_val)

d2kbEb_dXdY_vals_sumb = sum(d2kbEb_dXdY_vals[i, :, :] for i in 1:nb)
d2kbEb_dYdY_vals_sumb = sum(d2kbEb_dYdY_vals[i, :, :] for i in 1:nb)
dkbEb_dY_vals_sumb = sum(dkbEb_dY_vals[i, :] for i in 1:nb)

# Kernel matrix
eta_h_vals = [ScalarT(subs(eta_h[i], vals)) for i in 1:nh]
Ahh = Matrix(Diagonal(eta_h_vals))
hαβ = Hxx + transpose(dlogEh_dX_vals) * Ahh * dlogEh_dX_vals
invHessianXX = inv(Hxx)

ω = -dlogEh_dX_vals * invHessianXX * transpose(dlogEh_dX_vals)
ρ = inv(inv(Ahh) - ω)
Sij = -transpose(eListHT) * dlogEh_dX_vals * inv(hαβ) * transpose(dlogEh_dX_vals) * eListHT
κ = eListHT * inv(Sij) * transpose(eListHT)
M_kernel = ρ - ρ * inv(Ahh) * κ * inv(Ahh) * ρ

println("\nM_kernel:")
display(M_kernel)

# ------------------------------------------------------------
# Spinfoam linear / quadratic terms and Regge comparison
# ------------------------------------------------------------
println("[6/6] Computing linear/quadratic terms...")

Bα = transpose(transpose(dηdl_matrix) * dlogEh_dX_vals)

M_matrix = vcat(
    dlogEb_dY_vals[:, nb+1:end] - dlogEb_dX_vals * invHessianXX * d2kbEb_dXdY_vals_sumb[:, nb+1:end],
    -dlogEh_dX_vals * invHessianXX * d2kbEb_dXdY_vals_sumb[:, nb+1:end]
)

dmatrix = vcat(
    im * gamma_val/2 * DΘDl + dlogEb_dX_vals * invHessianXX * (Bα + d2kbEb_dXdY_vals_sumb[:, 1:nb] * dkbdl),
    im * gamma_val/2 * DϵDl + dlogEh_dX_vals * invHessianXX * (Bα + d2kbEb_dXdY_vals_sumb[:, 1:nb] * dkbdl)
)

DξDl = pinv(M_matrix, atol=tol, rtol=tol) * dmatrix
DYDl = vcat(dkbdl, DξDl)

SF_linear_bdry = transpose(dkbEb_dY_vals_sumb) * DYDl

Cα = d2kbEb_dXdY_vals_sumb * DYDl
BA = vcat(Bα + Cα, zeros(nt, 2))
Hαi = transpose(dlogEh_dX_vals) * eListHT
HIJ = [hαβ Hαi; transpose(Hαi) zeros(nt, nt)]
invHIJ = inv(HIJ)

SF_quadratic_form = -1/2 * transpose(BA[1:nX, :]) * invHIJ[1:nX, 1:nX] * BA[1:nX, :] +
                    1/2 * transpose(DYDl) * d2kbEb_dYdY_vals_sumb * DYDl

iSRegge_linear = im/2 * gamma_val * transpose(dkbdl) * dihedral_angles
iSRegge_quadratic = im * (gamma_val/4 * transpose(dηdl_matrix) * DϵDl +
                          gamma_val/4 * (transpose(dkbdl) * DΘDl + sum(dihedral_angles[i] * d2kb_dldl[i, :, :] for i in 1:nb)))
correction = -gamma_val^2 / 8 * transpose(DϵDl) * M_kernel * DϵDl

println("\nSpinfoam linear term:")
display(SF_linear_bdry)

println("\nSpinfoam quadratic term:")
display(SF_quadratic_form)

println("\nRegge linear term:")
display(iSRegge_linear)

println("\nRegge quadratic term:")
display(iSRegge_quadratic)

println("\nKernel correction term:")
display(correction)

println("\nDone.")
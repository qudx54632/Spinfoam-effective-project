# main_interactive_driver.jl
# Interactive driver for single-γ spinfoam / Regge computation

using Pkg
Pkg.activate("src/LorentzianSimplexSolver")
Pkg.instantiate()

using LorentzianSimplexSolver
using SymEngine
using JLD2
using LinearAlgebra
using Symbolics
using DoubleFloats

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
# Input helpers
# ------------------------------------------------------------

# Choose the numeric precision used throughout the computation.
function choose_scalartype()
    println("Choose scalar type:")
    println("  1) Float64")
    println("  2) BigFloat")
    print("> ")

    choice = strip(readline())

    if choice == "1"
        return Float64, 1e-8
    elseif choice == "2"
        return BigFloat, BigFloat(1e-30)
    else
        error("Invalid choice. Please enter 1, or 2.")
    end
end

# Read simplices from one line, e.g.
# [[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6]]
function read_simplices()
    println()
    println("Enter simplices:")
    println("[[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6],[1,3,4,5,6],[2,3,4,5,6]]")
    print("> ")

    line = strip(readline())
    isempty(line) && error("No simplices entered.")

    simplices = try
        Meta.parse(line) |> eval
    catch e
        error("Could not parse simplices input:\n$e")
    end

    if !(simplices isa Vector{<:Vector{<:Integer}})
        error("Input must be a vector of integer vectors.")
    end

    return [Int.(s) for s in simplices]
end

# Read one coordinate line per vertex.
# Example line:
# 0,-2.7,-0.9,-1.6
function read_coord_lines(nv::Int)
    println()
    println("Enter $nv coordinate lines (one per vertex):")
    println("Format: t,x,y,z")
    coord_lines = String[]

    for i in 1:nv
        print("[$i] ")
        line = strip(readline())
        isempty(line) && error("Empty coordinate line at entry $i.")
        push!(coord_lines, line)
    end

    return coord_lines
end

# ------------------------------------------------------------
# Start
# ------------------------------------------------------------

println("========================================")
println(" Spinfoam / Regge Interactive Driver")
println("========================================")
println()

ScalarT, tol = choose_scalartype()

if ScalarT === BigFloat
    setprecision(BigFloat, 100)
    LorentzianSimplexSolver.PrecisionUtils.set_big_precision!(100)
end
LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(tol)

println()
println("Using scalar type: $ScalarT")
println("Tolerance: $tol")

@variables γ

# ------------------------------------------------------------
# Read simplices
# ------------------------------------------------------------

simplices = read_simplices()

ns = length(simplices)
all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)
Nverts = length(all_vertices)

println()
println("Detected $ns simplex/s")
println("Vertices: ", all_vertices)

# ------------------------------------------------------------
# Read coordinates
# ------------------------------------------------------------

println()
println("Enter coordinates in the order of the vertices above.")

coord_lines = read_coord_lines(Nverts)

vertex_coords = Dict{Int, Vector{ScalarT}}()
for (i, v) in enumerate(all_vertices)
    nums = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(coord_lines[i], ScalarT)
    length(nums) == 4 || error("Vertex $v: expected 4 numbers, got $(length(nums))")
    vertex_coords[v] = nums
end

println()
println("Coordinates loaded.")

# ------------------------------------------------------------
# Read gamma
# ------------------------------------------------------------

println()
print("Enter γ: ")
gamma_val = parse(ScalarT, strip(readline()))

println("Using γ = $gamma_val")

# ------------------------------------------------------------
# Geometry and action setup
# ------------------------------------------------------------

println()
println("[1/6] Building geometry...")
geom = RunGeometry.run_geometry_pipeline(simplices, coord_lines, ScalarT, tol)

println("[2/6] Building action and Regge data...")
γsym = LorentzianSimplexSolver.DefineAction.γsym()
_, dihedral_angles, _, _, iRegge = LorentzianSimplexSolver.ReggeAction.run_Regge_action(
    geom, simplices, vertex_coords
)

sd, S_symbols, phase_soln = RunAction.run_action(geom, dihedral_angles, γsym)
S_no_phase = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_symbols, phase_soln)

println()
println("iRegge:")
display(iRegge)

# ------------------------------------------------------------
# Symbolic derivatives
# ------------------------------------------------------------

println("[3/6] Preparing symbolic derivatives...")
dlogEh_dX_sym = RunDlogEhDX.run_dlogEh_dX(geom)
dlogEb_dX_sym, dlogEb_dY_sym, Y_vars = RunDlogEhDX.run_dlogEb_dXY(geom)
dkbEb_dX_sym, dkbEb_dY_sym, d2kbEb_dXdY_sym, d2kbEb_dYdY_sym =
    RunDlogEhDX.run_kblogEb_dXY(geom, Y_vars)

# ------------------------------------------------------------
# Hessian and perturbation data
# ------------------------------------------------------------

println("[4/6] Constructing Hessian and perturbation data...")

g_vars = geom.varias[:g_var]
z_vars = geom.varias[:z_var]
η_vars = geom.varias[:η_var]
X_vars = vcat(g_vars, z_vars)
vars = vcat(g_vars, z_vars, η_vars)

H_symbols = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block_half(S_no_phase, vars)

η_h_vertices = DηDLUtils.get_bulk_faces_vertices(geom)
eta_h = [
    LorentzianSimplexSolver.DefineSymbols.make_symbol(
        "η_$(faces[1][1])$(faces[1][2])$(faces[1][3])"
    ) for faces in geom.connectivity[1]["OrderBulkFaces"]
]

bulk_edges, bdry_edges = DηDLUtils.get_bulk_edges(geom, η_h_vertices)

# Currently perturb only one boundary edge.
bdry_edges_perturb = [bdry_edges[1]]
perturb_edges = vcat(bulk_edges, bdry_edges_perturb)

nh = length(geom.connectivity[1]["OrderBulkFaces"])
nb = length(geom.connectivity[1]["OrderBDryFaces"])
nl = length(perturb_edges)
nt = nh - nl
nX = length(X_vars)

kb_vertices = [
    geom.connectivity[1]["TetFaces"][f[1][1]][f[1][2]][f[1][3]]
    for f in geom.connectivity[1]["OrderBDryFaces"]
]

DϵDl = DθDl_module.compute_dθDl(
    simplices, η_h_vertices, perturb_edges, vertex_coords, geom.connectivity[1]["Tets"], ScalarT
)

DΘDl = DθDl_module.compute_dθDl(
    simplices, kb_vertices, perturb_edges, vertex_coords, geom.connectivity[1]["Tets"], ScalarT
)

# ------------------------------------------------------------
# Evaluate at chosen gamma
# ------------------------------------------------------------

println("[5/6] Evaluating at γ = $gamma_val ...")

vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd, γsym; γval=gamma_val)

S_val = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_no_phase, vals)
SF_action = SymEngine.expand(S_val)

println()
println("Spinfoam action:")
display(SF_action)

H_eval = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_block(H_symbols, sd; γ=gamma_val)
Hxx = H_eval[1:nX, 1:nX]

println()
println("det(Hxx) = ")
println(det(Hxx))

dηdl_matrix = DηDLUtils.build_dηdl_matrix(η_h_vertices, perturb_edges, vertex_coords, ScalarT, gamma_val)
eListHT = TransverseBasis.compute_transverse_basis(dηdl_matrix, tol)

dkbdl, d2kb_dldl = Soln_dY_dX.dkb_dl(
    geom, nl, bdry_edges_perturb, vertex_coords; γ=gamma_val
)

dlogEh_dX_vals = RunDlogEhDX.evaluate_dlogEh_dX(dlogEh_dX_sym, geom, sd; γval=gamma_val)

dlogEb_dX_vals, dlogEb_dY_vals = RunDlogEhDX.evaluate_dlogEb_dXY(
    dlogEb_dX_sym, dlogEb_dY_sym, Y_vars, geom, sd, phase_soln; γval=gamma_val
)

dkbEb_dX_vals, dkbEb_dY_vals, d2kbEb_dXdY_vals, d2kbEb_dYdY_vals =
    RunDlogEhDX.evaluate_kblogEb_all(
        dkbEb_dX_sym, dkbEb_dY_sym, d2kbEb_dXdY_sym, d2kbEb_dYdY_sym,
        geom, sd, Y_vars, phase_soln; γval=gamma_val
    )

d2kbEb_dXdY_vals_sumb = sum(d2kbEb_dXdY_vals[i, :, :] for i in 1:nb)
d2kbEb_dYdY_vals_sumb = sum(d2kbEb_dYdY_vals[i, :, :] for i in 1:nb)
dkbEb_dY_vals_sumb = sum(dkbEb_dY_vals[i, :] for i in 1:nb)

# Build kernel-related matrices.
eta_h_vals = [ScalarT(subs(eta_h[i], vals)) for i in 1:nh]

Ahh = Matrix(Diagonal(eta_h_vals))
hαβ = Hxx + transpose(dlogEh_dX_vals) * Ahh * dlogEh_dX_vals
invHessianXX = inv(Hxx)

ω = -dlogEh_dX_vals * invHessianXX * transpose(dlogEh_dX_vals)
ρ = inv(inv(Ahh) - ω)
Sij = -transpose(eListHT) * dlogEh_dX_vals * inv(hαβ) * transpose(dlogEh_dX_vals) * eListHT
κ = eListHT * inv(Sij) * transpose(eListHT)
M_kernel = ρ - ρ * inv(Ahh) * κ * inv(Ahh) * ρ

println()
println("M_kernel:")
display(M_kernel)

# ------------------------------------------------------------
# Linear and quadratic terms
# ------------------------------------------------------------

println("[6/6] Computing linear and quadratic terms...")

Bα = transpose(transpose(dηdl_matrix) * dlogEh_dX_vals)

M_matrix = vcat(
    dlogEb_dY_vals[:, nb+1:end] - dlogEb_dX_vals * invHessianXX * d2kbEb_dXdY_vals_sumb[:, nb+1:end],
    -dlogEh_dX_vals * invHessianXX * d2kbEb_dXdY_vals_sumb[:, nb+1:end]
)

dmatrix = vcat(
    im * gamma_val / 2 * DΘDl +
    dlogEb_dX_vals * invHessianXX * (Bα + d2kbEb_dXdY_vals_sumb[:, 1:nb] * dkbdl),

    im * gamma_val / 2 * DϵDl +
    dlogEh_dX_vals * invHessianXX * (Bα + d2kbEb_dXdY_vals_sumb[:, 1:nb] * dkbdl)
)

DξDl = pinv(M_matrix, atol=tol, rtol=tol) * dmatrix
DYDl = vcat(dkbdl, DξDl)

SF_linear_bdry = transpose(dkbEb_dY_vals_sumb) * DYDl

Cα = d2kbEb_dXdY_vals_sumb * DYDl
BA = vcat(Bα + Cα, zeros(nt, nl))
Hαi = transpose(dlogEh_dX_vals) * eListHT
HIJ = [hαβ Hαi; transpose(Hαi) zeros(nt, nt)]
invHIJ = inv(HIJ)

DXDl = -invHessianXX * (
    Bα +
    d2kbEb_dXdY_vals_sumb[:, 1:nb] * dkbdl +
    d2kbEb_dXdY_vals_sumb[:, nb+1:end] * DξDl
)

kb_vals = [ScalarT(subs(eta_h[i], vals)) for i in 1:nh]
Mm = [kb_vals[b] * dlogEb_dY_vals[b, j] for b in 1:nb, j in nb+1:length(Y_vars)]

R = [
    dkbdl[i, j] * transpose(dlogEb_dY_vals[i, nb+1:end]) * DξDl[:, j] +
    transpose(DXDl[:, j]) * d2kbEb_dXdY_vals[i, :, nb+1:end] * DξDl[:, j] +
    transpose(DξDl[:, j]) * d2kbEb_dYdY_vals[i, nb+1:end, nb+1:end] * DξDl[:, j]
    for i in 1:nb, j in 1:nl
]

D2ξDl2 = -pinv(Mm, atol=tol, rtol=tol) * R[:, end]

nxi = length(Y_vars) - nb
D2ξDl2_matrix = [j == k == nl ? D2ξDl2[b] : 0 for b in 1:nxi, j in 1:nl, k in 1:nl]
D2YDl2_matrix = vcat(d2kb_dldl, D2ξDl2_matrix)

Iboundary_qadratic =
    1 / 2 * transpose(DYDl) * d2kbEb_dYdY_vals_sumb * DYDl +
    1 / 2 * sum(dkbEb_dY_vals_sumb[i] * D2YDl2_matrix[i, :, :] for i in 1:length(Y_vars))

SF_quadratic_form =
    -1 / 2 * transpose(BA[1:nX, :]) * invHIJ[1:nX, 1:nX] * BA[1:nX, :] +
    Iboundary_qadratic

iSRegge_linear = im / 2 * gamma_val * transpose(dkbdl) * dihedral_angles

iSRegge_quadratic = im * (
    gamma_val / 4 * transpose(dηdl_matrix) * DϵDl +
    gamma_val / 4 * (
        transpose(dkbdl) * DΘDl +
        sum(dihedral_angles[i] * d2kb_dldl[i, :, :] for i in 1:nb)
    )
)

correction = -gamma_val^2 / 8 * transpose(DϵDl) * M_kernel * DϵDl

println()
println("Spinfoam linear term:")
display(SF_linear_bdry)

println()
println("Spinfoam quadratic term:")
display(SF_quadratic_form)

println()
println("Regge linear term:")
display(iSRegge_linear)

println()
println("Regge quadratic term:")
display(iSRegge_quadratic)

println()
println("Kernel correction:")
display(correction)

println()
println("Done.")
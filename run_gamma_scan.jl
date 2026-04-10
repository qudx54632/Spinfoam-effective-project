using Pkg

Pkg.activate("src/LorentzianSimplexSolver")
Pkg.instantiate()

using LorentzianSimplexSolver
using DoubleFloats
using SymEngine

include("scripts/run_geometry.jl")
include("scripts/run_action.jl")
include("scripts/run_dlogEh_dg.jl")
include("scripts/run_djdl.jl")

include("src/perturbations/TransverseBasis.jl")
include("src/perturbations/LinearizedEOMs.jl")
include("src/perturbations/QuadraticRegge.jl")

using JLD2
using LinearAlgebra
using Symbolics

using .RunGeometry
using .RunAction
using .RunDlogEhDG
using .DJDLUtils
using .TransverseBasis
using .QuadraticRegge


# ------------------------------------------------------------
# Define the precision and tolerance for numerical operations
# ------------------------------------------------------------
const ScalarT = Double64
const tol = 1e-10
LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(tol)

# ============================================================
# User input section
# ============================================================
@variables γ

# Example gamma list
gamma_list = ScalarT(10.0) .^ range(ScalarT(0), ScalarT(-6), length=40)
#gamma_list = [ScalarT(1.0)]
# Output file
outfile = "data/gamma_scan_B2Wholes.jld2"

# ------------------------------------------------------------
# Put here the same geometry input used in your single-gamma code
# ------------------------------------------------------------
simplices = [
    [2, 5, 7, 8, 13], [3, 5, 6, 8, 13], [4, 5, 6, 7, 13], [2, 3, 5, 8, 13],
    [2, 4, 5, 7, 13], [3, 4, 5, 6, 13], [2, 3, 4, 5, 13], [3, 6, 8, 13, 14],
    [4, 6, 7, 13, 14], [3, 4, 6, 13, 14], [1, 6, 7, 8, 14], [1, 3, 6, 8, 14],
    [1, 4, 6, 7, 14], [1, 3, 4, 6, 14], [4, 7, 13, 14, 15], [1, 7, 8, 14, 15],
    [1, 4, 7, 14, 15], [2, 7, 8, 13, 15], [2, 4, 7, 13, 15], [1, 2, 7, 8, 15],
    [1, 2, 4, 7, 15], [1, 8, 14, 15, 16], [2, 8, 13, 15, 16], [1, 2, 8, 15, 16],
    [3, 8, 13, 14, 16], [1, 3, 8, 14, 16], [2, 3, 8, 13, 16], [1, 2, 3, 8, 16],
    [2, 9, 11, 12, 13], [3, 9, 10, 12, 13], [4, 9, 10, 11, 13], [2, 3, 9, 12, 13],
    [2, 4, 9, 11, 13], [3, 4, 9, 10, 13], [2, 3, 4, 9, 13], [3, 10, 12, 13, 14],
    [4, 10, 11, 13, 14], [3, 4, 10, 13, 14], [1, 10, 11, 12, 14], [1, 3, 10, 12, 14],
    [1, 4, 10, 11, 14], [1, 3, 4, 10, 14], [4, 11, 13, 14, 15], [1, 11, 12, 14, 15],
    [1, 4, 11, 14, 15], [2, 11, 12, 13, 15], [2, 4, 11, 13, 15], [1, 2, 11, 12, 15],
    [1, 2, 4, 11, 15], [1, 12, 14, 15, 16], [2, 12, 13, 15, 16], [1, 2, 12, 15, 16],
    [3, 12, 13, 14, 16], [1, 3, 12, 14, 16], [2, 3, 12, 13, 16], [1, 2, 3, 12, 16]
]
# simplices = [[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6],[1,3,4,5,6],[2,3,4,5,6]]

all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)

coords_lines = [
    "0, 9.4280904158206336586779248280646538571311458358463, 0, -3.3333333333333333333333333333333333333333333333333",
    "0, -4.7140452079103168293389624140323269285655729179232, 8.1649658092772603273242802490196379732198249355222, -3.3333333333333333333333333333333333333333333333333",
    "0, -4.7140452079103168293389624140323269285655729179232, -8.1649658092772603273242802490196379732198249355222, -3.3333333333333333333333333333333333333333333333333",
    "0, 0, 0, 10.000000000000000000000000000000000000000000000000",
    "1329.8215531505937217882374511637806643740013498050, -3790.0923471598947307885257808819908505667206260102, 0, 1340.0000000000000000000000000000000000000000000000",
    "1329.8215531505937217882374511637806643740013498050, 1895.0461735799473653942628904409954252833603130051, -3282.3162553294586515843606601058944652343696240799, 1340.0000000000000000000000000000000000000000000000",
    "1329.8215531505937217882374511637806643740013498050, 1895.0461735799473653942628904409954252833603130051, 3282.3162553294586515843606601058944652343696240799, 1340.0000000000000000000000000000000000000000000000",
    "1329.8215531505937217882374511637806643740013498050, 0, 0, -4020.0000000000000000000000000000000000000000000000",
    "1329.8215531505937217882374511637806643740013498050, -3790.0923471598947307885257808819908505667206260102, 0, 1340.0000000000000000000000000000000000000000000000",
    "1329.8215531505937217882374511637806643740013498050, 1895.0461735799473653942628904409954252833603130051, -3282.3162553294586515843606601058944652343696240799, 1340.0000000000000000000000000000000000000000000000",
    "1329.8215531505937217882374511637806643740013498050, 1895.0461735799473653942628904409954252833603130051, 3282.3162553294586515843606601058944652343696240799, 1340.0000000000000000000000000000000000000000000000",
    "1329.8215531505937217882374511637806643740013498050, 0, 0, -4020.0000000000000000000000000000000000000000000000",
    "1389.2747047417341491052201825950879169736927773464, -3959.7979746446661366447284277871546199950812510555, 0, 1400.0000000000000000000000000000000000000000000000",
    "1389.2747047417341491052201825950879169736927773464, 1979.8989873223330683223642138935773099975406255277, -3429.2856398964493374761977045882479487523264729193, 1400.0000000000000000000000000000000000000000000000",
    "1389.2747047417341491052201825950879169736927773464, 1979.8989873223330683223642138935773099975406255277, 3429.2856398964493374761977045882479487523264729193, 1400.0000000000000000000000000000000000000000000000",
    "1389.2747047417341491052201825950879169736927773464, 0, 0, -4200.0000000000000000000000000000000000000000000000"
]
# coords_lines = [
#     "0, 0, 0, 0",
#     "0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
#     "0, 0, 0, -3.398088489694245",
#     "-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
#     "0, 0, -2.942830956382712, -1.6990442448471226",
#     "-0.068,-0.27,-0.5,-1.3",
# ]

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

# println("Constructing symbolic Hessian block...")
# H_base_block = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block(S_base, vars)
println("Precomputing first derivatives...")
dS_precomp = [SymEngine.diff(S_base, v) for v in vars]

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

    # ========================================================
    # Evaluate action/Hessian at this gamma
    # ========================================================
    # evaluate Hessian
    println("Evaluating Hessian for gamma = $gamma_val")
    # HessianOld = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_ondemand(S_base, vars, sd_base; γ=gamma_val)
    HessianOld = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_from_dS(dS_precomp, vars, sd_base; γ=gamma_val)

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

end

# ============================================================
# Save
# ============================================================
mkpath(dirname(outfile))
@save outfile gamma_list Spinfoam_Quadratic1_list ReggeQuadratic_list kappa_matrix_list correction_list

println("Saved results to ", outfile)
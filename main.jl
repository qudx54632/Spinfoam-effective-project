using Pkg

Pkg.activate("src/LorentzianSimplexSolver")
Pkg.instantiate()

using LorentzianSimplexSolver
using Symbolics
using SymEngine

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

using Symbolics

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

ns = length(simplices)

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

#const ScalarT = Float64
# tol = 1e-10;
const ScalarT = BigFloat
const tol = parse(ScalarT, "1e-8")

if ScalarT === BigFloat
    setprecision(BigFloat, 100)
    LorentzianSimplexSolver.PrecisionUtils.set_big_precision!(100)
    # LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(sqrt(eps(BigFloat)))
    LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(tol)
else
    LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(tol)
end

gamma_vals = ScalarT(1e-6);

vertex_coords = Dict{Int, Vector{ScalarT}}()  

for (v, line) in zip(all_vertices, coords_lines)
    vertex_coords[v] = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(line, ScalarT)
end

geom_base = run_geometry_pipeline(simplices, coords_lines, ScalarT, tol)

γ = LorentzianSimplexSolver.DefineAction.γsym()
sd_base, S_base = RunAction.run_action(geom_base, γ)
vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd_base, γ; γval=gamma_vals)
S_val = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(S_base, vals);
S_simpl = SymEngine.expand(S_val) 
println("Evaluating action for $gamma_vals is $S_simpl")
    

g_vars = geom_base.varias[:g_var]
z_vars = geom_base.varias[:z_var]
j_vars = geom_base.varias[:j_var]
vars = vcat(g_vars, z_vars, j_vars)

println("Evaluating Hessian for gamma = $gamma_vals")
H_base = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block_half(S_base, vars)
H_base_eval = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_block(H_base, sd_base; γ=gamma_vals)

nh = length(j_vars)
dlogEh_dg_sym = RunDlogEhDG.run_dlogEh_dg(geom_base)
dlogEh_dg_vals = RunDlogEhDG.evaluate_dlogEh_dg(dlogEh_dg_sym, geom_base, sd_base; γval=gamma_vals)

# djdl_matrix is a nh x nl matrix
djdl_matrix, bulk_edges, j_h_vertices = DJDLUtils.build_djdl_matrix(geom_base, vertex_coords, ScalarT, gamma_vals)
nl = length(bulk_edges)
nt = nh - nl

# eListHT is a nh x nt matrix
eListHT = TransverseBasis.compute_transverse_basis(djdl_matrix, tol)

ng = length(g_vars)
nz = length(z_vars)
nx = ng + nz

HessianOld = H_base_eval
@variables dl[1:nl]
dl_vec = collect(dl) 
dYsoln, HYY = LinearizedEOMs.solve_linearized_eoms(HessianOld, eListHT, djdl_matrix, nx, ScalarT, dl_vec)
invHessianXX = inv(HessianOld[1:nx, 1:nx])[1:ng, 1:ng]

Wmatrix = -dlogEh_dg_vals * invHessianXX * transpose(dlogEh_dg_vals)

Smatrix = eListHT' * Wmatrix * eListHT
Smatrix_num = Complex{ScalarT}.(Symbolics.value.(Smatrix))

kappa_matrix = eListHT * inv(Smatrix_num) * eListHT'
quadratic_correct = - ScalarT(0.5) .* djdl_matrix' * Wmatrix * kappa_matrix * transpose(Wmatrix) * djdl_matrix
println("Quadratic correction from integrating out g and z: $quadratic_correct")

W_all_matrix = -dlogEh_dg_vals * inv(HYY)[1:ng, 1:ng] * transpose(dlogEh_dg_vals);
Spinfoam_Quadratic1 = ScalarT(0.5) * transpose(djdl_matrix) * W_all_matrix * djdl_matrix

dthetadl = QuadraticRegge.compute_dθDl(simplices, j_h_vertices, bulk_edges, vertex_coords, geom_base.connectivity[1]["Tets"], LorentzianSimplexSolver.Dihedral.minkowski_norm2, ScalarT)
dadl = gamma_vals .* djdl_matrix
ReggeQuadratic = im * ScalarT(0.5) * dadl' * dthetadl

correction = gamma_vals^2 * ScalarT(0.5) * dthetadl' * kappa_matrix * dthetadl
Spinfoam_Quadratic = ReggeQuadratic + correction
println("Spinfoam quadratic term is: $Spinfoam_Quadratic")
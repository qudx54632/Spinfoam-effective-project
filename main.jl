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

simplices = [[2,3,4,6,7],[1,3,4,6,7],[1,2,4,6,7],[1,2,3,6,7],[1,2,3,4,7],[2,3,5,6,8],[1,3,5,6,8],[1,2,5,6,8],[1,2,3,6,8],[1,2,3,5,8],[2,4,5,6,9],[1,4,5,6,9],[1,2,5,6,9],[1,2,4,6,9],[1,2,4,5,9],[3,4,5,6,10],[1,4,5,6,10],[1,3,5,6,10],[1,3,4,6,10],[1,3,4,5,10],[3,4,5,6,11],[2,4,5,6,11],[2,3,5,6,11],[2,3,4,6,11],[2,3,4,5,11]]

ns = length(simplices)

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

const ScalarT = Float64
# using DoubleFloats
# const ScalarT = Double64
# tol = 1e-10;
#const ScalarT = BigFloat
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

kappa_matrix = eListHT * inv(Smatrix) * eListHT'
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
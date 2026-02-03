# module RunHessian

# export run_hessian

# using LorentzianSimplexSolver
# using JLD2

# function run_hessian(geom_base, gamma_vals, outfile)
#     # ------------------------------------------------------------
#     # 6. Symbols and action (reference orientation)
#     # ------------------------------------------------------------
#     LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom_base)

#     sd_base, _ =
#         LorentzianSimplexSolver.SolveVars.run_solver(geom_base)

#     S_base =
#         LorentzianSimplexSolver.DefineAction.compute_action(geom_base)

#     # ------------------------------------------------------------
#     # 7. Hessian
#     # ------------------------------------------------------------
#     H_base =
#         LorentzianSimplexSolver.EOMsHessian.compute_Hessian(S_base, sd_base)

#     hess_base_fns =
#         LorentzianSimplexSolver.SymbolicToJulia.build_hessian_functions(H_base, sd_base)

#     H_base_eval, _ =
#         LorentzianSimplexSolver.EOMsHessian.evaluate_hessian(hess_base_fns, sd_base; γ = gamma_vals)

#     # ------------------------------------------------------------
#     # 8. Save result
#     # ------------------------------------------------------------
#     @save outfile H_base_eval

#     return nothing
# end

# end # module RunHessian

# ============================================================
# examples/Delta3-Hessian.jl
# ============================================================

# Optional: make sure we are using the package environment when running this file directly
# (safe even if already activated)
# try
#     using Pkg
#     Pkg.activate(joinpath(@__DIR__, ".."))
# catch
# end

using LorentzianSimplexSolver
using LinearAlgebra
using Printf
using Dates
using Symbolics
using PythonCall
using JLD2

# If you want sympy available (only if your package needs it at runtime)
sympy = pyimport("sympy")

# ------------------------------------------------------------
# 1. Precision choice (user-controlled)
# ------------------------------------------------------------
const ScalarT = Float64
LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(1e-10)

# ------------------------------------------------------------
# 2. Read simplices
# ------------------------------------------------------------
simplices = [[1,2,3,4,6], [1,2,3,5,6], [1,2,4,5,6],[1,2,3,4,7], [1,2,3,5,7], [1,2,4,5,7]]

ns = length(simplices)

all_vertices = unique(Iterators.flatten(simplices))
sort!(all_vertices)

Nverts = length(all_vertices)

# ------------------------------------------------------------
# 3. Read vertex coordinates
# ------------------------------------------------------------
vertex_coords = Dict{Int, Vector{ScalarT}}()    

coords_lines = [
    "0, 0, 0, 0",
    "-0.068000000000000005, -0.21988127663727278, -0.5316227766016838, -1.3316227766016839",
    "0, 0, 0, -3.398088489694245",
    "-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
    "0, 0, -2.942830956382712, -1.6990442448471226",
    "0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
    "-2.4696884592430974, -3.893218630529324, -1.3565336794679874, -1.9090667752920147",
]

for (v, line) in zip(all_vertices, coords_lines)
    vertex_coords[v] = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(line, ScalarT)
end

# ------------------------------------------------------------
# 4. Build geometry
# ------------------------------------------------------------
datasets = LorentzianSimplexSolver.GeometryTypes.GeometryDataset{ScalarT}[]

for (s, simplex) in enumerate(simplices)
    # println("\n--- Processing simplex $s with vertices $simplex ---")
    bdypoints = [vertex_coords[v] for v in simplex]
    ds = LorentzianSimplexSolver.GeometryPipeline.run_geometry_pipeline(bdypoints)
    push!(datasets, ds)
end

geom_base = LorentzianSimplexSolver.GeometryTypes.GeometryCollection(datasets);

# ------------------------------------------------------------
# 5. Connect simplices + face matching + gauge fixing
# ------------------------------------------------------------
LorentzianSimplexSolver.KappaOrientation.fix_kappa_signs!(simplices, geom_base)
conn = LorentzianSimplexSolver.FourSimplexConnectivity.build_global_connectivity(simplices, geom_base)
push!(geom_base.connectivity, conn)
LorentzianSimplexSolver.FaceXiMatching.run_face_xi_matching(geom_base; sector=:ref)
LorentzianSimplexSolver.GaugeFixingSU.run_su2_su11_gauge_fix(geom_base)

# ------------------------------------------------------------
# 6. Symbols and action (reference orientation)
# ------------------------------------------------------------
LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom_base)
sd_base, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom_base)
S_base = LorentzianSimplexSolver.DefineAction.compute_action(geom_base);

# ------------------------------------------------------------
# 7. Compute Hessian matrix
# ------------------------------------------------------------
H_base = LorentzianSimplexSolver.EOMsHessian.compute_Hessian(S_base, sd_base)
hess_base_fns = LorentzianSimplexSolver.SymbolicToJulia.build_hessian_functions(H_base, sd_base)
H_base_eval, _ = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian(hess_base_fns, sd_base; γ = one(ScalarT));

@save joinpath(pwd(), "Hessian_data/H_base_eval_2Delta3.jld2") H_base_eval
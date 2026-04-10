using LorentzianSimplexSolver
using LinearAlgebra
using Printf
using Dates
using Symbolics
using JLD2

# ------------------------------------------------------------
# 1. Precision choice (user-controlled)
# ------------------------------------------------------------
const ScalarT = Float64
LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(1e-10)

# ------------------------------------------------------------
# 2. Read simplices
# ------------------------------------------------------------
simplices = [[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6],[1,3,4,5,6],[2,3,4,5,6]]

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
    "0, -2.7745276335252114, -0.9809436521275706, -1.6990442448471226",
    "0, 0, 0, -3.398088489694245",
    "-0.24028114141347542, -0.6936319083813028, -0.9809436521275706, -1.6990442448471226",
    "0, 0, -2.942830956382712, -1.6990442448471226",
    "-0.068,-0.27,-0.5,-1.3",
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

@save joinpath(pwd(), "../../data/H_base_15move.jld2") H_base
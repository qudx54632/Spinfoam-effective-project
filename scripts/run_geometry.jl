module RunGeometry

export run_geometry_pipeline

using LorentzianSimplexSolver

function run_geometry_pipeline(
    simplices::Vector{<:AbstractVector{Int}},
    coords_lines::Vector{String},
    ScalarT,
    tol
)

    # ------------------------------------------------------------
    # 1. Precision
    # ------------------------------------------------------------
    LorentzianSimplexSolver.PrecisionUtils.set_tolerance!(tol)

    # ------------------------------------------------------------
    # 2. Vertices
    # ------------------------------------------------------------
    all_vertices = unique(Iterators.flatten(simplices))
    sort!(all_vertices)

    @assert length(all_vertices) == length(coords_lines) 
        "Number of coordinate lines must match number of vertices"

    # ------------------------------------------------------------
    # 3. Parse coordinates
    # ------------------------------------------------------------
    vertex_coords = Dict{Int, Vector{ScalarT}}()

    for (v, line) in zip(all_vertices, coords_lines)
        vertex_coords[v] = LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(line, ScalarT)
    end

    # ------------------------------------------------------------
    # 4. Build geometry datasets
    # ------------------------------------------------------------
    datasets = LorentzianSimplexSolver.GeometryTypes.GeometryDataset{ScalarT}[]

    for simplex in simplices
        bdypoints = [vertex_coords[v] for v in simplex]
        ds = LorentzianSimplexSolver.GeometryPipeline.run_geometry_pipeline(bdypoints)
        push!(datasets, ds)
    end

    geom_base =
        LorentzianSimplexSolver.GeometryTypes.GeometryCollection(datasets)

    # ------------------------------------------------------------
    # 5. Connectivity, matching, gauge fixing
    # ------------------------------------------------------------
    LorentzianSimplexSolver.KappaOrientation.fix_kappa_signs!(simplices, geom_base)

    conn = LorentzianSimplexSolver.FourSimplexConnectivity.build_global_connectivity(simplices, geom_base)

    push!(geom_base.connectivity, conn)

    LorentzianSimplexSolver.FaceXiMatching.run_face_xi_matching(geom_base; sector = :ref)

    LorentzianSimplexSolver.GaugeFixingSU.run_su2_su11_gauge_fix(geom_base)

    return geom_base
end

end # module
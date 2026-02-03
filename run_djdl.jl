module DJDLUtils

export build_djdl_matrix

using LorentzianSimplexSolver

# ------------------------------------------------------------
# 1. Geometry helpers
# ------------------------------------------------------------

function get_bulk_faces_vertices(geom_base)
    j_h_all = geom_base.connectivity[1]["OrderBulkFaces"]

    return [
        geom_base.connectivity[1]["TetFaces"][v[1][1]][v[1][2]][v[1][3]]
        for v in j_h_all
    ]
end


function get_bulk_edges(geom_base, j_h_vertices)
    # edges from bulk faces
    edges_jh = [
        sort([v[i], v[j]])
        for v in j_h_vertices
        for (i, j) in ((1,2), (1,3), (2,3))
    ]
    unique_edges_jh = unique(edges_jh)

    # boundary faces
    j_b_all = geom_base.connectivity[1]["OrderBDryFaces"]
    j_b_vertices = [
        geom_base.connectivity[1]["TetFaces"][v[1][1]][v[1][2]][v[1][3]]
        for v in j_b_all
    ]

    edges_bdry = [
        sort([v[i], v[j]])
        for v in j_b_vertices
        for (i, j) in ((1,2), (1,3), (2,3))
    ]
    unique_edges_bdry = unique(edges_bdry)

    return setdiff(unique_edges_jh, unique_edges_bdry)
end


# ------------------------------------------------------------
# 2. Combinatorial helper
# ------------------------------------------------------------

@inline function edge_position(face::Vector{Int}, edge::Vector{Int})
    a, b, c = face
    u, v = edge

    (u==a && v==b || u==b && v==a) && return 1
    (u==a && v==c || u==c && v==a) && return 2
    (u==b && v==c || u==c && v==b) && return 3
    return 0
end


# ------------------------------------------------------------
# 3. Area / derivative formulas
# ------------------------------------------------------------

@inline function Δ(l1, l2, l3)
    return -l1^4 - l2^4 - l3^4 +
           2*(l1^2*l2^2 + l1^2*l3^2 + l2^2*l3^2)
end


@inline function djdl(l1, l2, l3, γ)
    Δval = Δ(l1, l2, l3)
    pref = 1 / (2 * γ * sqrt(Δval))

    return pref .* [
        l1*(l2^2 + l3^2 - l1^2),
        l2*(l1^2 + l3^2 - l2^2),
        l3*(l1^2 + l2^2 - l3^2)
    ]
end


# ------------------------------------------------------------
# 4. Length extraction
# ------------------------------------------------------------

function compute_face_edge_lengths(j_h_vertices, vertex_coords)
    j_h_coords = [
        [vertex_coords[i], vertex_coords[j], vertex_coords[k]]
        for (i, j, k) in map(Tuple, j_h_vertices)
    ]

    return [
        sqrt.([
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(coords[1] - coords[2]),
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(coords[1] - coords[3]),
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(coords[2] - coords[3])
        ])
        for coords in j_h_coords
    ]
end


# ------------------------------------------------------------
# 5. Main driver
# ------------------------------------------------------------

function build_djdl_matrix(
    geom_base,
    vertex_coords,
    ScalarT::Type{<:Real},
    gamma_vals
)

    # faces and edges
    j_h_vertices = get_bulk_faces_vertices(geom_base)
    bulk_edges   = get_bulk_edges(geom_base, j_h_vertices)

    nh = length(j_h_vertices)
    nb = length(bulk_edges)

    # lengths
    j_h_length = compute_face_edge_lengths(j_h_vertices, vertex_coords)

    djdl_matrix = zeros(ScalarT, nh, nb)

    for h in 1:nh
        for b in 1:nb
            pos = edge_position(j_h_vertices[h], bulk_edges[b])
            pos == 0 && continue

            lens = j_h_length[h]
            djdl_matrix[h, b] =
                djdl(lens[1], lens[2], lens[3], gamma_vals)[pos]
        end
    end

    return djdl_matrix, bulk_edges, j_h_vertices
end

end # module
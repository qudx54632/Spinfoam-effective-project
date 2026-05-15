module DθDl_module

using Symbolics
using Combinatorics
using LinearAlgebra
using LorentzianSimplexSolver

export compute_dθDl

# -------------------------------------------------
# Cayley–Menger volumes (symbolic)
# -------------------------------------------------

function V4sq_symbolic(ls12, ls13, ls14, ls15,
                       ls23, ls24, ls25,
                       ls34, ls35,
                       ls45)
    (-1)^(5) *
    det([
        0      ls12   ls13   ls14   ls15   1;
        ls12   0      ls23   ls24   ls25   1;
        ls13   ls23   0      ls34   ls35   1;
        ls14   ls24   ls34   0      ls45   1;
        ls15   ls25   ls35   ls45   0      1;
        1      1      1      1      1      0
    ]) / (2^4 * factorial(4)^2)
end

function V3sq_symbolic(ls12, ls13, ls14,
                       ls23, ls24, ls34)
    (-1)^(4) *
    det([
        0     ls12  ls13  ls14  1;
        ls12  0     ls23  ls24  1;
        ls13  ls23  0     ls34  1;
        ls14  ls24  ls34  0     1;
        1     1     1     1     0
    ]) / (2^3 * factorial(3)^2)
end

function V2sq_symbolic(ls12, ls13, ls23)
    (-1)^(3) *
    det([
        0     ls12  ls13  1;
        ls12  0     ls23  1;
        ls13  ls23  0     1;
        1     1     1     0
    ]) / (2^2 * factorial(2)^2)
end

# -------------------------------------------------
# ∂V4² / ∂ℓ
# -------------------------------------------------

function dV4s_ds(i::Int, simplex_edges)
    @variables x
    edges = collect(simplex_edges)
    edges[i] = x

    V = expand(simplify(V4sq_symbolic(edges...)))
    dV = Symbolics.expand_derivatives(Differential(x)(V))

    return substitute(dV, x => simplex_edges[i])
end

function dθdl_single_pp(simplex_edges,
                     teta_edges,
                     tetb_edges,
                     ah_edges,
                     hbar_idx, l_var::Num,
                     lsq_vals_dict,
                     ::Type{T}) where {T<:Real}

    # Volumes
    Vsigma = V4sq_symbolic(simplex_edges...)
    Vtaua  = V3sq_symbolic(teta_edges...)
    Vtaub  = V3sq_symbolic(tetb_edges...)
    Vt     = V2sq_symbolic(ah_edges...)

    # Replace opposite edge symbolically
    @variables x
    simplex_edges_new = copy(simplex_edges)
    simplex_edges_new[hbar_idx] = x
    Vsigmanew = V4sq_symbolic(simplex_edges_new...)

    Vtauab = substitute(Symbolics.derivative(Vsigmanew, x), x => simplex_edges[hbar_idx])

    term1 = 4^2 * Vtauab / Vt
    term2 = Vtaua / Vt
    term3 = Vtaub / Vt

    z = (term1 + sqrt(term1^2 - term2 * term3)) /
        (sqrt(term2) * sqrt(term3))
    z_val = T(Symbolics.value(substitute(z, lsq_vals_dict)))

    dz_dlsq = Symbolics.derivative(z, l_var)
    l_val = T(sqrt(lsq_vals_dict[l_var]))
    dz_dl = T(Symbolics.toexpr(substitute(dz_dlsq, lsq_vals_dict) * (2 * l_val)))

    return - dz_dl/z_val
end

# -------------------------------------------------
# index bookkeeping
# -------------------------------------------------
function build_face_indices(simplex_vertices,
                            face_vertices,
                            tets_in_simplex,
                            perturb_edges)

    tets = [tet for tet in tets_in_simplex
            if all(v -> v in tet, face_vertices)]

    edges_simplex = [collect(e) for e in combinations(simplex_vertices, 2)]
    edges_face    = [collect(e) for e in combinations(face_vertices, 2)]
    edges_tets    = [[collect(e) for e in combinations(t,2)] for t in tets]

    hbar = setdiff(simplex_vertices, face_vertices)
    hbar_idx = findfirst(e -> e == sort(hbar), edges_simplex)

    h_edge_indices = [
        findfirst(e -> e == sort(be), edges_simplex)
        for be in perturb_edges
    ]

    return (; edges_simplex, edges_face, edges_tets, hbar_idx, h_edge_indices)
end

# -------------------------------------------------
# main driver
# -------------------------------------------------
function compute_dθDl(
    simplices,
    η_vertices,
    perturb_edges,
    vertex_coords,
    tets_per_simplex, ::Type{T}
) where {T<:Real}

    nh = length(η_vertices)
    nb = length(perturb_edges)

    DθDl = zeros(Complex{T}, nh, nb)
    
    @variables lsq[1:nb]
    lsq_vec = collect(lsq)

    ls_vals = Dict(
        lsq_vec[i] => LorentzianSimplexSolver.Dihedral.minkowski_norm2(vertex_coords[e[1]] - vertex_coords[e[2]])
        for (i, e) in enumerate(perturb_edges))


    for h in 1:nh, i in eachindex(simplices)

        if !all(v -> v in simplices[i], η_vertices[h])
            continue
        end

        idx = build_face_indices(
            simplices[i],
            η_vertices[h],
            tets_per_simplex[i],
            perturb_edges
        )

        if all(isnothing, idx.h_edge_indices)
            continue
        end
        
        ls_simplex = [
            (k in idx.h_edge_indices) ? lsq_vec[findfirst(x->x==e, perturb_edges)] :
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(vertex_coords[e[1]] - vertex_coords[e[2]])
            for (k,e) in enumerate(idx.edges_simplex)
        ]
        
        ls_tetA = [
            (e in perturb_edges) ? lsq_vec[findfirst(x->x==e, perturb_edges)] :
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(vertex_coords[e[1]] - vertex_coords[e[2]])
            for e in idx.edges_tets[1]
        ]


        ls_tetB = [
            (e in perturb_edges) ? lsq_vec[findfirst(x->x==e, perturb_edges)] :
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(vertex_coords[e[1]] - vertex_coords[e[2]])
            for e in idx.edges_tets[2]
        ]

        ls_face = [
            (e in perturb_edges) ? lsq_vec[findfirst(x->x==e, perturb_edges)] :
            LorentzianSimplexSolver.Dihedral.minkowski_norm2(vertex_coords[e[1]] - vertex_coords[e[2]])
            for e in idx.edges_face
        ]

        for b in 1:nb
            DθDl[h,b] += dθdl_single_pp(ls_simplex, ls_tetA, ls_tetB, ls_face, idx.hbar_idx, lsq_vec[b], ls_vals, T)
        end
    end

    return DθDl
end

end # module
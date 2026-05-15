module Soln_dY_dX

using LinearAlgebra, SymEngine
using LorentzianSimplexSolver
using DoubleFloats

export solve_δY_δX

@inline val_basic(x) = Basic(string(x))

# ------------------------------------------------------------
# Cayley–Menger matrix for triangle (2-simplex)
# ------------------------------------------------------------
function CM2D(ls12, ls13, ls23)
    # Return a SymEngine Matrix{Basic}
    Basic[
        0 ls12 ls13 1;
        ls12 0 ls23 1;
        ls13 ls23 0 1;
        1   1   1  0
    ]
end
# Triangle area squared
function V2sq(ls12, ls13, ls23)
    # (-1)^(2+1) = -1
    # V2 = - det(CM) / (2^2 * 2!^2) with n=2
    # 2!^2 = 2^2, so denominator = 4 * 4 = 16
    detCM = det(CM2D(ls12, ls13, ls23))
    return -detCM / 16
end

function build_distance_subs(edge_bdry_pairs, vertices)
    subs_dict = Dict{SymEngine.Basic, Any}()

    for (i,j) in edge_bdry_pairs
        d2 = LorentzianSimplexSolver.Volume.distance_sq(vertices[i], vertices[j])  # distance_sq
        edge_sym = SymEngine.Basic(Symbol("lb$(i)_$(j)"))
        subs_dict[edge_sym] = sqrt(d2)
    end

    return subs_dict
end

function dkb_dl(geom, nl_perturb, bdry_edges_perturb, vertices; γ=γsym())
    TetFaces = geom.connectivity[1]["TetFaces"]
    bdry_faces = geom.connectivity[1]["OrderBDryFaces"]
    
    kb_triangles = [TetFaces[v[1][1]][v[1][2]][v[1][3]] for v in bdry_faces]
    
    edge_symbols = Basic[]
    perturbed_set = Set((min(e[1], e[2]), max(e[1], e[2])) for e in bdry_edges_perturb)

    function edge_or_symbol(i, j)
        et = (min(i, j), max(i, j))
        if et in perturbed_set
            lb_sym = SymEngine.Basic(Symbol("lb$(et[1])_$(et[2])"))
            if !(lb_sym in edge_symbols)
                push!(edge_symbols, lb_sym)
            end
            return lb_sym^2
        else
            return val_basic(LorentzianSimplexSolver.Volume.distance_sq(vertices[i], vertices[j]))
        end
    end

    tri_area_sq = [
        sqrt(V2sq(
            edge_or_symbol(tri[1], tri[2]),
            edge_or_symbol(tri[1], tri[3]),
            edge_or_symbol(tri[2], tri[3])
        ))
        for tri in kb_triangles
    ]
    
    ntri = length(tri_area_sq)
    npert = length(edge_symbols)
    dkb_dl = Matrix{Any}(undef, ntri, nl_perturb)
    d2kb_dldl = Array{Any}(undef, ntri, nl_perturb, nl_perturb)
    subs_dict = build_distance_subs(bdry_edges_perturb, vertices)

    for k in 1:ntri
        # Bulk edge directions: zero
        for u in 1:(nl_perturb - npert)
            dkb_dl[k, u] = Basic(0)
            for j in 1:nl_perturb
                d2kb_dldl[k, u, j] = Basic(0)
            end
        end
        # Perturbed edge directions
        for uu in 1:npert
            u = nl_perturb - npert + uu
            sym_u = edge_symbols[uu]
            # first derivative
            d1 = SymEngine.diff(tri_area_sq[k], sym_u)

            if typeof(vertices[1][1]) == Double64
                dkb_dl[k, u] = Double64(parse(BigFloat, string(N(subs(d1, subs_dict))))) * val_basic(2) / γ
            else
                dkb_dl[k, u] = subs(d1, subs_dict) * 2 / γ
            end
            
            # second derivative wrt every j
            for jj in 1:npert
                j = nl_perturb - npert + jj
                sym_j = edge_symbols[jj]
                d2 = SymEngine.diff(SymEngine.diff(tri_area_sq[k], sym_u), sym_j)
                
                if typeof(vertices[1][1]) == Double64
                    dkb_dl[k, u] = Double64(parse(BigFloat, string(N(subs(d2, subs_dict))))) * val_basic(2) / γ
                else
                     d2kb_dldl[k, u, j] = subs(d2, subs_dict) * 2 / γ
                end
               
            end
            # bulk directions remain zero
            for j in 1:(nl_perturb - npert)
                d2kb_dldl[k, u, j] = Basic(0)
            end
        end
    end
    return dkb_dl, d2kb_dldl
end



end # module
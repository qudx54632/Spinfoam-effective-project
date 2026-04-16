module TransverseBasis

using LinearAlgebra

export compute_transverse_basis

"""
    compute_transverse_basis(djdl; tol=1e-12)

Compute a basis of vectors transverse to the Regge submanifold.

Input
-----
djdl :: Vector{Vector{T}}
    djdl[h][ℓ] = ∂ j_h / ∂ ℓ_ℓ

Keyword
-------
tol :: Real
    Numerical cutoff for null vectors

Returns
-------
eList      :: Vector{Vector{T}}   # transverse basis vectors (length nh)
eListHT    :: Matrix{T}            # transpose: nt × nh
Pperp      :: Matrix{T}            # projector
"""
function compute_transverse_basis(djdl::Matrix{T}, tol) where {T<:Real}

    nh, nl = size(djdl)
    nt = nh - nl

    # ------------------------------------------------------------
    # Projector
    # ------------------------------------------------------------
    J = djdl
    G = transpose(J) * J
    Pperp = Matrix{T}(I, nh, nh) - J * (G \ transpose(J))

    # ------------------------------------------------------------
    # Step 1: standard basis
    # ------------------------------------------------------------
    stdBasis = [Matrix{T}(I, nh, nh)[:, i] for i in 1:nh]

    # ------------------------------------------------------------
    # Step 2: eRaw = Pperp * basis vectors
    # ------------------------------------------------------------
    eRaw = [Pperp * v for v in stdBasis]

    # ------------------------------------------------------------
    # Step 3: Orthogonalize (Mathematica style)
    # ------------------------------------------------------------
    eListRaw = Vector{Vector{T}}()

    for v in eRaw
        w = copy(v)

        for u in eListRaw
            w -= (dot(u, w)) * u
        end

        if norm(w) > tol
            push!(eListRaw, w / norm(w))
        end
    end

    # ------------------------------------------------------------
    # Step 4: Select Norm > 0.5
    # ------------------------------------------------------------
    eList = [v for v in eListRaw if norm(v) > 0.5]

    # ------------------------------------------------------------
    # Step 5: Transpose
    # ------------------------------------------------------------
    eListHT = hcat(eList...)   # (nh × nt)

    return eListHT
end

end # module
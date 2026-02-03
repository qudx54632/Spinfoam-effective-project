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

    # ----------------------------
    # Build Jacobian J (nh × nl)
    # ----------------------------
    J = djdl

    # ----------------------------
    # Gram matrix G = Jᵀ J (nl × nl)
    # ----------------------------
    G = transpose(J) * J

    # ----------------------------
    # Projector P⊥ = I − J (G⁻¹ Jᵀ)
    # (use linear solve, not inverse)
    # ----------------------------
    Pperp = Matrix{T}(I, nh, nh) - J * (G \ transpose(J))

    # ----------------------------
    # Raw projected basis vectors
    # ----------------------------
    eRaw = [Pperp[:, i] for i in 1:nh]

    # ----------------------------
    # Select non-zero vectors
    # ----------------------------
    eNonZero = [v for v in eRaw if norm(v) > tol]

    @assert length(eNonZero) ≥ nt "Not enough transverse directions found"

    # ----------------------------
    # Orthonormalize
    # ----------------------------
    E = hcat(eNonZero[1:nt]...)
    eListHT = qr(E).Q[:, 1:nt]

    return eListHT
end

end # module
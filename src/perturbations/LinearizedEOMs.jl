module LinearizedEOMs

using LinearAlgebra

export solve_linearized_eoms

"""
    solve_linearized_eoms(
        HessianOld,
        eListHT,
        djdl,
        nx;
        tol = 1e-12
    )

Solve the linearized equations of motion

    HYY δY + HYℓ δℓ = 0

Returns the linear solution

    δY = - HYY⁻¹ HYℓ δℓ

Parameters
----------
HessianOld :: Matrix{T}
    Hessian in old variables (g,z,j)

eListHT :: Matrix{T}
    Transverse basis, size nt × nh

djdl :: Matrix{T}
    ∂j_h / ∂ℓ_s, size nh × nl

nx :: Int
    Number of (g,z) variables

Returns
-------
dYdl :: Matrix{T}
    Linear response matrix, size (nx+nt) × nl
"""
function solve_linearized_eoms(HessianOld, eListHT, djdl, nx, ScalarT, dl)

    nh, nl = size(djdl)
    nt = nh - nl

    # --------------------------------------------------
    # Build Jacobian J = ∂(g,z,j) / ∂(g,z,t,ℓ)
    # --------------------------------------------------
    J = [
        Matrix{ScalarT}(I, nx, nx)   zeros(ScalarT, nx, nt)   zeros(ScalarT, nx, nl);
        zeros(ScalarT, nh, nx)   eListHT      djdl
    ]

    Hnew = transpose(J) * HessianOld * J

    ny = nx + nt

    # --------------------------------------------------
    # Block decomposition
    # --------------------------------------------------
    HYY = Hnew[1:ny, 1:ny]
    HYl = Hnew[1:ny, ny+1:end]

    # --------------------------------------------------
    # Solve linear system (NO explicit inverse)
    # --------------------------------------------------
    dYdl = - inv(HYY) \ HYl

    return dYdl * dl, HYY

end

end # module
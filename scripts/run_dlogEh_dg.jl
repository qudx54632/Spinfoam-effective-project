module RunDlogEhDG

export run_dlogEh_dg, evaluate_dlogEh_dg

using SymEngine
using LorentzianSimplexSolver

# ------------------------------------------------------------
# 1. log E_h (UNCHANGED physics logic)
# ------------------------------------------------------------
function log_Eh(
    gvariablesall, zvariablesall, bdyxikappafa,
    OrderBulkFaces, metaxikappaf,
    kappa, sgndet, tetareasign;
    γ = LorentzianSimplexSolver.DefineAction.γsym()
)
    safe_log = SymEngine.log

    BDActions = Vector{Basic}(undef, length(OrderBulkFaces))

    for idx in eachindex(OrderBulkFaces)
        faces = OrderBulkFaces[idx]
        nfaces = length(faces)

        xilist  = Vector{Any}(undef, nfaces)
        zlist   = Vector{Any}(undef, nfaces)
        glist   = Vector{Any}(undef, nfaces)
        κlist   = Vector{Int}(undef, nfaces)
        metalist = Matrix{Any}(undef, nfaces, 2)
        sgndetlist = Vector{Int}(undef, nfaces)

        for r in 1:nfaces
            k,i,j = faces[r]
            xilist[r] = bdyxikappafa[k][i][j]
            zlist[r]  = zvariablesall[k][i][j]
            glist[r]  = gvariablesall[k][i]
            κlist[r]  = kappa[k][i][j]

            m = metaxikappaf[k][i][j]
            metalist[r,1] = m[1]
            metalist[r,2] = m[2]
            sgndetlist[r] = sgndet[k][i]
        end

        k1,i1,j1 = faces[1]
        areasign = tetareasign[k1][i1][j1]

        XiZ = areasign == -1 ?
            LorentzianSimplexSolver.DefineAction.bulkfaceactionttXiZ(
                xilist, zlist, glist, κlist, metalist
            ) :
            LorentzianSimplexSolver.DefineAction.bulkfaceactionsXiZ(
                xilist, zlist, glist, κlist, metalist, sgndetlist
            )

        ZZ = areasign == -1 ?
            LorentzianSimplexSolver.DefineAction.bulkfaceactionttZZ(
                xilist, zlist, glist, κlist, metalist; γ = γ
            ) :
            LorentzianSimplexSolver.DefineAction.bulkfaceactionsZZ(
                xilist, zlist, glist, κlist, metalist, sgndetlist; γ = γ
            )

        BDActions[idx] = 2*safe_log(XiZ) + ZZ
    end

    return BDActions
end

function compute_dlogEh_dg(S::Basic, g_vars::Vector{Basic})
    dS = Dict{Basic,Basic}()
    for v in g_vars
        dS[v] = SymEngine.diff(S, v)
    end
    return dS
end

# ------------------------------------------------------------
# 2. Main driver: ∂ log E_h / ∂ g_α
# ------------------------------------------------------------

function run_dlogEh_dg(geom_base)
    # --- unpack geometry ---
    xi_mat = geom_base.varias[:xi_mat]
    z_mat  = geom_base.varias[:z_mat]
    g_mat  = geom_base.varias[:g_mat]

    kappa       = [geom_base.simplex[i].kappa for i in eachindex(geom_base.simplex)]
    sgndet      = [geom_base.simplex[i].sgndet for i in eachindex(geom_base.simplex)]
    tetareasign = [geom_base.simplex[i].tetareasign for i in eachindex(geom_base.simplex)]
    tetn0sign   = [geom_base.simplex[i].tetn0sign for i in eachindex(geom_base.simplex)]

    OrderBulkFaces = geom_base.connectivity[1]["OrderBulkFaces"]
    metaxikappaf = LorentzianSimplexSolver.DefineAction.build_metaxikappaf(sgndet, tetareasign, tetn0sign)

    # --- symbolic log E_h ---
    γsym = LorentzianSimplexSolver.DefineAction.γsym()
    logEh_list = log_Eh(g_mat, z_mat, xi_mat,OrderBulkFaces, metaxikappaf,kappa, sgndet, tetareasign; γ = γsym)
    # --- background solution ---
    g_vars = geom_base.varias[:g_var]

    dlogEh_dg_sym = [compute_dlogEh_dg(S, g_vars) for S in logEh_list]

    return dlogEh_dg_sym
end

function evaluate_dlogEh_dg(dlogEh_dg_sym, geom_base, sd_base::LorentzianSimplexSolver.SolveVars.SolveData{T}; γval=nothing) where {T<:Real}

    γsym = LorentzianSimplexSolver.DefineAction.γsym()

    vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(
        sd_base, γsym; γval=γval
    )

    g_vars = geom_base.varias[:g_var]

    nh = length(dlogEh_dg_sym)
    ng = length(g_vars)

    dlogEh_dg_vals = Matrix{Complex{T}}(undef, nh, ng)

    for h in 1:nh
        for (i, v) in enumerate(g_vars)

            val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(
                dlogEh_dg_sym[h][v], vals
            )

            dlogEh_dg_vals[h,i] =  Complex{T}(
                T(N(real(val_sym))),
                T(N(imag(val_sym)))
            )
        end
    end

    return dlogEh_dg_vals
end

end # module
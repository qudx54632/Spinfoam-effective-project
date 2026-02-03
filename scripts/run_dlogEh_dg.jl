module RunDlogEhDG

export run_dlogEh_dg

using PythonCall
using Symbolics
using LorentzianSimplexSolver

# ------------------------------------------------------------
# 1. log E_h (UNCHANGED physics logic)
# ------------------------------------------------------------
function log_Eh(
    gvariablesall, zvariablesall, bdyxikappafa,
    OrderBulkFaces, metaxikappaf,
    kappa, sgndet, tetareasign;
    γ::Py = LorentzianSimplexSolver.DefineAction.γsym()
)
    sp = LorentzianSimplexSolver.DefineAction._sympy()
    safe_log = sp.log

    BDActions = Vector{Py}(undef, length(OrderBulkFaces))

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

function compute_dlogEh_dg(S::Py, g_vars::Vector{Py})
    sp = LorentzianSimplexSolver.DefineAction._sympy()
    dS = Dict{Py,Py}()

    for v in g_vars
        dS[v] = sp.diff(S, v)
    end

    return dS
end

# ------------------------------------------------------------
# 2. Main driver: ∂ log E_h / ∂ g_α
# ------------------------------------------------------------

function run_dlogEh_dg(geom_base, sd_base)
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

    dlogEh_dg_py = [compute_dlogEh_dg(S, g_vars) for S in logEh_list];

    # --- build gradient functions ---
    dlogEh_dg_func = [
        LorentzianSimplexSolver.SymbolicToJulia.build_gradient_functions(S, sd_base)
        for S in dlogEh_dg_py
    ]

    return dlogEh_dg_func
end

end # module
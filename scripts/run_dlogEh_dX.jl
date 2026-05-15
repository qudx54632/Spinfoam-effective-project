module RunDlogEhDX

export run_dlogEh_dX, evaluate_dlogEh_dX

using SymEngine
using LorentzianSimplexSolver

# ------------------------------------------------------------
# 1. log E_h symbolic
# ------------------------------------------------------------
function log_Eh(gvariablesall, zvariablesall, kappaMat, OrderBulkFaces; γ=γsym())

    logEh_list = Vector{Basic}(undef, length(OrderBulkFaces))

    for idx in eachindex(OrderBulkFaces)
        faces = OrderBulkFaces[idx]
        zlist = [zvariablesall[k][i][j] for (k,i,j) in faces]
        glist = [gvariablesall[k][i] for (k,i,j) in faces]
        κlist = [kappaMat[k][i][j] for (k,i,j) in faces]

        prodEh = LorentzianSimplexSolver.DefineAction.symone(κlist[1])
        for i in 1:2:length(faces)
            zim1 = (i == 1) ? length(zlist) : i - 1
            prodEh *= LorentzianSimplexSolver.DefineAction.Eehss((zlist[zim1], zlist[i+1]),
                            (glist[i], glist[i+1]),
                            (κlist[i], κlist[i+1]); γ=γ)
        end

        logEh = LorentzianSimplexSolver.DefineAction.slog(prodEh)

        logEh_list[idx] = logEh
    end

    return logEh_list
end

function compute_dlogEh_dX(logEh::Basic, X_vars::Vector{Basic})
    dlogEh_dX = Dict{Basic,Basic}()
    for v in X_vars
        dlogEh_dX[v] = SymEngine.diff(logEh, v)
    end
    return dlogEh_dX
end

function run_dlogEh_dX(geom)
    z_mat  = geom.varias[:z_mat]
    g_mat  = geom.varias[:g_mat]

    kappa  = [geom.simplex[i].kappa for i in eachindex(geom.simplex)]

    OrderBulkFaces = geom.connectivity[1]["OrderBulkFaces"]

    # --- symbolic log E_h ---
    γsym = LorentzianSimplexSolver.DefineAction.γsym()
    logEh_list = log_Eh(g_mat, z_mat, kappa, OrderBulkFaces; γ = γsym)
    
    # --- background solution ---
    X_vars = vcat(geom.varias[:g_var], geom.varias[:z_var])

    dlogEh_dX_sym = [compute_dlogEh_dX(logEh, X_vars) for logEh in logEh_list]

    return dlogEh_dX_sym
end

function evaluate_dlogEh_dX(dlogEh_dX_sym, geom, sd::LorentzianSimplexSolver.SolveVars.SolveData{T}; γval=nothing) where {T<:Real}

    γsym = LorentzianSimplexSolver.DefineAction.γsym()

    vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(
        sd, γsym; γval=γval
    )

    X_vars = vcat(geom.varias[:g_var], geom.varias[:z_var])

    nh = length(dlogEh_dX_sym)
    ng = length(X_vars)

    dlogEh_dX_vals = Matrix{Complex{T}}(undef, nh, ng)

    for h in 1:nh
        for (i, v) in enumerate(X_vars)

            val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(
                dlogEh_dX_sym[h][v], vals
            )

            dlogEh_dX_vals[h,i] =  Complex{T}(
                T(N(real(val_sym))),
                T(N(imag(val_sym)))
            )
        end
    end

    return dlogEh_dX_vals
end

# ------------------------------------------------------------
# 2. log E_b
# ------------------------------------------------------------
function log_Eb(gvariablesall, zvariablesall, xivariablesall, kappaMat, OrderBDryFaces; γ=γsym())

    logEb_list = Vector{Basic}(undef, length(OrderBDryFaces))

    for idx in eachindex(OrderBDryFaces)
        faces = OrderBDryFaces[idx]

        zlist  = [zvariablesall[k][i][j] for (k,i,j) in faces]
        glist  = [gvariablesall[k][i] for (k,i,j) in faces]
        xilist = [xivariablesall[k][i][j] for (k,i,j) in faces]
        κlist  = [kappaMat[k][i][j] for (k,i,j) in faces]

        prodEh = LorentzianSimplexSolver.DefineAction.symone(κlist[1])

        for i in 2:2:(length(faces)-1)
            prodEh *= LorentzianSimplexSolver.DefineAction.Eehss((zlist[i-1], zlist[i+1]),
                            (glist[i], glist[i+1]),
                            (κlist[i], κlist[i+1]); γ=γ)
        end

        term = LorentzianSimplexSolver.DefineAction.EebssSource(glist[1], zlist[1], xilist[1], κlist[1]; γ=γ) *
               prodEh *
               LorentzianSimplexSolver.DefineAction.EebssTarget(glist[end], zlist[end-1], xilist[end], κlist[end]; γ=γ)

        logEb_list[idx] = LorentzianSimplexSolver.DefineAction.slog(term)
    end

    return logEb_list
end


function compute_dlogEb_dXY(logEb::Basic, X_vars::Vector{Basic}, Y_vars::Vector{Basic})
    dlogEb_dX  = Dict{Basic,Basic}()
    dlogEb_dY  = Dict{Basic,Basic}()

    for v in X_vars
        dlogEb_dX[v] = SymEngine.diff(logEb, v)
    end

    for v in Y_vars
        dlogEb_dY[v] = SymEngine.diff(logEb, v)
    end

    return dlogEb_dX, dlogEb_dY
end


function run_dlogEb_dXY(geom)
    z_mat  = geom.varias[:z_mat]
    g_mat  = geom.varias[:g_mat]
    xi_mat = geom.varias[:xi_mat]

    kappa  = [geom.simplex[i].kappa for i in eachindex(geom.simplex)]
    OrderBDryFaces = geom.connectivity[1]["OrderBDryFaces"]

    γsym = LorentzianSimplexSolver.DefineAction.γsym()

    logEb_list = log_Eb(g_mat, z_mat, xi_mat, kappa, OrderBDryFaces; γ=γsym)

    # flatten variable containers
    X_vars = vcat(vec(geom.varias[:g_var]), vec(geom.varias[:z_var]))

    eta_bdry = [LorentzianSimplexSolver.DefineSymbols.make_symbol("η_$(faces[1][1])$(faces[1][2])$(faces[1][3])") for faces in OrderBDryFaces]
    
    xi_bdry = Basic[]
    for i in eachindex(OrderBDryFaces)
        faces = OrderBDryFaces[i]
        a,i,j = faces[1]
        za = LorentzianSimplexSolver.DefineSymbols.make_symbol("zeta_$(a)$(i)$(j)$(:a)")
        zb = LorentzianSimplexSolver.DefineSymbols.make_symbol("zeta_$(a)$(i)$(j)$(:b)")

        a,i,j = faces[end]
        zaend = LorentzianSimplexSolver.DefineSymbols.make_symbol("zeta_$(a)$(i)$(j)$(:a)")
        zbend = LorentzianSimplexSolver.DefineSymbols.make_symbol("zeta_$(a)$(i)$(j)$(:b)")
        push!(xi_bdry, za, zb, zaend, zbend)
    end

    Y_vars = vcat(vec(eta_bdry), vec(xi_bdry))

    dlogEb_dXY_sym = [
        compute_dlogEb_dXY(logEb, X_vars, Y_vars)
        for logEb in logEb_list
    ]

    dlogEb_dX_sym  = [t[1] for t in dlogEb_dXY_sym]
    dlogEb_dY_sym  = [t[2] for t in dlogEb_dXY_sym]

    return dlogEb_dX_sym, dlogEb_dY_sym, Y_vars
end

function evaluate_dlogEb_dXY(dlogEb_dX_sym, dlogEb_dY_sym, Y_vars,
                               geom, sd::LorentzianSimplexSolver.SolveVars.SolveData{T}, phase_soln;
                               γval=nothing) where {T<:Real}

    γsym = LorentzianSimplexSolver.DefineAction.γsym()

    vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(
        sd, γsym; γval=γval
    )

    X_vars = vcat(vec(geom.varias[:g_var]), vec(geom.varias[:z_var]))
    
    nh = length(dlogEb_dX_sym)
    nx = length(X_vars)
    ny = length(Y_vars)
    
    dlogEb_dX_vals  = Matrix{Complex{T}}(undef, nh, nx)
    dlogEb_dY_vals  = Matrix{Complex{T}}(undef, nh, ny)
    
    for h in 1:nh
        for (i, v) in enumerate(X_vars)
            val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(dlogEb_dX_sym[h][v], vals)
            val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(val_sym, phase_soln);
            dlogEb_dX_vals[h, i] = Complex{T}(T(N(real(val_sym))), T(N(imag(val_sym))))
        end

        for (i, v) in enumerate(Y_vars)
            val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(dlogEb_dY_sym[h][v], vals)
            val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(val_sym, phase_soln);
            dlogEb_dY_vals[h, i] = Complex{T}(T(N(real(val_sym))), T(N(imag(val_sym))))
        end

    end

    return dlogEb_dX_vals, dlogEb_dY_vals
end

# ------------------------------------------------------------
# 3. log(k_b E_b)
# ------------------------------------------------------------
function logkbEb_func(gvariablesall, zvariablesall, ηlabelsMat, zetabdryall, kappaMat, OrderBDryFaces; γ=γsym())

    kbEb_list = Vector{Basic}(undef, length(OrderBDryFaces))

    for idx in eachindex(OrderBDryFaces)
        faces = OrderBDryFaces[idx]

        xilist  = [zetabdryall[k][i][j] for (k,i,j) in faces]
        zlist   = [zvariablesall[k][i][j] for (k,i,j) in faces]
        glist   = [gvariablesall[k][i] for (k,i,j) in faces]
        κlist   = [kappaMat[k][i][j] for (k,i,j) in faces]

        ηvals = ηlabelsMat[faces[1][1]][faces[1][2]][faces[1][3]]

        prodEh = LorentzianSimplexSolver.DefineAction.symone(κlist[1])

        for i in 2:2:(length(faces)-1)
            prodEh *= LorentzianSimplexSolver.DefineAction.Eehss((zlist[i-1], zlist[i+1]),
                            (glist[i], glist[i+1]),
                            (κlist[i], κlist[i+1]); γ=γ)
        end

        term = LorentzianSimplexSolver.DefineAction.EebssSource(glist[1], zlist[1], xilist[1], κlist[1]; γ=γ) *
               prodEh *
               LorentzianSimplexSolver.DefineAction.EebssTarget(glist[end], zlist[end-1], xilist[end], κlist[end]; γ=γ)

        kbEb_list[idx] = ηvals * LorentzianSimplexSolver.DefineAction.slog(term)
    end

    return kbEb_list
end

function compute_dkblogEb_dXY(logkbEb::Basic, X_vars::Vector{Basic}, Y_vars::Vector{Basic})
    dkbEb_dX = Dict{Basic,Basic}()
    dkbEb_dY = Dict{Basic,Basic}()

    for v in X_vars
        dkbEb_dX[v] = SymEngine.diff(logkbEb, v)
    end

    for v in Y_vars
        dkbEb_dY[v] = SymEngine.diff(logkbEb, v)
    end

    return dkbEb_dX, dkbEb_dY
end

function compute_d2kblogEb_dXdY(logkbEb::Basic, X_vars::Vector{Basic}, Y_vars::Vector{Basic})
    d2kbEb_dXdY = Dict{Tuple{Basic,Basic},Basic}()

    for x in X_vars
        for y in Y_vars
            d2kbEb_dXdY[(x,y)] = SymEngine.diff(SymEngine.diff(logkbEb, x), y)
        end
    end

    return d2kbEb_dXdY
end

function compute_d2kblogEb_dYdY(logkbEb::Basic, Y_vars::Vector{Basic})
    d2kbEb_dYdY = Dict{Tuple{Basic,Basic},Basic}()

    for y1 in Y_vars
        for y2 in Y_vars
            d2kbEb_dYdY[(y1,y2)] = SymEngine.diff(SymEngine.diff(logkbEb, y1), y2)
        end
    end

    return d2kbEb_dYdY
end

function run_kblogEb_dXY(geom, Y_vars)
    z_mat      = geom.varias[:z_mat]
    g_mat      = geom.varias[:g_mat]
    xi_mat    = geom.varias[:xi_mat]
    ηlabelsMat = geom.varias[:η_mat]
    kappa      = [geom.simplex[i].kappa for i in eachindex(geom.simplex)]
    OrderBDryFaces = geom.connectivity[1]["OrderBDryFaces"]

    γsym_ = LorentzianSimplexSolver.DefineAction.γsym()

    kbEb_list = logkbEb_func(g_mat, z_mat, ηlabelsMat, xi_mat, kappa, OrderBDryFaces; γ=γsym_)

    # flatten variables
    X_vars = vcat(geom.varias[:g_var], geom.varias[:z_var])

    dkbEb_dXY_sym = [
        compute_dkblogEb_dXY(kbEb, X_vars, Y_vars)
        for kbEb in kbEb_list
    ]

    d2kbEb_dXdY_sym = [
        compute_d2kblogEb_dXdY(kbEb, X_vars, Y_vars)
        for kbEb in kbEb_list
    ]

    d2kbEb_dYdY_sym = [
        compute_d2kblogEb_dYdY(kbEb, Y_vars)
        for kbEb in kbEb_list
    ]

    dkbEb_dX_sym = [t[1] for t in dkbEb_dXY_sym]
    dkbEb_dY_sym = [t[2] for t in dkbEb_dXY_sym]

    return dkbEb_dX_sym, dkbEb_dY_sym, d2kbEb_dXdY_sym, d2kbEb_dYdY_sym
end

function evaluate_kblogEb_all(
    dkbEb_dX_sym,
    dkbEb_dY_sym,
    d2kbEb_dXdY_sym,
    d2kbEb_dYdY_sym,
    geom,
    sd::LorentzianSimplexSolver.SolveVars.SolveData{T}, Y_vars, phase_soln;
    γval=nothing
) where {T<:Real}

    γsym_ = LorentzianSimplexSolver.DefineAction.γsym()

    vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(
        sd, γsym_; γval=γval
    )

    # --------------------------------------------------
    # Variables
    # --------------------------------------------------
    X_vars = vcat(vec(geom.varias[:g_var]), vec(geom.varias[:z_var]))

    nfaces = length(dkbEb_dX_sym)
    nx = length(X_vars)
    ny = length(Y_vars)

    # --------------------------------------------------
    # Allocate outputs
    # --------------------------------------------------
    dkbEb_dX_vals   = Matrix{Complex{T}}(undef, nfaces, nx)
    dkbEb_dY_vals   = Matrix{Complex{T}}(undef, nfaces, ny)
    d2kbEb_dXdY_vals = Array{Complex{T}}(undef, nfaces, nx, ny)
    d2kbEb_dYdY_vals = Array{Complex{T}}(undef, nfaces, ny, ny)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    for h in 1:nfaces
        # first derivatives wrt X
        for (i, v) in enumerate(X_vars)
            val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(
                dkbEb_dX_sym[h][v], vals
            )
            val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(
                val_sym, phase_soln
            )
            dkbEb_dX_vals[h, i] = Complex{T}(T(N(real(val_sym))), T(N(imag(val_sym))))
        end

        # first derivatives wrt Y
        for (i, v) in enumerate(Y_vars)
            val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(
                dkbEb_dY_sym[h][v], vals
            )
            val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(
                val_sym, phase_soln
            )
            dkbEb_dY_vals[h, i] = Complex{T}(T(N(real(val_sym))), T(N(imag(val_sym))))
        end

        # second derivatives wrt X and Y
        for (ix, x) in enumerate(X_vars)
            for (iy, y) in enumerate(Y_vars)
                expr = d2kbEb_dXdY_sym[h][(x, y)]
                val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(expr, vals)
                val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(
                    val_sym, phase_soln
                )
                d2kbEb_dXdY_vals[h, ix, iy] = Complex{T}(T(N(real(val_sym))), T(N(imag(val_sym))))
            end
        end

        # second derivatives wrt Y and Y
        for (iy1, y1) in enumerate(Y_vars)
            for (iy2, y2) in enumerate(Y_vars)
                expr = d2kbEb_dYdY_sym[h][(y1, y2)]
                val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(expr, vals)
                val_sym = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(
                    val_sym, phase_soln
                )
                d2kbEb_dYdY_vals[h, iy1, iy2] = Complex{T}(T(N(real(val_sym))), T(N(imag(val_sym))))
            end
        end
    end

    return dkbEb_dX_vals, dkbEb_dY_vals, d2kbEb_dXdY_vals, d2kbEb_dYdY_vals
end

end # module
module EffectiveSpinfoamWorkflow

using LinearAlgebra
using SymEngine

using LorentzianSimplexSolver

include(joinpath(@__DIR__, "..", "scripts", "run_dlogEh_dX.jl"))
include(joinpath(@__DIR__, "..", "scripts", "run_deta_dl.jl"))
include(joinpath(@__DIR__, "perturbations", "TransverseBasis.jl"))
include(joinpath(@__DIR__, "perturbations", "Soln_dY_dX.jl"))
include(joinpath(@__DIR__, "perturbations", "DθDl.jl"))

using .RunDlogEhDX
using .DηDLUtils
using .TransverseBasis
using .Soln_dY_dX
using .DθDl_module

export build_vertex_coordinates,
       build_effective_data,
       compute_effective_terms,
       print_effective_terms,
       scan_m_kernel

function build_vertex_coordinates(simplices, coords_lines, ::Type{T}) where {T<:Real}
    vertices = sort(unique(Iterators.flatten(simplices)))
    length(vertices) == length(coords_lines) ||
        error("Expected $(length(vertices)) coordinate lines, got $(length(coords_lines)).")

    return Dict{Int, Vector{T}}(
        v => LorentzianSimplexSolver.PrecisionUtils.parse_numeric_line(line, T)
        for (v, line) in zip(vertices, coords_lines)
    )
end

function build_effective_data(
    simplices,
    vertex_coords;
    boundary_edges_perturbed=nothing,
    verbose::Bool=false,
)
    geom = LorentzianSimplexSolver.construct_geometry(simplices, vertex_coords; verbose=verbose)
    LorentzianSimplexSolver.prepare_global_geometry!(geom, simplices; verbose=verbose)

    regge = LorentzianSimplexSolver.compute_regge_action(geom, simplices, vertex_coords)
    spinfoam = LorentzianSimplexSolver.compute_spinfoam_action(geom, regge)

    g_vars = geom.varias[:g_var]
    z_vars = geom.varias[:z_var]
    eta_vars = geom.varias[:η_var]
    X_vars = vcat(g_vars, z_vars)
    all_vars = vcat(X_vars, eta_vars)

    hessian_symbols = LorentzianSimplexSolver.EOMsHessian.compute_Hessian_block_half(
        spinfoam.action_no_phase,
        all_vars,
    )
    dlogEh_dX_symbols = RunDlogEhDX.run_dlogEh_dX(geom)

    eta_h_vertices = DηDLUtils.get_bulk_faces_vertices(geom)
    bulk_edges, boundary_edges = DηDLUtils.get_bulk_edges(geom, eta_h_vertices)
    selected_boundary_edges =
        boundary_edges_perturbed === nothing ? [boundary_edges[1]] : boundary_edges_perturbed
    perturb_edges = vcat(bulk_edges, selected_boundary_edges)

    eta_symbols = [
        LorentzianSimplexSolver.DefineSymbols.make_symbol("η_$(face[1][1])$(face[1][2])$(face[1][3])")
        for face in geom.connectivity[1]["OrderBulkFaces"]
    ]

    boundary_face_vertices = [
        geom.connectivity[1]["TetFaces"][face[1][1]][face[1][2]][face[1][3]]
        for face in geom.connectivity[1]["OrderBDryFaces"]
    ]

    T = eltype(first(values(vertex_coords)))
    DεDl = DθDl_module.compute_dθDl(
        simplices,
        eta_h_vertices,
        perturb_edges,
        vertex_coords,
        geom.connectivity[1]["Tets"],
        T,
    )
    DΘDl = DθDl_module.compute_dθDl(
        simplices,
        boundary_face_vertices,
        perturb_edges,
        vertex_coords,
        geom.connectivity[1]["Tets"],
        T,
    )

    dkb_dl, d2kb_dldl = Soln_dY_dX.dkb_dl(
        geom,
        length(perturb_edges),
        selected_boundary_edges,
        vertex_coords;
        γ=spinfoam.gamma_symbol,
    )

    return (
        geom=geom,
        vertex_coords=vertex_coords,
        regge=regge,
        spinfoam=spinfoam,
        gamma_symbol=spinfoam.gamma_symbol,
        solve_data=spinfoam.solve_data,
        action_no_phase=spinfoam.action_no_phase,
        hessian_symbols=hessian_symbols,
        dlogEh_dX_symbols=dlogEh_dX_symbols,
        eta_symbols=eta_symbols,
        eta_h_vertices=eta_h_vertices,
        bulk_edges=bulk_edges,
        boundary_edges=boundary_edges,
        boundary_edges_perturbed=selected_boundary_edges,
        perturb_edges=perturb_edges,
        DεDl=DεDl,
        DΘDl=DΘDl,
        dkb_dl=dkb_dl,
        d2kb_dldl=d2kb_dldl,
        X_variables=X_vars,
        all_variables=all_vars,
    )
end

function compute_effective_terms(
    data,
    gamma_value;
    tolerance=LorentzianSimplexSolver.PrecisionUtils.get_tolerance(),
)
    sd = data.solve_data
    γ = data.gamma_symbol
    T = real_type(sd)

    vals = LorentzianSimplexSolver.ActionEvaluation.build_value_dict(sd, γ; γval=gamma_value)
    action_value = LorentzianSimplexSolver.ActionEvaluation.eval_symbolic(data.action_no_phase, vals)
    spinfoam_action = SymEngine.expand(action_value)

    hessian = LorentzianSimplexSolver.EOMsHessian.evaluate_hessian_block(
        data.hessian_symbols,
        sd;
        γ=gamma_value,
    )

    nX = length(data.X_variables)
    Hxx = hessian[1:nX, 1:nX]
    invHxx = inv(Hxx)

    dηdl = DηDLUtils.build_dηdl_matrix(
        data.eta_h_vertices,
        data.perturb_edges,
        data.vertex_coords,
        T,
        gamma_value,
    )
    e_hat = TransverseBasis.compute_transverse_basis(dηdl, tolerance)

    dlogEh_dX = RunDlogEhDX.evaluate_dlogEh_dX(
        data.dlogEh_dX_symbols,
        data.geom,
        sd;
        γval=gamma_value,
    )

    eta_values = [convert(T, subs(eta, vals)) for eta in data.eta_symbols]
    Ahh = Matrix(Diagonal(eta_values))

    hαβ = Hxx + transpose(dlogEh_dX) * Ahh * dlogEh_dX
    w = -dlogEh_dX * invHxx * transpose(dlogEh_dX)
    rho = inv(inv(Ahh) - w)

    S = -transpose(e_hat) * dlogEh_dX * inv(hαβ) * transpose(dlogEh_dX) * e_hat
    kappa = e_hat * inv(S) * transpose(e_hat)
    M_kernel = rho - rho * inv(Ahh) * kappa * inv(Ahh) * rho

    dkb_dl, d2kb_dldl = evaluate_boundary_derivatives(data, gamma_value, T)

    linear_regge =
        im / 2 * gamma_value * transpose(dkb_dl) * data.regge.dihedral_angles

    quadratic_regge = im * (
        gamma_value / 4 * transpose(dηdl) * data.DεDl +
        gamma_value / 4 * (
            transpose(dkb_dl) * data.DΘDl +
            sum(
                data.regge.dihedral_angles[i] * d2kb_dldl[i, :, :]
                for i in eachindex(data.regge.dihedral_angles)
            )
        )
    )

    non_regge_correction =
        -gamma_value^2 / 8 * transpose(data.DεDl) * M_kernel * data.DεDl

    return (
        linear_regge=linear_regge,
        quadratic_regge=quadratic_regge,
        non_regge_correction=non_regge_correction,
        spinfoam_quadratic=quadratic_regge + non_regge_correction,
        spinfoam_action=spinfoam_action,
        M_kernel=M_kernel,
        rho=rho,
        kappa=kappa,
        hessian=hessian,
        Hxx=Hxx,
        dηdl=dηdl,
        transverse_basis=e_hat,
        dlogEh_dX=dlogEh_dX,
    )
end

function print_effective_terms(result; io::IO=stdout, show_matrices::Bool=false)
    print_term(io, "iS_Regge^(1): linear Regge term", result.linear_regge; show_matrix=show_matrices)
    print_term(io, "iS_Regge^(2): quadratic Regge term", result.quadratic_regge; show_matrix=show_matrices)
    print_term(io, "DeltaS_nonRegge^(2): non-Regge correction term", result.non_regge_correction; show_matrix=show_matrices)
    return nothing
end

function scan_m_kernel(data, gamma_values; tolerance=LorentzianSimplexSolver.PrecisionUtils.get_tolerance())
    return [
        compute_effective_terms(data, gamma; tolerance=tolerance).M_kernel
        for gamma in gamma_values
    ]
end

function print_term(io::IO, title::AbstractString, value; show_matrix::Bool=false)
    println(io)
    println(io, title)
    println(io, "size = ", size(value))
    println(io, "sum  = ", sum(value))
    println(io, "norm = ", norm(value))
    if show_matrix
        show(io, "text/plain", value)
        println(io)
    end
    return nothing
end

function evaluate_boundary_derivatives(data, gamma_value, ::Type{T}) where {T<:Real}
    vals = Dict{Basic,Basic}(data.gamma_symbol => Basic(gamma_value))
    dkb = map(x -> to_complex_T(subs(x, vals), T), data.dkb_dl)
    d2kb = map(x -> to_complex_T(subs(x, vals), T), data.d2kb_dldl)
    return dkb, d2kb
end

real_type(::LorentzianSimplexSolver.SolveVars.SolveData{T}) where {T<:Real} = T

function to_complex_T(x, ::Type{T}) where {T<:Real}
    return Complex{T}(T(N(real(x))), T(N(imag(x))))
end

end # module

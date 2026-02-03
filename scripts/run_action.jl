module RunAction

export run_action

using LorentzianSimplexSolver

function run_action(geom_base, γ)
    # ------------------------------------------------------------
    # 6. Symbols and action (reference orientation)
    # ------------------------------------------------------------
    LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom_base)

    sd_base, _ =
        LorentzianSimplexSolver.SolveVars.run_solver(geom_base)

    S_base =
        LorentzianSimplexSolver.DefineAction.compute_action(geom_base)

    S_base_fn, labels_base =
        LorentzianSimplexSolver.SymbolicToJulia.build_action_function(S_base, sd_base)

    args_base =
        LorentzianSimplexSolver.SymbolicToJulia.build_argument_vector(sd_base, labels_base, γ)

    return sd_base, S_base_fn, args_base, labels_base
end

end # module RunAction
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

    return sd_base, S_base
end

end # module RunAction
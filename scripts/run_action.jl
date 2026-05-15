module RunAction

export run_action

using LorentzianSimplexSolver

function run_action(geom, dihedral_angles, γ)
    # ------------------------------------------------------------
    # 6. Symbols and action (reference orientation)
    # ------------------------------------------------------------
    LorentzianSimplexSolver.DefineSymbols.run_define_variables(geom)

    sd, _ = LorentzianSimplexSolver.SolveVars.run_solver(geom);
    S_no_phase, phase_soln = LorentzianSimplexSolver.SFaction_no_phase.compute_action_no_bdry_phase(geom, sd, dihedral_angles; γ=γ);

    return sd, S_no_phase, phase_soln
end

end # module RunAction
# Effective Spinfoam

This repository contains Julia scripts for the numerical evaluation of the higher-order correction to the effective spinfoam action. It builds on `LorentzianSimplexSolver` for Lorentzian geometry reconstruction, Regge data, critical spinfoam action data, equations of motion, and Hessian blocks.

The paper-level workflow is:

```julia
using Pkg
Pkg.activate("../LorentzianSimplexSolver")

using LorentzianSimplexSolver
include("src/EffectiveSpinfoamWorkflow.jl")
using .EffectiveSpinfoamWorkflow

configure_precision!(Float64)

vertex_coords = build_vertex_coordinates(simplices, coords_lines, Float64)
data = build_effective_data(simplices, vertex_coords)
result = compute_effective_terms(data, 0.1)

M = result.M_kernel
correction = result.non_regge_correction
quadratic_action = result.spinfoam_quadratic
```

`build_effective_data` performs the gamma-independent construction: geometry, face matching, Regge action, critical Spinfoam action, Hessian symbols, deficit-angle derivatives, and perturbation data. `compute_effective_terms` evaluates the Hessian, the correction matrix `M_kernel`, and the three terms used in the paper: `linear_regge`, `quadratic_regge`, and `non_regge_correction`.

## Dependency Layout

This project uses the sibling package:

```text
../LorentzianSimplexSolver
```

The scripts activate this package environment directly. The older embedded copy under `src/LorentzianSimplexSolver/` is kept only for reference and should not be used by the current drivers.

## Main Files

- `main_interactive_driver.jl`: interactive single-gamma calculation.
- `run_gamma_scan_Mkernel.jl`: batch scan of the kernel over a list of gamma values.
- `src/EffectiveSpinfoamWorkflow.jl`: compact workflow for computing `M_kernel`, the Regge terms, and the non-Regge correction.
- `src/perturbations/`: perturbation utilities used by the workflow.
- `scripts/`: lower-level symbolic derivative helpers.
- `results/`: saved kernel scans and figures.

## Running

From this directory:

```bash
julia main_interactive_driver.jl
```

or for a gamma scan:

```bash
julia run_gamma_scan_Mkernel.jl
```

The scan script periodically saves `gamma_list` and `M_kernel_matrix_list` to the selected `.jld2` file.

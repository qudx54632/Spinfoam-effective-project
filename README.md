# Spinfoam Effective Project

This repository contains numerical tools for studying **effective and perturbative dynamics in Lorentzian spinfoam models**, including the quadratic Regge action, linearized equations of motion, and transverse spin fluctuations.

The project is organized as a lightweight Julia codebase, with a **Jupyter notebook (main.ipynb) serving as the main driver** for running calculations and exploring results.

---

## Repository structure

- src/  
  Core library code  
  - perturbations/  
    Effective and perturbative dynamics  
    - QuadraticRegge.jl  
    - LinearizedEOMs.jl  
    - TransverseBasis.jl  
  - hessian/  
    Hessian construction and related code  
  - LorentzianSimplexSolver/  
    Lorentzian simplex and spinfoam solvers  

- scripts/  
  Standalone Julia scripts for batch runs  
  - run_geometry.jl  
  - run_action.jl  
  - run_djdl.jl  
  - run_dlogEh_dg.jl  

- data/  
  Precomputed numerical data  

- main.ipynb  
  Main entry point (Jupyter notebook)  

- Manifest.toml  
  Julia environment lock file  

- .CondaPkg/  
  Local Conda environment (not required)  

---

## Requirements

- Julia 1.9 or newer (recommended)
- Jupyter Notebook
- IJulia

All Julia dependencies are specified in Manifest.toml.

---

## Installation

Clone the repository:

git clone https://github.com/qudx54632/Spinfoam-effective-project.git  
cd Spinfoam-effective-project

(Optional) install the Jupyter kernel:

using IJulia  
notebook()

---

## Dependencies

This project relies on the **LorentzianSimplexSolver** package for Lorentzian
simplex geometry and spinfoam-related computations. A local copy of the package
is included under `src/LorentzianSimplexSolver/` for convenience and
reproducibility.

---

## Running the calculations (recommended)

The main workflow is via the Jupyter notebook.

Start Jupyter and open the notebook:

jupyter notebook main.ipynb

Inside `main.ipynb`, the project environment is activated and all core
modules are loaded from the `src/` directory.

The notebook performs the following tasks:

- sets up the geometry and boundary data  
- constructs the effective (quadratic) action  
- computes linearized equations of motion and transverse modes  
- performs numerical evaluation and visualization  

---

## Scripts

The `scripts/` directory contains standalone Julia scripts for specific
tasks or batch-style runs, such as:

- geometry setup  
- action evaluation  
- Jacobian and Hessian-related computations  

These scripts are not required to run `main.ipynb`, but may be useful
for automation, testing, or debugging.

Example usage:

julia --project scripts/run_action.jl

---

## Data

The `data/` directory contains **precomputed Hessian matrices** for
specific Pachner moves and triangulations, including:

- the **1–5 Pachner move**  
- the **Double–Delta–3 configuration**  

Each data file corresponds to a specific choice of vertices, simplices,
and triangulation orientation.

**Important:**  
When using or extending these datasets, the data file name should be
updated to explicitly reflect the corresponding vertices and simplices,
to avoid ambiguity and to ensure consistency with the geometry used in
the calculation.

---

## Hessian computation (optional)

The `src/hessian/` directory contains the file `run_hessian.jl`, which is
used to compute Hessian matrices for different triangulation configurations.

To generate Hessian data for a new configuration, modify the following
items in `run_hessian.jl`:

- the definition of the **simplices**
- the definition of **coords_lines**
- the output file name in  
  `@save joinpath(pwd(), "data/<new_name>.jld2") H_base_eval`

The output file name should be chosen to clearly reflect the corresponding
vertices and simplices.

After making these changes, run the Hessian computation in Julia as follows:

1. Start Julia in the project root directory.
2. Add the Lorentzian simplex solver to the load path:

push!(LOAD_PATH, joinpath(pwd(), "src/LorentzianSimplexSolver", "src"))

3.	Load the solver and execute the Hessian script:

using LorentzianSimplexSolver

include("src/hessian/run_hessian.jl")

Once the script finishes, the computed Hessian matrix will be automatically
saved in the `data/` directory.

---

## Status

This repository is under active development.

The current structure is intended to support future extensions to:

- higher-order effective actions  
- refined triangulations  
- cosmological and black-hole spinfoam models  

---

## License

To be specified.
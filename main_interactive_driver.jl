# Interactive driver for computing the effective spinfoam correction.

using Pkg

const SOLVER_PROJECT = normpath(joinpath(@__DIR__, "src", "LorentzianSimplexSolver"))
isdir(SOLVER_PROJECT) || error("LorentzianSimplexSolver project not found at $SOLVER_PROJECT")
Pkg.activate(SOLVER_PROJECT; io=devnull)

using LorentzianSimplexSolver

include(joinpath(@__DIR__, "src", "EffectiveSpinfoamWorkflow.jl"))
using .EffectiveSpinfoamWorkflow

function choose_scalartype()
    println("Choose scalar type:")
    println("  1) Float64")
    println("  2) BigFloat")
    print("> ")

    choice = strip(readline())
    choice == "1" && return Float64, 1e-8
    choice == "2" && return BigFloat, BigFloat(1e-30)

    error("Invalid choice. Please enter 1 or 2.")
end

function parse_simplices(line::AbstractString)
    text = replace(strip(line), r"\s+" => "")
    startswith(text, "[[") && endswith(text, "]]") ||
        error("Input must look like [[1,2,3,4,5],[1,2,3,4,6]].")

    body = text[3:end-2]
    isempty(body) && error("No simplices entered.")

    return [parse_simplex(chunk) for chunk in split(body, "],[")]
end

function parse_simplex(text::AbstractString)
    vertices = parse.(Int, split(text, ","; keepempty=false))
    length(vertices) == 5 || error("Each 4-simplex must contain 5 vertices, got $(length(vertices)).")
    length(unique(vertices)) == 5 || error("Simplex vertices must be distinct: $vertices")
    return vertices
end

function read_simplices()
    println()
    println("Enter simplices:")
    println("[[1,2,3,4,6],[1,2,3,5,6],[1,2,4,5,6]]")
    print("> ")

    line = strip(readline())
    isempty(line) && error("No simplices entered.")
    return parse_simplices(line)
end

function read_coordinate_lines(vertices)
    println()
    println("Enter coordinates in vertex order $(collect(vertices)).")
    println("Format: t,x,y,z")

    lines = String[]
    for v in vertices
        print("[$v] ")
        line = strip(readline())
        isempty(line) && error("Empty coordinate line for vertex $v.")
        push!(lines, line)
    end
    return lines
end

function compute_from_input(
    simplices,
    coord_lines,
    gamma_value,
    ::Type{ScalarT};
    tolerance=ScalarT === BigFloat ? BigFloat(1e-30) : 1e-8,
    verbose::Bool=false,
) where {ScalarT<:Real}
    LorentzianSimplexSolver.configure_precision!(ScalarT; precision=100, tolerance=tolerance)
    vertex_coords = EffectiveSpinfoamWorkflow.build_vertex_coordinates(
        simplices,
        coord_lines,
        ScalarT,
    )
    data = EffectiveSpinfoamWorkflow.build_effective_data(
        simplices,
        vertex_coords;
        verbose=verbose,
    )
    result = EffectiveSpinfoamWorkflow.compute_effective_terms(
        data,
        ScalarT(gamma_value);
        tolerance=tolerance,
    )

    return (
        data=data,
        terms=result,
        vertex_coords=vertex_coords,
        gamma_value=ScalarT(gamma_value),
        tolerance=tolerance,
    )
end

println("========================================")
println(" Effective Spinfoam Interactive Driver")
println("========================================")

ScalarT, tol = choose_scalartype()
LorentzianSimplexSolver.configure_precision!(ScalarT; precision=100, tolerance=tol)

simplices = read_simplices()
vertices = sort(unique(Iterators.flatten(simplices)))
println()
println("Detected $(length(simplices)) simplices with $(length(vertices)) vertices.")

coord_lines = read_coordinate_lines(vertices)

println()
print("Enter gamma: ")
gamma_value = parse(ScalarT, strip(readline()))

println()
println("[1/2] Building geometry and Hessian data.")
result = compute_from_input(
    simplices,
    coord_lines,
    gamma_value,
    ScalarT;
    tolerance=tol,
    verbose=false,
)
terms = result.terms

println("[2/2] Evaluated terms at gamma = $gamma_value.")
# Set show_matrices=true here if the full matrices are needed for debugging.
EffectiveSpinfoamWorkflow.print_effective_terms(terms)

println()
println("Done.")

nothing

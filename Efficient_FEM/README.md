# Finite Elements

Highly efficient parallel implementation of Finite Element Analysis of the Poisson equation. 

Run the .ipynb notebook ('Zac_Keskin Phase 2 - Final') using `jupyter notebook`

The implementation uses a SciPy LinearOperator wrapper around OpenCL parallel computation for efficient Matrix-Vector multiplication, used to solve the Finite Element Equation.

Parameters for diverse load vector forces are provided, with the plot provided displaying the heat distribution under a sinusoidal forcing term over the 2D 'L-shape' surface.

We use the code developed during Phase 1 of the project, to read in mesh data from .vtk files and calculate Jacobians for the isoparametric transformation. This implementation makes use of code written by Timo Betcke, in the interest of guaranteeing an agreed start point for Phase 2 of the project.
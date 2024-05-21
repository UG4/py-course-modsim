import ug4py.ugcore as ug4
import ug4py.pylimex as limex
import ug4py.pyconvectiondiffusion as cd
import ug4py.pysuperlu as slu


# Setup:
# defining needed subsets, gird and number of refinements
requiredSubsets = ["LIP", "COR", "BOTTOM_SC", "TOP_SC"] # defining subsets
gridName = "skin2d-aniso.ugx"  # Grid created with ProMesh
numRefs = 2  # Number of Refinement steps on given grid

# Choosing a domain object
# (either 1d, 2d or 3d suffix)
dom = ug4.Domain2d()

# Loading the given grid into the domain
print(f"Loading Domain {gridName} ...")
ug4.LoadDomain(dom, gridName)
print("Domain loaded.")

# Optional: Refining the grid
if numRefs > 0:
    print("Refining ...")
    refiner = ug4.GlobalDomainRefiner(dom)
    for i in range(numRefs):
        ug4.TerminateAbortedRun()
        refiner.refine()
        print(f"Refining step {i} ...")

    print("Refining done")

# checking if geometry has the needed subsets of the probelm
sh = dom.subset_handler()
for e in requiredSubsets:
    if sh.get_subset_index(e) == -1:
        print(f"Domain does not contain subset {e}.")
        sys.exit(1)

# Create approximation space which describes the unknowns in the equation
fct = "u"  # name of the function
type = "Lagrange"
order = 1  # polynom order for lagrange

approxSpace = ug4.ApproximationSpace2d(dom)
approxSpace.add_fct(fct, type, order)
approxSpace.init_levels()
approxSpace.init_surfaces()  
approxSpace.init_top_surface()
approxSpace.print_statistic()

# Create discretization for the subsets LIP and COR
# perform the discretization on the actual elements
# using first order finite volumes (FV1) on the 2d grid
KDcor=1.0
KCor=1.0

elemDisc={}
# creating instance of a convection diffusion equation
elemDisc["COR"] = cd.ConvectionDiffusionFV12d("u", "COR")
elemDisc["COR"].set_diffusion(KDcor)
elemDisc["COR"].set_mass_scale(KCor)

elemDisc["LIP"] = cd.ConvectionDiffusionFV12d("u", "LIP")
elemDisc["LIP"].set_diffusion(KDcor)
elemDisc["LIP"].set_mass_scale(KCor)

# ug4 separates the boundary value and the discretization
# boundary conditions can be enforced through a post-process (dirichlet).
# To init at boundary, the value, function name from the Approximationspace
# and the subset name are needed
dirichletBND = ug4.DirichletBoundary2dCPU1()
dirichletBND.add(1.0, "u", "TOP_SC")
dirichletBND.add(0.0, "u", "BOTTOM_SC")


# create the discretization object which combines all the
# separate discretizations into one domain discretization.
domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)
domainDisc.add(elemDisc["COR"])
domainDisc.add(elemDisc["LIP"])
domainDisc.add(dirichletBND)


# Create Solver
# In this case we use an LU (Lower Upper) Solver for an exact solution
lsolver=ug4.LUCPU1()
lsolver=slu.SuperLUCPU1()

# Solve the transient problem
# Use the Approximationspace to
# create a vector of unknowns and a vector
# which contains the right hand side
usol = ug4.GridFunction2dCPU1(approxSpace)

# Init the vector representing the unknowns with 0 to obtain
# reproducable results
ug4.Interpolate(0.0, usol, "u")

# Define start time, end time and step size
startTime = 0.0
endTime = 1000.0
dt = 25.0

# create a time discretization with the theta-scheme
# takes in domain and a theta
# theta = 1.0 -> Implicit Euler, 
# 0.0 -> Explicit Euler 
timeDisc=ug4.ThetaTimeStepCPU1(domainDisc, 1.0) 

# creating Time Iterator, setting the solver and step size
timeInt = limex.LinearTimeIntegrator2dCPU1(timeDisc)
#timeInt = limex.ConstStepLinearTimeIntegrator2dCPU1(timeDisc)
timeInt.set_linear_solver(lsolver)
timeInt.set_time_step(dt)


# Exporting the result to a vtu-file
# can be visualized in paraview or with a python extension
# We add a corresponding observer to the time integrator.
def MyCallback(usol, step, time, dt) :
    ug4.WriteGridFunctionToVTK(usol, "SkinDiffusion_"+str(int(step)).zfill(5)+".vtu")
pyobserver =ug4.PythonCallbackObserver2dCPU1(MyCallback) 
timeInt.attach_observer(pyobserver)

# Solving the transient problem
try:
    timeInt.apply(usol, endTime, usol, startTime)
except Exception as inst:
    print(inst)



# Plot the result using pyvista
import pyvista

result = pyvista.read('Solution_SkinDif_Pybind.vtu')
print()
print("Pyvista input: ")
print(result)
result.plot(scalars="u", show_edges=True, cmap='coolwarm')
#scalars="node_value", categories=True

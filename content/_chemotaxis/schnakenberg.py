# Load utility scripts (e.g. from from ugcore/scripts)
#!/usr/bin/env python
# coding: utf-8

# [<img src="../../header.svg">](../index.ipynb)
# 
# ---
# Typical Schnakenberg model.
# Governing equations:
#   \frac{\partial u}{\partial t} = D_u \nabla^2 u + \gamma (a - u + u^2 v)
#   \frac{\partial v}{\partial t} = D_v \nabla^2 v + \gamma (b - u^2 v)
# plus boundary and initial condiitions.

# Reference: Schnakenberg, J. (1979). Simple chemical reaction systems with
# limit cycle behaviour. Journal of Theoretical Biology, 81(3), 389–400.
# The Schnakenberg model is a prototypical reaction-diffusion system that exhibits Turing patterns. It consists of two chemical species, u and v, that interact through nonlinear reactions and diffuse in space.
# Here, u is the activator, v is the inhibitor, and a,b,\gamma are feed/decay
# parameters that determine the reaction kinetics and pattern-forming regime.
# Often the system is solved with constant boundary conditions or homogeneous
# boundary conditions on a 2D domain.


# ## Setup

# System imports.
import sys
import random

# UG4 imports.
import ug4py.pyugcore as ug4
import ug4py.pylimex as limex
import ug4py.pyconvectiondiffusion as cd

# Own imports.
sys.path.append("..")
import modsimtools as util


# ## Grid and domain
requiredSubsets = ["Inner", "Boundary"]     # defining subsets
gridName = "square2d.ugx"                   # Mesh created with ProMesh
numRefs = 5                                 # Number of refinements

# Choosing a domain object (either 1d, 2d or 3d suffix)
print("Loading Domain "+ gridName + "...")
dom = ug4.Domain2d()
ug4.LoadDomain(dom, gridName)

# Grid refinement (optional)
if numRefs > 0:
    print("Refining ...")
    refiner = ug4.GlobalDomainRefiner(dom)
    for i in range(numRefs):
        ug4.TerminateAbortedRun()
        refiner.refine()
        print("Refining step "+ str(i) +" ...")

    print("Refining done")

# Check, if geometry has the needed subsets of the problem
sh = dom.subset_handler()
for e in requiredSubsets:
    if sh.get_subset_index(e) == -1:
        print("Domain does not contain subset {e}.")
        sys.exit(1)


#-----------------------------------------
#-- A) Model parameters
#-----------------------------------------


a = 0.1
b = 0.9  
gamma = 10.0

diffusion ={}
diffusion["u"] = 1.0 
diffusion["v"] = 10.0

uinfty = a+b
vinfty = b/(uinfty*uinfty)

#-----------------------------------------
#-- B) Ansatz spaces
#-----------------------------------------
approxSpace = ug4.ApproximationSpace2d(dom)
approxSpace.add_fct("u", "Lagrange", 1)   # lineare Ansatzfunktionen
approxSpace.add_fct("v", "Lagrange", 1)
approxSpace.init_levels()
approxSpace.init_top_surface()

print("approximation space:")
approxSpace.print_statistic()


#-----------------------------------------
#-- C) Finite volume element discretization
#-----------------------------------------
elemDisc = {}
elemDisc["u"] = util.CreateDiffusionElemDisc("u", "Inner", 1.0, diffusion["u"], 0.0)
elemDisc["v"] = util.CreateDiffusionElemDisc("v", "Inner", 1.0, diffusion["v"], 0.0)


# -----------------------------------------
# -- C).2 Reaction terms
# -----------------------------------------

def SchnakenbergU(u, v):
    return -gamma * (a - u + u * u * v)

def SchnakenbergU_dU(u, v):
    return -gamma * (-1.0 + 2.0 * u * v)

def SchnakenbergU_dV(u, v):
    return -gamma * (u * u)


def SchnakenbergV(u, v):
    return -gamma * (b - u * u * v)

def SchnakenbergV_dU(u, v):
    return -gamma * (-2.0 * u * v)

def SchnakenbergV_dV(u, v):
    return -gamma * (u * u)


# activator u
schnackenberg_func = {}
schnackenberg_func["u"] = ug4.PythonUserFunction2d(SchnakenbergU, 2)
schnackenberg_func["u"].set_input_and_deriv(0, elemDisc["u"].value(), SchnakenbergU_dU)
schnackenberg_func["u"].set_input_and_deriv(1, elemDisc["v"].value(), SchnakenbergU_dV)

# inhibitor v
schnackenberg_func["v"] = ug4.PythonUserFunction2d(SchnakenbergV, 2)
schnackenberg_func["v"].set_input_and_deriv(0, elemDisc["u"].value(), SchnakenbergV_dU)
schnackenberg_func["v"].set_input_and_deriv(1, elemDisc["v"].value(), SchnakenbergV_dV)

elemDisc["u"].set_reaction(schnackenberg_func["u"])  #-- DEBUG 
elemDisc["v"].set_reaction(schnackenberg_func["v"])  #-- DEBUG 



#-----------------------------------------
#-- D) Initial values ("Anfangswerte")
#-----------------------------------------
def MyInitialValueU(x, y, t, si):
    return uinfty + 0.01 * random.randint(-1, 1) 

def MyInitialValueV(x, y, t, si):
    return vinfty + 0.01 * random.randint(-1, 1) 

#-----------------------------------------
#-- E) Boundary values
#-----------------------------------------

#dirichletBND = ug4.DirichletBoundary2dCPU1()
#dirichletBND.add(uinfty, "u", "Boundary")
#dirichletBND.add(vinfty, "v", "Boundary")


#-----------------------------------------
#-- F) Domain discretization
#-----------------------------------------

domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)
domainDisc.add(elemDisc["u"])
domainDisc.add(elemDisc["v"])
# domainDisc.add(dirichletBND) # We use noflux 

#-----------------------------------------
#-- G) Solver setup
#-----------------------------------------

# Setup linear solver
lsolver = ug4.LUCPU1()

# Time stepping config.
# n = 1: Backward Euler, n > 1: LIMEX mit n Stufen
nstages = 2
dt = 0.01   # Setup time step

if (nstages == 1):
    print("Only LIMEX supported (nstages>1). Exiting.")
    sys.exit(1)
else:
    # LIMEX config.
    timeInt = limex.LimexTimeIntegrator2dCPU1(nstages)
    nlsolver = limex.LimexNewtonSolverCPU1()
    nlsolver.set_linear_solver(lsolver)
    for i in range(nstages):
        timeInt.add_stage(i+1, nlsolver, domainDisc)

    # Time stepping config.
    timeInt.set_time_step(dt)
    timeInt.set_dt_min(dt*1e-8)
    timeInt.set_dt_max(dt*1e3)
    timeInt.set_increase_factor(1.5) # max. Faktor, um den dt erhöht werden darf
    timeInt.disable_matrix_cache()

    # LIMEX w/ error estimation
    TOL = 1e-4
    timeInt.set_tolerance(TOL)
    errorEst = limex.Norm2EstimatorCPU1() # Euclidean norm.
    timeInt.add_error_estimator(errorEst)

    # TODO: Verbesserter Schaetzer, z.B. basierend auf den Massen in den Schichten oder Flüssen über die Ränder.    
    #metricSpace = ug4.CompositeSpace2dCPU1()
    #metricSpace.add(ug4.L2ComponentSpace2dCPU1("u", 2))
    # metricSpace.add(ug4.L2ComponentSpace2dCPU1("u", 2, "DEPOS"))
    # metricSpace.add(ug4.L2ComponentSpace2dCPU1("u", 2, "SC"))
    #timeInt.set_space(metricSpace)

    # errorEst = limex.CompositeGridFunctionEstimator2dCPU1()
    # errorEst.add(metricSpace)
    

# Create callback observer for file I/O.
def MyVTKCallback(usol, step, time, dt) :
    ug4.WriteGridFunctionToVTK(usol, "Schnakenberg"+str(int(step)).zfill(5)+".vtu")
    
vtkobserver = ug4.PythonCallbackObserver2dCPU1(MyVTKCallback) 
timeInt.attach_observer(vtkobserver)   
#-----------------------------------------
#-- H) Solve transient problem
#-----------------------------------------
usol = ug4.GridFunction2dCPU1(approxSpace)


# Set initial values.
print("Setting initial values ...")
valU0 =ug4.PythonUserNumber2d(MyInitialValueU)
valV0 =ug4.PythonUserNumber2d(MyInitialValueV)

ug4.Interpolate(valU0, usol, "u")
ug4.Interpolate(valV0, usol, "v")


# Solve transient problem.
print("Solving transient problem ...")

startTime = 0
endTime = 5.0
try:
    timeInt.apply(usol, endTime, usol, startTime)
except Exception as e:
    print("Error during time stepping: ", e)
    sys.exit(1)


print("done")

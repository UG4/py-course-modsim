# Load utility scripts (e.g. from from ugcore/scripts)
#!/usr/bin/env python
# coding: utf-8

# [<img src="../../header.svg">](../index.ipynb)
# 
# ---
# # Bromine diffusion problem in 3D
# 
# Governing equations:
# $$ \frac{\partial u}{\partial t} + \nabla \cdot [-D \nabla u] = f
# $$
# plus boundary and initial condiitions.
# 
# ## Setup

# In[ ]:


import sys
sys.path.append("..")
import random

import ug4py.pyugcore as ug4
import ug4py.pylimex as limex
import ug4py.pyconvectiondiffusion as cd
import modsimtools as util


# ## Grid and domain

# In[ ]:


# Setup:
# defining needed subsets, grid and number of refinements
requiredSubsets = ["Inner", "Boundary"] # defining subsets
gridName = "square2d.ugx"  # Grid created with ProMesh
numRefs = 2  # Number of Refinement steps on given grid

# Choosing a domain object
# (either 1d, 2d or 3d suffix)
dom = ug4.Domain2d()

# Loading the given grid into the domain
print("Loading Domain "+ gridName + "...")
ug4.LoadDomain(dom, gridName)
print("Domain loaded.")

# Optional: Refining the grid
if numRefs > 0:
    print("Refining ...")
    refiner = ug4.GlobalDomainRefiner(dom)
    for i in range(numRefs):
        ug4.TerminateAbortedRun()
        refiner.refine()
        print("Refining step "+ str(i) +" ...")

    print("Refining done")

# checking if geometry has the needed subsets of the problem
sh = dom.subset_handler()
for e in requiredSubsets:
    if sh.get_subset_index(e) == -1:
        print("Domain does not contain subset {e}.")
        sys.exit(1)







#-----------------------------------------
#-- A) Model parameters
#-----------------------------------------
diffusionTensor ={}
diffusionTensor["v"] = 1.0
diffusionTensor["w"] = 40.0
diffusionTensor["u"] = 10 

x0 = 4.0

a = 0.2
b = 1.0  
c = 1.3
gamma = 490.0 

cc = -0.1
ee = 0.1
uinfty = 1.0  #-- u0

k1 = a*uinfty
k2 = c*uinfty

 
vinfty = (k1+k2)/b  #-- aux
winfty = k2/(vinfty*vinfty)


#-----------------------------------------
#-- B) Ansatz spaces
#-----------------------------------------
approxSpace = ug4.ApproximationSpace2d(dom)
approxSpace.add_fct("u", "Lagrange", 1)   # lineare Ansatzfunktionen
approxSpace.add_fct("v", "Lagrange", 1)
approxSpace.add_fct("w", "Lagrange", 1)
approxSpace.init_levels()
approxSpace.init_top_surface()

print("approximation space:")
approxSpace.print_statistic()


#-----------------------------------------
#-- C) Finite volume element discretization
#-----------------------------------------
elemDisc = {}
elemDisc["u"] = util.CreateDiffusionElemDisc("u", "Inner", 1.0, diffusionTensor["u"], 0.0)
elemDisc["v"] = util.CreateDiffusionElemDisc("v", "Inner", 1.0, diffusionTensor["v"], 0.0)
elemDisc["w"] = util.CreateDiffusionElemDisc("w", "Inner", 1.0, diffusionTensor["w"], 0.0)

# Adding velocity term to elemDisc["u"]
x0_vel=ug4.ScaleAddLinkerVector2d(); x0_vel.add(x0, elemDisc["v"].gradient())
elemDisc["u"].set_upwind(ug4.FullUpwind2d()) #-- NoUpwind() -- FullUpwind()

# -----------------------------------------
# -- C).2 Reaction terms
# -----------------------------------------

def ReactionV(u, v, w):
    return -1.0 * gamma * (a * u - b * v + w * v * v)

def ReactionV_dU(u, v, w):
    return -1.0 * gamma * a

def ReactionV_dV(u, v, w):
    return -1.0 * gamma * (-b + 2.0 * w * v)

def ReactionV_dW(u, v, w):
    return -1.0 * gamma * (v * v)

nonlinearGrowth = {}
nonlinearGrowth["v"] = ug4.PythonUserFunction2d(ReactionV, 3)
nonlinearGrowth["v"].set_input_and_deriv(0, elemDisc["u"].value(), ReactionV_dU)
nonlinearGrowth["v"].set_input_and_deriv(1, elemDisc["v"].value(), ReactionV_dV)
nonlinearGrowth["v"].set_input_and_deriv(2, elemDisc["w"].value(), ReactionV_dW)


def ReactionW(u, v, w):
    return -1.0 * gamma * (c * u - v * v * w)

def ReactionW_dU(u, v, w):
    return -1.0 * gamma * c

def ReactionW_dV(u, v, w):
    return -1.0 * gamma * (-2.0 * v * w)

def ReactionW_dW(u, v, w):
    return 1.0 * gamma * (v * v)

nonlinearGrowth["w"] = ug4.PythonUserFunction2d(ReactionW, 3)
nonlinearGrowth["w"].set_input_and_deriv(0, elemDisc["u"].value(), ReactionW_dU)
nonlinearGrowth["w"].set_input_and_deriv(1, elemDisc["v"].value(), ReactionW_dV)
nonlinearGrowth["w"].set_input_and_deriv(2, elemDisc["w"].value(), ReactionW_dW)

elemDisc["v"].set_reaction(nonlinearGrowth["v"])  #-- DEBUG 
elemDisc["w"].set_reaction(nonlinearGrowth["w"])  



#-----------------------------------------
#-- D) Initial values ("Anfangswerte")
#-----------------------------------------
def MyInitialValueU(x, y, t, si):
    return uinfty + 0.1 * ((ee - cc) * random.randint(-1, 1) + cc)

def MyInitialValueV(x, y, t, si):
    return vinfty + 0.1 * ((ee - cc) * random.randint(-1, 1) + cc)

def MyInitialValueW(x, y, t, si):
    return winfty + 0.1 * ((ee - cc) * random.randint(-1, 1) + cc)


#-----------------------------------------
#-- E) Boundary values
#-----------------------------------------
dirichletBND = ug4.DirichletBoundary2dCPU1()
dirichletBND.add(uinfty, "u", "Boundary")
dirichletBND.add(vinfty, "v", "Boundary")
dirichletBND.add(winfty, "w", "Boundary")


#-----------------------------------------
#-- F) Domain discretization
#-----------------------------------------
domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)
domainDisc.add(elemDisc["u"])
domainDisc.add(elemDisc["v"])
domainDisc.add(elemDisc["w"])
# domainDisc.add(dirichletBND)

#-----------------------------------------
#-- G) Solver setup
#--    (using 'util/solver_util.lua')
#-----------------------------------------

# Neue Schrittweitenberechnung für Testzwecke auskommentiert
# Zeitintegrator
# n = 1: Backward Euler, n > 1: LIMEX mit n Stufen
nstages = 2

# Setup linear solver
lsolver = ug4.LUCPU1()

# Setup time step
dt = 0.01

if (nstages == 1):
    # Backward Euler
    timeDisc = ug4.ThetaTimeStepCPU1(domainDisc, 1.0)
    timeInt = limex.ConstStepLinearTimeIntegrator2dCPU1(timeDisc)
    timeInt.set_linear_solver(lsolver)
    timeInt.set_time_step(dt)
else:
    # LIMEX config.
    timeInt = limex.LimexTimeIntegrator2dCPU1(nstages)
    nlsolver = limex.LimexNewtonSolverCPU1()
    nlsolver.set_linear_solver(lsolver)
    for i in range(nstages):
        timeInt.add_stage(i+1, nlsolver, domainDisc)

    # Time stepping config.
    timeInt.set_time_step(dt)
    timeInt.set_dt_min(dt*1e-3)
    timeInt.set_dt_max(dt*1e3)
    timeInt.set_increase_factor(1.5) # max. Faktor, um den dt erhöht werden darf
    timeInt.disable_matrix_cache()


    # LIMEX w/ error estimation
    TOL = 1e-3
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
    ug4.WriteGridFunctionToVTK(usol, "Chemotaxis"+str(int(step)).zfill(5)+".vtu")
    
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
valW0 =ug4.PythonUserNumber2d(MyInitialValueW)

ug4.Interpolate(valU0, usol, "u")
ug4.Interpolate(valV0, usol, "v")
ug4.Interpolate(valW0, usol, "w")


# Solve transient problem.
print("Solving transient problem ...")

startTime = 0
endTime = 0.5
try:
    timeInt.apply(usol, endTime, usol, startTime)
except Exception as e:
    print("Error during time stepping: ", e)
    sys.exit(1)


print("done")

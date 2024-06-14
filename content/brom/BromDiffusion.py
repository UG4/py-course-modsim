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

# In[2]:


import sys
sys.path.append("..")

import ug4py.pyugcore as ug4
import ug4py.pylimex as limex
import ug4py.pyconvectiondiffusion as cd


# ## Grid and domain

# In[3]:


# Setup:
# defining needed subsets, grid and number of refinements
requiredSubsets = ["INNER", "WALL", "IN"] # defining subsets
gridName = "brom-dose.ugx"  # Grid created with ProMesh
numRefs = 2  # Number of Refinement steps on given grid

# Choosing a domain object
# (either 1d, 2d or 3d suffix)
dom = ug4.Domain3d()

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

# checking if geometry has the needed subsets of the probelm
sh = dom.subset_handler()
for e in requiredSubsets:
    if sh.get_subset_index(e) == -1:
        print("Domain does not contain subset {e}.")
        sys.exit(1)



# ## Approximation space 

# In[4]:


# Create approximation space which describes the unknowns in the equation
fct = "u"  # name of the function
type = "Lagrange"
order = 1  # polynom order for lagrange

approxSpace = ug4.ApproximationSpace3d(dom)
approxSpace.add_fct(fct, type, order)
approxSpace.init_levels()
approxSpace.init_surfaces()  
approxSpace.init_top_surface()
approxSpace.print_statistic()




# ## Finite Volume Discretization

# In[ ]:


# Create discretization for the Convection-Diffusion Equation
# perform the discretization on the actual elements
# using first order finite volumes (FV1) on the 3d grid
dif = 10

# creating instance of a convection diffusion equation
elemDisc = cd.ConvectionDiffusionFV13d("u", "INNER")
elemDisc.set_diffusion(dif)

# ug4 separates the boundary value and the discretization
# boundary conditions can be enforced through a post-process (dirichlet).
# To init at boundary, the value, function name from the Approximationspace
# and the subset name are needed
dirichletBND = ug4.DirichletBoundary3dCPU1()
dirichletBND.add(1.0, "u", "TOP")  # Zufluss
dirichletBND.add(0.0, "u", "IN")  # "offene Dose"


# create the discretization object which combines all the
# separate discretizations into one domain discretization.
domainDisc = ug4.DomainDiscretization3dCPU1(approxSpace)
domainDisc.add(elemDisc)
domainDisc.add(dirichletBND)


# In[4]:


# Technical stuff
def MyFunction (u,v):
    return u*v

def MyFunction_u (u,v):
    return v

def MyFunction_v (u,v):
    return u

pyFunction=ug4.PythonUserFunction3d(MyFunction, 2)
pyFunction.set_input_and_deriv(0,elemDisc.value(), MyFunction_u)
pyFunction.set_input_and_deriv(1,elemDisc.value(), MyFunction_v)
# elemDisc.set_reaction(pyFunction)



# ## Solve transient problem

# In[5]:


# Use the Approximationspace to
# create a vector of unknowns and a vector
# which contains the right hand side
usol = ug4.GridFunction3dCPU1(approxSpace)

# Init the vector representing the unknowns with function
def MyInitialValue(x, y, z, t, si):
    return 0.0 if (z<1.75) else 1.0

try:
    # Passing the function as a string for the C++ Backend
    pyInitialValue=ug4.PythonUserNumber3d(MyInitialValue)
    # pyInitialValue.evaluate(0.0, ug4.Vec3d(1.0, 0.0, 0.0), 0.0, 4) # Test
    ug4.Interpolate(pyInitialValue, usol, "u") 

except Exception as inst:
    print("EXCEPTION:")
    print(inst)
ug4.WriteGridFunctionToVTK(usol, "BromInitial")


# In[ ]:


# Create Solver
# In this case we use an LU (Lower Upper) Solver for an exact solution
lsolver=ug4.LUCPU1()


# In[6]:


# Define start time, end time and step size
startTime = 0.0
endTime = 1.0
dt = 0.00625

# create a time discretization with the theta-scheme
# takes in domain and a theta
# theta = 1.0 -> Implicit Euler, 
# 0.0 -> Explicit Euler 
timeDisc=ug4.ThetaTimeStepCPU1(domainDisc, 1.0) 

# creating Time Iterator, setting the solver and step size
timeInt = limex.LinearTimeIntegrator3dCPU1(timeDisc)
#timeInt = limex.ConstStepLinearTimeIntegrator3dCPU1(timeDisc)
timeInt.set_linear_solver(lsolver)
timeInt.set_time_step(dt)

# Solving the transient problem
try:
    timeInt.apply(usol, endTime, usol, startTime)
except Exception as inst:
    print("EXCEPTION:")
    # print(type(inst))
    print(inst)
    
    


# ## Exporting the result 

# In[8]:


# Exporting the result to a vtu-file
# can be visualized in paraview or with a python extension
ug4.WriteGridFunctionToVTK(usol, "BromFinal")

# Plotting the result using pyvista
import pyvista
result = pyvista.read('BromFinal.vtu')
print()
print("Pyvista input: ")
print(result)
result.plot(scalars="u", show_edges=True, cmap='hot')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# [<img src="../../header.svg">](../../index.ipynd)
# 
# ---
# 
# # Drug Transport across a Virtual Skin Membrane: Binding of substances.
# 
# In this example we solve a coupled system
# 
# $$ \partial_t c + \nabla \cdot [-D \nabla c] + k_f \cdot c - k_r \cdot b = 0 $$
# $$ \partial_t b   - k_f \cdot c + k_r \cdot b = 0 $$
# 
# The steady-state is given by $ b = \frac{k_f}{k_r} c $
# 

# ### Initialize UG4 

# In[85]:


# Importing 
import ug4py.pyugcore  as ugcore
import ug4py.pylimex as limex
import ug4py.pyconvectiondiffusion as cd

import sys
sys.path.append('..')
import modsimtools as util


# ### Create Domain

# In[69]:


requiredSubsets = {"LIP", "COR", "BOTTOM_SC", "TOP_SC"}
gridName = "skin2d-aniso.ugx"
numRefs = 2


# In[70]:


dom = util.CreateDomain(gridName, numRefs, requiredSubsets)


# ### Create Approximation space

# In[71]:


approxSpace = ugcore.ApproximationSpace2d(dom)
approxSpace.add_fct("c","Lagrange", 1) 
approxSpace.add_fct("b","Lagrange", 1) ## NEW: Two functions are required
approxSpace.init_levels()
approxSpace.init_top_surface()
print("Approximation space:")
approxSpace.print_statistic()


# ## Create a convection-diffusion-equation
# 
# * Define model parameters

# In[72]:


# Define model parameter
D={ "COR":  0.01, "LIP":1.0 }

kf = 2.0
kr = 1.0


# * Print characteristic times:

# In[73]:


L=17.6 # characteristic length
print("Characteristic times:")
print("Binding (c->b):\t\t" + str(1.0/kf))
print("Binding (b->c):\t\t" + str(1.0/kr))
print("Diffusion [LIP]:\t" + str(L*L/D["LIP"]))
print("Diffusion [COR]:\t" + str(L*L/D["COR"]))


# * The system requires **separate** element discs **for each species**:

# In[74]:


# Create element discretizations for free species 
elemDiscC ={}
elemDiscC["COR"] = util.CreateDiffusionElemDisc("c", "COR", 1.0, D["COR"], 0.0)
elemDiscC["LIP"] = util.CreateDiffusionElemDisc("c", "LIP", 1.0, D["LIP"], 0.0)


# In[75]:


# Create element discretizations for bound species
elemDiscB ={}
elemDiscB["COR"] = util.CreateDiffusionElemDisc("b", "COR", 1.0, 0.0, 0.0)
elemDiscB["LIP"] = util.CreateDiffusionElemDisc("b", "LIP", 1.0, 0.0, 0.0)


# * Add corneocyte binding, i.e., reactions in corneocytes

# In[76]:


#myReactionC = kf*elemDiscC["COR"]:value()-kr*elemDiscB["COR"]:value()
#myReactionB = -kf*elemDiscC["COR"]:value()+kr*elemDiscB["COR"]:value()

def MyFunction (b,c): return kf*c - kr*b
def MyFunction_b (b,c): return kf
def MyFunction_c (b,c): return -kr

pyReactionC=ugcore.PythonUserFunction2d(MyFunction, 2)
pyReactionC.set_input_and_deriv(0,elemDiscC["COR"].value(), MyFunction_b)
pyReactionC.set_input_and_deriv(1,elemDiscB["COR"].value(), MyFunction_c)

pyReactionB =  ugcore.ScaleAddLinkerNumber2d()
pyReactionB.add(-1.0, pyReactionC)

elemDiscC["COR"].set_reaction(pyReactionC)
elemDiscB["COR"].set_reaction(pyReactionB)


# * Boundary conditions:

# In[77]:


dirichletBnd = ugcore.DirichletBoundary2dCPU1()
dirichletBnd.add(1.0, "c", "TOP_SC")
dirichletBnd.add(0.0, "c", "BOTTOM_SC")
dirichletBnd.add(0.0, "b", "TOP_SC") #NEW
dirichletBnd.add(0.0, "b", "BOTTOM_SC")


# * Summarize everything in domain discretization:

# In[78]:


try:
    domainDisc = ugcore.DomainDiscretization2dCPU1(approxSpace)
    domainDisc.add(dirichletBnd)
    domainDisc.add(elemDiscC["LIP"]) 
    domainDisc.add(elemDiscC["COR"])
    domainDisc.add(elemDiscB["LIP"])
    domainDisc.add(elemDiscB["COR"])
except RuntimeError as inst:    
    print(type(inst))    # the exception type
    print(inst.args)     # arguments stored in .args
    print(inst)  


# ## Solve transient problem
# * Define parameters for time stepping:

# In[79]:


startTime = 0.0
endTime = 500.0
dt=endTime/100.0


# * Define grid function:

# In[80]:


usol = ugcore.GridFunction2dCPU1(approxSpace)
ugcore.Interpolate(0.0, usol, "c")
ugcore.Interpolate(0.0, usol, "b")


# * Create solver

# In[81]:


lsolver=ugcore.LUCPU1()


# * Use C++-Objects

# In[82]:


timeDisc=ugcore.ThetaTimeStepCPU1(domainDisc, 1.0)
timeInt = limex.ConstStepLinearTimeIntegrator2dCPU1(timeDisc)
timeInt.set_linear_solver(lsolver)
timeInt.set_time_step(dt)


# In[83]:


vtkObserver = limex.VTKOutputObserver2dCPU1("vtk/BindingSol.vtk", ugcore.VTKOutput2d())
timeInt.attach_finalize_observer(vtkObserver)


# In[84]:


try:
    timeInt.apply(usol, endTime, usol, startTime)
except Exception as inst:    
    print(type(inst))    # the exception type
    print(inst.args)     # arguments stored in .args
    print(inst)  


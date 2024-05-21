#!/usr/bin/env python
# coding: utf-8

# [<img src="../header.svg">](../index.ipynd)
# 
# ---
# 
# # Drug Transport across a Virtual Skin Membrane:  Extended version
# [Previous Version](SkinDiffusion.ipybnd)

# ## 1. Setup
# 
# ### Load existing modules 

# In[1]:


import modsimtools as util

import ug4py as ug4
import pylimex as limex
import pyconvectiondiffusion as cd
import pysuperlu as slu


# ### Create Domain

# In[2]:


requiredSubsets = {"LIP", "COR", "BOTTOM_SC", "TOP_SC"}
gridName = "skin2d-aniso.ugx"
numRefs = 2


# In[3]:


dom = util.CreateDomain(gridName, numRefs, requiredSubsets)


# ### Create Approximation space

# In[4]:


approxSpaceDesc = dict(fct = "u", type = "Lagrange", order = 1)
approxSpace = util.CreateApproximationSpace(dom, approxSpaceDesc)


# ### Create discretization
# 

# In[5]:


# Define model parameter
K={ "COR": 1e-3, "LIP": 1.0 }
D={ "COR":  1.0, "LIP":1.0 }


# In[6]:


# Create element discretizations for lipids and corneocytes.
elemDisc ={}
elemDisc["COR"] = util.CreateDiffusionElemDisc("u", "COR", K["COR"], K["COR"]*D["COR"], 0.0)
elemDisc["LIP"] = util.CreateDiffusionElemDisc("u", "LIP", K["LIP"], K["LIP"]*D["LIP"], 0.0)


# In[7]:


# Set Dirichlet boundary conditions.
dirichletBnd = ug4.DirichletBoundary2dCPU1()
dirichletBnd.add(1.0, "u", "TOP_SC")
dirichletBnd.add(0.0, "u", "BOTTOM_SC")


# In[8]:


# Create the global discretization object.
domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)
domainDisc.add(elemDisc["LIP"])
domainDisc.add(elemDisc["COR"])
domainDisc.add(dirichletBnd)


# ## 2. Steady state problem
# Flux is computed from steady state. Since configuration of a multigrid solver is somewhat tricky, we use an LU decomposition here:

# In[9]:


A = ug4.AssembledLinearOperatorCPU1(domainDisc)
u = ug4.GridFunction2dCPU1(approxSpace)
b = ug4.GridFunction2dCPU1(approxSpace)

ug4.Interpolate(0.0, u, "u")

domainDisc.assemble_linear(A, b)
domainDisc.adjust_solution(u)

#lsolver = ug4.LUCPU1()
lsolver =slu.SuperLUCPU1()

lsolver.init(A, u)
lsolver.apply(u, b)

ug4.WriteGridFunctionToVTK(u, "SkinSteadyState.vtk")


# Compute $J_\infty=J(t=\infty)$ for
# $$ J(t)=\frac{1}{|\Gamma|}\int_\Gamma (-KD \nabla u(t,x)) \cdot \vec n dA$$

# In[10]:


areaG=ug4.Integral(1.0, u, "BOTTOM_SC")
print("Surface area [um^2]:")
print(areaG)

surfaceFlux = {}
surfaceFlux["BOT"] = K["LIP"]*D["LIP"]*ug4.IntegrateNormalGradientOnManifold(u, "u", "BOTTOM_SC", "LIP")
surfaceFlux["TOP"] = K["LIP"]*D["LIP"]*ug4.IntegrateNormalGradientOnManifold(u, "u", "TOP_SC", "LIP")

print("Surface fluxes [kg/s]:")
print(surfaceFlux["TOP"])
print(surfaceFlux["BOT"])


# In[11]:


print("Normalized Fluxes [kg / (mu^2 * s)]:")
print(surfaceFlux["TOP"]/areaG)
print(surfaceFlux["BOT"]/areaG)


# In[12]:


Jref = 1.0/17.6

print("Relative Fluxes [1]:")
print(surfaceFlux["TOP"]/areaG/Jref)
print(surfaceFlux["BOT"]/areaG/Jref)


# ## 3. Transient problem

# After each time-step, we execute a a callback function `MyPostProcess`. In this function, print the solution and compute
# $$
# m(t_k):= \int_0^{t_k} J(s) \, ds \approx \sum_{i=1}^k(t_{i}- t_{i-1}) \frac{J(t_{i-1}) +J(t_i)}{2} 
# $$
# using the trapezoid rule. Moreover, we also compute the lag time $\tau$ from $m(t_k) = J_\infty(t_k - \tau)$.
# 

# In[13]:


# auxiliary variables for book-keeping
import numpy as np
import os

tPoints = np.array([0.0])
mPoints = np.array([0.0])
jPoints = np.array([0.0])

def MyPostProcess(u, step, time, dt):
    
    global tPoints
    global mPoints
    global jPoints 
    
    tOld = tPoints[-1]
    mOld = mPoints[-1]
    jOld = jPoints[-1]
    
    # 1) Compute fluxes.
    gradFlux={}
    gradFlux["BOT"] = ug4.IntegrateNormalGradientOnManifold(u, "u", "BOTTOM_SC", "LIP")
    gradFlux["TOP"] = ug4.IntegrateNormalGradientOnManifold(u, "u", "TOP_SC", "LIP")
  
    jTOP = K["LIP"]*D["LIP"]*gradFlux["TOP"]
    jBOT = K["LIP"]*D["LIP"]*gradFlux["BOT"]
    print ("flux_top (\t" + str(time) + "\t)=\t" + str(jTOP))
    print ("flux_bot (\t" + str(time) + "\t)=\t" + str(jBOT))
  
    # 2) Compute mass.
    dt = time - tOld
    mass = mOld + (time - tOld)*(jBOT + jOld)/2.0
    print("mass_bot (\t"+ str(time) +"\t)=\t"+ str(mass))
  
    # 3) Compute lag time.
    print("tlag="+ str(time - mass/jBOT) )
  
    # 4) Updates
    tPoints = np.append(tPoints, time)
    jPoints = np.append(jPoints, jBOT)
    mPoints = np.append(mPoints, mass)
    
    os.system('df -k /')
    return True
    


# In[14]:


pyobserver= ug4.PythonCallbackObserver2dCPU1(MyPostProcess)


# Additionally, we define a callback for vtk files.

# In[15]:


# Callback for file I/O.
vtk=ug4.VTKOutput2d()
vtkobserver=limex.VTKOutputObserver2dCPU1("vtk/SkinData.vtu", vtk)


# ### Define time integrator

# In[16]:


# Create time discretization.
timeDisc=ug4.ThetaTimeStepCPU1(domainDisc, 1.0) 


# In[17]:


# Settings for time stepping.
startTime = 0.0
endTime = 2500.0
dt=25.0
dtMin=2.5

ug4.Interpolate(0.0, u, "u")


# In[18]:


# Create time integrator.
timeInt = limex.ConstStepLinearTimeIntegrator2dCPU1(timeDisc)
timeInt.set_linear_solver(lsolver)
timeInt.set_time_step(dt)


# In[19]:


# Attach observers.
timeInt.attach_observer(vtkobserver)
timeInt.attach_observer(pyobserver)


# In[20]:


# Create progress widget.
import ipywidgets as widgets
from IPython.display import display

wProgress=widgets.FloatProgress(value=startTime, min=startTime, max=endTime, 
                                description='Progress:', style={'bar_color': 'blue'})
display(wProgress)

# Create callback and attach observer.
def MyProgress(u, step, time, dt):
    wProgress.value=time
    
timeInt.attach_observer(ug4.PythonCallbackObserver2dCPU1(MyProgress)) 


# ### Solve transient problem

# In[ ]:


# Solve problem.
try:
    timeInt.apply(u, endTime, u, startTime)
except Exception as inst:
    print(inst)

wProgress.style={'bar_color': 'red'}


# ## 4. Visualization
# 

# In[ ]:


import matplotlib.pyplot as pyplot
pyplot.xlabel("time")
pyplot.ylabel("mass")
pyplot.plot(tPoints, mPoints)
pyplot.show()


# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# [<img src="../../header.svg">](../index.ipynb)
# 
# ---
# # Illustration: Basic properties of iterative methods
# 
# This example shows the basic steps to solve $-\triangle u = f$ on the unit square using Dirichlet boundary conditions. 
# 
# An extension is given in the [following example](./tutorial-fem-02.ipynb).
# 
# ## Initialization
# Loading modules:

# In[ ]:


import sys
sys.path.append("../.")

import modsimtools as util
import ug4py.pyugcore as ugcore
import ug4py.pyconvectiondiffusion as cd

history = {}


# In[ ]:


myRefs = 5
dom =  util.CreateDomain("./grids/laplace_sample_grid_2d-tri.ugx", myRefs, {"Inner","Boundary"})
approxSpace = util.CreateApproximationSpace(dom, dict(fct = "u", type = "Lagrange", order = 1))


# In[ ]:


# Order vertices (and dofs) lexicographically
ugcore.OrderLex(approxSpace, "x")


# In[ ]:


# Import problem
from config.ConvectionDiffusionProblem import ConvectionDiffusionProblem
mydesc = {"eps":1, "velx":0.0, "vely": 0.0}
problem=ConvectionDiffusionProblem(mydesc)


# In[ ]:


# Select discretization
domainDisc = ugcore.DomainDiscretization2dCPU1(approxSpace)
problem.add_element_discretizations(domainDisc, "u")
problem.add_boundary_conditions(domainDisc, "u")


# ## Defining iterative methods

# In[ ]:


# Select smoothers
preconditioners = dict(
    jac = ugcore.JacobiCPU1(0.66),              # Jacobi (w/ damping)
    gs  = ugcore.GaussSeidelCPU1(),             # Gauss-Seidel (forward only)
    sgs = ugcore.SymmetricGaussSeidelCPU1(),    # Symmetric Gauss-Seidel (forward + backward)
    ilu = ugcore.ILUCPU1(),                     # ILU (w/ pattern based),
   # ilut = ug4.ILUT(0.1)                       # ILU (w/ threshold)
)


# In[ ]:


# Additional config.
preconditioners["ilu"].set_sort(False)
preconditioners["ilu"].set_beta(0.01)


baseLU = ugcore.LUCPU1()
# baseSLU = SuperLU()



# ## Testing preconditioners

# In[ ]:


# Solve problem with max 100 iterations, 
# reducing the error by 6 orders of magnitude  
convCheck = ugcore.ConvCheckCPU1()
convCheck.set_maximum_steps(100)
convCheck.set_reduction(1e-6)
convCheck.set_minimum_defect(1e-15)


# In[ ]:


solverID= "gs"

# Linear solver
linSolver = ugcore.LinearSolverCPU1()
linSolver.set_preconditioner(preconditioners[solverID])
linSolver.set_convergence_check(convCheck)


# In[ ]:


# CG method
cgSolver = ugcore.CGCPU1()
cgSolver.set_preconditioner(preconditioners[solverID])
cgSolver.set_convergence_check(convCheck)


# Aktiviert man die folgenden Zeilen, so können wir anhand der Dateien *LS_Solution_IterNN* nachzuvollziehen, wie der Fehler abnimmt:

# In[ ]:


#dbgWriter = ugcore.GridFunctionDebugWriter2dCPU1(approxSpace)
#linSolver.set_debug(dbgWriter)


# Die Konvergenz dieser einfachen Iterationen hängt von der Gitterweite ab. Die einfachen Glätter konvergieren mit zunehmender Verfeinerung immer schlechter.  Für $h\rightarrow 0$ geht diese gegen 1.

# In[ ]:


def SolverTest(domainDisc, approxSpace, solver, ioname):
    # Assemble linear system
    A = ugcore.AssembledLinearOperatorCPU1(domainDisc)
    u = ugcore.GridFunction2dCPU1(approxSpace)
    b = ugcore.GridFunction2dCPU1(approxSpace)
    
    ugcore.VecScaleAssign(u, 0.0, u)  #u.set(0.0)
    try:
        domainDisc.assemble_linear(A, b)
    except:
        print ("Assembly error!")
        
    clock = ugcore.CuckooClock()
  
   
    ugcore.VecScaleAssign(b, 0.0, b)  # b.set(0.0)

    # u.set_random(-1.0, 1.0)
    import random
    random.seed(5)
    ugcore.Interpolate(ugcore.PythonUserNumber2d(lambda x,y,t,si : random.uniform(-1.0,1.0)), u, "u")  
    
    # Call solver
    clock.tic()
    try:
        solver.init(A, u)
        solver.apply_return_defect(u,b)
    except:
        print ("Solver failed!")

    nvec = convCheck.step()
    print ("Results: " + str(nvec) + " steps (rho=" +str(convCheck.avg_rate()) +")")
    print("Time: " + str(clock.toc()) +" seconds")
    
    # Post processing (print results):
    ugcore.WriteGridFunctionToVTK(u, ioname)
    ugcore.SaveMatrixForConnectionViewer(u, A, "MatrixA.mat")
    # SaveVectorForConnectionViewer(u, "VectorU.vec")
    
    # Store history (e.g. for plotting)
    import numpy as np
    history = np.zeros(nvec)
    history[0] = convCheck.defect()/convCheck.reduction() # Initial defect.
    for i in range(convCheck.step()):
        history[i] = convCheck.get_defect(i)
    print(ioname)
    return history



# In[ ]:


# Test solver.
resID = solverID + "-ref" + str(myRefs) + "lin"
history[resID] = SolverTest(domainDisc, approxSpace, linSolver, "Error_" + solverID)


# In[ ]:


def PlotResults(history, title):
    import matplotlib.pyplot as plt
    for key in history:
        plt.plot(range(history[key].size),history[key], label=key)

    plt.title(title)
    plt.xlabel("Step k")
    plt.yscale("log")
    plt.ylabel("Defect |b-Ax|")
    plt.legend()
    


# In[ ]:


PlotResults(history, "Gauss-Seidel")


# In[1]:


try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

try:  
    # Activate the following lines binder, colab etc.
    pyvista.start_xvfb()
    pyvista.set_jupyter_backend('static')

    # Activate locally, if trame exists.
    # pyvista.set_jupyter_backend('trame')
    
    result = pyvista.read( "Error_gs.vtu")
    result.plot(scalars="u", show_edges=True, cmap='jet')
except:
     print("Plotting failed.")


# In[ ]:





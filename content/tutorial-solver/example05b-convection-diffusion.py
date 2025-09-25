#!/usr/bin/env python
# coding: utf-8

# 
# # Multigrid for convection-diffusion problems
# 
# In this example, we consider the equation
# 
# $$
# - \frac{\partial^2 u}{\partial x^2} - \epsilon \frac{\partial^2 u}{\partial y^2} = f
# $$
# 
# 
# The scaling factor $\epsilon$ in diffusion ensures that the **approximation property** is destroyed. The following holds only:
# 
# $$ \|A_h^{-1} - p A_{H}^{-1} r \| \le C_A \frac{h^2}{\epsilon}
# $$
# For convergence to occur, a smoother must be used, whose **smoothing property** compensates for this accordingly. Thus, it should hold:
# $$ \|A_h S_h^{\nu}\| \le C_s \frac{\epsilon }{h^2} \eta(\nu)
# $$
# A method with this property is referred to as a **robust smoother**.
# 
# ## Initialization
# First, we start as usual:
# 

# In[23]:


import sys
sys.path.append("..")
import modsimtools as util


# In[24]:


import ug4py.pyugcore as ugcore
import ug4py.pyconvectiondiffusion as cd


# In[25]:


import json


# In[26]:


myconfig = { 
    "geometry": {
        "numRefs": 6,
        "gridName": "grids/laplace_sample_grid_2d.ugx",
        "requiredSubsets": ["Inner", "Boundary"]
    },
        
    "order":  None,
    
    "problem": {
        #"id": "ConvectionDiffusionProblem",
        #"parameters": {"cmp": "u", "eps": 1e-6, "velx": -1.0, "vely": 0.0, "upwind" : True}
        
        "id": "RecirculatingFlowProblem",
        "parameters": {"cmp": "u", "eps": 1e-6, "upwind" : True}
        
    }
}



# In[27]:


jconfig = json.loads(json.dumps(myconfig))
jgeometry= jconfig["geometry"]

jgeometry["gridName"]
jconfig["problem"]


# In[28]:


dom =  util.CreateDomain(jgeometry["gridName"], jgeometry["numRefs"], jgeometry["requiredSubsets"])
approxSpace = util.CreateApproximationSpace(dom, dict(fct = "u", type = "Lagrange", order = 1))


# ## Ordering of the unknowns
# The following command reorders the unknowns:

# In[29]:


ugcore.OrderLex(approxSpace, "y")    # order vertices (and dofs) lexicographically


# In[30]:


# import problem from configs

problem_desc = jconfig["problem"]

if problem_desc['id'] == 'ConvectionDiffusionProblem':
    print(problem_desc["parameters"])
    from config.ConvectionDiffusionProblem import ConvectionDiffusionProblem as Problem 

if problem_desc['id'] == 'RecirculatingFlowProblem':
    print(problem_desc["parameters"])
    from config.RecirculatingFlowProblem import RecirculatingFlowProblem as Problem 

# problem=getattr(Problem, problem_desc["id"])
problem = Problem(**problem_desc["parameters"])
#problem = Problem(**problem_desc["parameters"])

domainDisc = ugcore.DomainDiscretization2dCPU1(approxSpace)
problem.add_element_discretizations(domainDisc)
problem.add_boundary_conditions(domainDisc)


# ## Gauß-Seidel method
# 
# The Gauß-Seidel method converges. For $\epsilon \rightarrow 0$ it does not provide a **robust smoother** however:

# In[31]:


#solve problem with max 50 iterations, 
#reducing the error by 6 orders of magnitude  
convCheck = ugcore.ConvCheckCPU1()
convCheck.set_maximum_steps(30)
convCheck.set_reduction(1e-6)
convCheck.set_minimum_defect(1e-15)
convCheck.set_supress_unsuccessful(True)


solver = ugcore.LinearSolverCPU1()
solver.set_convergence_check(convCheck)

# dbgWriter = ugcore.GridFunctionDebugWriter2dCPU1(approxSpace)
# solver:set_debug(dbgWriter)


# # Gauß-Seidel

# In[32]:


import toolbox as tools


# In[33]:


gs = ugcore.GaussSeidelCPU1()
solver.set_preconditioner(gs)
tools.SolverTest(domainDisc, approxSpace, solver, convCheck, "ConvDiff_Error_GS.vtk");


# In[34]:


import pyvista
pyvista.set_jupyter_backend('trame')

result = pyvista.read("ConvDiff_Error_GS.vtu")
result.plot(scalars="u", show_edges=True, cmap='jet')


# # Symmetric Gauß-Seidel

# In[ ]:


sgs = ugcore.SymmetricGaussSeidelCPU1()
solver.set_preconditioner(sgs)
tools.SolverTest(domainDisc, approxSpace, solver, convCheck, "ConvDiff_Error_SGS.vtk");


# In[ ]:


result = pyvista.read("ConvDiff_Error_SGS.vtu")
result.plot(scalars="u", show_edges=True, cmap='jet')


# ## ILU method
# The ILU scheme is robust. For the limit case $\epsilon \rightarrow 0$ an exact solver is obtained:

# In[ ]:


ilu =ugcore.ILUCPU1()
solver.set_preconditioner(ilu)
tools.SolverTest(domainDisc, approxSpace, solver, convCheck,"ConvDiff_Error_ILU.vtk");


# In[ ]:


result = pyvista.read("ConvDiff_Error_ILU.vtu")
result.plot(scalars="u", show_edges=True, cmap='jet')


# ## Mehrgitterverfahren
# 

# In[ ]:


mgConfig = {
    "type" : "linear",  # linear solver type ["bicgstab", "cg", "linear"]
    "precond" : 
    {
        "type"        : "gmg",    # preconditioner ["gmg", "ilu", "ilut", "jac", "gs", "sgs"]
        
        "smoother"    : "gs",     # gmg-smoother ["ilu", "ilut", "jac", "gs", "sgs"]
        "cycle"       : "V",      # gmg-cycle ["V", "F", "W"]
        "preSmooth"   : 1,        # number presmoothing steps
        "postSmooth"  : 1,        # number postsmoothing steps
        "rap"         : False,    # computes RAP-product instead of assembling if true 
        
        "baseLevel"   : 0,        # gmg - baselevel
        "baseSolver"  : "lu",
    },

    "convCheck" : "standard"
}


# In[ ]:


def CreateMultiGridSolver(approxSpace, domainDisc, mysmoother) :
    mg = ugcore.GeometricMultiGrid2dCPU1(approxSpace)  # Konstruktor
    mg.set_discretization(domainDisc)
    mg.set_base_level(0)          # Multi-grid

    # Configuration of the smoother.
    mg.set_smoother(mysmoother)                  
    mg.set_num_presmooth(1)   # Vorglaettung
    mg.set_num_postsmooth(1)  #  Nachglaettung
    mg.set_rap(False)          # fuer Galerkinprodukt A_H=RAP, if true (alternative: assemble coarse system)

    # Konfiguration Grobgitterloeser
    baseSolver = ugcore.LUCPU1()
    mg.set_base_level(0)          # Multi-grid
    mg.set_base_solver(baseSolver)     

    mg.set_cycle_type("V")       # Select cycle type "V,W,F".
    return mg


# In[ ]:


mg = CreateMultiGridSolver(approxSpace, domainDisc,gs)
solver.set_preconditioner(mg)
tools.SolverTest(domainDisc, approxSpace, solver, convCheck,"Error_GMG.vtk");


# Wir testen, indem wir eine Nullfolge $\epsilon_k$ betrachten.

# In[ ]:


def robustness_test(solver, conv_check, eps_start=1.0, eps_stop=1e-6, eps_factor=3.16227766016):

    myeps=[]
    rates =[]
    # Print solver config.
    print(solver.config_string())
    
    # Set initial values for the loop
    eps = eps_start
    
    while eps >= eps_stop:
        # Update the configuration parameter
        myconfig["problem"]["parameters"]['eps'] = eps
        print(f"1/eps={1.0 / myconfig["problem"]["parameters"]['eps']}")

        # Initialize variables to store results
        my_residuals = None
        my_rate = None
        my_steps = None

        # Create domain discretization and run the solver test
        print(myconfig["problem"]["parameters"])
        problem = Problem(**myconfig["problem"]["parameters"])

        domainDisc = ugcore.DomainDiscretization2dCPU1(approxSpace)
        problem.add_element_discretizations(domainDisc)
        problem.add_boundary_conditions(domainDisc)
       
        my_residuals = tools.SolverTest(domainDisc, approxSpace, solver, convCheck, "ErrorMGV")
        my_rate=convCheck.avg_rate()
        my_steps=convCheck.step()
        
        # Store the results
        myeps.append(1.0 / myconfig["problem"]["parameters"]['eps'])
        rates.append(my_rate)
        print("=========")

        # Update epsy for the next iteration
        eps /= eps_factor

        print(myeps)
        print(rates)
    return myeps, rates


# In[ ]:


eps = []
rates_gs=[]
mg = CreateMultiGridSolver(approxSpace, domainDisc,gs)
solver.set_preconditioner(mg)
eps, rates_gs = robustness_test(solver, convCheck, 1.0, 1e-8, 3.16227766016)


# In[ ]:


eps = []
rates_sgs=[]
mg = CreateMultiGridSolver(approxSpace, domainDisc,sgs)
solver.set_preconditioner(mg)
eps, rates_sgs = robustness_test(solver, convCheck, 1.0, 1e-8, 3.16227766016)


# In[ ]:


eps = []
rates_ilu=[]
mg = CreateMultiGridSolver(approxSpace, domainDisc,ilu)
solver.set_preconditioner(mg)
eps, rates_ilu = robustness_test(solver, convCheck, 1.0, 1e-8, 3.16227766016)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(eps, rates_gs, label="GS")
plt.plot(eps, rates_sgs, label="SGS")
plt.plot(eps, rates_ilu, label="ILU")

plt.title("Convergence rate multigrid")
plt.xscale("log")
plt.xlabel("1/eps")

plt.ylabel("Avg. rate ")
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





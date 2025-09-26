#!/usr/bin/env python
# coding: utf-8

# 
# # Mehrgitter für singulär gestörte Probleme
# 
# Hier betrachten wir die Gleichung
# 
# ## Initialisierung
# Zunächst starten wir wie gehabt:
# 

# In[1]:


-- Initialize UG.
InitUG(2, AlgebraType("CPU", 1));

-- Load convenience scripts.
ug_load_script("../modsim2/toolbox.lua")

history = {}


# In[2]:


configDesc = {
    geometry = {
        numRefs = 6, 
        gridName = "../grids/laplace_sample_grid_2d-tri.ugx", -- grid name
        requiredSubsets ={"Inner", "Boundary"},
    },
    
    order = nil, -- Lex
   
    problem = {
        id = "PoissonProblem",
        parameters = {
            epsx = 1.0,
            epsy = 1.0, 
        }   
    }
}

-- Create discretization.
domainDisc, approxSpace = util.modsim2.CreateDomainDisc(configDesc)

OrderLex(approxSpace, "x")    -- order vertices (and dofs) lexicographically


# ## Gauß-Seidel-Verfahren

# In[3]:


-- solve problem with max 50 iterations, 
-- reducing the error by 6 orders of magnitude  
convCheck = ConvCheck()
convCheck:set_maximum_steps(50)
convCheck:set_reduction(1e-6)
convCheck:set_minimum_defect(1e-15)

solver = LinearSolver()
solver:set_preconditioner(GaussSeidel())
solver:set_convergence_check(convCheck)

dbgWriter = GridFunctionDebugWriter(approxSpace)
solver:set_debug(dbgWriter)


# In[4]:


solver:set_preconditioner(GaussSeidel())
util.modsim2.SolverTest(domainDisc, approxSpace, solver, "SmootherError_GS.vtk");


# In[5]:


solver:set_preconditioner(ILU())
util.modsim2.SolverTest(domainDisc, approxSpace, solver, "SmootherError_ILU.vtk");


# ## Mehrgitterverfahren
# 
# Obwohl die Konvergenzrate des Zweigitterverfahrens $h$-unabhängig ist, limitiert letztlich die Größe des  Grobgitterproblems immer noch die Komplexität. Ersetzt man dies durch eine rekursive Approximation, ergibt sich das **Mehrgitterverfahren**:

# In[10]:


local myBaseLevel = 1
-- local smoother = SymmetricGaussSeidel()
local smoother = ILU()
local nu =1
-- Konstruktion (s.o., aber beachte baseLevel!)
mgv = GeometricMultiGrid(approxSpace)  -- Konstruktor
mgv:set_discretization(domainDisc)
mgv:set_base_level(0)          -- Mehrgitterverfahren!

-- Konfiguration des Glaetters
mgv:set_smoother(smoother)                  
mgv:set_num_presmooth(nu)  -- Vorglaettung
mgv:set_num_postsmooth(nu)  -- Nachglaettung
mgv:set_rap(true)          -- fuer Galerkinprodukt A_H=RAP, if true (alternative: assemble coarse system)

-- Konfiguration Grobgitterloeser
baseSolver = LU()
mgv:set_base_level(myBaseLevel)          -- Zweigitterverfahren
mgv:set_base_solver(baseSolver)     

mgv:set_cycle_type("V")       -- Select cycle type "V,W,F".


-- dbgWriter = GridFunctionDebugWriter(approxSpace)
get_ipython().run_line_magic('pinfo', 'REQ')
solver:set_preconditioner(mgv)


# Auch hiermit können wir jetzt testen:

# In[11]:


historyMGV = {}
configDesc.problem.parameters.epsy =1 
for nu=1,6 do
    print("1/eps="..1/configDesc.problem.parameters.epsy )
  
    local domainDisc=util.modsim2.CreateDiscretization(approxSpace,configDesc)
    historyMGV[nu] = util.modsim2.SolverTest(domainDisc, approxSpace, solver, "ErrorMGV")
    print()
   
    print()
    print("=========")
    configDesc.problem.parameters.epsy = configDesc.problem.parameters.epsy/10
end


# In[9]:


dataTab = util.modsim2.CreateGnuplotDataTable(historyMGV)
plots = {
   { label = "MGV, eps=1",  data = dataTab[1], style = "linespoints", 1, 2}, 
   { label = "eps=10^-1",  data = dataTab[2], style = "linespoints", 1, 2}, 
   { label = "eps=10^-2",  data = dataTab[3], style = "linespoints", 1, 2}, 
   { label = "eps=10^-3",  data = dataTab[4], style = "linespoints", 1, 2}, 
   { label = "eps=10^-4",  data = dataTab[4], style = "linespoints", 1, 2}, 
   { label = "eps=10^-5",  data = dataTab[4], style = "linespoints", 1, 2}, 
}

options = {
    title = "Konvergenz MGV", 
    label = {x = "Iteration k", y = "Defekt |b-Ax^k|"},
    logscale = {x = false, y = true},
    terminal = "x11"
}

gnuplot.plot("Konvergenz_MGV.png", plots, options)


# In[ ]:





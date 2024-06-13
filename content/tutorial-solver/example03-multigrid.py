#!/usr/bin/env python
# coding: utf-8

# [<img src="../../header.svg">](../index.ipynb)
# 
# ---
# 
# # Multigrid method

# 
# Obwohl die Konvergenzrate des Zweigitterverfahrens $h$-unabhängig ist, limitiert letztlich die Größe des  Grobgitterproblems immer noch die Komplexität. Ersetzt man dies durch eine rekursive Approximation, ergibt sich das **Mehrgitterverfahren**:
# 
# 
# 
# <ins>Algorithmus:</ins> `MGV`($A_l, x_l, b_l$)
# * Falls $l=0$:
# >
# > `Löse` $A_0 x_0 = b_0$;
# >
# > `return` $x_0$;
# 
# * Vorglättung:
# >for i=1:$\nu_1$ do
# >
# >    $\hspace{3em}x_l := x_l + M_l^{-1}(b_l-A_l x_l)$ 
# >
# > end 
# 
# * **NEU:** _Approximative_ Grobgitterkorrektur (rekursiv!) 
# > $d_{l-1} := r (b_l-A_l x_l); $ 
# >
# > $e_{l-1} := 0;$ 
# >
# > for i=1:$\gamma$ do
# >
# > $\hspace{3em}$ e_l:=`MGV`($A_{l-1}, e_{l-1}, d_{l-1}$)
# >
# > end
# >
# > $x_l := x_l + p e_{l-1}$ 
# 
# * Nachglättung (optional, aus Symmetriegründen):
# >for i=1:$\nu_2$ do
# >
# >    $\hspace{3em}x_l := x_l + M_l^{-1}(b_l-A_h x_l)$ 
# >
# > end 
# 
# * Rückgabe:
# > `return` $x_l$;
# 

# ## Configuration

# In[23]:


get_ipython().run_line_magic('run', 'example01-smoothers.ipynb')


# In[24]:


myBaseLevel = 1

mg = ugcore.GeometricMultiGrid2dCPU1(approxSpace)  # Konstruktor
mg.set_discretization(domainDisc)
mg.set_base_level(0)          # Multi-grid

# Configuration of the smoother.
mg.set_smoother(preconditioners["jac"])                  
mg.set_num_presmooth(1)   # Vorglaettung
mg.set_num_postsmooth(1)  #  Nachglaettung
mg.set_rap(True)          # fuer Galerkinprodukt A_H=RAP, if true (alternative: assemble coarse system)

# Konfiguration Grobgitterloeser
baseSolver = ugcore.LUCPU1()
mg.set_base_level(myBaseLevel)          # Multi-grid
mg.set_base_solver(baseSolver)     

mg.set_cycle_type(1)       # Select cycle type "V,W,F".


# In[25]:


# dbgWriter = GridFunctionDebugWriter(approxSpace)
# mgv:set_debug(dbgWriter) -- REQ?
linSolver.set_preconditioner(mg)


# Auch hiermit können wir jetzt testen:

# In[26]:


historyMGV = {}

for nu in range(1,6):
    mg.set_num_presmooth(nu)
    historyMGV[nu] = SolverTest(domainDisc, approxSpace, linSolver, "ErrorMGV")
    print()
    print("1/nu=" + str(1.0/nu))
    print()
    print("=========")


# In[ ]:





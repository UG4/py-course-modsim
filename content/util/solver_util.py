
import sys
sys.path.append("..")
import modsimtools as util

import ug4py.pyugcore as ug4
import ug4py.pylimex as limex
import ug4py.pyconvectiondiffusion as cd
# import pysuperlu as slu

# dbgWriter = ug4.GridFunctionDebugWriter2dCPU1(approxSpace)

# Convergence checks
convCheck = ug4.ConvCheckCPU1()
convCheck.set_maximum_steps(50)
convCheck.set_reduction(1e-12)
convCheck.set_minimum_defect(1e-15)
    
baseConvCheck = ug4.ConvCheckCPU1()
baseConvCheck.set_maximum_steps(5000)
baseConvCheck.set_reduction(1e-12)
baseConvCheck.set_verbose(False)

# select smoother 
jac = ug4.JacobiCPU1(0.66)             # Jacobi (w/ damping)
gs  = ug4.GaussSeidelCPU1()           # Gauss-Seidel (forward only)
sgs = ug4.SymmetricGaussSeidelCPU1()  # Symmetric Gauss-Seidel (forward + backward)
ilu = ug4.ILUCPU1()                   # ILU (w/ pattern based)

# base solvers
baseSolver = ug4.BiCGStabCPU1()
baseSolver.set_preconditioner(jac)
baseSolver.set_convergence_check(baseConvCheck)

#baseSolver = slu.SuperLUCPU1()

baseSolver = ug4.LUCPU1()



def CreateMultigridPrecond(approxSpace, domainDisc, cycle, nu1, nu2) :
    #Geometric Multi Grid
    gmg = ug4.GeometricMultiGrid2dCPU1(approxSpace)
    gmg.set_discretization(domainDisc)
    # gmg:set_rap(True)                   # use Galerkin A_H=RAP, if true (alternative: assemble coarse system)


    gmg.set_base_level(0)        # select level for coarsest grid
    gmg.set_base_solver(baseSolver)          # select base solver

    gmg.set_smoother(sgs)      # select smoother
    gmg.set_cycle_type(cycle)    # select cycle type "V,W,F"
    gmg.set_num_presmooth(nu1)   # select nu1
    gmg.set_num_postsmooth(nu2)  # select nu2
    
    return gmg

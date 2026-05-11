
# (Navier-)Stokes test problem on the unit square using dirichlet conditions, cf.
#
# 	Nigon, P., Une nouvelle classe de methodes multigrilles pour 
#					les problemes mixtes, E.C.L. 84-19. Lyon 1984
#	Wittum, G., Multi-Grid Methods for Stokes and Navier-Stokes Equations,
# 				Numer. Math. 54, 543-563, 1989


# System imports.
import sys
import math 
print("✓ System libraries loaded (sys, random, math)")

# Own imports.
sys.path.append("..")

# Some users may also want to specify their own dir for ug4py
# sys.path =["$UG4_ROOT/ug4-git/bin/plugins/"] + sys.path 

print("✓ sys.path updated")
print("  sys.path:", sys.path)

# UG4 imports.
import ug4py.pyugcore as ug4
print("✓ Loaded ug4py.pyugcore")

import ug4py.pynavierstokes as ns
print("✓ Loaded ug4py.pynavierstokes")

# Try loading pysuperlu (optional)
slu = None
try:
    import ug4py.pysuperlu as slu
    print("✓ Loaded ug4py.pysuperlu")
except ImportError:
    print("x Failed ug4py.pysuperlu not available")

import modsimtools as util
print("✓ Loaded modsimtools")

 
#-----------------------------------------
#-- A) Model parameters
#-----------------------------------------

type = "fe"
uorder = 2 
porder = 1
velCmp = "u, v"
fctCmp = velCmp + ", p"

#-----------------------------------------
#-- FE ansatz spaces
#-----------------------------------------
def CreateApproxSpace(dom, velCmp, uorder, porder):
     # Create approximation space
     approxSpace = ug4.ApproximationSpace2d(dom)
     approxSpace.add_fct(velCmp, "Lagrange", uorder)   # lineare Ansatzfunktionen
     approxSpace.add_fct("p", "Lagrange", porder)
     approxSpace.init_levels()
     approxSpace.init_top_surface()

     print("approximation space:")
     approxSpace.print_statistic()
     return approxSpace


# -------------------------------------------------------
# -- User defined terms (e.g. for boundary & sources.)
# -------------------------------------------------------
def mySource_(x, y, t, si): 
	return [36 * math.sin(3*(x+y)), 0.0]				
myUserSource =ug4.PythonUserVector2d(mySource_)

# Exact solution of the problem
def exactSolU2d(x, y, t, si):
    return math.sin(3*(x+y)) 
mySolutionU =ug4.PythonUserNumber2d(exactSolU2d)

def exactSolV2d(x, y, t, si):
    return -math.sin(3*(x+y)) 
mySolutionV =ug4.PythonUserNumber2d(exactSolV2d)

def exactSolP2d(x, y, t, si): 
    return -6*math.cos(3*(x+y))	
mySolutionP =ug4.PythonUserNumber2d(exactSolP2d)

def exactSolVel2d(x, y, t, si):
	return [exactSolU2d(x,y,t,si), exactSolV2d(x,y,t,si)]
mySolutionVel =ug4.PythonUserVector2d(exactSolVel2d)
print("✓ Defined python callbacks for source and exact solution")




#-----------------------------------------
#-- Domain discretization
#-----------------------------------------
def CreateDomainDisc(approxSpace, fctCmp, uorder, porder, type):   
    domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)  
    
    # create NavierStokes disc
    elemDisc= ns.NavierStokesFE2d(fctCmp, "Inner")
    elemDisc.set_exact_jacobian(True)
    elemDisc.set_stokes(True)
    elemDisc.set_laplace(True)
    elemDisc.set_kinematic_viscosity(1)
    elemDisc.set_source(myUserSource)

    # FEM must be stabilized for (Pk, Pk) space
    if (type == "fe") and (porder == uorder):
	    elemDisc.set_stabilization(3)
    
    #fixPressureDisc = ug4.DirichletBoundary2dCPU1()
    #fixPressureDisc.add(0, "p", "Boundary, PressureNode")

    bndDisc = ns.NavierStokesInflowFE2dCPU1(elemDisc)
    bndDisc.add(mySolutionVel, "Boundary, PressureNode")

    domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)
    domainDisc.add(elemDisc)
    # domainDisc.add(fixPressureDisc)
    domainDisc.add(bndDisc)

    return domainDisc

#-----------------------------------------
#-- Define test.
#-----------------------------------------
def test_nigon(numRefs, lsolver):  
    
    # Create domain.
    dom = util.CreateDomain("nigon.ugx",numRefs, requiredSubsets=["Inner", "Boundary"])
    approxSpace = CreateApproxSpace(dom, velCmp, uorder, porder)
    domainDisc = CreateDomainDisc(approxSpace, fctCmp, uorder, porder, type)

    # Solve Problem.    
    A = ug4.MatrixOperatorCPU1()
    u = ug4.GridFunction2dCPU1(approxSpace)
    b = ug4.GridFunction2dCPU1(approxSpace)

    domainDisc.assemble_linear(A, b)
    domainDisc.adjust_solution(u)

    lsolver.init(A, u)
    lsolver.apply(u,b)

    # Compute error.
    errU=ug4.L2Error(mySolutionU, u, "u", 0.0, 4, "Inner")
    errV=ug4.L2Error(mySolutionV, u, "v", 0.0, 4, "Inner")
    print( "Error: ", [errU, errV] )
    return [errU, errV]
            
#-----------------------------------------
#-- Execute tests
#-----------------------------------------
if __name__ == "__main__":
    test_nigon(numRefs=2, lsolver=ug4.LUCPU1())
    test_nigon(numRefs=3, lsolver=ug4.LUCPU1())
    if (slu is not None):   
        test_nigon(numRefs=4, lsolver=slu.SuperLUCPU1())


# TODO: Test multilevel solvers...

# Upscaling of Stokes flow in a porous medium.
# The example is based on the pressure-drop approach.

# References:

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


#-----------------------------------------
#-- Domain discretization
#-----------------------------------------
 # create NavierStokes FEM disc
def CreateDomainDisc(approxSpace, fctCmp, uorder, porder, type, g):   
      
    elemDisc= ns.NavierStokesFE2d(fctCmp, "Inner")
    elemDisc.set_exact_jacobian(True)
    elemDisc.set_stokes(True)
    elemDisc.set_laplace(True)
    elemDisc.set_kinematic_viscosity(1.0)
    myGrad=ug4.ConstUserVector2d()
    myGrad.set_entry(0, g[0])
    myGrad.set_entry(1, g[1])
    elemDisc.set_source(myGrad)

    # FEM must be stabilized for (Pk, Pk) space
    if (type == "fe") and (porder == uorder):
	    elemDisc.set_stabilization(3)

    domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)
    domainDisc.add(elemDisc)
  
    return domainDisc

def CreateDirichletBC(bndDicts):
    pressureBnd = ug4.DirichletBoundary2dCPU1() 
    for bnd in bndDicts: 
        print(bnd)    
        pressureBnd.add(bnd["value"], bnd["cmp"], bnd["subset"])
    return pressureBnd

#-----------------------------------------
#-- Define test.
#-----------------------------------------
def test_upscale(numRefs, lsolver, gdir=[1.0, 0.0], filename=None):  

    # Create domain.
    dom = util.CreateDomain("upscale.ugx", numRefs, requiredSubsets=["Inner"])
    approxSpace = CreateApproxSpace(dom, velCmp, uorder, porder)
    domainDisc = CreateDomainDisc(approxSpace, fctCmp, uorder, porder, type, gdir)
    #domainDisc.add(CreateDirichletBC(bndDesc))

    # No slip boundary conditions on grains.
    noSlipDesc =  [
        {"value": 0.0, "cmp": "u", "subset": "Grain"},
        {"value": 0.0, "cmp": "v", "subset": "Grain"}
    ]       
    domainDisc.add(CreateDirichletBC(noSlipDesc))

    # Fix pressure at one point to avoid singularity.
    fixPDesc =  [
        {"value": 0.0, "cmp": "p", "subset": "Vertex_SE"}
    ] 
    domainDisc.add(CreateDirichletBC(fixPDesc))

    # Solve Problem.    
    A = ug4.MatrixOperatorCPU1()
    u = ug4.GridFunction2dCPU1(approxSpace)
    b = ug4.GridFunction2dCPU1(approxSpace)

    import traceback
    try:
        domainDisc.assemble_linear(A, b)
        domainDisc.adjust_solution(u)
    except Exception as e:
        traceback.print_exc()
        print("Exception during assembly: ", e)
        raise
        
    try:
        lsolver.init(A, u)
        lsolver.apply(u,b)
    except Exception as e:
        traceback.print_exc()
        print("Exception during solution: ", e)
        raise
    
    if (filename):
        # Print solution as vtk.
        ug4.WriteGridFunctionToVTK(u, filename)

    # Compute averages.
    area = ug4.Integral(1.0, u, "Inner", 4)
    avgU=ug4.Integral(u, "u", "Inner", 4)/area
    avgV=ug4.Integral(u, "v", "Inner", 4)/area

    print( "Average: ", [avgU, avgV] )
    return [avgU, avgV]
            
#-----------------------------------------------------
#-- Execute tests (if this module is called directly)
#-----------------------------------------------------

if __name__ == "__main__":
    # test_upscale(numRefs=2, lsolver=ug4.LUCPU1(), gdir=[1.0, 0.0], filename="upscale_gradx_2.vtu")
    test_upscale(numRefs=3, lsolver=ug4.LUCPU1(), gdir=[1.0, 0.0], filename="upscale_gradx_3.vtu")
    test_upscale(numRefs=3, lsolver=ug4.LUCPU1(), gdir=[0.0, 1.0], filename="upscale_grady_3.vtu")
    if (slu is not None):   
        test_upscale(numRefs=4, lsolver=slu.SuperLUCPU1())

# TODO: Test multilevel solvers...
#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.append("..")
print(sys.path)



import ug4py.pyugcore as ugcore
#import ug4py.pyconvectiondiffusion as cd
import ug4py.pysmallstrainmechanics as mech
import ug4py.pyutil as pyutil


import math  # For sqrt, sin, exp, ...
import modsimtools as util


class units:
    MPa = 1e+6  # Mega Pascal
    GPa = 1e+9  # Giga Pascal



ptop = 2.0*units.MPa
pside = 2.0*units.MPa

def cylinder_normal_x(x, y, z, t, si):
    """Return the x component of the outward unit normal on the cylinder shell."""
    radius = math.hypot(x, y)
    if radius == 0.0:
        raise ValueError("The cylinder normal is undefined on its axis.")
    return pside*x / radius

def cylinder_normal_y(x, y, z, t, si):
    """Return the y component of the outward unit normal on the cylinder shell."""
    radius = math.hypot(x, y)
    if radius == 0.0:
        raise ValueError("The cylinder normal is undefined on its axis.")
    return pside*y/ radius


# ## Problem definitions
class MyProblem:
    # Geometrie
    gridName= "cylinder_small.ugx" 
    requiredSubsets = {"Inner", "Shell", "Top", "Bottom", "BottomRing", "TopRing"}
    
    numRefs= 1
    
    # Constructor
    def __init__(self, E, nu):
        self.EYoung = E
        self.nu = nu

        matLaw = mech.HookeLaw3d()
        matLaw.set_hooke_elasticity_tensor_E_nu(self.EYoung, self.nu)
        
        elemDisc = mech.SmallStrainMechanics3d("ux, uy, uz", "Inner")
        elemDisc.set_quad_order(2)  # TODO: must be increased for higher order elements.

        elemDisc.set_material_law(matLaw)
        # elemDisc:set_volume_forces(volumeForceField)
        elemDisc.set_mass_scale(0.0)

        self.elemDisc = elemDisc

        self.dirichletBND = ugcore.DirichletBoundary3dCPU1()
      
         # Fix z deformation at the bottom of the cylinder.
        #self.dirichletBND.add(0.0, "uz", "Bottom") 

        # Avoid rigid body motion by fixing the bottom pole in all directions.
        self.dirichletBND.add(0.0, "ux", "Bottom, BottomPole") 
        self.dirichletBND.add(0.0, "uy", "Bottom, BottomPole") 
        self.dirichletBND.add(0.0, "uz", "Bottom, BottomPole") 


        # Add Neumann boundary condition, e.g., for compressions of the side walls 
        self.neumannBNDX = ugcore.NeumannBoundaryFE3d("ux")
        self.neumannBNDY = ugcore.NeumannBoundaryFE3d("uy")
        self.neumannBNDZ = ugcore.NeumannBoundaryFE3d("uz")
        #self.neumannBNDX.add(1000.0, "ux", "Shell")  # Apply a force of 1000 N in the x-direction on the top ring of the cylinder.
        
       
        #self.neumannBNDY.add(0.0, "uy", "Shell")  # Apply a force of 0 N in the y-direction on the top ring of the cylinder.
            
        # Compress cylinder by force:
        if True:
            self.neumannBNDX.add(0.0, "Top", "Inner")  # Apply a force of 0 N in the z-direction on the top ring of the cylinder.
            self.neumannBNDY.add(0.0, "Top", "Inner")
            self.neumannBNDZ.add(ptop, "Top", "Inner")
        else:
            # Compress (negative value) the top of the cylinder by a Dirichlet boundary condition:
            self.dirichletBND.add(-0.1, "uz", "Top")  # Cylinder
       

        self.neumannBNDX.add(ugcore.PythonUserNumber3d(cylinder_normal_x), "Shell", "Inner")  # Apply a force of 0 N in the z-direction on the top ring of the cylinder.
        self.neumannBNDY.add(ugcore.PythonUserNumber3d(cylinder_normal_y), "Shell", "Inner")
        self.neumannBNDZ.add(0.0, "Shell", "Inner")
    
    # API function
    def CreateDomainDisc(self,approxSpace):   
        domainDisc = ugcore.DomainDiscretization3dCPU1(approxSpace)  
        domainDisc.add(self.elemDisc)
        domainDisc.add(self.dirichletBND)

        # Neumann boundaries are optional.
        if self.neumannBNDX is not None:
            domainDisc.add(self.neumannBNDX)
        
        if self.neumannBNDY is not None:
            domainDisc.add(self.neumannBNDY)
        
        if self.neumannBNDZ is not None:
            domainDisc.add(self.neumannBNDZ)
            
        return domainDisc
    
    # Callback fuer rechte Seite
    def SomeCallback(self, x, y, t, si):
        return 47.11


problem = MyProblem(E=1e+9, nu=0.3) # E in GPa

result = ", ".join(problem.requiredSubsets)
print(result)
dom = pyutil.CreateDomain3d(problem.gridName, problem.numRefs, )

approxSpace = ugcore.ApproximationSpace3d(dom)
approxSpace.add_fct("ux", "Lagrange", 1)
approxSpace.add_fct("uy", "Lagrange", 1)
approxSpace.add_fct("uz", "Lagrange", 1)
approxSpace.init_levels()
approxSpace.init_top_surface()

domainDisc = problem.CreateDomainDisc(approxSpace)


# ### Assemble and solve linear system (as usual)
Ah = ugcore.AssembledLinearOperatorCPU1(domainDisc)
uh = ugcore.GridFunction3dCPU1(approxSpace)
bh = ugcore.GridFunction3dCPU1(approxSpace)

import traceback
try:
    domainDisc.assemble_linear(Ah, bh)
    domainDisc.adjust_solution(uh)
except Exception as inst:
    traceback.print_exc()
    print(str(inst))



# Create multigrid solver.
sys.path.append("..")
import util.solver_util as util_solver

gmg = util_solver.CreateMultigridPrecond3D(approxSpace, domainDisc, "V",2,2) # TODO: GMG requires a block solver.
gmg.set_discretization(domainDisc)

solver = ugcore.LinearSolverCPU1()
solver.set_preconditioner(gmg)
solver.set_convergence_check(util_solver.convCheck)

# Solve the linear system
solver = ugcore.LUCPU1()

# In[124]:


try:
    solver.init(Ah, uh)
    solver.apply(uh, bh)
except Exception as inst:
    traceback.print_exc()
    print(inst)
    


# In[125]:


ugcore.WriteGridFunctionToVTK(uh, "cylinder.vtu")

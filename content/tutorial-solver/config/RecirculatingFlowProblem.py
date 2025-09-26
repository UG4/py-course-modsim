import ug4py.pyugcore as ugcore
import ug4py.pyconvectiondiffusion as cd
import time

class RecirculatingFlowProblem:

    eps = 1.0
    upwind = ugcore.NoUpwind2d()

    @staticmethod
    def create_elem_disc(fct) : 
        return cd.ConvectionDiffusionFV12d(fct, "Inner")

    def __init__(self, cmp, eps, upwind):
        self.cmp = cmp
        self.eps = eps
        if upwind:
            self.upwind = ugcore.FullUpwind2d()
        else:
            self.upwind = ugcore.NoUpwind2d()

    def dirichlet_values(self, x, y, t):
        from math import cos, pi
        s = 1.0*pi
        return cos(s*x) + cos(s*y)
    
    def velocity_field(self, x, y, t, si):
        print(x)
        print(y)
        time.sleep(10)
        urecirc = 4.0*x*(1-x)*(1-2*y) 
        vrecirc = -4.0*y*(1-y)*(1-2*x)
        return [urecirc, vrecirc]

    # Element discretization
    def add_element_discretizations(self,domainDisc):

        elemDisc = self.create_elem_disc(self.cmp)
        elemDisc.set_diffusion(self.eps)

        pyVelocity = lambda x,y,t,si : self.velocity_field(x,y,t,si)
        myVelocity =  ugcore.PythonUserVector2d(pyVelocity)
        elemDisc.set_velocity(myVelocity)
  
        if (self.upwind): 
            elemDisc.set_upwind(self.upwind)
            
        domainDisc.add(elemDisc)

    # Boundary conditions
    def add_boundary_conditions(self,domainDisc):
        dirichletBND = ugcore.DirichletBoundary2dCPU1()
        pyBoundary = lambda x, y, t : self.dirichlet_values(x,y,t)
        #ugcore.PythonUserNumber2d(myBoundary),
        dirichletBND.add(ugcore.ConstUserNumber2d(0.0),self.cmp, "Boundary")
        domainDisc.add(dirichletBND)



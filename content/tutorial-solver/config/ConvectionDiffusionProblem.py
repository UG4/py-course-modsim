import ug4py.pyugcore as ugcore
import ug4py.pyconvectiondiffusion as cd

class ConvectionDiffusionProblem:

    eps = 1.0
    velx = 0.0
    vely = 0.0
    upwind = ugcore.NoUpwind2d()

    

    @staticmethod
    def create_elem_disc(fct) : 
        return cd.ConvectionDiffusionFV12d(fct, "Inner")

    def __init__(self,desc):
        self.eps = desc["eps"]
        self.velx = desc["velx"]
        self.vely = desc["vely"]
        if "upwind" in desc:
            self.upwind = desc["upwind"]
    

    def dirichlet_values(self, x, y, t):
        from math import cos, pi
        s = 1.0*pi
        return cos(s*x) + cos(s*y)
    
    def velocity_field(self, x, y, t):
        return  self.velx, self.vely

    # Element discretization
    def add_element_discretizations(self,domainDisc, fct):
        elemDisc = self.create_elem_disc(fct)
        elemDisc.set_diffusion(self.eps)
        myVelocity = lambda x,y,t : self.velocity_field(x,y,t)
        myVelocity =  ugcore.ConstUserVector2d()
        myVelocity:set_entry(0, self.velx)
        myVelocity:set_entry(0, self.vely)
        elemDisc.set_velocity(
            myVelocity
            #ugcore.PythonUserNumber2d(myVelocity)
        )
  
        if (self.upwind): 
            elemDisc.set_upwind(self.upwind)
        domainDisc.add(elemDisc)

    # Boundary conditions
    def add_boundary_conditions(self,domainDisc,fct):
        dirichletBND = ugcore.DirichletBoundary2dCPU1()
        myBoundary = lambda x, y, t : self.dirichlet_values(x,y,t)
        dirichletBND.add(
            ugcore.ConstUserNumber2d(0.0),
            #ugcore.PythonUserNumber2d(myBoundary),
            "u", "Boundary")
        domainDisc.add(dirichletBND)



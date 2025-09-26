import math

import ug4py.pyugcore as ugcore
import ug4py.pyconvectiondiffusion as cd

class PoissonProblem:
    def __init__(self, cmp, epsx=1.0, epsy=1.0, mu = 2, nu = 4):
        self.cmp = cmp
        self.epsx = epsx
        self.epsy = epsy
        self.mu = mu
        self.nu = nu
       
        
    #@staticmethod
    def poisson_initial_error_2d(x, y, t):
        fx = self.mu * math.pi
        fy = self.nu * math.pi
        return math.sin(fx * x) * math.sin(fy * y)

    #@staticmethod
    def poisson_source_2d(x, y, t):
        s = 2 * math.pi
        print(s)
        return s * s * (math.sin(s * x) + math.sin(s * y))

    #@staticmethod
    def poisson_dirichlet_value_2d(x, y, t):
        s = 2 * math.pi
        return True, math.sin(s * x) + math.sin(s * y)

    def set_initial_values(self, u):
        ugcore.Interpolate(ugcore.PythonUserNumber2d(lambda x,y,t : self.poisson_initial_error_2d(x,y,t)), u, self.cmp)

    def add_element_discretizations(self, domain_disc):
        # Anisotropic diffusion matrix
        print(f"epsx={self.epsx}")
        print(f"epsy={self.epsy}")

        anisotropic_diffusion = ugcore.ConstUserMatrix2d(0.0)
        anisotropic_diffusion.set_entry(0, 0, self.epsx)
        anisotropic_diffusion.set_entry(1, 1, self.epsy)

        elem_disc = cd.ConvectionDiffusionFE2d(self.cmp, "Inner")
        elem_disc.set_diffusion(anisotropic_diffusion)

         # TODO: This throws an exception!
        #elem_disc.set_source(ugcore.PythonUserNumber2d(lambda x,y,t : self.poisson_source_2d(x,y,t)))

        domain_disc.add(elem_disc)

    def add_boundary_conditions(self, domain_disc):
        dirichlet_bnd = ugcore.DirichletBoundary2dCPU1()
        dirichlet_bnd.add(0.0, self.cmp, "Boundary")
        # TODO: This throws an exception!
        #dirichlet_bnd.add(ugcore.PythonCondUserNumber2d(lambda x,y,t : self.poisson_dirichlet_value_2d(x,y,t)), 
        #                  self.cmp, "Boundary")
        domain_disc.add(dirichlet_bnd)


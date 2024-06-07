import sys
sys.path.append("..")

import modsimtools as util
import ug4py.pyugcore as ug4
import ug4py.pylimex as limex
import ug4py.pyconvectiondiffusion as cd
# import pysuperlu as slu
import math 

class SquareConfig:
    # Geometrie
    gridName= "grids/unit_square_tri.ugx" # "grids/unit_square_tri.ugx",
    requiredSubsets = {"Inner", "Boundary"}
    numRefs= 2
    
    # Constructor
    def __init__(self, mu, nu):
        self.mu = mu
        self.nu = nu
        
        self.elemDisc = cd.ConvectionDiffusionFE2d("u", "Inner")
        self.elemDisc.set_diffusion(1.0)
        self.elemDisc.set_source(lambda x,y,t : self.SourceCallback(x,y,t))
        
        self.dirichletBND = ug4.DirichletBoundary2dCPU1()
        # self.dirichletBND.add(MyDirichletBndCallback, "u", "Boundary")
        self.dirichletBND.add(0.0, "u", "Boundary")
        
    # Randbedingungen
    def MyDirichletBndCallback1(x, y, t):
        if (y==1) : return true, 0.0 
        elif (y==0) : return true, math.sin(math.pi*1*x)
        else : return false, 0.0 
    
    # API function
    def CreateDomainDisc(self,approxSpace):   
        domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)  
        domainDisc.add(self.elemDisc)
        domainDisc.add(self.dirichletBND)
        return domainDisc
    
    
    # Callback fuer rechte Seite
    def SourceCallback(self, x, y, t):
        mu = self.mu
        nu = self.nu
        scale =  (mu*mu + nu*nu)*(math.pi)*(math.pi)
        return scale*math.sin(math.pi*mu*x)* math.sin(math.pi*nu*y)

   
    def SolutionCallback(self,x,y,t):
        return math.sin(math.pi*self.mu*x)* math.sin(self.math.pi*self.nu*y)


# Callback function boundary values.
def SectorDirichletSol(x, y, t, si):
    r = math.sqrt(x*x+y*y);
    phi = math.atan2(y,x);
    if (phi<0) : phi = phi + 2*math.pi; 
    val=math.pow(r,(2/3))*math.sin(phi/3.0*2);
    return val

class SectorProblem:
    # Geometrie
    gridName= "grids/sectorTest.ugx"
    requiredSubsets = {"Inner", "Circle", "Cut"}
    numRefs= 2
    
    # Constructor
    def __init__(self):
        self.mu = 1.0
        self.nu = 4.0
        
        self.elemDisc = cd.ConvectionDiffusionFE2d("u", "Inner")
        self.elemDisc.set_diffusion(1.0)
        self.elemDisc.set_source(0.0)
    
        self.dirichletBND = ug4.DirichletBoundary2dCPU1()
        #self.dirichletBND.add(ug4.PythonUserNumber2d(SectorDirichletSol), "u", "Boundary")
       #self.dirichletBND.add(0.0, "u", "Cut")
        self.dirichletBND.add(ug4.PythonUserNumber2d(SectorDirichletSol), "u", "Circle")
        
    # Randbedingungen
    def MyDirichletBndCallback1(x, y, t):
        if (y==1) : return true, 0.0 
        elif (y==0) : return true, math.sin(math.pi*1*x)
        else : return False, 0.0 
    
    # API function
    def CreateDomainDisc(self,approxSpace):   
        domainDisc = ug4.DomainDiscretization2dCPU1(approxSpace)  
        domainDisc.add(self.elemDisc)
        domainDisc.add(self.dirichletBND)
        return domainDisc
    
    # Solution
    def SolutionCallback(self,x,y,t,si):
        return SectorDirichletSol(x,y,t,si)
    
   
   

CONFIG = SectorProblem()

dom = util.CreateDomain(CONFIG.gridName, CONFIG.numRefs, CONFIG.requiredSubsets)

approxSpace = util.CreateApproximationSpace(dom, dict(fct = "u", type = "Lagrange", order = 1))

domainDisc = CONFIG.CreateDomainDisc(approxSpace)


Ah = ug4.AssembledLinearOperatorCPU1(domainDisc)
uh = ug4.GridFunction2dCPU1(approxSpace)
bh = ug4.GridFunction2dCPU1(approxSpace)
#ug4.Interpolate(0.0, uh, "u")


#import traceback
#try:
domainDisc.assemble_linear(Ah, bh)
domainDisc.adjust_solution(uh)
#except Exception as inst:
#    traceback.print_exc()
#    print(str(inst))


solver = ug4.LUCPU1()
try:
    solver.init(Ah, uh)
    solver.apply(uh, bh)
except Exception as inst:
    print(inst)
    
print("Done!")
ug4.WriteGridFunctionToVTK(uh, "tmp/u_solution")

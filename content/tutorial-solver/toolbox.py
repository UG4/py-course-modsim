import ug4py.pyugcore as ugcore
import traceback

def SolverTest(domainDisc, approxSpace, solver, convCheck, ioname):
    # Assemble linear system
    A = ugcore.AssembledLinearOperatorCPU1(domainDisc)
    u = ugcore.GridFunction2dCPU1(approxSpace)
    b = ugcore.GridFunction2dCPU1(approxSpace)
    
    ugcore.VecScaleAssign(u, 0.0, u)  #u.set(0.0)
    try:
        domainDisc.assemble_linear(A, b)
    except Exception as e:
        print ("Assembly error:")
        print(repr(e))
        traceback.print_exc()
        print ("END")
              
    clock = ugcore.CuckooClock()
  
    ugcore.VecScaleAssign(b, 0.0, b)  # b.set(0.0)

    # u.set_random(-1.0, 1.0)
    import random
    random.seed(5)
    ugcore.Interpolate(ugcore.PythonUserNumber2d(lambda x,y,t,si : random.uniform(-1.0,1.0)), u, "u")  
    ugcore.WriteGridFunctionToVTK(u, "error0.vtk")
    
    # Call solver
    clock.tic()
    try:
        solver.init(A, u)
        solver.apply_return_defect(u,b)
    except Exception as e:
        print ("Solver failed:")
        print (repr(e))

    nvec = convCheck.step()
    print ("Results: " + str(nvec) + " steps (rho=" +str(convCheck.avg_rate()) +")")
    print("Time: " + str(clock.toc()) +" seconds")
    
    # Post processing (print results):
    ugcore.WriteGridFunctionToVTK(u, ioname)
    ugcore.SaveMatrixForConnectionViewer(u, A, "MatrixA.mat")
    # SaveVectorForConnectionViewer(u, "VectorU.vec")
    
    # Store history (e.g. for plotting)
    import numpy as np
    print(nvec)
    history = np.zeros(nvec)
    history[0] = convCheck.defect()/convCheck.reduction() # Initial defect.
    for i in range(convCheck.step()):
        history[i] = convCheck.get_defect(i)
    print(ioname)
    return history


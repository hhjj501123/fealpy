#import numpy as np
import pytest
#import scipy.sparse as sp
#import scipy.sparse.linalg as spla
from fealpy.backend import backend_manager as bm
from fealpy.solver import gmres,fealpy_gmres
from fealpy.sparse import COOTensor, CSRTensor
from gamg_solver_data import * 
import time 


from fealpy.pde.semilinear_2d import SemilinearData
from fealpy.mesh.quadrangle_mesh import QuadrangleMesh
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
        BilinearForm, ScalarDiffusionIntegrator,LinearForm,DirichletBC
    )  

#tmr = timer()
#next(tmr)

pde = SemilinearData(domain=[0,1,0,1])  # 初始化 PDE 数据

def get_Af(mesh,p):
    space = LagrangeFESpace(mesh,p=p)

    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator())
    A_without_apply = bform.assembly()

    lform = LinearForm(space)
    F_without_apply = lform.assembly()

    A1,f = DirichletBC(space,gd = pde.solution).apply(A_without_apply,F_without_apply)
    A = A1.tocsr()
    #if tmr is not None:
    #    tmr.send(f"组装 Poisson 方程离散系统")

    return A,f

class TestGSSolver:
    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", test_data)
    def test_gmres(self, backend):
        bm.set_backend(backend)
        
        #生成A,f,x0
        nx_1 = ny_1 =32
        p = 1
        mesh_1 = TriangleMesh.from_box(domain,nx_1,ny_1) 
        node_1 = mesh_1.entity('node')
        A,f = get_Af(mesh_1,p)
        sol_1 = pde.solution(node_1)
        print(A.shape)
        
        x0 = bm.zeros(A.shape[1])

        #fealpy的函数fealpy_gmres求解
        start_time  = time.time()
        phi,stop_res,niter = fealpy_gmres(A,f,x0)
        end_time = time.time()
        
        err = bm.linalg.norm(phi-x0)/bm.linalg.norm(x0)

        #scipy的函数gmres求解
        from scipy.sparse.linalg import gmres
        A = A.toarray()
        start_time_contrast  = time.time()
        phi_1,info = gmres(A,f,x0,restart=20)
        end_time_contrast = time.time()
        print("差:",bm.all(bm.abs(phi_1 - phi))<1e-5)
        
        #res_0 = bm.linalg.norm(bm.array(f))
        #stop_res = res/res_0

        # 输出误差和相对残差
        print('err:', err)
        print('stop_res:',stop_res )

        # 判断收敛
        rtol = 1e-4  # 设置收敛阈值
        if stop_res <= rtol:
            print("Converged: True")
            converged = True
        else:
            print("Converged: False")
            converged = False

        # 断言确保收敛
        assert converged, f"Gmres solver did not converge: stop_res = {stop_res} > rtol = {rtol}"

        print("Time cost: ",end_time-start_time)
        print("Time cost of contrast: ",end_time_contrast-start_time_contrast)


if __name__ == '__main__':
    test = TestGSSolver() 
    test.test_gmres('numpy')
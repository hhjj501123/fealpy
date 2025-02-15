import numpy as np
import pytest
from fealpy.backend import backend_manager as bm
from fealpy.solver import fealpy_minres
from fealpy.sparse import COOTensor, CSRTensor
#from minres_solver_data import * 
import time 
#from fealpy.utils import timer

from fealpy.pde.helmholtz_2d import HelmholtzData2d
from fealpy.mesh.quadrangle_mesh import QuadrangleMesh
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
        BilinearForm, ScalarDiffusionIntegrator,LinearForm,DirichletBC
    )  

#tmr = timer()
#next(tmr)
pde = HelmholtzData2d()  # 初始化 PDE 数据
domain = pde.domain() 

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
    #@pytest.mark.parametrize("data", test_data)

    def test_minres(self, backend):
        bm.set_backend(backend)

        #生成A,f,x0
        nx_1 = ny_1 = 200
        p = 1
        mesh_1 = TriangleMesh.from_box(domain,nx_1,ny_1) 
        node_1 = mesh_1.entity('node')
        A,f = get_Af(mesh_1,p)
        sol_1 = pde.solution(node_1)
        print(A.shape)
        
        x0 = bm.zeros(A.shape[1])

        #fealpy的函数fealpy_minres求解
        start_time  = time.time()
        #if tmr is not None:
        #    tmr.send(f"数据准备")
        #if tmr is not None:
        #    tmr.send(f"Minres 求解")
        phi,info = fealpy_minres(A,f,x0)
        #tmr.send(None)
        end_time = time.time()

        #scipy的函数minres求解
        from scipy.sparse.linalg import minres
        A = A.toarray()
        start_time_contrast  = time.time()
        phi_1,stop_res_1= minres(A,f,x0)
        end_time_contrast = time.time()
        print("差:",bm.all(bm.abs(phi_1 - phi))<1e-5)
        
        print("stop_res_1",stop_res_1)

        err = bm.linalg.norm(phi-x0)/bm.linalg.norm(x0)

        res_0 = bm.linalg.norm(bm.array(f))
        stop_res = info['residual']/res_0

        # 输出迭代次数
        print('niter',info['niter'])
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
        assert converged, f"Minres solver did not converge: stop_res = {stop_res} > rtol = {rtol}"

        print("Time cost: ",end_time-start_time)
        print("Time cost of contrast: ",end_time_contrast-start_time_contrast)



if __name__ == '__main__':
    test = TestGSSolver() 
    test.test_minres('numpy')
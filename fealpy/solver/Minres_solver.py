from typing import Optional, Protocol

from ..backend import backend_manager as bm
from ..backend import TensorLike
from .mumps import spsolve, spsolve_triangular
from ..sparse.coo_tensor import COOTensor
from ..sparse.csr_tensor import CSRTensor

from .. import logger
from fealpy.utils import timer

class SupportsMatmul(Protocol):
    def __matmul__(self, other: TensorLike) -> TensorLike: ...



def minres(A: SupportsMatmul, b: TensorLike, x0: Optional[TensorLike]=None,
       atol: float=1e-12, rtol: float=1e-8,
       maxit: Optional[int]=10000) -> TensorLike:
    """
    Solve a linear system Ax = b using the minres method.
    Parameters:
        A (SupportsMatmul): The coefficient matrix of the linear system.
        b (TensorLike): The right-hand side vector of the linear system, can be a 1D or 2D tensor.
        x0 (TensorLike): Initial guess for the solution, a 1D or 2D tensor.\
        Must have the same shape as b when reshaped appropriately.
        atol (float, optional): Absolute tolerance for convergence. Default is 1e-12.
        rtol (float, optional): Relative tolerance for convergence. Default is 1e-8.
        maxiter (int, optional): Maximum number of iterations allowed. Default is 10000.\
        If not provided, the method will continue until convergence based on the given tolerances.

    Returns:
        Tensor: The approximate solution to the system Ax = b.
        info: Including residual and iteration count.

    Raises:
        ValueError: If inputs do not meet the specified conditions (e.g., A is not sparse, dimensions mismatch).

    Note:
        This implementation assumes that A is a symmetric matrix,
        which is a common requirement for the minres method to work correctly.
    """
  
    assert isinstance(b, TensorLike), "b must be a Tensor"
    if x0 is not None:
        assert isinstance(x0, TensorLike), "x0 must be a Tensor if not None"
    single_vector = b.ndim == 1

    if b.ndim not in {1, 2}:
        raise ValueError("b must be a 1D or 2D dense tensor")

    if x0 is None:
        x0 = bm.zeros_like(b)
    else:
        if x0.shape != b.shape:
            raise ValueError("x0 and b must have the same shape")
    
    info = {}

    m,n = A.shape
    b_norm = bm.linalg.norm(b,ord=2)
    
    r = b - A @ x0
    beta = bm.linalg.norm(r,ord=2)

    #判断A为对称矩阵
    w = A @ r
    w_1 = A @ w
    s = w.T @ w
    a = r.T @ w_1 
    if bm.abs(s -a) > 1e-5:
        raise ValueError("A must be symmetric matrix")

    t_1 = beta
    #P = bm.zeros((m,max_iter))
    
    x = x0 
    q_1 = r/beta 
    q_0 = bm.zeros_like(q_1)
    q_2 = bm.zeros_like(q_1) 

    T = bm.zeros((5,5))

    p_1 = bm.zeros_like(q_1)
    p_0 = bm.zeros_like(q_1)
    p_2 = bm.zeros_like(q_1)

    for niter in range(maxit):
        #Lanczos
        if niter<2:
            w = A@q_1  #w = A@Q[:,k]
            #T[k,k] = q_1.T @ w
            T = bm.set_at(T,(niter,niter),q_1.T @ w)
        else:
            w = A@q_2
            if niter==3 or niter==2:
                #T[k,k] = q_2.T @ w
                T = bm.set_at(T,(niter,niter),q_2.T @ w)
            if niter>3:
                T[:4,:4] = T[1:,1:]
                #T[2,4] = 0
                T = bm.set_at(T,(2,4),0)
                #T[3,3] = q_2.T @ w
                T = bm.set_at(T,(3,3),q_2.T @ w)

        #print("q_0:",q_0)
        #print("q_1:",q_1) 
        if niter==0:
            w = w - T[niter,niter] * q_1  #w = w - T[k,k] * Q[:,k]
        else:
            if niter ==1:
                w = w - T[niter,niter] * q_1 - T[niter,niter-1] * q_0
            if niter>1 and niter<4:
                w = w - T[niter,niter] * q_2 - T[niter,niter-1] * q_1
            if niter>3:
                w = w - T[3,3] * q_2 - T[3,2] * q_1
        
        if niter<4:
            #T[k+1,k] = bm.linalg.norm(w,ord=2)
            T = bm.set_at(T,(niter+1,niter),bm.linalg.norm(w,ord=2))

        else:
            #T[4,3] = bm.linalg.norm(w,ord=2)
            T = bm.set_at(T,(4,3),bm.linalg.norm(w,ord=2))

        
        if niter<maxit-1 and niter<4:
            #T[k,k+1] = T[k+1,k]
            T = bm.set_at(T,(niter,niter+1),T[niter+1,niter])
        else:
            #T[3,4] = T[4,3]
            T = bm.set_at(T,(3,4),T[4,3])

        
        if niter==0:
            q_0 = q_1
            q_1 = w/T[niter+1,niter]
        else:
            if niter != 1 :
                q_0 = q_1
                q_1 = q_2
            if niter<4:
                q_2 = w/T[niter+1,niter]  #Q[:,k+1] = w/T[k+1,k] .
            else:
                q_2 = w/T[4,3]  #Q[:,k+1] = w/T[k+1,k] .

                
        if niter < 1:
        #    print("--------------------------------------------")
            continue
        
        if niter < 4:
            c = T[niter-1,niter-1]/(T[niter,niter-1]**2+T[niter-1,niter-1]**2)**0.5
            s = T[niter,niter-1]/(T[niter,niter-1]**2+T[niter-1,niter-1]**2)**0.5
        else:
            c = T[2,2]/(T[3,2]**2+T[2,2]**2)**0.5
            s = T[3,2]/(T[3,2]**2+T[2,2]**2)**0.5

        t_0 = t_1
        t_1 = -s * t_0
        t_0 = c * t_0
            
        #T = G_1 @ T
        for j in range(T.shape[1]):
            if niter < 4:
                a = T[niter-1,j]
                #T[k-1,j] = c*T[k-1,j]+s*T[k,j]
                #T[k,j] = -s*a+c*T[k,j]
                T = bm.set_at(T,(niter-1,j),c*T[niter-1,j]+s*T[niter,j])
                T = bm.set_at(T,(niter,j),-s*a+c*T[niter,j])
            else:
                a = T[2,j]
                #T[2,j] = c*T[2,j]+s*T[3,j]
                #T[3,j] = -s*a+c*T[3,j]
                T = bm.set_at(T,(2,j),c*T[2,j]+s*T[3,j])
                T = bm.set_at(T,(3,j),-s*a+c*T[3,j])

        if niter == 1:
            p_0 = q_0/T[0,0]
        if niter == 2:
            p_1 = (q_0 - T[0,1]*p_0) / T[1,1]
        if niter > 2 and niter < 4:
            p_2 = (q_0 - T[niter-2,niter-1]*p_1 - T[niter-3,niter-1]*p_0) / T[niter-1,niter-1]
            p_0 = p_1
            p_1 = p_2
        if niter > 3:
            p_2 = (q_0 - T[1,2]*p_1 - T[0,2]*p_0) / T[2,2]
            p_0 = p_1
            p_1 = p_2

        
        if niter == 1:
            x = x + t_0*p_0
        else:
            x = x + t_0*p_1

        if niter == maxit-1:
            if niter < 4:
                c = T[niter,niter]/(T[niter+1,niter]**2+T[niter,niter]**2)**0.5
                s = T[niter+1,niter]/(T[niter+1,niter]**2+T[niter,niter]**2)**0.5
            else:
                c = T[3,3]/(T[4,3]**2+T[3,3]**2)**0.5
                s = T[4,3]/(T[4,3]**2+T[3,3]**2)**0.5

            t_0 = t_1
            t_1 = -s * t_0
            t_0 = c * t_0
            
            #T = G_1 @ T
            if niter < 4:
                a = T[niter,niter]
                #T[k,k] = c*T[k,k]+s*T[k+1,k]
                #T[k+1,k] = -s*a+c*T[k+1,k]
                T = bm.set_at(T,(niter,niter),c*T[niter,niter]+s*T[niter+1,niter])
                T = bm.set_at(T,(niter+1,niter),-s*a+c*T[niter+1,niter])
            else:
                a = T[3,3]
                #T[3,3] = c*T[3,3]+s*T[4,3]
                #T[4,3] = -s*a+c*T[4,3]
                T = bm.set_at(T,(3,3),c*T[3,3]+s*T[4,3])
                T = bm.set_at(T,(4,3),-s*a+c*T[4,3])

            #进行求解
            if niter <4:
                p_2 = (q_1 - T[niter-1,niter]*p_1 - T[niter-2,niter]*p_0) / T[niter,niter]
            else:
                p_2 = (q_1 - T[2,3]*p_1 - T[1,3]*p_0) / T[3,3]
            p_0 = p_1
            p_1 = p_2
            x = x + t_0*p_1
            
        T[bm.abs(T) < 1e-6] = 0
        
        r = b - A@x
        res = bm.linalg.norm(r)

        if res < atol:
            logger.info(f"minres: converged in {niter} iterations, "
                        "stopped by absolute tolerance.")
            break

        if res < rtol * b_norm:
            logger.info(f"minres: converged in {niter} iterations, "
                        "stopped by relative tolerance.")
            break

        if (maxit is not None) and (niter >= maxit):
            logger.info(f"minres: failed, stopped by maxiter ({maxit}).")
            break

        
    info['residual'] = res
    info['niter'] = niter
    return x,info
        


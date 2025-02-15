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

def gmres(A: SupportsMatmul, b: TensorLike, x0: Optional[TensorLike]=None,
       atol: float=1e-12, rtol: float=1e-8, restart: Optional[int]=20, 
       maxit: Optional[int]=10000) -> TensorLike:
    
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
    
    m,n = A.shape
    H = bm.zeros((restart+1,restart),dtype=bm.float64)
    Q = bm.zeros((n,restart+1),dtype=bm.float64)
    t = bm.zeros(restart+1,dtype=bm.float64)
        
    tmr = timer()
    next(tmr)

    for niter in range(maxit):
        #Arnoldi得到Q,H,k
        r0 = b - A.matmul(x0)
        beta = bm.linalg.norm(r0)
        q0 = r0/beta
        
        x = x0
        Q[:,0] = q0
        for j in range(restart):
            w = A.matmul(Q[:,j])
            for i in range(j+1):
                H[i,j] = Q[:,i] @ w
                w = w - H[i,j]*Q[:,i]
            H[j+1,j] = bm.linalg.norm(w)
            Q[:,j+1] = w/H[j+1,j]
        #print("Krylov矩阵的Q因子:\n", Q)
        #print("AQ=QH的H:\n", H)
        #print(type(H))
        t[0] = beta

        if tmr is not None:
            tmr.send(f"Arnoldi")
    
        #对H进行QR分解
        for i in range(restart):
            if H[i+1,i] ==0:
                continue
            c = H[i,i]/bm.sqrt(H[i+1,i]**2+H[i,i]**2)
            s = H[i+1,i]/bm.sqrt(H[i+1,i]**2+H[i,i]**2)

            t[i+1] = -s * t[i]
            t[i] = c * t[i]
        
            ##H = G_1 @ H 
            for j in range(H.shape[1]):
                a = H[i,j]
                H[i,j] = c*H[i,j]+s*H[i+1,j]
                H[i+1,j] = -s*a+c*H[i+1,j] 
        H[bm.abs(H) < 1e-10] = 0
        if tmr is not None:
            tmr.send(f"对H进行QR分解") 
        
        #求得y
        for k in range():
            
        #y = spsolve_triangular(H_CSR,t[:-1])
        #print("y:\n",y)
        #print(Q[:,:-1]@y)
        #求得x
        x = x0+Q[:,:-1]@y
        #print("x:",x)

        if tmr is not None:
            tmr.send(f"求解x") 

        r = b - A@x
        stop_rule = bm.linalg.norm(r)/bm.linalg.norm(b)
        if stop_rule < rtol:
            tmr.send(None)
            return x,stop_rule,niter
        else:
            x0 = x 
            
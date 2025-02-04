import numpy as np
import torch


def mSSRM_PGA(m:int, iternum:int, tol:float, matR:torch.Tensor, vecmu:torch.Tensor) -> torch.Tensor:
    T, N = matR.shape
    RE = 100
    eI = torch.finfo(torch.float32).eps * torch.eye(N, device = matR.device)
    p = vecmu
    Q = (1/np.sqrt(T - 1)) * (matR - (1/T) * torch.ones((T, T), device = matR.device) @ matR)
    QeI = Q.mT @ Q + eI
    alpha = 0.999 / QeI.norm(2)
    w = vecmu
    k = 1
    while k < iternum and RE > tol:
        w1 = w
        w_pre = w - alpha * (QeI @ w - p)
        w_pre = w_pre.clamp(min = 0)
        itw = torch.argsort(w_pre, descending = True)
        w = torch.zeros((N), device = matR.device)
        w[itw[:m]] = w_pre[itw[:m]]
        RE = (w - w1).norm(2) / w1.norm(2)
        k += 1

    if sum(w) == 0:
        return torch.ones((N), device = matR.device) / N
    else:
        return w / sum(w)
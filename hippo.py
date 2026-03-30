# @title hippo.py down
import math
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def unroll(A, u, s=None): # https://github.com/HazyResearch/hippo-code/blob/master/model/unroll.py
    if s is None: s = torch.zeros_like(u[0])
    out = []
    for (A_, u_) in zip(torch.unbind(A, dim=0), torch.unbind(u, dim=0)):
        s = (A_ @ s.unsqueeze(-1)).squeeze(-1)
        s = s + u_
        out.append(s)
    return torch.stack(out, dim=0)

def LegS(N): # Legendre (scaled)
    q = torch.arange(N, dtype=torch.float32)
    col, row = torch.meshgrid(q, q, indexing="xy")
    M = -((row >= col)*(2*q+1) - torch.diag(q))
    T = torch.diag(2*q+1)**.5
    A = T @ M @ torch.linalg.inv(T)
    B = torch.diag(T).unsqueeze(-1)
    return A, B

class HiPPO_LegS(nn.Module): # https://github.com/HazyResearch/hippo-code/blob/master/model/hippo.py
    def __init__(self, N, max_length=1024):
        super().__init__()
        A, B = LegS(N)
        B = B.squeeze(-1)
        self.A_stacked, self.B_stacked = torch.empty((max_length, N, N)), torch.empty((max_length, N))
        for t in range(1, max_length + 1):
            self.A_stacked[t-1] = torch.matrix_exp(A * (math.log(t + 1) - math.log(t))) # ZOH
            self.B_stacked[t-1] = torch.linalg.solve_triangular(A, (self.A_stacked[t-1] @ B - B).unsqueeze(-1), upper=False).squeeze(-1) # ZOH
        self.eval_matrix = (B[:, None] * torch.special.legendre_polynomial_p(torch.linspace(-1, 1, max_length), torch.arange(N)[:, None])).T

    def forward(self, inputs): # [T,d]
        T = inputs.shape[0]
        u = inputs.unsqueeze(-1) * self.B_stacked[:T].unsqueeze(1) # Td1*T1N->TdN
        result = unroll(self.A_stacked[:T], u) # og fast=F # TNN,TdN->TdN
        return result # TdN where N is the order of the HiPPO projection

    def reconstruct(self, c): # [b,N]
        return c @ self.eval_matrix.T

# N, T, d = 256, 200, 1
# f = torch.randn(T, d)

# legs = HiPPO_LegS(N, T)
# legsf = legs(f)[-1] # TdN->dN
# print('legsf', legsf.shape)

# f_legs = legs.reconstruct(legsf).T # Td
# # print('f f_legs', f.shape, f_legs.shape)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 2))
# vals = torch.linspace(0, 1, T)
# plt.plot(vals, f[:,-1])
# vals = torch.linspace(0, 1, T-1)
# plt.plot(vals, f_legs[1:,-1])
# plt.show()

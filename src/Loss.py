import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def physics_loss(model, x, EI, q):
    u = model(x)
    
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    u_xxxx = torch.autograd.grad(u_xxx, x, torch.ones_like(u_xxx), create_graph=True)[0]
    
    f = EI * u_xxxx - q
    
    return torch.mean(f**2)

def boundary_loss(model, EI, L, apoio_esq, apoio_dir):
    bc_loss = 0.0

    x0 = torch.tensor([[0.0]], requires_grad=True)
    xL = torch.tensor([[L]], requires_grad=True)

    u_0 = model(x0)
    u_L = model(xL)

    u0_x = torch.autograd.grad(u_0, x0, torch.ones_like(u_0), create_graph=True)[0]
    u0_xx = torch.autograd.grad(u0_x, x0, torch.ones_like(u0_x), create_graph=True)[0]
    
    uL_x = torch.autograd.grad(u_L, xL, torch.ones_like(u_L), create_graph=True)[0]
    uL_xx = torch.autograd.grad(uL_x, xL, torch.ones_like(uL_x), create_graph=True)[0]

    if apoio_esq[1] == 1:
        bc_loss += u_0**2

    if apoio_esq[2] == 1: 
        bc_loss += u0_x**2
    else:
        bc_loss += u0_xx**2

    if apoio_dir[1] == 1:
        bc_loss += u_L**2

    if apoio_dir[2] == 1:
        bc_loss += uL_x**2
    else:
        bc_loss += uL_xx**2

    return bc_loss.mean()
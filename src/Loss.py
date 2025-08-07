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

def boundary_loss(model, EI, L=1.0): 
    # Mantenha os tensores de contorno como folhas com requires_grad=True
    x0 = torch.tensor([[0.0]], requires_grad=True)
    xL = torch.tensor([[L]], requires_grad=True)
    
    u0 = model(x0)
    uL = model(xL)
    
    loss_u0 = u0**2
    loss_uL = uL**2
    
    u0_output = model(x0)
    u0_x = torch.autograd.grad(u0_output, x0, torch.ones_like(u0_output), create_graph=True)[0]
    u0_xx = torch.autograd.grad(u0_x, x0, torch.ones_like(u0_x), create_graph=True)[0]
    loss_u0_xx = u0_xx**2
    
    uL_output = model(xL)
    uL_x = torch.autograd.grad(uL_output, xL, torch.ones_like(uL_output), create_graph=True)[0]
    uL_xx = torch.autograd.grad(uL_x, xL, torch.ones_like(uL_x), create_graph=True)[0]
    loss_uL_xx = uL_xx**2
    
    bc_loss = loss_u0 + loss_uL + loss_u0_xx + loss_uL_xx
    
    return bc_loss.mean()
import torch
import torch.nn as nn
import numpy as np

def physics_loss(model, x, L, EI, q):
    u = model(x)
    
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    u_xxxx = torch.autograd.grad(u_xxx, x, torch.ones_like(u_xxx), create_graph=True)[0]
    
    f = EI * u_xxxx - q
    
    return torch.sum(f**2)
    #  return torch.mean(f**2)

def boundary_loss(model, q, L, apoio_esq, apoio_dir):
    bc_loss = 0.0

    x0 = torch.tensor([[0.0]], requires_grad=True)
    xL = torch.tensor([[L]], requires_grad=True)

    u_0 = model(x0)
    u_L = model(xL)

    u0_x = torch.autograd.grad(u_0, x0, torch.ones_like(u_0), create_graph=True)[0]
    u0_xx = torch.autograd.grad(u0_x, x0, torch.ones_like(u0_x), create_graph=True)[0]
    u0_xxx = torch.autograd.grad(u0_xx, x0, torch.ones_like(u0_x), create_graph=True)[0]
    u0_xxxx = torch.autograd.grad(u0_xxx, x0, torch.ones_like(u0_x), create_graph=True)[0]

    uL_x = torch.autograd.grad(u_L, xL, torch.ones_like(u_L), create_graph=True)[0]
    uL_xx = torch.autograd.grad(uL_x, xL, torch.ones_like(uL_x), create_graph=True)[0]
    uL_xxx = torch.autograd.grad(uL_xx, xL, torch.ones_like(uL_xx), create_graph=True)[0]
    uL_xxxx = torch.autograd.grad(uL_xxx, xL, torch.ones_like(uL_xxx), create_graph=True)[0]

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

    # bc_loss += (u0_xxxx ** 2 + uL_xxxx ** 2)

    return bc_loss.sum()
# return bc_loss.mean()

class PINNViga(nn.Module):
    def __init__(self):
        super(PINNViga, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.log_pde_weight = nn.Parameter(torch.tensor(0.0))
        self.log_bc_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x)
    
    @property
    def pde_weight(self):
        return torch.exp(self.log_pde_weight)

    @property
    def bc_weight(self):
        return torch.exp(self.log_bc_weight)

def run_model(apoio_esq, apoio_dir, EI, q, L, num_epochs=1000, pde_weight = 1.0, bc_weight = 10.0):
    L = float(L)
    model = PINNViga()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    u_plot_variation = []

    # x_train_raw = torch.rand(101, 1)
    min_val = 0
    max_val = 1
    x_scaled = min_val + (max_val - min_val) * torch.rand(101, 1)
    # x_train_raw = torch.tensor(x_scaled).reshape(101,1).float()
    # x_train_raw = torch.tensor(np.ones(101) * 0.5).reshape(101,1).float()

    # x_train = (x_train_raw * L).requires_grad_(True)
    x_train = (x_scaled * L).requires_grad_(True)

    for epoch in range(num_epochs + 1):
        
        loss_pde = physics_loss(model, x_train, L, EI, q)
        loss_bc = boundary_loss(model, q, L, apoio_esq, apoio_dir)

        optimizer.zero_grad()

        loss = pde_weight * loss_pde + bc_weight * loss_bc

        loss.backward()
        optimizer.step()
        
        if epoch % int(num_epochs / 10) == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}")

        x_plot = torch.linspace(0, L, 100).view(-1,1)
        u_plot = model(x_plot).detach().numpy()

        u_plot_variation.append((epoch, u_plot))

    return x_plot.numpy(), u_plot, u_plot_variation

def run_model_adaptive(apoio_esq, apoio_dir, EI, q, L, num_epochs=1000):
    L = float(L)
    model = PINNViga()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    u_plot_variation = []

    x_train_raw = torch.rand(101, 1)
    x_train = (x_train_raw * L).requires_grad_(True)

    for epoch in range(num_epochs + 1):
        optimizer.zero_grad()
        
        loss_pde = physics_loss(model, x_train, EI, q)
        loss_bc = boundary_loss(model, L, apoio_esq, apoio_dir)
        
        loss = model.pde_weight * loss_pde + model.bc_weight * loss_bc

        loss.backward()
        optimizer.step()
        
        if epoch % int(num_epochs / 10) == 0:
            print(f"Epoch {epoch}, Total Loss: {loss.item():.6f}, "
                  f"PDE Loss: {loss_pde.item():.6f}, BC Loss: {loss_bc.item():.6f}, "
                  f"PDE Weight: {model.pde_weight.item():.4f}, BC Weight: {model.bc_weight.item():.4f}")

        x_plot = torch.linspace(0, L, 100).view(-1,1)
        u_plot = model(x_plot).detach().numpy()

        u_plot_variation.append((epoch, u_plot))

    return x_plot.numpy(), u_plot, u_plot_variation
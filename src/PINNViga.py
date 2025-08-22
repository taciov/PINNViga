import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Garantir reprodutibilidade no cuDNN (GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def physics_loss(model, x):
    u = model(x)
    
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    u_xxxx = torch.autograd.grad(u_xxx, x, torch.ones_like(u_xxx), create_graph=True)[0]
    
    f = u_xxxx - 1.0

    return torch.mean((f) ** 2)

def boundary_loss(model, apoio_esq, apoio_dir):
    x0 = torch.tensor([[0.0]], requires_grad=True)
    xL = torch.tensor([[1.0]], requires_grad=True)

    u_0 = model(x0)
    u_L = model(xL)

    u0_x = torch.autograd.grad(u_0, x0, torch.ones_like(u_0), create_graph=True)[0]
    u0_xx = torch.autograd.grad(u0_x, x0, torch.ones_like(u0_x), create_graph=True)[0]
    u0_xxx = torch.autograd.grad(u0_xx, x0, torch.ones_like(u0_xx), create_graph=True)[0]

    uL_x = torch.autograd.grad(u_L, xL, torch.ones_like(u_L), create_graph=True)[0]
    uL_xx = torch.autograd.grad(uL_x, xL, torch.ones_like(uL_x), create_graph=True)[0]
    uL_xxx = torch.autograd.grad(uL_xx, xL, torch.ones_like(uL_xx), create_graph=True)[0]

    bc_loss = 0.0

    if apoio_esq[1] == 1:
        bc_loss += (u_0)**2
    else:
        bc_loss += (u0_xxx)**2

    if apoio_esq[2] == 1: 
        bc_loss += (u0_x)**2
    else:
        bc_loss += (u0_xx)**2

    if apoio_dir[1] == 1:
        bc_loss += (u_L)**2
    else:
        bc_loss += (uL_xxx)**2

    if apoio_dir[2] == 1:
        bc_loss += (uL_x)**2
    else:
        bc_loss += (uL_xx)**2

    return bc_loss.mean()

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

    def run_model(self, apoio_esq, apoio_dir, EI, q, L, num_epochs=1000, pde_weight = 1.0, bc_weight = 1.0, min_delta=1e-6, tol = 1e-5,tam = 101):
        u_ref = q * (L**4) / EI
        set_seed(1)
        L = float(L)
        self.model = PINNViga()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.u_plot_variation = []
        self.loss_pde_variation = []
        self.loss_bc_variation = []

        best_loss = float('inf')
        best_state = None

        torch.manual_seed(1)
        min_val = 0
        max_val = 1
        x_scaled = min_val + (max_val - min_val) * torch.rand(tam, 1)
        x_train = x_scaled.requires_grad_(True)

        for epoch in range(num_epochs + 1):
            loss_pde = physics_loss(self.model, x_train)
            loss_bc = boundary_loss(self.model, apoio_esq, apoio_dir)

            optimizer.zero_grad()

            loss = pde_weight * loss_pde + bc_weight * loss_bc

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                best_state = self.model.state_dict()
            elif (current_loss > best_loss + min_delta) and (float(loss_bc) <= tol) and (float(loss_pde) <= tol):
                print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}")
                print(f"Treinamento interrompido na época {epoch} devido à falta de melhoria.")
                break
                       
            if epoch % int(num_epochs / 10) == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}")

            x_plot = torch.linspace(0, 1, tam).view(-1,1)
            u_plot_star = self.model(x_plot).detach().numpy()
            u_plot = u_plot_star * u_ref
            self.x_plot = (x_plot.numpy() * L)
            self.u_plot = u_plot
            self.u_plot_variation.append((epoch, self.u_plot))
            self.loss_pde_variation.append(loss_pde)
            self.loss_bc_variation.append(loss_bc)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.loss = self.loss_pde_variation + self.loss_bc_variation

    def plot_errors(self, scale):
        valores_epoch = np.arange(0, len(self.u_plot_variation), 1)
        valores_bc = [float(loss) for loss in self.loss_bc_variation]
        valores_pde = [float(loss) for loss in self.loss_pde_variation]
        valores_total = [bc + pde for bc, pde in zip(valores_bc, valores_pde)]
        plt.plot(valores_epoch, valores_bc, label = 'BC Loss', color = 'red', lw = 1)
        plt.plot(valores_epoch, valores_pde, label = 'PDE Loss', color = 'blue', lw = 1)
        plt.plot(valores_epoch, valores_total, label = 'Total Loss', color = 'green', lw = 1)
        plt.xlabel('Epoch')
        plt.ylabel('Erro')
        plt.xlim(0, max(valores_epoch))
        plt.ylim(0, 1.05 * max(valores_total) / scale)
        plt.legend()
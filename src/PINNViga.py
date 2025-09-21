import copy
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

def obter_trechos_cargas(lista_cargas, apoio_esq, apoio_dir):
    posicoes_x = []

    for carga in lista_cargas:
        if carga.x0 not in posicoes_x:
            posicoes_x.append(carga.x0)
        if carga.x1 not in posicoes_x:
            posicoes_x.append(carga.x1)

    for apoio in [apoio_esq, apoio_dir]:
        if apoio.x not in posicoes_x:
            posicoes_x.append(apoio.x)

    posicoes_x = sorted(posicoes_x)

    trechos_cargas = []

    for k in range(len(posicoes_x) -1):
        dict_trecho = {
            "x0" : posicoes_x[k],
            "x1" : posicoes_x[k + 1],
            "qx" : 0,
            "qy" : 0
        }

        for carga in lista_cargas:
            if carga.x0 <= posicoes_x[k] and carga.x1 >= posicoes_x[k+1]:
                dict_trecho['qx'] += carga.qx
                dict_trecho['qy'] += carga.qy

        trechos_cargas.append(dict_trecho)

    return trechos_cargas

def physics_loss(model, x, q):
    u = model(x)
    
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    u_xxxx = torch.autograd.grad(u_xxx, x, torch.ones_like(u_xxx), create_graph=True)[0]
    
    f = u_xxxx - q

    return torch.mean((f) ** 2)

def interface_loss(model, x):

    eps = 1e-6
    x_left = torch.tensor([[x - eps]], requires_grad=True)
    x_right = torch.tensor([[x + eps]], requires_grad=True)

    u_l = model(x_left)
    uxl = torch.autograd.grad(u_l, x_left, torch.ones_like(u_l), create_graph=True)[0]
    uxxl = torch.autograd.grad(uxl, x_left, torch.ones_like(uxl), create_graph=True)[0]
    uxxxl = torch.autograd.grad(uxxl, x_left, torch.ones_like(uxxl), create_graph=True)[0]

    u_r = model(x_right)
    uxr = torch.autograd.grad(u_r, x_right, torch.ones_like(u_r), create_graph=True)[0]
    uxxr = torch.autograd.grad(uxr, x_right, torch.ones_like(uxr), create_graph=True)[0]
    uxxxr = torch.autograd.grad(uxxr, x_right, torch.ones_like(uxxr), create_graph=True)[0]

    # Continuidade de u, u', u''
    cont_loss = ((u_l - u_r) ** 2 +
                (uxl - uxr) ** 2 + 
                (uxxl - uxxr) ** 2)

    return cont_loss

def trecho_physics_loss(model, trechos_cargas, x):

    loss_pde = 0
    loss_int = 0

    max_qy = max([abs(carga['qy']) for carga in trechos_cargas])

    for k, carga in enumerate(trechos_cargas):
        x0 = carga['x0']
        x1 = carga['x1']

        qx = carga['qx']
        qy = carga['qy'] / max_qy

        filter = (x >= x0) & (x <= x1)
        x_train_subdomain = x[filter].view(-1, 1)

        x_train_subdomain.requires_grad_(True)
        if x_train_subdomain.numel() > 0:
            loss_pde += physics_loss(model, x_train_subdomain, qy)
        if x0 != min([carga['x0'] for carga in trechos_cargas]):
            loss_int += interface_loss(model, x0)

    return loss_pde

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

    def forward(self, x):
        return self.net(x)
    
    # def closure():
    #     optimizer.zero_grad()

    def run_model(self, apoio_esq, apoio_dir, lista_cargas, EI, L,
                  num_epochs=5000, pde_weight=1.0, bc_weight=1.0,
                  min_delta=1e-6, tol=2e-5, tam=101,
                  patience=200):
        
        set_seed(1)
        L = float(L)
        self.model = PINNViga()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.u_plot_variation = []
        self.loss_pde_variation = []
        self.loss_bc_variation = []

        best_loss = float('inf')
        best_state = None
        best_epoch = 0

        wait = 0
        prev_loss = None

        torch.manual_seed(1)
        min_val = 0
        max_val = 1
        x_scaled = min_val + (max_val - min_val) * torch.rand(tam, 1)
        x_train = x_scaled.requires_grad_(True)

        self.trechos_cargas = obter_trechos_cargas(lista_cargas, apoio_esq, apoio_dir)
        max_qy = max_qy = max([abs(carga['qy']) for carga in self.trechos_cargas])

        u_ref = max_qy * (L**4) / EI

        for epoch in range(num_epochs + 1):
            # torch.manual_seed(1)
            # min_val = 0
            # max_val = 1
            # x_scaled = min_val + (max_val - min_val) * torch.rand(tam, 1)
            # x_train = x_scaled.requires_grad_(True)
            # loss_pde = physics_loss(self.model, x_train)

            # if epoch >= max([0.3 * num_epochs, 350]):
            #     optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.001, max_iter=int(round(0.7 * num_epochs,0)))

            optimizer.zero_grad()

            loss_pde = trecho_physics_loss(self.model, self.trechos_cargas, x_train)
            loss_bc = boundary_loss(self.model, apoio_esq.graus, apoio_dir.graus)

            loss = pde_weight * loss_pde + bc_weight * loss_bc

            loss.backward()            
            optimizer.step()
            current_loss = loss.item()

            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                best_state = self.model.state_dict()
                best_epoch = epoch
                wait = 0
            else:
                if prev_loss is not None:
                    delta = abs(current_loss - prev_loss) / prev_loss
                    if delta < 1e-3:
                        wait += 1
                    else:
                        wait = 0

            if (float(loss_bc) <= tol and float(loss_pde) <= tol) or (wait >= patience):
                print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}")
                print(f"Treinamento interrompido na época {epoch}.")
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

            self.x_train = x_train

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Melhor estado restaurado da época {best_epoch} com perda {best_loss:.6e}")

        self.loss = self.loss_pde_variation + self.loss_bc_variation

    def plot_errors(self, scale, nome = "erros.png"):
        valores_epoch = np.arange(0, len(self.u_plot_variation), 1)
        valores_bc = [float(loss) for loss in self.loss_bc_variation]
        valores_pde = [float(loss) for loss in self.loss_pde_variation]
        valores_total = [bc + pde for bc, pde in zip(valores_bc, valores_pde)]

        plt.figure(figsize=(8, 4.5))
        plt.plot(valores_epoch, valores_bc, label = 'Perda CC', color = 'red', lw = 1)
        plt.plot(valores_epoch, valores_pde, label = 'Perda ED', color = 'blue', lw = 1)
        plt.plot(valores_epoch, valores_total, label = 'Perda Total', color = 'green', lw = 1)
        plt.title("Variação do erro durante o treinamento")
        plt.xlabel('Época')
        plt.ylabel('Erro')
        plt.xlim(0, max(valores_epoch))
        plt.ylim(0, 1.05 * max(valores_total) / scale)
        plt.legend()
        plt.savefig(nome, dpi = 300)
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

def obter_trechos_cargas(lista_apoios, lista_cargas):
    posicoes_x = []

    for carga in lista_cargas:
        if carga.x0 not in posicoes_x:
            posicoes_x.append(carga.x0)
        if carga.x1 not in posicoes_x:
            posicoes_x.append(carga.x1)

    for apoio in lista_apoios:
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
                (uxl - uxr) ** 2)

    return cont_loss

def trecho_physics_loss(model, trechos_cargas):

    loss_pde = 0
    loss_int = 0

    max_qy = max([abs(carga['qy']) for carga in trechos_cargas])

    for k, carga in enumerate(trechos_cargas):
        L = 2

        x0 = carga['x0'] / L
        x1 = carga['x1'] / L

        qx = carga['qx']
        qy = carga['qy'] / max_qy

        # filter = (x >= x0) & (x <= x1)
        # x_train_subdomain = x[filter].view(-1, 1).requires_grad_(True)

        tam = max(int(round(abs(x1 - x0) * 100, 0)), 41)

        torch.manual_seed(1)
        min_val = 0
        max_val = 1
        x_scaled = min_val + (max_val - min_val) * torch.rand(tam, 1)
        x_train_subdomain = x_scaled.requires_grad_(True)

        if x_train_subdomain.numel() > 0:
            loss_pde += physics_loss(model, x_train_subdomain, qy)
        if x0 != min([carga['x0'] for carga in trechos_cargas]):
            loss_int += interface_loss(model, x0)

    return loss_pde, loss_int

def boundary_loss(model, dados_apoios):
    loss_bc = 0.0
    for k, dict_apoio in enumerate(dados_apoios):

        x0 = torch.tensor([[dict_apoio['x']]], requires_grad=True)

        u_0 = model(x0)

        u0_x = torch.autograd.grad(u_0, x0, torch.ones_like(u_0), create_graph=True)[0]
        u0_xx = torch.autograd.grad(u0_x, x0, torch.ones_like(u0_x), create_graph=True)[0]
        u0_xxx = torch.autograd.grad(u0_xx, x0, torch.ones_like(u0_xx), create_graph=True)[0]

        if dict_apoio['graus'][1] == 1:
            loss_bc += (u_0)**2
        else:
            loss_bc += (u0_xxx)**2

        if dict_apoio['graus'][2] == 1: 
            loss_bc += (u0_x)**2
        else:
            loss_bc += (u0_xx)**2

    return loss_bc.mean()

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
    
    def tratar_apoios(self, lista_apoios):
        valores_x = []
        for apoio in lista_apoios:
            valores_x.append(apoio.x)

        x0 = min(valores_x)
        xf = max(valores_x)

        L = xf - x0

        dados_apoios = []

        for apoio in lista_apoios:
            x_norm = (apoio.x - x0) / L
            dict_temp = {
                "x" : x_norm,
                "graus" : apoio.graus
            }
            dados_apoios.append(dict_temp)

        return float(L), dados_apoios
    
    # def ver_apoios():


    # def closure():
    #     optimizer.zero_grad()

    def run_model(self, lista_apoios, lista_cargas, EI, 
                  num_epochs=5000, pde_weight=1.0, bc_weight=1.0,
                  tol=1e-5, tol_apoio = 1e-5, tam=101):
        
        L, dados_apoios = self.tratar_apoios(lista_apoios)

        set_seed(1)

        self.model = PINNViga()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.u_plot_variation = []
        self.loss_pde_variation = []
        self.loss_bc_variation = []
        self.loss_int_variation = []

        self.trechos_cargas = obter_trechos_cargas(lista_apoios, lista_cargas)
        max_qy = max_qy = max([abs(carga['qy']) for carga in self.trechos_cargas])

        u_ref = max_qy * (L**4) / EI

        for epoch in range(num_epochs + 1):

            # if epoch >= max([0.3 * num_epochs, 350]):
            #     optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.001, max_iter=int(round(0.7 * num_epochs,0)))

            optimizer.zero_grad()

            loss_pde, loss_int = trecho_physics_loss(self.model, self.trechos_cargas)
            loss_bc = boundary_loss(self.model, dados_apoios)

            loss = pde_weight * loss_pde + bc_weight * loss_bc + loss_int

            loss.backward()            
            optimizer.step()

            if (float(loss_bc) <= tol) and (float(loss_pde) <= tol):
                print("Treinamento concluído pelo critério de tolerância da perda")
                print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}")
                print(f"Treinamento interrompido na época {epoch}.")
                break
                       
            if epoch % int(num_epochs / 10) == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}")

            x_plot = torch.linspace(0, 1, tam).view(-1,1)
            u_plot_star = self.model(x_plot).detach().numpy()
            self.u_plot = u_plot_star * u_ref
            self.x_plot = (x_plot.numpy() * L)
            self.u_plot_variation.append((epoch, self.u_plot))
            self.loss_pde_variation.append(loss_pde)
            self.loss_bc_variation.append(loss_bc)
            self.loss_int_variation.append(loss_int)

            ver_apoio = 0
            lista_x = [float(x) for x in self.x_plot]
            lista_u = [float(u) for u in self.u_plot]

            for k, apoio in enumerate(lista_apoios):
                idx = int(lista_x.index(float(apoio.x)))
                if abs(lista_u[idx]) <= tol_apoio:
                    ver_apoio += 1

            if ver_apoio == len(lista_apoios):
                print("Treinamento concluído pelo critério do deslocamento nos apoios")
                print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}")
                print(f"Treinamento interrompido na época {epoch}.")
                break

        self.loss = self.loss_pde_variation + self.loss_bc_variation + self.loss_int_variation

    def plot_errors(self, scale, nome = "erros.png"):
        valores_epoch = np.arange(0, len(self.u_plot_variation), 1)
        valores_bc = [float(loss) for loss in self.loss_bc_variation]
        valores_pde = [float(loss) for loss in self.loss_pde_variation]
        valores_int = [float(loss) for loss in self.loss_int_variation]
        valores_total = [bc + pde for bc, pde, int in zip(valores_bc, valores_pde, valores_int)]

        plt.figure(figsize=(8, 4.5))
        plt.plot(valores_epoch, valores_bc, label = 'Perda CC', color = 'red', lw = 1)
        plt.plot(valores_epoch, valores_pde, label = 'Perda ED', color = 'blue', lw = 1)
        plt.plot(valores_epoch, valores_int, label = 'Perda INT', color = 'orange', lw = 1)
        plt.plot(valores_epoch, valores_total, label = 'Perda Total', color = 'green', lw = 1)
        plt.title("Variação do erro durante o treinamento")
        plt.xlabel('Época')
        plt.ylabel('Erro')
        plt.xlim(0, max(valores_epoch))
        plt.ylim(0, 1.05 * max(valores_total) / scale)
        plt.legend()
        plt.savefig(nome, dpi = 300)
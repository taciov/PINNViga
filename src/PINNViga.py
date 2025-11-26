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

def inserir_apoios_no_trecho(lista_apoios, trechos_cargas):
    for k, trecho in enumerate(trechos_cargas):
        x0 = trecho['x0']
        x1 = trecho['x1']

        for apoio in lista_apoios:
            if apoio.x == x0:
                trecho['apoio0'] = apoio
                apoio.trecho_dir = k

            if apoio.x == x1:
                trecho['apoio1'] = apoio
                apoio.trecho_esq = k

    for apoio in lista_apoios:
        if apoio.trecho_esq is not None and apoio.trecho_dir is not None:
            apoio.tipo = "int"
        else:
            apoio.tipo = "ext"

    return trechos_cargas

def obter_trechos_cargas(lista_apoios, lista_cargas, EI):
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
            "qy" : 0,
            "L" : abs(posicoes_x[k + 1] - posicoes_x[k]),
            "EI" : EI
        }

        for carga in lista_cargas:
            if carga.x0 <= posicoes_x[k] and carga.x1 >= posicoes_x[k+1]:
                dict_trecho['qx'] += carga.qx
                dict_trecho['qy'] += carga.qy

        trechos_cargas.append(dict_trecho)

    trechos_cargas = inserir_apoios_no_trecho(lista_apoios, trechos_cargas)

    for carga in trechos_cargas:
            
        tam = max(int(round(abs(carga['x1'] - carga['x0']) * 100, 0)), 21)

        torch.manual_seed(1)
        min_val = 0
        max_val = 1
        x_scaled = min_val + (max_val - min_val) * torch.rand(tam, 1)

        carga['vetor_x'] = x_scaled.requires_grad_(True)
        carga["u_ref"] = carga['qy'] * carga['L'] ** 4 / (carga['EI'])


    return trechos_cargas

class FormaFraca:

    def u(self, model, x):
         return model(x)

    def physics_loss_trecho(self, x, u, q):
       
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx, x, torch.ones_like(u_xxx), create_graph=True)[0]
        
        f = u_xxxx - q

        return torch.mean((f) ** 2)

    def physics_loss(self, model, trechos_cargas, qy_max):

        loss_pde = 0

        for k, carga in enumerate(trechos_cargas):

            qy = carga['qy'] / qy_max
            x = carga['vetor_x']
 
            u = model(x)
            u_local = u[:, k:k+1]

            loss_pde += self.physics_loss_trecho(x, u_local, qy)

        return loss_pde
    
    def boundary_loss_apoio(self, u, x, apoio):

        variacao_loss_bc = 0

        u0_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u0_xx = torch.autograd.grad(u0_x, x, torch.ones_like(u0_x), create_graph=True)[0]
        u0_xxx = torch.autograd.grad(u0_xx, x, torch.ones_like(u0_xx), create_graph=True)[0]

        if apoio.graus[1] == 1:
            variacao_loss_bc += (u)**2
        else:
            variacao_loss_bc += (u0_xxx)**2

        if apoio.graus[2] == 1: 
            variacao_loss_bc += (u0_x)**2
        else:
            if apoio.tipo == "ext":
                variacao_loss_bc += (u0_xx)**2
            else:
                pass

        return variacao_loss_bc
    
    def boundary_loss(self, model, lista_apoios):
        loss_bc = 0.0
        for apoio in lista_apoios:

            if apoio.trecho_esq is not None:

                x = torch.tensor([[1.0]], requires_grad=True)
                u_esq = model(x)[:, apoio.trecho_esq : apoio.trecho_esq + 1]
                loss_bc += self.boundary_loss_apoio(u_esq, x, apoio)

            if apoio.trecho_dir is not None:

                x = torch.tensor([[0.0]], requires_grad=True)
                u_dir = model(x)[:, apoio.trecho_dir : apoio.trecho_dir + 1]
                loss_bc += self.boundary_loss_apoio(u_dir, x, apoio)

        return loss_bc

    def interface_loss_apoio(self, u_esq, u_dir, x_esq, x_dir, graus):

        loss_cont = 0

        u1_esq = torch.autograd.grad(u_esq, x_esq, torch.ones_like(u_esq), create_graph=True)[0]
        u2_esq = torch.autograd.grad(u1_esq, x_esq, torch.ones_like(u1_esq), create_graph=True)[0]

        u1_dir = torch.autograd.grad(u_dir, x_dir, torch.ones_like(u_dir), create_graph=True)[0]
        u2_dir = torch.autograd.grad(u1_dir, x_dir, torch.ones_like(u1_dir), create_graph=True)[0]

        loss_cont += (u_esq - u_dir)**2
        loss_cont += (u1_esq - u1_dir)**2
        loss_cont += (u2_esq - u2_dir)**2

        return loss_cont

    def interface_loss(self, model, lista_apoios):
        loss_int = 0.0
        for apoio in lista_apoios:

            if apoio.tipo == 'int':

                x_esq = torch.tensor([[1.0]], requires_grad=True)
                u_esq = model(x_esq)[:, apoio.trecho_esq:apoio.trecho_esq + 1]
                x_dir = torch.tensor([[0.0]], requires_grad=True)
                u_dir = model(x_dir)[:, apoio.trecho_dir:apoio.trecho_dir + 1]
                loss_int += self.interface_loss_apoio(u_esq, u_dir, x_esq, x_dir, apoio.graus)
   
        if torch.is_tensor(loss_int):
            return loss_int.mean()
        else:
            return torch.tensor(0.0, requires_grad=True)

class PINN(nn.Module):
    def __init__(self, num_outputs=1, width=32, depth=3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, num_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PINNViga:
    def __init__(self, lista_apoios, lista_cargas, E, I,
                 formulacao = "fraca", width = 32, depth = 3):

        self.lista_apoios = lista_apoios
        self.lista_cargas = lista_cargas
        self.EI = E * I

        self.trechos_cargas = obter_trechos_cargas(self.lista_apoios, self.lista_cargas, self.EI)
        num_camadas_output = len(self.trechos_cargas)

        self.qy_max = max([abs(carga['qy']) for carga in self.trechos_cargas])
        self.L_total = sum([carga['L'] for carga in self.trechos_cargas])

        if formulacao == 'fraca':
            self.formulacao = FormaFraca()
        else:
            raise NotImplementedError("Somente 'fraca' implementado nesta versão.")
        
        self.model = PINN(num_outputs=num_camadas_output, width=width, depth = depth)
    
    def run_model(self, num_epochs=5000, pde_weight=1.0, bc_weight=1.0, int_weight = 1.0,
                  tol=1e-5, tol_apoio = 1e-5, tam=51):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.u_plot_variation = []
        self.loss_pde_variation = []
        self.loss_bc_variation = []
        self.loss_int_variation = []

        for epoch in range(num_epochs + 1):
            optimizer.zero_grad()

            loss_pde = self.formulacao.physics_loss(self.model, self.trechos_cargas, self.qy_max)
            loss_bc = self.formulacao.boundary_loss(self.model, self.lista_apoios)
            loss_int = self.formulacao.interface_loss(self.model, self.lista_apoios)
        
            loss = pde_weight * loss_pde + bc_weight * loss_bc + int_weight * loss_int

            loss.backward()            
            optimizer.step()

            if (float(loss_bc) <= tol) and (float(loss_pde) <= tol):
                print("Treinamento concluído pelo critério de tolerância da perda")
                print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}, INT Loss: {loss_int.item():.12f}")
                print(f"Treinamento interrompido na época {epoch}.")
                break
                       
            if epoch % int(num_epochs / 10) == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}, INT Loss: {loss_int.item():.12f}")

            # x_plot = torch.linspace(0, 1, tam).view(-1,1)
            # u_plot_star = self.model(x_plot).detach().numpy()
            # self.u_plot = u_plot_star
            # self.x_plot = (x_plot.numpy() * self.L)
            # self.u_plot_variation.append((epoch, self.u_plot))
            self.loss_pde_variation.append(loss_pde)
            self.loss_bc_variation.append(loss_bc)
            self.loss_int_variation.append(loss_int)

            # ver_apoio = 0
            # lista_x = [float(x) for x in self.x_plot]
            # lista_u = [float(u) for u in self.u_plot]

            # for k, apoio in enumerate(self.lista_apoios):
            #     idx = int(lista_x.index(float(apoio.x)))
            #     if abs(lista_u[idx]) <= tol_apoio:
            #         ver_apoio += 1

            # if ver_apoio == len(self.lista_apoios):
            #     print("Treinamento concluído pelo critério do deslocamento nos apoios")
            #     print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}")
            #     print(f"Treinamento interrompido na época {epoch}.")
            #     break

        self.loss = self.loss_pde_variation + self.loss_bc_variation + self.loss_int_variation

    def plot_errors(self, scale, nome = "erros.png"):
        valores_epoch = np.arange(0, len(self.loss_bc_variation), 1)
        valores_bc = [float(loss) for loss in self.loss_bc_variation]
        valores_pde = [float(loss) for loss in self.loss_pde_variation]
        valores_int = [float(loss) for loss in self.loss_int_variation]
        valores_total = [bc + pde + int for bc, pde, int in zip(valores_bc, valores_pde, valores_int)]

        plt.figure(figsize=(8, 4.5))
        plt.plot(valores_epoch, valores_bc, label = 'Perda CC', color = 'red', lw = 1)
        plt.plot(valores_epoch, valores_pde, label = 'Perda EDO', color = 'blue', lw = 1)
        plt.plot(valores_epoch, valores_int, label = 'Perda CONT', color = 'orange', lw = 1)
        plt.plot(valores_epoch, valores_total, label = 'Perda Total', color = 'green', lw = 1)
        plt.title("Variação do erro durante o treinamento")
        plt.xlabel('Época')
        plt.ylabel('Erro')
        plt.xlim(0, max(valores_epoch))
        plt.ylim(0, 1.05 * max(valores_total) / scale)
        plt.legend()
        plt.savefig(nome, dpi = 300)

    # def plot_deslocamento(self, nome = "deslocamento.png"):
    #     x_plot = torch.linspace(0, 1, 101).view(-1,1)
    #     u_plot_star = self.model(x_plot).detach().numpy()
    #     self.u_plot = u_plot_star
    #     self.x_plot = (x_plot.numpy() * L)

    #     u_pred = self.model(x_plot).detach().numpy()

    #     for k, trecho in enumerate(self.trechos_cargas):

    #         print(trecho)
    #         x0 = trecho['x0']
    #         x1 = trecho['x1']
    #         u_trecho = u_pred[:, k : k+1]
    #         x_trecho = np.linspace(x0, x1, len(u_trecho))
    #         plt.plot(x_trecho, u_trecho, color = 'blue')
    #         plt.xlabel("Comprimento (m)")
    #         plt.ylabel("Deslocamento (m)")
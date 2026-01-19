import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from anastruct import SystemElements
from src import Output

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
        
class FormaForte(FormaFraca):

    def u(self, model, x, apoio_esq=None, apoio_dir=None):

        g = torch.ones_like(x)

        # apoio esquerdo com deslocamento imposto
        if apoio_esq is not None and apoio_esq.graus[1] == 1:
            g = g * x

        # apoio direito com deslocamento imposto
        if apoio_dir is not None and apoio_dir.graus[1] == 1:
            g = g * (1 - x)

        return g * model(x)

def physics_loss(self, model, trechos_cargas, qy_max):

    loss_pde = 0.0

    for k, carga in enumerate(trechos_cargas):

        qy = carga['qy'] / qy_max
        x = carga['vetor_x']

        apoio_esq = carga.get('apoio_esq', None)
        apoio_dir = carga.get('apoio_dir', None)

        u_raw = model(x)[:, k:k+1]
        u = self.u(lambda x_: u_raw, x, apoio_esq, apoio_dir)

        loss_pde += self.physics_loss_trecho(x, u, qy)

    return loss_pde

def boundary_loss_apoio(self, u, x, apoio):

    variacao_loss_bc = 0.0

    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]

    if apoio.graus[2] == 1:
        variacao_loss_bc += (u_x)**2
    else:
        if apoio.tipo == "ext":
            variacao_loss_bc += (u_xx)**2

    return variacao_loss_bc

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
                 formulacao = "fraca", otimizador = "adam", lr_adam = 0.001, width = 32, depth = 3):
        
        self.verif_analitica = False
        self.otimizador = otimizador
        self.lr_adam = lr_adam

        self.lista_apoios = lista_apoios
        self.lista_cargas = lista_cargas
        self.EI = E * I

        self.trechos_cargas = obter_trechos_cargas(self.lista_apoios, self.lista_cargas, self.EI)
        num_camadas_output = len(self.trechos_cargas)

        self.qy_max = max([abs(carga['qy']) for carga in self.trechos_cargas])
        self.L_total = sum([carga['L'] for carga in self.trechos_cargas])

        if formulacao == 'fraca':
            self.formulacao = FormaFraca()
        elif formulacao == 'forte':
            self.formulacao = FormaForte()
        else:
            raise NotImplementedError("Formulação deve ser 'fraca' ou 'forte'.")
        
        self.model = PINN(num_outputs=num_camadas_output, width=width, depth = depth)

    def calc_loss(self, pde_weight, bc_weight, int_weight):
        loss_pde = self.formulacao.physics_loss(self.model, self.trechos_cargas, self.qy_max)
        loss_bc = self.formulacao.boundary_loss(self.model, self.lista_apoios)
        loss_int = self.formulacao.interface_loss(self.model, self.lista_apoios)
    
        loss = pde_weight * loss_pde + bc_weight * loss_bc + int_weight * loss_int

        return loss, loss_pde, loss_bc, loss_int
    
    def adam_step(self, optimizer, pde_weight, bc_weight, int_weight):
        optimizer.zero_grad()

        loss, loss_pde, loss_bc, loss_int = self.calc_loss(pde_weight, bc_weight, int_weight)

        loss.backward()
        optimizer.step()

        return loss, loss_pde, loss_bc, loss_int
    
    def lbfgs_step(self, optimizer, pde_weight, bc_weight, int_weight):

        def closure():
            optimizer.zero_grad()
            loss, _, _, _ = self.calc_loss(pde_weight, bc_weight, int_weight)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        loss, loss_pde, loss_bc, loss_int = self.calc_loss(pde_weight, bc_weight, int_weight)

        return loss, loss_pde, loss_bc, loss_int
    
    def run_model(self, num_epochs=5000, pde_weight=1.0, bc_weight=1.0, int_weight = 1.0,
                  tol=1e-5, tol_apoio = 1e-5, print_progresso = True):
        
        self.u_plot_variation = []
        self.loss_pde_variation = []
        self.loss_bc_variation = []
        self.loss_int_variation = []

        if self.otimizador == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr_adam)
            step_fn = self.adam_step

        elif self.otimizador == "lbfgs":
            optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )
            step_fn = self.lbfgs_step

        else:
            raise ValueError("otimizador deve ser 'adam' ou 'lbfgs'" )

        for epoch in range(num_epochs + 1):

            loss, loss_pde, loss_bc, loss_int = step_fn(optimizer, pde_weight, bc_weight, int_weight)

            if (float(loss_bc) <= tol) and (float(loss_pde) <= tol):
                print("Treinamento concluído pelo critério de tolerância da perda")
                print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}, INT Loss: {loss_int.item():.12f}")
                print(f"Treinamento interrompido na época {epoch}.")
                break

            if print_progresso is False:
                pass
            else:
                denominador = 10 if num_epochs >= 10 else 2
                if epoch % int(num_epochs / denominador) == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}, INT Loss: {loss_int.item():.12f}")

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

        self.output = Output.Output(self)

    def run_sol_analitica(self, tam = 20):
        self.output.run_sol_analitica(tam)
        self.output.save_values()

    def plot_errors(self, scale, nome = "erros.png"):
        self.output.plot_errors(scale, nome)

    def plot_deslocamento(self, plot_analitico = True, nome = "deslocamento.png"):
        self.output.plot_deslocamento(plot_analitico, nome)

    def plot_rotacao(self, plot_analitico = True, nome = "rotacao.png"):
        self.output.plot_rotacao(plot_analitico, nome)

    def plot_momento(self, plot_analitico = True, nome = "momento.png"):
        self.output.plot_momento(plot_analitico, nome)

    def plot_cortante(self, plot_analitico = True, nome = "cortante.png"):
        self.output.plot_cortante(plot_analitico, nome)
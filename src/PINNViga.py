import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from anastruct import SystemElements

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
        
        self.verif_analitica = False

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
                  tol=1e-5, tol_apoio = 1e-5):
        
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

    def run_sol_analitica(self, tam = 20):
        sistema = SystemElements(EI = self.EI, EA=1e12)

        for k, trecho in enumerate(self.trechos_cargas):
            x0 = trecho['x0']
            x1 = trecho['x1']
            dx = (x1 - x0) / tam

            for i in range(tam):
                sistema.add_element([[x0 + i*dx,0], [x0 + (i+1) * dx,0]])
                sistema.q_load(q = trecho['qy'], element_id = sistema.id_last_element, direction = 'element')

        lista_ids_apoios = [1 + k * tam for k in range(len(self.trechos_cargas) + 1)]

        for id_atual, apoio in zip(lista_ids_apoios, self.lista_apoios):
            if apoio.graus == [1, 1, 0]:
                sistema.add_support_hinged(node_id=id_atual)
            elif apoio.graus == [0, 1, 0]:
                sistema.add_support_roll(node_id = id_atual)
            elif apoio.graus == [1, 1, 1]:
                sistema.add_support_fixed(node_id = id_atual)

        sistema.solve()

        self.x_analitico = [vars(list(sistema._vertices.keys())[k])['coordinates'][0] for k in range(len(vars(sistema)['_vertices']))]
        self.x_analitico2 = []
        for k in range(len(self.x_analitico) - 1):
            self.x_analitico2.append([self.x_analitico[k + 1], self.x_analitico[k]])
        self.uy_analitico = [float(- sistema.get_node_results_system()[id]['uy']) for id in range(len(sistema.get_node_results_system()))]
        self.rz_analitico = [float(sistema.get_node_results_system()[id]['phi_z']) for id in range(len(sistema.get_node_results_system()))]
        self.fy_analitico = [float(sistema.get_node_results_system()[id]['Fy']) for id in range(len(sistema.get_node_results_system()))]
        self.fy_analitico = []
        for k, barra in enumerate(sistema.get_element_results()):
            self.fy_analitico.append([float(-barra['Qmax']), float(-barra['Qmin'])])
        self.mf_analitico = []
        for k, barra in enumerate(sistema.get_element_results()):
            self.mf_analitico.append([float(-barra['Mmin']), float(-barra['Mmax'])])

        for k in range(1, len(self.mf_analitico)):
            if round(self.mf_analitico[k][1], 2) != round(self.mf_analitico[k - 1][0], 2):
                temp = [self.mf_analitico[k][1], self.mf_analitico[k][0]]
                self.mf_analitico[k] = temp

        self.verif_analitica = True

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

    def plot_deslocamento(self, plot_analitico = True, nome = "deslocamento.png"):
        x_plot = torch.linspace(0, 1, 101).view(-1,1)
        u_plot_star = self.model(x_plot).detach().numpy()
        self.u_plot = u_plot_star

        u_pred = self.model(x_plot).detach().numpy()
        plt.figure(figsize=(7.5,4.5))

        for k, trecho in enumerate(self.trechos_cargas):
            x0 = trecho['x0']
            x1 = trecho['x1']
            u_trecho = u_pred[:, k : k+1] * trecho['u_ref'] * (-1)
            x_trecho = np.linspace(x0, x1, len(u_trecho))
            if k == 0:
                plt.plot(x_trecho, u_trecho, color = 'blue', label = "Solução PINN")

            else:
                plt.plot(x_trecho, u_trecho, color = 'blue')

        if plot_analitico is True:
            if self.verif_analitica is False:
                self.run_sol_analitica()
            plt.plot(self.x_analitico, self.uy_analitico, color = 'red', label = "Solução analítica", ls = '--')
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Deslocamento (m)")
        plt.legend()
        plt.savefig(nome, dpi = 300)

    def plot_rotacao(self, plot_analitico = True, nome="rotacao.png"):
        x_plot = torch.linspace(0, 1, 101).view(-1,1)
        plt.figure(figsize=(7.5,4.5))

        for k, trecho in enumerate(self.trechos_cargas):

            x = x_plot.clone().requires_grad_(True)
            u = self.model(x)[:, k:k+1]

            u1 = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]  # rotação

            theta_ref = trecho['qy'] * trecho['L'] ** 3 / trecho['EI']

            theta = theta_ref * (u1).detach().numpy()
            xx = np.linspace(trecho['x0'], trecho['x1'], len(theta))

            if k == 0:
                plt.plot(xx, theta, color="green", label="Solução PINN")
            else:
                plt.plot(xx, theta, color="green")

        if plot_analitico is True:
            if self.verif_analitica is False:
                self.run_sol_analitica()
            plt.plot(self.x_analitico, self.rz_analitico, color = 'red', label = "Solução analítica", ls = '--')
        plt.title("Rotação")
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Rotação (rad)")
        plt.legend()
        plt.savefig(nome, dpi=300)

    def plot_momento(self, plot_analitico = True, nome="momento.png"):
        x_plot = torch.linspace(0, 1, 101).view(-1,1)
        plt.figure(figsize=(7.5,4.5))

        for k, trecho in enumerate(self.trechos_cargas):

            x = x_plot.clone().requires_grad_(True)
            u = self.model(x)[:, k:k+1]

            u1 = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            u2 = torch.autograd.grad(u1, x, torch.ones_like(u1), create_graph=True)[0]

            m_ref = trecho['qy'] * trecho['L'] ** 2

            M = - m_ref * (u2).detach().numpy()  # ← convencional em viga Euler-Bernoulli
            xx = np.linspace(trecho['x0'], trecho['x1'], len(M))

            if k == 0:
                plt.plot(xx, M, color="blue", label="Solução PINN")
            else:
                plt.plot(xx, M, color="blue")

        if plot_analitico is True:
            if self.verif_analitica is False:
                self.run_sol_analitica()
            for k, (x, M) in enumerate(zip(self.x_analitico2, self.mf_analitico)):
                if k == 0:
                    plt.plot(x, M, color = 'red', label = "Solução analítica", ls = '--')
                else:
                    plt.plot(x, M, color = 'red', ls = '--')

        plt.title("Momento Fletor")
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Momento Fletor (N.m)")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(nome, dpi=300)

    def plot_cortante(self, plot_analitico = True, nome="cortante.png"):
        x_plot = torch.linspace(0, 1, 101).view(-1,1)
        plt.figure(figsize=(7.5,4.5))

        for k, trecho in enumerate(self.trechos_cargas):

            x = x_plot.clone().requires_grad_(True)
            u = self.model(x)[:, k:k+1]

            u1 = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            u2 = torch.autograd.grad(u1, x, torch.ones_like(u1), create_graph=True)[0]
            u3 = torch.autograd.grad(u2, x, torch.ones_like(u2), create_graph=True)[0]

            v_ref = trecho['qy'] * trecho['L']

            V = - v_ref * (u3).detach().numpy()  # sinal direto — manteve padrão clássico
            xx = np.linspace(trecho['x0'], trecho['x1'], len(V))

            if k == 0:
                plt.plot(xx, V, color="blue", label="Solução PINN")
            else:
                plt.plot(xx, V, color="blue")

        if plot_analitico is True:
            if self.verif_analitica is False:
                self.run_sol_analitica()
            for k, (x, Q) in enumerate(zip(self.x_analitico2, self.fy_analitico)):
                if k == 0:
                    plt.plot(x, Q, color = 'red', label = "Solução analítica", ls = '--')
                else:
                    plt.plot(x, Q, color = 'red', ls = '--')

        plt.title("Esforço Cortante")
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Esforço Cortante (N)")
        plt.legend()
        plt.savefig(nome, dpi=300)



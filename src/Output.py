import matplotlib.pyplot as plt
from anastruct import SystemElements
import torch
import numpy as np

class Output:

    def __init__(self, input_pinn):
        self.pinn = input_pinn
        
    def run_sol_analitica(self, tam = 20):
        sistema = SystemElements(EI = self.pinn.EI, EA=1e12)

        for k, trecho in enumerate(self.pinn.trechos_cargas):
            x0 = trecho['x0']
            x1 = trecho['x1']
            dx = (x1 - x0) / tam

            for i in range(tam):
                sistema.add_element([[x0 + i*dx,0], [x0 + (i+1) * dx,0]])
                sistema.q_load(q = trecho['qy'], element_id = sistema.id_last_element, direction = 'element')

        lista_ids_apoios = [1 + k * tam for k in range(len(self.pinn.trechos_cargas) + 1)]

        for id_atual, apoio in zip(lista_ids_apoios, self.pinn.lista_apoios):
            if apoio.graus == [1, 1, 0]:
                sistema.add_support_hinged(node_id=id_atual)
            elif apoio.graus == [0, 1, 0]:
                sistema.add_support_roll(node_id = id_atual)
            elif apoio.graus == [1, 1, 1]:
                sistema.add_support_fixed(node_id = id_atual)

        sistema.solve()

        self.pinn.x_analitico = [vars(list(sistema._vertices.keys())[k])['coordinates'][0] for k in range(len(vars(sistema)['_vertices']))]
        self.pinn.x_analitico2 = []
        for k in range(len(self.pinn.x_analitico) - 1):
            self.pinn.x_analitico2.append([self.pinn.x_analitico[k + 1], self.pinn.x_analitico[k]])
        self.pinn.uy_analitico = [float(- sistema.get_node_results_system()[id]['uy']) for id in range(len(sistema.get_node_results_system()))]
        self.pinn.rz_analitico = [float(sistema.get_node_results_system()[id]['phi_z']) for id in range(len(sistema.get_node_results_system()))]
        self.pinn.fy_analitico = [float(sistema.get_node_results_system()[id]['Fy']) for id in range(len(sistema.get_node_results_system()))]
        self.pinn.fy_analitico = []
        for k, barra in enumerate(sistema.get_element_results()):
            self.pinn.fy_analitico.append([float(-barra['Qmax']), float(-barra['Qmin'])])
        self.pinn.mf_analitico = []
        for k, barra in enumerate(sistema.get_element_results()):
            self.pinn.mf_analitico.append([float(-barra['Mmin']), float(-barra['Mmax'])])

        for k in range(1, len(self.pinn.mf_analitico)):
            if round(self.pinn.mf_analitico[k][1], 2) != round(self.pinn.mf_analitico[k - 1][0], 2):
                temp = [self.pinn.mf_analitico[k][1], self.pinn.mf_analitico[k][0]]
                self.pinn.mf_analitico[k] = temp

        self.pinn.verif_analitica = True

    def save_values(self):
        x_plot = torch.linspace(0, 1, 101).view(-1,1)
        u_plot_star = self.pinn.model(x_plot).detach().numpy()
        self.pinn.u_plot = u_plot_star

        u_pred = self.pinn.model(x_plot).detach().numpy()
        plt.figure(figsize=(7.5,4.5))

        for k, trecho in enumerate(self.pinn.trechos_cargas):
            x0 = trecho['x0']
            x1 = trecho['x1']
            u_trecho = u_pred[:, k : k+1] * trecho['u_ref'] * (-1)
            x_trecho = np.linspace(x0, x1, len(u_trecho))
            trecho['valores_x'] = x_trecho
            trecho['valores_u'] = u_trecho

            x = x_plot.clone().requires_grad_(True)
            u = self.pinn.model(x)[:, k:k+1]
            u1 = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            u2 = torch.autograd.grad(u1, x, torch.ones_like(u1), create_graph=True)[0]
            u3 = torch.autograd.grad(u2, x, torch.ones_like(u2), create_graph=True)[0]

            theta_ref = trecho['qy'] * trecho['L'] ** 3 / trecho['EI']
            theta = theta_ref * (u1).detach().numpy()
            trecho['valores_theta'] = theta

            m_ref = trecho['qy'] * trecho['L'] ** 2
            M = - m_ref * (u2).detach().numpy()
            trecho['valores_mf'] = M

            v_ref = trecho['qy'] * trecho['L']
            V = - v_ref * (u3).detach().numpy()
            trecho['valores_V'] = V

    def plot_errors(self, scale, nome = "erros.png"):
        valores_epoch = np.arange(0, len(self.pinn.loss_bc_variation), 1)
        valores_bc = [float(loss) for loss in self.pinn.loss_bc_variation]
        valores_pde = [float(loss) for loss in self.pinn.loss_pde_variation]
        valores_int = [float(loss) for loss in self.pinn.loss_int_variation]
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
        u_plot_star = self.pinn.model(x_plot).detach().numpy()
        self.pinn.u_plot = u_plot_star

        u_pred = self.pinn.model(x_plot).detach().numpy()
        plt.figure(figsize=(7.5,4.5))

        for k, trecho in enumerate(self.pinn.trechos_cargas):
            x0 = trecho['x0']
            x1 = trecho['x1']
            u_trecho = u_pred[:, k : k+1] * trecho['u_ref'] * (-1)
            x_trecho = np.linspace(x0, x1, len(u_trecho))
            trecho['valores_x'] = x_trecho
            trecho['valores_u'] = u_trecho
            if k == 0:
                plt.plot(x_trecho, u_trecho, color = 'blue', label = "Solução PINN")

            else:
                plt.plot(x_trecho, u_trecho, color = 'blue')

        if plot_analitico is True:
            if self.pinn.verif_analitica is False:
                self.pinn.run_sol_analitica()
            plt.plot(self.pinn.x_analitico, self.pinn.uy_analitico, color = 'red', label = "Solução analítica", ls = '--')
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Deslocamento (m)")
        plt.title("Linha Elástica")
        plt.legend()
        plt.savefig(nome, dpi = 300)

    def plot_rotacao(self, plot_analitico = True, nome="rotacao.png"):
        x_plot = torch.linspace(0, 1, 101).view(-1,1)
        plt.figure(figsize=(7.5,4.5))

        for k, trecho in enumerate(self.pinn.trechos_cargas):

            x = x_plot.clone().requires_grad_(True)
            u = self.pinn.model(x)[:, k:k+1]

            u1 = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]  # rotação

            theta_ref = trecho['qy'] * trecho['L'] ** 3 / trecho['EI']

            theta = theta_ref * (u1).detach().numpy()
            xx = np.linspace(trecho['x0'], trecho['x1'], len(theta))

            trecho['valores_x'] = xx
            trecho['valores_theta'] = theta

            if k == 0:
                plt.plot(xx, theta, color="blue", label="Solução PINN")
            else:
                plt.plot(xx, theta, color="blue")

        if plot_analitico is True:
            if self.pinn.verif_analitica is False:
                self.pinn.run_sol_analitica()
            plt.plot(self.pinn.x_analitico, self.pinn.rz_analitico, color = 'red', label = "Solução analítica", ls = '--')
        plt.title("Rotação")
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Rotação (rad)")
        plt.legend()
        plt.savefig(nome, dpi=300)

    def plot_momento(self, plot_analitico = True, nome="momento.png"):
        x_plot = torch.linspace(0, 1, 101).view(-1,1)
        plt.figure(figsize=(7.5,4.5))

        for k, trecho in enumerate(self.pinn.trechos_cargas):

            x = x_plot.clone().requires_grad_(True)
            u = self.pinn.model(x)[:, k:k+1]

            u1 = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            u2 = torch.autograd.grad(u1, x, torch.ones_like(u1), create_graph=True)[0]

            m_ref = trecho['qy'] * trecho['L'] ** 2

            M = - m_ref * (u2).detach().numpy()
            xx = np.linspace(trecho['x0'], trecho['x1'], len(M))

            trecho['valores_x'] = xx
            trecho['valores_mf'] = M

            if k == 0:
                plt.plot(xx, M, color="blue", label="Solução PINN")
            else:
                plt.plot(xx, M, color="blue")

        if plot_analitico is True:
            if self.pinn.verif_analitica is False:
                self.pinn.run_sol_analitica()
            for k, (x, M) in enumerate(zip(self.pinn.x_analitico2, self.pinn.mf_analitico)):
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

        for k, trecho in enumerate(self.pinn.trechos_cargas):

            x = x_plot.clone().requires_grad_(True)
            u = self.pinn.model(x)[:, k:k+1]

            u1 = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            u2 = torch.autograd.grad(u1, x, torch.ones_like(u1), create_graph=True)[0]
            u3 = torch.autograd.grad(u2, x, torch.ones_like(u2), create_graph=True)[0]

            v_ref = trecho['qy'] * trecho['L']

            V = - v_ref * (u3).detach().numpy()
            xx = np.linspace(trecho['x0'], trecho['x1'], len(V))

            trecho['valores_x'] = xx
            trecho['valores_V'] = V

            if k == 0:
                plt.plot(xx, V, color="blue", label="Solução PINN")
            else:
                plt.plot(xx, V, color="blue")

        if plot_analitico is True:
            if self.pinn.verif_analitica is False:
                self.pinn.run_sol_analitica()
            for k, (x, Q) in enumerate(zip(self.pinn.x_analitico2, self.pinn.fy_analitico)):
                if k == 0:
                    plt.plot(x, Q, color = 'red', label = "Solução analítica", ls = '--')
                else:
                    plt.plot(x, Q, color = 'red', ls = '--')

        plt.title("Esforço Cortante")
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Esforço Cortante (N)")
        plt.legend()
        plt.savefig(nome, dpi=300)
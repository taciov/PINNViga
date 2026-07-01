import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from anastruct import SystemElements
import torch
import numpy as np
import copy

class Output:

    def __init__(self, input_pinn):
        self.pinn = input_pinn
        
    def run_sol_analitica(self, tam = 20):
        sistema = SystemElements(EI = self.pinn.EI, EA=1e12)

        if isinstance(self.pinn.trechos_cargas, list):
            trechos_cargas = self.pinn.trechos_cargas
        else:
            trechos_cargas = self.pinn.trechos_cargas['resultante']

        for k, trecho in enumerate(trechos_cargas):
            x0 = trecho['x0']
            x1 = trecho['x1']
            dx = (x1 - x0) / tam

            for i in range(tam):
                sistema.add_element([[x0 + i*dx,0], [x0 + (i+1) * dx,0]])
                sistema.q_load(q = trecho['qy'], element_id = sistema.id_last_element, direction = 'element')

        lista_ids_apoios = [1 + k * tam for k in range(len(trechos_cargas) + 1)]

        for id_atual, apoio in zip(lista_ids_apoios, self.pinn.lista_apoios):
            if apoio.graus == [1, 1, 0]:
                sistema.add_support_hinged(node_id=id_atual)
            elif apoio.graus == [0, 1, 0]:
                sistema.add_support_roll(node_id = id_atual)
            elif apoio.graus == [1, 1, 1] or apoio.graus == [1, 1, 1]:
                sistema.add_support_fixed(node_id = id_atual)

        for carga in self.pinn.lista_cargas:
            if carga.tipo == 'pontual':
                if carga.fy != 0:
                    sistema.point_load(node_id=sistema.find_node_id([carga.x, carga.y]), Fy = carga.fy)
            if carga.tipo == 'momento':
                if carga.mz != 0:
                    sistema.moment_load(node_id=sistema.find_node_id([carga.x, carga.y]), Ty = carga.mz)

        self.pinn.sistema = sistema ## TESTES
        sistema.solve()

        self.pinn.x_analitico = [vars(list(sistema._vertices.keys())[k])['coordinates'][0] for k in range(len(vars(sistema)['_vertices']))]
        self.pinn.x_analitico2 = []
        for k in range(len(self.pinn.x_analitico) - 1):
            self.pinn.x_analitico2.append([self.pinn.x_analitico[k + 1], self.pinn.x_analitico[k]])
        self.pinn.uy_analitico = [float(- sistema.get_node_results_system()[id]['uy'] * 1e3) for id in range(len(sistema.get_node_results_system()))]
        self.pinn.rz_analitico = [float(sistema.get_node_results_system()[id]['phi_z']) for id in range(len(sistema.get_node_results_system()))]
        self.pinn.fy_analitico = [float(sistema.get_node_results_system()[id]['Fy'] * 1e-3) for id in range(len(sistema.get_node_results_system()))]
        self.pinn.fy_analitico = []
        for k, barra in enumerate(sistema.get_element_results()):
            self.pinn.fy_analitico.append([float(-barra['Qmax']) * 1e-3, float(-barra['Qmin']) * 1e-3])
        self.pinn.mf_analitico = []
        for k, barra in enumerate(sistema.get_element_results()):
            self.pinn.mf_analitico.append([float(-barra['Mmin'] * 1e-3), float(-barra['Mmax'] * 1e-3)])

        for k in range(1, len(self.pinn.mf_analitico)):
            if round(self.pinn.mf_analitico[k][1], 2) != round(self.pinn.mf_analitico[k - 1][0], 2):
                temp = [self.pinn.mf_analitico[k][1], self.pinn.mf_analitico[k][0]]
                self.pinn.mf_analitico[k] = temp

        self.pinn.verif_analitica = True

    def save_values(self):
        x_plot = torch.linspace(0, 1, 101).view(-1,1)

        if isinstance(self.pinn.trechos_cargas, list):
            u_pred = self.pinn.model(x_plot).detach().numpy()
            for k, trecho in enumerate(self.pinn.trechos_cargas):
                self.save_values_por_carga(k, trecho, x_plot, u_pred, self.pinn.model)
        else:
            for key in self.pinn.trechos_cargas:
                trechos = self.pinn.trechos_cargas[key]
                model = self.pinn.model[key]
                u_pred = model(x_plot).detach().numpy()
                for k, trecho in enumerate(trechos):
                    self.save_values_por_carga(k, trecho, x_plot, u_pred, model)

            lista_keys = [k for k in self.pinn.trechos_cargas.keys() if k != 'resultante']
            self.pinn.trechos_cargas['resultante'] = list()
            num_trechos = len(self.pinn.trechos_cargas[lista_keys[0]])
            
            for num in range(num_trechos):
                dict_num = {
                    'valores_x' : None,
                    'valores_u' : None,
                    'valores_theta' : None,
                    'valores_mf' : None,
                    'valores_v' : None
                }

                for idx, key in enumerate(lista_keys):
                    if idx == 0:
                        dict_num['x0'] = copy.deepcopy(self.pinn.trechos_cargas[key][num]['x0'])
                        dict_num['x1'] = copy.deepcopy(self.pinn.trechos_cargas[key][num]['x1'])
                        dict_num['qy'] = copy.deepcopy(self.pinn.trechos_cargas[key][num]['qy'])
                        dict_num['valores_x'] = copy.deepcopy(self.pinn.trechos_cargas[key][num]['valores_x'])
                        dict_num['valores_u'] = copy.deepcopy(self.pinn.trechos_cargas[key][num]['valores_u'])
                        dict_num['valores_theta'] = copy.deepcopy(self.pinn.trechos_cargas[key][num]['valores_theta'])
                        dict_num['valores_mf'] = copy.deepcopy(self.pinn.trechos_cargas[key][num]['valores_mf'])
                        dict_num['valores_v'] = copy.deepcopy(self.pinn.trechos_cargas[key][num]['valores_v'])
                    
                    else:
                        dict_num['qy'] += self.pinn.trechos_cargas[key][num]['qy']
                        dict_num['valores_u'] += self.pinn.trechos_cargas[key][num]['valores_u']
                        dict_num['valores_theta'] += self.pinn.trechos_cargas[key][num]['valores_theta']
                        dict_num['valores_mf'] += self.pinn.trechos_cargas[key][num]['valores_mf']
                        dict_num['valores_v'] += self.pinn.trechos_cargas[key][num]['valores_v']

                self.pinn.trechos_cargas['resultante'].append(dict_num)

    def save_values_por_carga(self, k, trecho, x_plot, u_pred, model):
        x0 = trecho['x0']
        x1 = trecho['x1']
        u_trecho_adim = u_pred[:, k : k+1]
        u_trecho_carga_dist = u_pred[:, k : k+1] * (trecho['u_ref_dist']) * (-1)
        u_trecho_carga_pontual = u_pred[:, k : k+1] * (trecho['u_ref_pontual']) * (-1)
        quant = 0
        for u_ref in [trecho['u_ref_dist'], trecho['u_ref_pontual'], trecho['u_ref_momento']]:
            if u_ref != 0:
                quant += 1

        if quant == 0:
            u_ref = 0
        else:
            u_ref = sum([trecho['u_ref_dist'], trecho['u_ref_pontual'], trecho['u_ref_momento']]) / quant

        trecho['u_ref_medio'] = u_ref
        u_trecho = u_pred[:, k : k+1] * u_ref * (-1)
        trecho['valores_u'] = u_trecho
        trecho['valores_u_adim'] = u_trecho_adim
        trecho['valores_u_carga_dist'] = u_trecho_carga_dist
        trecho['valores_u_carga_pontual'] = u_trecho_carga_pontual

        x_trecho = np.linspace(x0, x1, len(u_trecho))
        trecho['valores_x'] = x_trecho

        x = x_plot.clone().requires_grad_(True)
        u = model(x)[:, k:k+1]
        u1 = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u2 = torch.autograd.grad(u1, x, torch.ones_like(u1), create_graph=True)[0]
        u3 = torch.autograd.grad(u2, x, torch.ones_like(u2), create_graph=True)[0]

        quant = 0
        for theta_ref in [trecho['theta_ref_dist'], trecho['theta_ref_pontual'], trecho['theta_ref_momento']]:
            if theta_ref != 0:
                quant += 1

        if quant == 0:
            theta_ref = 0
        else:
            theta_ref = sum([trecho['theta_ref_dist'], trecho['theta_ref_pontual'], trecho['theta_ref_momento']]) / quant

        trecho['theta_ref_medio'] = theta_ref
        theta = theta_ref * (u1).detach().numpy()
        trecho['valores_theta'] = theta

        quant = 0
        for m_ref in [trecho['m_ref_dist'], trecho['m_ref_pontual'], trecho['m_ref_momento']]:
            if m_ref != 0:
                quant += 1

        if quant == 0:
            m_ref = 0
        else:
            m_ref = sum([trecho['m_ref_dist'], trecho['m_ref_pontual'], trecho['m_ref_momento']]) / quant

        trecho['m_ref_medio'] = m_ref
        M = - m_ref * (u2).detach().numpy()
        trecho['valores_mf'] = M

        quant = 0
        for v_ref in [trecho['v_ref_dist'], trecho['v_ref_pontual'], trecho['v_ref_momento']]:
            if v_ref != 0:
                quant += 1

        if quant == 0:
            v_ref = 0
        else:
            v_ref = sum([trecho['v_ref_dist'], trecho['v_ref_pontual'], trecho['v_ref_momento']]) / quant

        trecho['v_ref_medio'] = v_ref
        V = - v_ref * (u3).detach().numpy()
        trecho['valores_v'] = V

    def plot_errors(self, scale, nome = "erros.png"):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12 
        plt.figure(figsize=(9,4.5))
        valores_epoch = np.arange(0, len(self.pinn.loss_bc_variation), 1)
        valores_bc = [float(loss) for loss in self.pinn.loss_bc_variation]
        valores_pde = [float(loss) for loss in self.pinn.loss_pde_variation]
        valores_int = [float(loss) for loss in self.pinn.loss_int_variation]
        valores_jump = [float(loss) for loss in self.pinn.loss_jump_variation]
        valores_total = [bc + pde + int + jump for bc, pde, int, jump in zip(valores_bc, valores_pde, valores_int, valores_jump)]

        plt.figure(figsize=(8, 4.5))
        plt.plot(valores_epoch, valores_bc, label = 'Loss BC', color = 'red', lw = 1)
        plt.plot(valores_epoch, valores_pde, label = 'Loss PDE', color = 'blue', lw = 1)
        plt.plot(valores_epoch, valores_total, label = 'Loss Total', color = 'green', lw = 1)
        plt.title("Variação das perdas")
        plt.xlabel('Épocas')
        plt.ylabel('Perdas')
        plt.xlim(0, max(valores_epoch))
        plt.ylim(0, 1.05 * max(valores_total) / scale)
        plt.legend()
        plt.savefig(nome, dpi = 300)

    def plot_errors_novo(self, scale, scale_zoom, nome = "erros.png"):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12 
        
        valores_epoch = np.arange(0, len(self.pinn.loss_bc_variation), 1)
        valores_bc = [float(loss) for loss in self.pinn.loss_bc_variation]
        valores_pde = [float(loss) for loss in self.pinn.loss_pde_variation]
        valores_int = [float(loss) for loss in self.pinn.loss_int_variation]
        valores_jump = [float(loss) for loss in self.pinn.loss_jump_variation]
        valores_total = [bc + pde + int + jump for bc, pde, int, jump in zip(valores_bc, valores_pde, valores_int, valores_jump)]

        fig, ax1 = plt.subplots(figsize=(8, 4.5))

        ax1.plot(valores_epoch, valores_bc, label = 'Loss BC', color = 'red', lw = 1)
        ax1.plot(valores_epoch, valores_pde, label = 'Loss PDE', color = 'blue', lw = 1)
        ax1.plot(valores_epoch, valores_total, label = 'Loss Total', color = 'green', lw = 1)
        
        ax1.set_title("Variação das perdas")
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Perdas')
        ax1.set_xlim(0, max(valores_epoch))
        ax1.set_ylim(-0.05 * max(valores_total) / scale, 1.05 * max(valores_total) / scale)

        yticks = np.linspace(0, 1.05 * max(valores_total) / scale, 11)
        xticks = np.linspace(0, max(valores_epoch), 11)
        ax1.set_yticks(yticks)
        ax1.set_xticks(xticks)
        
        formatter1 = mticker.ScalarFormatter(useMathText=True)
        formatter1.set_scientific(True)
        formatter1.set_powerlimits((-1,1))
        ax1.yaxis.set_major_formatter(formatter1)

        ax1.legend()

        ax_ins = fig.add_axes([0.55, 0.25, 0.30, 0.35])
        ax_ins.plot(valores_epoch, valores_bc, color = 'red', lw = 1)
        ax_ins.plot(valores_epoch, valores_pde, color = 'blue', lw = 1)
        ax_ins.plot(valores_epoch, valores_total, color = 'green', lw = 1)

        ax_ins.set_xlim(0, max(valores_epoch))
        lim_superior_zoom = 1.05 * max(valores_total) / scale_zoom
        lim_inferior_zoom = - 0.05 * max(valores_total) / scale_zoom 
        ax_ins.set_ylim(lim_inferior_zoom, lim_superior_zoom)

        yticks_zoom = np.linspace(0, lim_superior_zoom, 6)
        xticks_zoom = np.linspace(0, max(valores_epoch), 6)
        ax_ins.set_yticks(yticks_zoom)
        ax_ins.set_xticks(xticks_zoom)

        formatter2 = mticker.ScalarFormatter(useMathText=True)
        formatter2.set_scientific(True)
        formatter2.set_powerlimits((-1,1))
        ax_ins.yaxis.set_major_formatter(formatter2)
        
        plt.savefig(nome, dpi = 300, bbox_inches='tight')

    def plot_deslocamento(self, plot_analitico = True, transparencia = 0.3, nome = "deslocamento.png"):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12 
        plt.figure(figsize=(9,4.5))

        if isinstance(self.pinn.trechos_cargas, list):
            trechos_cargas = self.pinn.trechos_cargas
        else:
            trechos_cargas = self.pinn.trechos_cargas['resultante']

        u_abs = []

        for k, trecho in enumerate(trechos_cargas):
            x_trecho = trecho['valores_x']
            u_trecho = np.squeeze(trecho['valores_u']) * 1e3
            zeros_trecho = np.zeros_like(u_trecho)
            if k == 0:
                plt.plot(x_trecho, u_trecho, color = 'blue', label = "Solução PINN")

            else:
                plt.plot(x_trecho, u_trecho, color = 'blue')

            plt.hlines(y = 0, xmin = x_trecho[0], xmax=x_trecho[-1], colors='gray', linestyles='--')

            plt.fill_between(
                x_trecho, u_trecho, zeros_trecho,
                where=(u_trecho >= 0),
                color='blue',
                alpha=transparencia,
                interpolate=True
            )

            plt.fill_between(
                x_trecho, u_trecho, zeros_trecho,
                where=(u_trecho < 0),
                color='red',
                alpha=transparencia,
                interpolate=True
            )

            u_trecho_abs = [abs(v) for v in u_trecho]
            u_abs.append(max(u_trecho_abs))             

        if plot_analitico is True:
            if self.pinn.verif_analitica is False:
                self.pinn.run_sol_analitica()
            plt.plot(self.pinn.x_analitico, self.pinn.uy_analitico, color = 'red', label = "Solução analítica", ls = '--')
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Deslocamento (mm)")
        plt.title("Linha Elástica")
        range_y = 1.1 * max(u_abs)
        plt.ylim(-range_y, range_y)
        plt.legend()
        plt.savefig(nome, dpi = 300)

    def plot_rotacao(self, plot_analitico = True, transparencia = 0.3, nome="rotacao.png"):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12 
        plt.figure(figsize=(9,4.5))

        if isinstance(self.pinn.trechos_cargas, list):
            trechos_cargas = self.pinn.trechos_cargas
        else:
            trechos_cargas = self.pinn.trechos_cargas['resultante']

        theta_abs = []

        for k, trecho in enumerate(trechos_cargas):
            x_trecho = trecho['valores_x']
            theta_trecho = np.squeeze(trecho['valores_theta'])
            zeros_trecho = np.zeros_like(theta_trecho)

            if k == 0:
                plt.plot(x_trecho, theta_trecho, color="blue", label="Solução PINN")
            else:
                plt.plot(x_trecho, theta_trecho, color="blue")

            plt.hlines(y = 0, xmin = x_trecho[0], xmax=x_trecho[-1], colors='gray', linestyles='--')

            plt.fill_between(
                x_trecho, theta_trecho, zeros_trecho,
                where=(theta_trecho >= 0),
                color='blue',
                alpha=transparencia,
                interpolate=True
            )

            # Região onde u < 0 (vermelho)
            plt.fill_between(
                x_trecho, theta_trecho, zeros_trecho,
                where=(theta_trecho < 0),
                color='red',
                alpha=transparencia,
                interpolate=True
            )

            theta_trecho_abs = [abs(v) for v in theta_trecho]
            theta_abs.append(max(theta_trecho_abs))

        if plot_analitico is True:
            if self.pinn.verif_analitica is False:
                self.pinn.run_sol_analitica()
            plt.plot(self.pinn.x_analitico, self.pinn.rz_analitico, color = 'red', label = "Solução analítica", ls = '--')
        plt.title("Rotação")
        range_y = 1.1 * max(theta_abs)
        plt.ylim(-range_y, range_y)
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Rotação (rad)")
        plt.legend()
        plt.savefig(nome, dpi=300)

    def plot_momento(self, plot_analitico = True, transparencia = 0.3, nome="momento.png"):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12 
        plt.figure(figsize=(9,4.5))

        if isinstance(self.pinn.trechos_cargas, list):
            trechos_cargas = self.pinn.trechos_cargas
        else:
            trechos_cargas = self.pinn.trechos_cargas['resultante']

        m_abs = []

        for k, trecho in enumerate(trechos_cargas):
            x_trecho = trecho['valores_x']
            m_trecho = np.squeeze(trecho['valores_mf']) * 1e-3
            zeros_trecho = np.zeros_like(m_trecho)

            if k == 0:
                plt.plot(x_trecho, m_trecho, color="blue", label="Solução PINN")
            else:
                plt.plot(x_trecho, m_trecho, color="blue")

            plt.hlines(y = 0, xmin = x_trecho[0], xmax=x_trecho[-1], colors='gray', linestyles='--')

            m_trecho_abs = [abs(v) for v in m_trecho]
            m_abs.append(max(m_trecho_abs))

            plt.fill_between(
                x_trecho, m_trecho, zeros_trecho,
                where=(m_trecho >= 0),
                color='blue',
                alpha=transparencia,
                interpolate=True
            )

            plt.fill_between(
                x_trecho, m_trecho, zeros_trecho,
                where=(m_trecho < 0),
                color='red',
                alpha=transparencia,
                interpolate=True
            )

        if plot_analitico is True:
            if self.pinn.verif_analitica is False:
                self.pinn.run_sol_analitica()
            for k, (x, M) in enumerate(zip(self.pinn.x_analitico2, self.pinn.mf_analitico)):
                if k == 0:
                    plt.plot(x, M, color = 'red', label = "Solução analítica", ls = '--')
                else:
                    plt.plot(x, M, color = 'red', ls = '--')
        plt.title("Momento Fletor")
        range_y = 1.1 * max(m_abs)
        plt.ylim(-range_y, range_y)
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Momento Fletor (kN.m)")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(nome, dpi=300)

    def plot_cortante(self, plot_analitico = True, transparencia = 0.3, nome="cortante.png"):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12 
        plt.figure(figsize=(9,4.5))

        if isinstance(self.pinn.trechos_cargas, list):
            trechos_cargas = self.pinn.trechos_cargas
        else:
            trechos_cargas = self.pinn.trechos_cargas['resultante']

        v_abs = []

        for k, trecho in enumerate(trechos_cargas):
            x_trecho = trecho['valores_x']
            v_trecho = np.squeeze(trecho['valores_v']) * 1e-3
            zeros_trecho = np.zeros_like(v_trecho)

            if k == 0:
                plt.plot(x_trecho, v_trecho, color="blue", label="Solução PINN")
            else:
                plt.plot(x_trecho, v_trecho, color="blue")

            plt.hlines(y = 0, xmin = x_trecho[0], xmax=x_trecho[-1], colors='gray', linestyles='--')

            v_trecho_abs = [abs(v) for v in v_trecho]
            v_abs.append(max(v_trecho_abs))

            plt.fill_between(
                x_trecho, v_trecho, zeros_trecho,
                where=(v_trecho >= 0),
                color='blue',
                alpha=transparencia,
                interpolate=True
            )

            plt.fill_between(
                x_trecho, v_trecho, zeros_trecho,
                where=(v_trecho < 0),
                color='red',
                alpha=transparencia,
                interpolate=True
            )

        if plot_analitico is True:
            if self.pinn.verif_analitica is False:
                self.pinn.run_sol_analitica()
            for k, (x, Q) in enumerate(zip(self.pinn.x_analitico2, self.pinn.fy_analitico)):
                if k == 0:
                    plt.plot(x, Q, color = 'red', label = "Solução analítica", ls = '--')
                else:
                    plt.plot(x, Q, color = 'red', ls = '--')

        plt.title("Esforço Cortante")
        range_y = 1.1 * max(v_abs)
        plt.ylim(-range_y, range_y)
        plt.xlabel("Comprimento (m)")
        plt.ylabel("Esforço Cortante (kN)")
        plt.legend()
        plt.savefig(nome, dpi=300)
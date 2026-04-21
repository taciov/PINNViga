import torch
import torch.nn as nn
import numpy as np
import copy
from src import DadosEntrada
from src import Formas
from src import Output

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
                 formulacao = "fraca", tipos_cargas = 'simples', otimizador = "adam", lr_adam = 0.001, width = 32, depth = 3):
        
        self.verif_analitica = False
        self.otimizador = otimizador
        self.lr_adam = lr_adam

        self.lista_apoios = lista_apoios
        self.lista_cargas = lista_cargas
        self.EI = E * I

        if tipos_cargas == 'simples':
            self.tipos_cargas = 'simples'
        elif tipos_cargas == 'superposicao':
            self.tipos_cargas = 'superposicao'
        else:
            raise NotImplementedError("tipos_cargas deve ser 'simples' ou 'superposicao'.")

        if self.tipos_cargas == 'superposicao':
            
            lista_filtros = list(set([carga.tipo for carga in self.lista_cargas]))
            self.trechos_cargas = dict.fromkeys(lista_filtros)
            for filtro in lista_filtros:
                lista_cargas_filtrada = [carga for carga in self.lista_cargas if carga.tipo == filtro]
                trecho_filtro = DadosEntrada.obter_trechos_cargas(self.lista_apoios, lista_cargas_filtrada, self.EI, filtro)
                self.trechos_cargas[filtro] = trecho_filtro
                for trecho in trecho_filtro:
                    trecho['lista_cargas_filtrada'] = lista_cargas_filtrada

        else:
            self.trechos_cargas = DadosEntrada.obter_trechos_cargas(self.lista_apoios, self.lista_cargas, self.EI, filtro = None)

        for apoio in self.lista_apoios:
            if isinstance(self.trechos_cargas, list):
                apoio.atribuir_propriedades_do_trecho(self.trechos_cargas)
            else:
                apoio.atribuir_propriedades_do_trecho(self.trechos_cargas[list(self.trechos_cargas.keys())[0]])
        if isinstance(self.trechos_cargas, list):
            num_camadas_output = len(self.trechos_cargas)
        else:
            num_camadas_output = len(self.trechos_cargas[list(self.trechos_cargas.keys())[0]])
        if isinstance(self.trechos_cargas, list):
            self.qy_max = max([abs(carga['qy']) for carga in self.trechos_cargas])
        else:
            self.qy_max = max([abs(carga['qy']) for carga in self.trechos_cargas[list(self.trechos_cargas.keys())[0]]])

        self.fy_max = 0

        for idx, carga in enumerate(self.lista_cargas):
            if carga.tipo == 'pontual':
                if abs(carga.fy) > self.fy_max:
                    self.fy_max = carga.fy

        lista_valores_cargas = []
        for carga in self.lista_cargas:
            if carga.tipo == 'pontual':
                lista_valores_cargas.append(carga.fy)
                lista_valores_cargas.append(carga.mz)
            elif carga.tipo == 'distribuida':
                lista_valores_cargas.append(carga.qy)

        if formulacao == 'fraca':
            self.formulacao = Formas.FormaFraca()
        elif formulacao == 'forte':
            self.formulacao = Formas.FormaForte()
        else:
            raise NotImplementedError("Formulação deve ser 'fraca' ou 'forte'.")
        
        if isinstance(self.trechos_cargas, list):
            self.model = PINN(num_outputs=num_camadas_output, width=width, depth = depth)
        else:
            self.model = dict.fromkeys(self.trechos_cargas.keys())
            for key in self.model:
                self.model[key] = PINN(num_outputs=num_camadas_output, width=width, depth = depth)

    def calc_loss(self, pde_weight, bc_weight, int_weight, jump_weight, input_model, input_trechos_cargas, key):
        loss_pde = self.formulacao.physics_loss(input_model, input_trechos_cargas, self.qy_max)
        loss_bc = self.formulacao.boundary_loss(input_model, self.lista_apoios)
        loss_int, loss_jump = self.formulacao.interface_loss(input_model, self.lista_apoios, self.fy_max, key)
    
        loss = pde_weight * loss_pde + bc_weight * loss_bc + int_weight * loss_int + jump_weight * loss_jump

        return loss, loss_pde, loss_bc, loss_int, loss_jump
    
    def adam_step(self, optimizer, pde_weight, bc_weight, int_weight, jump_weight):
        if isinstance(self.trechos_cargas, list):
            optimizer.zero_grad()

            loss, loss_pde, loss_bc, loss_int, loss_jump = self.calc_loss(pde_weight, bc_weight, int_weight, jump_weight, self.model, self.trechos_cargas, key = None)


            loss.backward()
            optimizer.step()

            return loss, loss_pde, loss_bc, loss_int, loss_jump

        else:
            soma_loss = 0
            soma_loss_pde = 0
            soma_loss_bc = 0
            soma_loss_int = 0
            soma_loss_jump = 0
            for key in self.model.keys():
                optimizer[key].zero_grad()

                loss, loss_pde, loss_bc, loss_int, loss_jump = self.calc_loss(pde_weight, bc_weight, int_weight, jump_weight, self.model[key], self.trechos_cargas[key], key)

                soma_loss += loss
                soma_loss_pde += loss_pde
                soma_loss_bc += loss_bc
                soma_loss_int += loss_int
                soma_loss_jump += loss_jump

                loss.backward()
                optimizer[key].step()

            return soma_loss, soma_loss_pde, soma_loss_bc, soma_loss_int, soma_loss_jump
    
    # def lbfgs_step(self, optimizer, pde_weight, bc_weight, int_weight, jump_weight):

    #     def closure():
    #         optimizer.zero_grad()
    #         loss, _, _, _, _ = self.calc_loss(pde_weight, bc_weight, int_weight, jump_weight)
    #         loss.backward()
    #         return loss

    #     loss = optimizer.step(closure)

    #     loss, loss_pde, loss_bc, loss_int, loss_jump = self.calc_loss(pde_weight, bc_weight, int_weight, jump_weight)

    #     return loss, loss_pde, loss_bc, loss_int, loss_jump

    def lbfgs_step(self, optimizer, pde_weight, bc_weight, int_weight, jump_weight):

    # =========================
    # CASO SIMPLES
    # =========================
        if isinstance(self.trechos_cargas, list):

            def closure():
                optimizer.zero_grad()

                loss, _, _, _, _ = self.calc_loss(pde_weight, bc_weight, int_weight, jump_weight, self.model, self.trechos_cargas, key=None)

                loss.backward()
                return loss

            loss = optimizer.step(closure)

            loss, loss_pde, loss_bc, loss_int, loss_jump = self.calc_loss(pde_weight, bc_weight, int_weight, jump_weight, self.model, self.trechos_cargas, key=None)

            return loss, loss_pde, loss_bc, loss_int, loss_jump

        # =========================
        # CASO COMPOSTO
        # =========================
        else:
            soma_loss = 0
            soma_loss_pde = 0
            soma_loss_bc = 0
            soma_loss_int = 0
            soma_loss_jump = 0

            for key in self.model.keys():

                def closure():
                    optimizer[key].zero_grad()

                    loss, _, _, _, _ = self.calc_loss(
                        pde_weight,
                        bc_weight,
                        int_weight,
                        jump_weight,
                        self.model[key],
                        self.trechos_cargas[key],
                        key
                    )

                    loss.backward()
                    return loss

                loss = optimizer[key].step(closure)

                loss, loss_pde, loss_bc, loss_int, loss_jump = self.calc_loss(
                    pde_weight,
                    bc_weight,
                    int_weight,
                    jump_weight,
                    self.model[key],
                    self.trechos_cargas[key],
                    key
                )

                soma_loss += loss
                soma_loss_pde += loss_pde
                soma_loss_bc += loss_bc
                soma_loss_int += loss_int
                soma_loss_jump += loss_jump

            return soma_loss, soma_loss_pde, soma_loss_bc, soma_loss_int, soma_loss_jump
    
    def run_model(self, num_epochs=5000, pde_weight=1.0, bc_weight=1.0, int_weight = 1.0, jump_weight = 1.0,
                  tol=1e-5, tol_apoio = 1e-5, print_progresso = True, salvar_historico = False, intervalo_historico = 10, manter_historico_apoio = False, tipos_cargas = 'simples'):
        
        self.u_plot_variation = []
        self.loss_pde_variation = []
        self.loss_bc_variation = []
        self.loss_int_variation = []
        self.loss_jump_variation = []
        self.historico = []

        self.formulacao.manter_historico_apoio = manter_historico_apoio
        if self.formulacao.manter_historico_apoio is True:
            for apoio in self.lista_apoios:
                apoio.historico_bc = []
                apoio.historico_cont = []

        if self.otimizador == "adam":
            if isinstance(self.trechos_cargas, list):
                optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr_adam)
            else:
                optimizer = dict.fromkeys(self.model.keys())
                for key in self.model.keys():
                    optimizer[key] = torch.optim.Adam(self.model[key].parameters(), lr = self.lr_adam)
            step_fn = self.adam_step

        elif self.otimizador == "lbfgs":
            if isinstance(self.trechos_cargas, list):
                optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=20,
                history_size=100,
                line_search_fn="strong_wolfe"
            )
            else:
                optimizer = dict.fromkeys(self.model.keys())
                for key in self.model.keys():
                    optimizer[key] = torch.optim.LBFGS(
                    self.model[key].parameters(),
                    lr=1.0,
                    max_iter=20,
                    history_size=100,
                    line_search_fn="strong_wolfe")

        #     optimizer = torch.optim.LBFGS(
        #     self.model.parameters(),
        #     lr=1.0,
        #     max_iter=20,
        #     history_size=100,
        #     line_search_fn="strong_wolfe"
        # )
            step_fn = self.lbfgs_step

        else:
            raise ValueError("otimizador deve ser 'adam' ou 'lbfgs'" )
        
        for epoch in range(num_epochs + 1):

            loss, loss_pde, loss_bc, loss_int, loss_jump = step_fn(optimizer, pde_weight, bc_weight, int_weight, jump_weight)

            if salvar_historico and (epoch % intervalo_historico == 0 or epoch == num_epochs):

                estado_atual = self._capturar_estado()
                
                snapshot = {
                    'epoch': epoch,
                    'loss': loss.item(),
                    'dados_trechos': estado_atual
                }
                self.historico.append(snapshot)

            if (float(loss_bc) <= tol) and (float(loss_pde) <= tol):
                print("Treinamento concluído pelo critério de tolerância da perda")
                print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}, INT Loss: {loss_int.item():.12f}, JUMP Loss: {loss_jump.item():.12f}")
                print(f"Treinamento interrompido na época {epoch}.")
                break

            if print_progresso is False:
                pass
            else:
                denominador = 10 if num_epochs >= 10 else 2
                if epoch % int(num_epochs / denominador) == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.12f}, PDE Loss: {loss_pde.item():.12f}, BC Loss: {loss_bc.item():.12f}, INT Loss: {loss_int.item():.12f}, JUMP Loss: {loss_jump.item():.12f}")

            self.loss_pde_variation.append(loss_pde)
            self.loss_bc_variation.append(loss_bc)
            self.loss_int_variation.append(loss_int)
            self.loss_jump_variation.append(loss_jump)

        self.loss = self.loss_pde_variation + self.loss_bc_variation + self.loss_int_variation + self.loss_jump_variation

        self.output = Output.Output(self)

    def run_sol_analitica(self, tam = 20):
        self.output.save_values()
        self.output.run_sol_analitica(tam)
        

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

    def _capturar_estado(self):
        """
        Calcula e retorna uma CÓPIA dos dados físicos (u, theta, M, V) 
        para o estado atual do modelo.
        """
        x_plot = torch.linspace(0, 1, 101).view(-1, 1).requires_grad_(True)
        
        u = self.model(x_plot) 
        
        u1 = torch.autograd.grad(u, x_plot, torch.ones_like(u), create_graph=True)[0]
        u2 = torch.autograd.grad(u1, x_plot, torch.ones_like(u1), create_graph=True)[0]
        u3 = torch.autograd.grad(u2, x_plot, torch.ones_like(u2), create_graph=False)[0]

        lista_estado_trechos = []

        u_np = u.detach().numpy()
        u1_np = u1.detach().numpy()
        u2_np = u2.detach().numpy()
        u3_np = u3.detach().numpy()

        for k, trecho in enumerate(self.trechos_cargas):
            dados_trecho = {}
            
            dados_trecho['x0'] = trecho['x0']
            dados_trecho['x1'] = trecho['x1']
            
            u_trecho = u_np[:, k : k+1] * (trecho['u_ref_dist'] + trecho['u_ref_pontual']) * (-1)
            x_trecho = np.linspace(trecho['x0'], trecho['x1'], len(u_trecho))
            
            dados_trecho['valores_x'] = x_trecho
            dados_trecho['valores_u'] = u_trecho

            theta_ref_dist = trecho['qy'] * trecho['L'] ** 3 / trecho['EI']
            theta_ref_pontual = trecho['fy'] * trecho['L'] ** 2 / trecho['EI']
            theta = (theta_ref_dist + theta_ref_pontual) * u1_np[:, k:k+1]
            dados_trecho['valores_theta'] = theta

            m_ref_dist = trecho['qy'] * trecho['L'] ** 2
            m_ref_pontual = trecho['fy'] * trecho['L']
            M = - (m_ref_dist + m_ref_pontual) * u2_np[:, k:k+1]
            dados_trecho['valores_mf'] = M

            v_ref = trecho['qy'] * trecho['L']
            v_ref_pontual = trecho['fy']
            V = - (v_ref + v_ref_pontual) * u3_np[:, k:k+1]
            dados_trecho['valores_V'] = V
            
            lista_estado_trechos.append(dados_trecho)
            
        return lista_estado_trechos
import torch
import torch.nn as nn
import numpy as np
import random
import copy
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
        if carga.tipo == 'distribuida':
            if carga.x0 not in posicoes_x:
                posicoes_x.append(carga.x0)
            if carga.x1 not in posicoes_x:
                posicoes_x.append(carga.x1)
        elif carga.tipo == 'pontual':
            if carga.x not in posicoes_x:
                posicoes_x.append(carga.x)

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
            if carga.tipo == 'distribuida':
                if carga.x0 <= posicoes_x[k] and carga.x1 >= posicoes_x[k+1]:
                    dict_trecho['qx'] += carga.qx
                    dict_trecho['qy'] += carga.qy

        trechos_cargas.append(dict_trecho)

    trechos_cargas = inserir_apoios_no_trecho(lista_apoios, trechos_cargas)

    for carga in trechos_cargas:
            
        tam = max(int(round(abs(carga['x1'] - carga['x0']) * 10, 0)), 21)

        torch.manual_seed(1)
        min_val = 0
        max_val = 1
        x_scaled = min_val + (max_val - min_val) * torch.rand(tam, 1)

        # ## TESTE

        # x_scaled = torch.linspace(carga['x0'], carga['x1'], tam).view(-1, 1)
        # x_scaled.requires_grad = True

        carga['vetor_x'] = x_scaled.requires_grad_(True)
        carga["u_ref_dist"] = carga['qy'] * carga['L'] ** 4 / (carga['EI'])
        carga['theta_ref_dist'] = carga['qy'] * carga['L'] ** 3 / carga['EI']
        carga['m_ref_dist'] = carga['qy'] * carga['L'] ** 2
        carga['v_ref_dist'] = carga['qy'] * carga['L']

        lista_cargas_trecho = (carga['apoio0'].lista_cargas + carga['apoio1'].lista_cargas)
        if len(lista_cargas_trecho) > 0:
            lista_fy = [carga.fy for carga in lista_cargas_trecho]
            carga['fy'] = sum(lista_fy)
        else:
            carga['fy'] = 0

        carga['u_ref_pontual'] = carga['fy'] * carga['L'] ** 3 / carga['EI']
        carga['theta_ref_pontual'] = carga['fy'] * carga['L'] ** 2 / carga['EI']
        carga['m_ref_pontual'] = carga['fy'] * carga['L']
        carga['v_ref_pontual'] = carga['fy']

        carga['tam'] = tam

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

            qy = carga['qy'] / qy_max if qy_max != 0 else 0
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
            if apoio.tipo == 'ext':
                variacao_loss_bc += (u0_xxx)**2
            else:
                pass

        if apoio.graus[2] == 1: 
            variacao_loss_bc += (u0_x)**2
        else:
            if apoio.tipo == "ext":
                variacao_loss_bc += (u0_xx)**2
            else:
                pass

        if self.manter_historico_apoio is True:
            apoio.historico_bc.append(variacao_loss_bc)

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
    
    def interface_loss_carga_pontual(self, u2_esq, u2_dir, u3_esq, u3_dir, fy_norm, apoio):

        jump_u3 = u3_esq - u3_dir
        # loss_jump_u3 = torch.mean((jump_u3 - (fy_norm / apoio.EI)) ** 2)
        loss_jump_u3 = torch.mean((jump_u3 - fy_norm) ** 2)

        return loss_jump_u3

    def interface_loss_apoio(self, u_esq, u_dir, x_esq, x_dir, apoio, fy_max):

        u1_esq = torch.autograd.grad(u_esq, x_esq, torch.ones_like(u_esq), create_graph=True)[0]
        u2_esq = torch.autograd.grad(u1_esq, x_esq, torch.ones_like(u1_esq), create_graph=True)[0]
        u3_esq = torch.autograd.grad(u2_esq, x_esq, torch.ones_like(u2_esq), create_graph=True)[0]

        u1_dir = torch.autograd.grad(u_dir, x_dir, torch.ones_like(u_dir), create_graph=True)[0]
        u2_dir = torch.autograd.grad(u1_dir, x_dir, torch.ones_like(u1_dir), create_graph=True)[0]
        u3_dir = torch.autograd.grad(u2_dir, x_dir, torch.ones_like(u2_dir), create_graph=True)[0]

        loss_cont = torch.mean((u_esq - u_dir)**2) + torch.mean((u1_esq - u1_dir)**2) + torch.mean((u2_esq - u2_dir)**2)

        if len(apoio.lista_cargas) != 0:
            for idx, carga in enumerate(apoio.lista_cargas):
                if idx == 0:
                    soma_fy = carga.fy
                else:
                    soma_fy += carga.fy
            fy_norm = soma_fy / fy_max if fy_max != 0 else 0
            loss_jump = self.interface_loss_carga_pontual(u2_esq, u2_dir, u3_esq, u3_dir, fy_norm, apoio)
        else:
            loss_jump = torch.tensor(0.0, requires_grad=True)
            if apoio.graus[2] == 0:
                loss_cont += torch.mean((u2_esq - u2_dir)**2)
            if apoio.graus[1] == 0:
                loss_cont += torch.mean((u3_esq - u3_dir)**2)

        if self.manter_historico_apoio is True:
            apoio.historico_cont.append(loss_cont)

        return loss_cont, loss_jump

    def interface_loss(self, model, lista_apoios, fy_max):
        loss_int = 0.0
        loss_jump = 0.0
        for apoio in lista_apoios:

            if apoio.tipo == 'int':

                x_esq = torch.tensor([[1.0]], requires_grad=True)
                u_esq = model(x_esq)[:, apoio.trecho_esq:apoio.trecho_esq + 1]
                x_dir = torch.tensor([[0.0]], requires_grad=True)
                u_dir = model(x_dir)[:, apoio.trecho_dir:apoio.trecho_dir + 1]
                var_int, var_jump = self.interface_loss_apoio(u_esq, u_dir, x_esq, x_dir, apoio, fy_max)

                loss_int += var_int
                loss_jump += var_jump
        
        if torch.is_tensor(loss_int):
            pass
        else:
            loss_int = torch.tensor(0.0, requires_grad=True)

        if torch.is_tensor(loss_jump):
            pass
        else:
            loss_jump = torch.tensor(0.0, requires_grad=True)

        return loss_int, loss_jump
    
    # def jump_loss(self, model, lista_apoios):
        
# class FormaForte(FormaFraca): ## REVISAR

#     def u(self, model, x, apoio_esq=None, apoio_dir=None):

#         g = torch.ones_like(x)

#         # apoio esquerdo com deslocamento imposto
#         if apoio_esq is not None and apoio_esq.graus[1] == 1:
#             g = g * x

#         # apoio direito com deslocamento imposto
#         if apoio_dir is not None and apoio_dir.graus[1] == 1:
#             g = g * (1 - x)

#         return g * model(x)

# def physics_loss(self, model, trechos_cargas, qy_max):

#     loss_pde = 0.0

#     for k, carga in enumerate(trechos_cargas):

#         qy = carga['qy'] / qy_max
#         x = carga['vetor_x']

#         apoio_esq = carga.get('apoio_esq', None)
#         apoio_dir = carga.get('apoio_dir', None)

#         u_raw = model(x)[:, k:k+1]
#         u = self.u(lambda x_: u_raw, x, apoio_esq, apoio_dir)

#         loss_pde += self.physics_loss_trecho(x, u, qy)

#     return loss_pde

# def boundary_loss_apoio(self, u, x, apoio):

#     variacao_loss_bc = 0.0

#     u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
#     u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
#     u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]

#     if apoio.graus[2] == 1:
#         variacao_loss_bc += (u_x)**2
#     else:
#         if apoio.tipo == "ext":
#             variacao_loss_bc += (u_xx)**2

#     if self.manter_historico_apoio is True:
#         apoio.historico_bc.append(variacao_loss_bc)

#     return variacao_loss_bc

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

        for apoio in self.lista_apoios:
            apoio.atribuir_propriedades_do_trecho(self.trechos_cargas)

        num_camadas_output = len(self.trechos_cargas)

        self.qy_max = max([abs(carga['qy']) for carga in self.trechos_cargas])

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

        self.ref_value = max([abs(valor) for valor in lista_valores_cargas])

        self.L_total = sum([carga['L'] for carga in self.trechos_cargas])

        if formulacao == 'fraca':
            self.formulacao = FormaFraca()
        elif formulacao == 'forte':
            self.formulacao = FormaForte()
        else:
            raise NotImplementedError("Formulação deve ser 'fraca' ou 'forte'.")
        
        self.model = PINN(num_outputs=num_camadas_output, width=width, depth = depth)

    def calc_loss(self, pde_weight, bc_weight, int_weight, jump_weight):
        loss_pde = self.formulacao.physics_loss(self.model, self.trechos_cargas, self.qy_max)
        loss_bc = self.formulacao.boundary_loss(self.model, self.lista_apoios)
        loss_int, loss_jump = self.formulacao.interface_loss(self.model, self.lista_apoios, self.fy_max)
    
        loss = pde_weight * loss_pde + bc_weight * loss_bc + int_weight * loss_int + jump_weight * loss_jump

        return loss, loss_pde, loss_bc, loss_int, loss_jump
    
    def adam_step(self, optimizer, pde_weight, bc_weight, int_weight, jump_weight):
        optimizer.zero_grad()

        loss, loss_pde, loss_bc, loss_int, loss_jump = self.calc_loss(pde_weight, bc_weight, int_weight, jump_weight)

        loss.backward()
        optimizer.step()

        return loss, loss_pde, loss_bc, loss_int, loss_jump
    
    def lbfgs_step(self, optimizer, pde_weight, bc_weight, int_weight, jump_weight):

        def closure():
            optimizer.zero_grad()
            loss, _, _, _, _ = self.calc_loss(pde_weight, bc_weight, int_weight, jump_weight)
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        loss, loss_pde, loss_bc, loss_int, loss_jump = self.calc_loss(pde_weight, bc_weight, int_weight, jump_weight)

        return loss, loss_pde, loss_bc, loss_int, loss_jump
    
    def run_model(self, num_epochs=5000, pde_weight=1.0, bc_weight=1.0, int_weight = 1.0, jump_weight = 1.0,
                  tol=1e-5, tol_apoio = 1e-5, print_progresso = True, salvar_historico = True, intervalo_historico = 10, manter_historico_apoio = False):
        
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
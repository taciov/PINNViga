import torch

class FormaFraca:

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

    def interface_loss_apoio(self, u_esq, u_dir, x_esq, x_dir, apoio, fy_max, key):

        u1_esq = torch.autograd.grad(u_esq, x_esq, torch.ones_like(u_esq), create_graph=True)[0]
        u2_esq = torch.autograd.grad(u1_esq, x_esq, torch.ones_like(u1_esq), create_graph=True)[0]
        u3_esq = torch.autograd.grad(u2_esq, x_esq, torch.ones_like(u2_esq), create_graph=True)[0]

        u1_dir = torch.autograd.grad(u_dir, x_dir, torch.ones_like(u_dir), create_graph=True)[0]
        u2_dir = torch.autograd.grad(u1_dir, x_dir, torch.ones_like(u1_dir), create_graph=True)[0]
        u3_dir = torch.autograd.grad(u2_dir, x_dir, torch.ones_like(u2_dir), create_graph=True)[0]

        loss_cont = torch.mean((u_esq - u_dir)**2) + torch.mean((u1_esq - u1_dir)**2) + torch.mean((u2_esq - u2_dir)**2)

        if key in ['pontual', 'momento', None]:
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
        else:
            loss_jump = torch.tensor(0.0, requires_grad=True)
            if apoio.graus[2] == 0:
                loss_cont += torch.mean((u2_esq - u2_dir)**2)
            if apoio.graus[1] == 0:
                loss_cont += torch.mean((u3_esq - u3_dir)**2)

        if self.manter_historico_apoio is True:
            apoio.historico_cont.append(loss_cont)

        return loss_cont, loss_jump

    def interface_loss(self, model, lista_apoios, fy_max, key):
        loss_int = 0.0
        loss_jump = 0.0
        for apoio in lista_apoios:

            if apoio.tipo == 'int':

                x_esq = torch.tensor([[1.0]], requires_grad=True)
                u_esq = model(x_esq)[:, apoio.trecho_esq:apoio.trecho_esq + 1]
                x_dir = torch.tensor([[0.0]], requires_grad=True)
                u_dir = model(x_dir)[:, apoio.trecho_dir:apoio.trecho_dir + 1]
                var_int, var_jump = self.interface_loss_apoio(u_esq, u_dir, x_esq, x_dir, apoio, fy_max, key)

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
        
class FormaForte(FormaFraca): ## REVISAR

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

    if self.manter_historico_apoio is True:
        apoio.historico_bc.append(variacao_loss_bc)

    return variacao_loss_bc
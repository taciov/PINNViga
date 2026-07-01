import torch

class FormaFraca:

    def calculo_derivadas(self, x, u):
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx, x, torch.ones_like(u_xxx), create_graph=True)[0]

        return u_x, u_xx, u_xxx, u_xxxx

    def physics_loss_trecho(self, x, u, q):
        _, _, _, u_xxxx = self.calculo_derivadas(x, u)
        
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

        u0_x, u0_xx, u0_xxx, _ = self.calculo_derivadas(x, u)

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
    
    def interface_loss_carga_momento(self, u2_esq, u2_dir, mz_norm):

        jump_u2 = u2_esq - u2_dir
        loss_jump_u2 = torch.mean((jump_u2 - mz_norm) ** 2)

        return loss_jump_u2

    def interface_loss_carga_pontual(self, u3_esq, u3_dir, fy_norm):

        jump_u3 = u3_esq - u3_dir
        loss_jump_u3 = torch.mean((jump_u3 - fy_norm) ** 2)

        return loss_jump_u3
    
    def obter_valores_normalizados(self, lista_cargas, fy_max, mz_max):
        for idx, carga in enumerate(lista_cargas):
            if idx == 0:
                soma_fy = carga.fy
                soma_mz = carga.mz
            else:
                soma_fy += carga.fy
                soma_mz += carga.mz
        fy_norm = soma_fy / fy_max if fy_max != 0 else 0
        mz_norm = soma_mz / mz_max if mz_max != 0 else 0

        return fy_norm, mz_norm

    def interface_loss_apoio(self, u_esq, u_dir, x_esq, x_dir, apoio, fy_max, mz_max, key):
        u1_esq, u2_esq, u3_esq, _ = self.calculo_derivadas(x_esq, u_esq)
        u1_dir, u2_dir, u3_dir, _ = self.calculo_derivadas(x_dir, u_dir)
        loss_cont = torch.mean((u_esq - u_dir)**2) + torch.mean((u1_esq - u1_dir)**2)

        if key in ['pontual', 'momento', None]:
            if len(apoio.lista_cargas) != 0:
                fy_norm, mz_norm = self.obter_valores_normalizados(apoio.lista_cargas, fy_max, mz_max)
                loss_jump = self.interface_loss_carga_pontual(u3_esq, u3_dir, fy_norm) if fy_norm != 0 else torch.mean((u3_esq - u3_dir)**2)
                loss_jump += self.interface_loss_carga_momento(u2_esq, u2_dir, mz_norm) if mz_norm != 0 else torch.mean((u2_esq - u2_dir)**2)
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

    def interface_loss(self, model, lista_apoios, fy_max, mz_max, key):
        loss_int = 0.0
        loss_jump = 0.0
        for apoio in lista_apoios:

            if apoio.tipo == 'int':

                x_esq = torch.tensor([[1.0]], requires_grad=True)
                u_esq = model(x_esq)[:, apoio.trecho_esq:apoio.trecho_esq + 1]
                x_dir = torch.tensor([[0.0]], requires_grad=True)
                u_dir = model(x_dir)[:, apoio.trecho_dir:apoio.trecho_dir + 1]
                var_int, var_jump = self.interface_loss_apoio(u_esq, u_dir, x_esq, x_dir, apoio, fy_max, mz_max, key)

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
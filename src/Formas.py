import torch

class FormaFraca:

    def calculo_derivadas(self, x, u, ordem = 4):
        # Calcula só até a ordem pedida: apoios e interfaces usam até u''' e
        # dispensam a derivada de 4ª ordem, que é a mais cara.
        derivadas = []
        atual = u
        for _ in range(ordem):
            atual = torch.autograd.grad(atual, x, torch.ones_like(atual), create_graph=True)[0]
            derivadas.append(atual)

        return tuple(derivadas)

    def avaliar_contornos(self, model, lista_apoios):
        # Avalia a rede e as derivadas até 3ª ordem em todos os pontos de apoio
        # numa única passada, em vez de um forward e quatro autograds por ponto.
        # Cada linha de x depende só de si mesma, então o gradiente da soma
        # devolve a derivada ponto a ponto.
        chaves = []
        posicoes = []
        colunas = []
        for apoio in lista_apoios:
            if apoio.trecho_esq is not None:
                chaves.append((id(apoio), 'esq'))
                posicoes.append(1.0)
                colunas.append(apoio.trecho_esq)
            if apoio.trecho_dir is not None:
                chaves.append((id(apoio), 'dir'))
                posicoes.append(0.0)
                colunas.append(apoio.trecho_dir)

        x = torch.tensor(posicoes).view(-1, 1).requires_grad_(True)
        indices = torch.tensor(colunas).view(-1, 1)
        u = model(x).gather(1, indices)
        u1, u2, u3 = self.calculo_derivadas(x, u, ordem = 3)

        return {chave: (u[i:i+1], u1[i:i+1], u2[i:i+1], u3[i:i+1]) for i, chave in enumerate(chaves)}

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
    
    def boundary_loss_apoio(self, valores, apoio):

        variacao_loss_bc = 0

        u, u0_x, u0_xx, u0_xxx = valores

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
            # float() solta o grafo; guardar o tensor manteria o grafo de cada época vivo
            apoio.historico_bc.append(float(variacao_loss_bc.detach()) if torch.is_tensor(variacao_loss_bc) else float(variacao_loss_bc))

        return variacao_loss_bc
    
    def boundary_loss(self, model, lista_apoios, contornos = None):
        if contornos is None:
            contornos = self.avaliar_contornos(model, lista_apoios)

        loss_bc = 0.0
        for apoio in lista_apoios:

            if apoio.trecho_esq is not None:
                loss_bc += self.boundary_loss_apoio(contornos[(id(apoio), 'esq')], apoio)

            if apoio.trecho_dir is not None:
                loss_bc += self.boundary_loss_apoio(contornos[(id(apoio), 'dir')], apoio)

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

    def interface_loss_apoio(self, valores_esq, valores_dir, apoio, fy_max, mz_max, key):
        u_esq, u1_esq, u2_esq, u3_esq = valores_esq
        u_dir, u1_dir, u2_dir, u3_dir = valores_dir
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
            apoio.historico_cont.append(float(loss_cont.detach()))

        return loss_cont, loss_jump

    def interface_loss(self, model, lista_apoios, fy_max, mz_max, key, contornos = None):
        if contornos is None:
            contornos = self.avaliar_contornos(model, lista_apoios)

        loss_int = 0.0
        loss_jump = 0.0
        for apoio in lista_apoios:

            if apoio.tipo == 'int':

                var_int, var_jump = self.interface_loss_apoio(contornos[(id(apoio), 'esq')], contornos[(id(apoio), 'dir')],
                                                              apoio, fy_max, mz_max, key)

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
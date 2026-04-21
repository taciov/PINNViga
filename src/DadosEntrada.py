import torch
# import torch.nn as nn
import numpy as np
import random
# import copy
# from src import Output

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

def obter_trechos_cargas(lista_apoios, lista_cargas, EI, filtro):
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
            if carga.tipo == 'distribuida' and filtro in ['distribuida', None]:
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

        carga['vetor_x'] = x_scaled.requires_grad_(True)
        carga["u_ref_dist"] = carga['qy'] * carga['L'] ** 4 / (carga['EI'])
        carga['theta_ref_dist'] = carga['qy'] * carga['L'] ** 3 / carga['EI']
        carga['m_ref_dist'] = carga['qy'] * carga['L'] ** 2
        carga['v_ref_dist'] = carga['qy'] * carga['L']

        if filtro in [None, 'pontual']:
            lista_cargas_trecho = (carga['apoio0'].lista_cargas + carga['apoio1'].lista_cargas)
            if len(lista_cargas_trecho) > 0:
                lista_fy = [carga.fy for carga in lista_cargas_trecho]
                carga['fy'] = sum(lista_fy)
            else:
                carga['fy'] = 0
        else:
            carga['fy'] = 0

        carga['u_ref_pontual'] = carga['fy'] * carga['L'] ** 3 / carga['EI']
        carga['theta_ref_pontual'] = carga['fy'] * carga['L'] ** 2 / carga['EI']
        carga['m_ref_pontual'] = carga['fy'] * carga['L']
        carga['v_ref_pontual'] = carga['fy']

        carga['tam'] = tam

    return trechos_cargas
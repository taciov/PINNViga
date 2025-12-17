from src import PINNViga
from src import Carga
from src import Apoio
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import time
import torch
import torch.nn as nn
from itertools import product
from tqdm import tqdm

class Teste:

    def run_lbfgs(lista_apoios, lista_cargas, E, I, lista_epoch, lista_largura, lista_profundidade, run_analitica = True, export_json = True, nome = 'resultados_LBFG-S'):

        lista_sim = []
        
        combinacoes = list(product(
            lista_profundidade,
            lista_largura,
            lista_epoch
        ))

        for profundidade, largura, epoch in tqdm(combinacoes, desc="Simulações PINN (LBFGS)", unit="caso"):
            inicio = time.time()
            pinn = PINNViga.PINNViga(lista_apoios, lista_cargas, E, I, otimizador="lbfgs", depth=profundidade, width=largura)
            pinn.run_model(num_epochs=epoch, tol = 1e-7, print_progresso=False)
            fim = time.time()
            duracao = fim - inicio
            pinn.save_values()
            dict_temp = {
                'otimizador' : 'LBFG-S',
                'profundidade' : profundidade,
                'largura' : largura,
                'n_epochs' : epoch,
                'u_max' : float(np.max([pinn.trechos_cargas[k]['valores_u'] for k in range(len(pinn.trechos_cargas))]))*1e3,
                'u_min' : float(np.min([pinn.trechos_cargas[k]['valores_u'] for k in range(len(pinn.trechos_cargas))]))*1e3,
                'theta_max' : float(np.max([pinn.trechos_cargas[k]['valores_theta'] for k in range(len(pinn.trechos_cargas))])),
                'theta_min' : float(np.min([pinn.trechos_cargas[k]['valores_theta'] for k in range(len(pinn.trechos_cargas))])),
                'V_max' : float(np.max([pinn.trechos_cargas[k]['valores_V'] for k in range(len(pinn.trechos_cargas))]))*1e-3,
                'V_min' : float(np.min([pinn.trechos_cargas[k]['valores_V'] for k in range(len(pinn.trechos_cargas))]))*1e-3,
                'mf_max' : float(np.max([pinn.trechos_cargas[k]['valores_mf'] for k in range(len(pinn.trechos_cargas))]))*1e-3,
                'mf_min' : float(np.min([pinn.trechos_cargas[k]['valores_mf'] for k in range(len(pinn.trechos_cargas))]))*1e-3,
                'tempo' : duracao
                }
            
            lista_sim.append(dict_temp)

        if run_analitica is True:

            inicio = time.time()
            pinn.run_sol_analitica(tam = 101)
            fim = time.time()
            duracao = fim - inicio

            dict_temp = {
                'otimizador' : 'analitico',
                'profundidade' : 101,
                'largura' : 'analitico',
                'n_epochs' : 'analitico',
                'u_max' : float(np.max(pinn.uy_analitico))*1e3,
                'u_min' : float(np.min(pinn.uy_analitico))*1e3,
                'theta_max' : float(np.max(pinn.rz_analitico)),
                'theta_min' : float(np.min(pinn.rz_analitico)),
                'V_max' : float(np.max(pinn.fy_analitico))*1e-3,
                'V_min' : float(np.min(pinn.fy_analitico))*1e-3,
                'mf_max' : float(np.max(pinn.mf_analitico))*1e-3,
                'mf_min' : float(np.min(pinn.mf_analitico))*1e-3,
                'tempo' : duracao
                }
            
            lista_sim.append(dict_temp)

        if export_json is True:
            pd.DataFrame(lista_sim).to_json(f"{nome}.json", orient="records", indent=4)

        return lista_sim
import numpy as np
import matplotlib.pyplot as plt

class CargaPontual:
    tipo = "pontual"
    def __init__(self, vetor, apoio):
        self.fx = vetor[0]
        self.fy = vetor[1]
        self.mz = vetor[2]

        self.apoio = apoio
        self.x = apoio.x
        self.y = apoio.y
        apoio.lista_cargas.append(self)

    def atribuir_propriedades_do_trecho(self, trechos_cargas):
        if (self.apoio.trecho_esq is None) or (self.apoio.trecho_dir is None):
            if self.apoio.trecho_esq is None:
                self.EI = trechos_cargas[self.apoio.trecho_dir]['EI']
                self.L = trechos_cargas[self.apoio.trecho_dir]['L']
            else:
                self.EI = trechos_cargas[self.apoio.trecho_esq]['EI']
                self.L = trechos_cargas[self.apoio.trecho_esq]['L']
        else:
            self.EI = max(trechos_cargas[self.apoio.trecho_esq]['EI'], trechos_cargas[self.apoio.trecho_dir]['EI'])
            self.L = max(trechos_cargas[self.apoio.trecho_esq]['L'], trechos_cargas[self.apoio.trecho_dir]['L'])      

class CargaDistribuida:
    tipo = "distribuida"
    def __init__(self, vetor, apoio0, apoio1):
        self.qx = vetor[0]
        self.qy = vetor[1]

        self.x0 = apoio0.x
        self.y0 = apoio0.y
        self.x1 = apoio1.x
        self.y1 = apoio1.y
import numpy as np
import matplotlib.pyplot as plt

class Apoio:
    def __init__(self, graus, posicao):
        self.graus = graus
        self.ux = graus[0]
        self.uy = graus[1]
        self.rz = graus[2]

        self.x = posicao[0]
        self.y = posicao[1]

        self.tipo = None

        self.trecho_esq = None
        self.trecho_dir = None

        self.lista_cargas = []

    def atribuir_propriedades_do_trecho(self, trechos_cargas):
        if (self.trecho_esq is None) or (self.trecho_dir is None):
            if self.trecho_esq is None:
                self.EI = trechos_cargas[self.trecho_dir]['EI']
                self.L = trechos_cargas[self.trecho_dir]['L']
            else:
                self.EI = trechos_cargas[self.trecho_esq]['EI']
                self.L = trechos_cargas[self.trecho_esq]['L']
        else:
            self.EI = max(trechos_cargas[self.trecho_esq]['EI'], trechos_cargas[self.trecho_dir]['EI'])
            self.L = max(trechos_cargas[self.trecho_esq]['L'], trechos_cargas[self.trecho_dir]['L']) 
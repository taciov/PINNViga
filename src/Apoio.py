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
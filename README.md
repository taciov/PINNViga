# PINNViga

**PINNViga** é um mini-framework *open-source* desenvolvido em **Python** para a análise estrutural de vigas utilizando **Redes Neurais Informadas por Física (Physics-Informed Neural Networks - PINNs)**.

O projeto foi desenvolvido como parte de um Trabalho de Conclusão de Curso (TCC) e tem como objetivo resolver problemas de flexão de vigas a partir da teoria de **Euler-Bernoulli**, utilizando aprendizado de máquina guiado por leis físicas.

Além da solução por PINNs, o framework realiza a **validação automática** dos resultados por meio do **Método da Rigidez Direta**, utilizando a biblioteca **anastruct**.

---

## ✨ Principais funcionalidades

- Análise estrutural de vigas em regime estático linear;
- Implementação baseada em **Physics-Informed Neural Networks (PINNs)**;
- Suporte a:
  - diferentes condições de contorno;
  - múltiplos apoios;
  - cargas distribuídas;
  - cargas pontuais;
  - momentos concentrados;
- Divisão automática do domínio em múltiplos trechos;
- Tratamento automático de descontinuidades através de perdas de continuidade (*interface losses*) e perdas de salto (*jump losses*);
- Treinamento utilizando os otimizadores **Adam** e **L-BFGS**;
- Comparação automática com solução analítica via **anastruct**;
- Geração de gráficos de:
  - deslocamento;
  - rotação;
  - momento fletor;
  - esforço cortante;
  - evolução da função de perda.

---

# Estrutura do projeto

O framework foi desenvolvido seguindo princípios de **Programação Orientada a Objetos (POO)**, buscando modularidade e facilidade de manutenção.

```
src/
│
├── Apoio.py
├── Carga.py
├── DadosEntrada.py
├── Formas.py
├── PINNViga.py
└── Output.py
```

## Classes principais

### `Apoio.py`

Responsável pela definição das condições de contorno da estrutura, incluindo restrições de deslocamento e rotação.

---

### `Carga.py`

Define os carregamentos aplicados na estrutura:

- cargas pontuais;
- cargas distribuídas;
- momentos fletores concentrados.

---

### `DadosEntrada.py`

Realiza o pré-processamento do problema estrutural.

Entre suas responsabilidades estão:

- identificação automática de descontinuidades;
- subdivisão do domínio em trechos;
- preparação dos dados utilizados durante o treinamento da PINN.

---

### `Formas.py`

Implementa a formulação matemática das funções de perda da PINN.

São calculadas:

- perda física (PDE Loss);
- perdas de condições de contorno (Boundary Condition Loss);
- perdas de continuidade;
- perdas de salto.

---

### `PINNViga.py`

Classe principal do framework.

Responsável por:

- criação da rede neural;
- gerenciamento dos hiperparâmetros;
- treinamento da PINN;
- execução dos otimizadores Adam e L-BFGS;
- gerenciamento do processo completo de aprendizado.

---

### `Output.py`

Módulo responsável pelo pós-processamento dos resultados.

Permite:

- geração dos diagramas estruturais;
- comparação com a solução analítica;
- criação dos gráficos de treinamento;
- validação utilizando o pacote **anastruct**.

---

# Instalação

Clone o repositório:

```bash
git clone https://github.com/SEU_USUARIO/PINNViga.git
cd PINNViga
```

Instale as dependências:

```bash
pip install torch numpy matplotlib anastruct
```

---

# Dependências

O projeto utiliza principalmente:

- PyTorch
- NumPy
- Matplotlib
- anastruct

---

# Exemplo de utilização

O exemplo abaixo cria uma viga biapoiada com uma carga distribuída e uma carga pontual, treina a PINN e gera os diagramas estruturais.

```python
from src import PINNViga
from src import Carga
from src import Apoio

# Comprimento da viga
L = 1.0

# Propriedades da seção
b = 0.2
h = 0.4

E = 30e9
I = b * (h ** 3) / 12
EI = E * I

# Apoios
apoio_esq = Apoio.Apoio([0, 1, 0], [0, 0])
apoio_meio = Apoio.Apoio([0, 0, 0], [L / 2, 0])
apoio_dir = Apoio.Apoio([0, 1, 0], [L, 0])

lista_apoios = [
    apoio_esq,
    apoio_meio,
    apoio_dir
]

# Carregamentos
p_valor = -10e3
p = Carga.CargaPontual([0, p_valor, 0], apoio_meio)

q_valor = -10e2
q = Carga.CargaDistribuida(
    [0, q_valor],
    apoio_esq,
    apoio_dir
)

cargas = [q, p]

# Criação da PINN
viga = PINNViga.PINNViga(
    lista_apoios,
    cargas,
    E,
    I,
    tipos_cargas="superposicao",
    depth=3,
    width=16,
    otimizador="lbfgs"
)

# Treinamento
viga.run_model(
    num_epochs=5,
    tol=1e-5
)

# Solução analítica
viga.run_sol_analitica(tam=20)

# Gráficos
viga.plot_errors(10)

viga.plot_deslocamento()
viga.plot_rotacao()
viga.plot_momento()
viga.plot_cortante()
```

---

# Fluxo de funcionamento

```
Definição da estrutura
          │
          ▼
 Criação dos apoios e cargas
          │
          ▼
 Pré-processamento do domínio
          │
          ▼
 Construção da PINN
          │
          ▼
 Treinamento
(Adam / L-BFGS)
          │
          ▼
 Validação via anastruct
          │
          ▼
 Pós-processamento
          │
          ▼
 Diagramas e resultados
```

---

# Aplicações

O framework pode ser utilizado para:

- estudos sobre PINNs;
- ensino de Mecânica das Estruturas;
- pesquisa em Engenharia Civil;
- pesquisa em Engenharia Mecânica;
- comparação entre métodos numéricos e PINNs;
- experimentação com arquiteturas de redes neurais.

---

# Tecnologias

- Python
- PyTorch
- NumPy
- Matplotlib
- anastruct

---

# Licença

Este projeto é distribuído sob a licença **MIT**.

---

# Autor

Desenvolvido por **Tácio Valmir** como parte do Trabalho de Conclusão de Curso:

> **Análise Estrutural de Vigas em Regime Estático Linear com Redes Neurais Informadas por Física (PINNs)**

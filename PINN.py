import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# 🔹 1. Definição do modelo PINN
# ============================================================

class PINN_Viga(nn.Module):
    def __init__(self):
        super(PINN_Viga, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# ============================================================
# 🔹 2. Função de perda baseada na equação diferencial da viga
#      EI * u'''' = q
# ============================================================

def physics_loss(model, x, EI, q):
    # Remova x.requires_grad = True daqui, pois x já deve vir como tal do loop de treinamento
    u = model(x)
    
    # Primeira derivada
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    # Segunda derivada
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    # Terceira derivada
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    # Quarta derivada
    u_xxxx = torch.autograd.grad(u_xxx, x, torch.ones_like(u_xxx), create_graph=True)[0]
    
    # Equação de viga: EI u'''' = q  => EI u'''' - q = 0
    f = EI * u_xxxx - q
    
    return torch.mean(f**2)

# ============================================================
# 🔹 3. Função para imposição de condições de contorno
#      Viga biapoiada: u(0)=0, u(L)=0, u''(0)=0, u''(L)=0
# ============================================================

def boundary_loss(model, EI, L=1.0): 
    # Mantenha os tensores de contorno como folhas com requires_grad=True
    x0 = torch.tensor([[0.0]], requires_grad=True)
    xL = torch.tensor([[L]], requires_grad=True)
    
    # Condições de deflexão nula
    u0 = model(x0)
    uL = model(xL)
    
    loss_u0 = u0**2
    loss_uL = uL**2
    
    # Condições de momento fletor nulo (segunda derivada nula)
    # Para x=0
    u0_output = model(x0) # Use x0, que é leaf
    u0_x = torch.autograd.grad(u0_output, x0, torch.ones_like(u0_output), create_graph=True)[0]
    u0_xx = torch.autograd.grad(u0_x, x0, torch.ones_like(u0_x), create_graph=True)[0]
    loss_u0_xx = u0_xx**2
    
    # Para x=L
    uL_output = model(xL) # Use xL, que é leaf
    uL_x = torch.autograd.grad(uL_output, xL, torch.ones_like(uL_output), create_graph=True)[0]
    uL_xx = torch.autograd.grad(uL_x, xL, torch.ones_like(uL_x), create_graph=True)[0]
    loss_uL_xx = uL_xx**2
    
    bc_loss = loss_u0 + loss_uL + loss_u0_xx + loss_uL_xx
    return bc_loss.mean()

# ============================================================
# 🔹 4. Inicialização do modelo, parâmetros e otimizador
# ============================================================

model = PINN_Viga()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EI = 1.0  # Rigidez flexional (N.m²) - escolha valor real conforme viga
q = 1.0   # Carga distribuída (N/m) - escolha valor real conforme viga
L = 1.0   # Comprimento da viga (m)

# ============================================================
# 🔹 5. Loop de treinamento
# ============================================================

num_epochs = 5000 
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Geração de pontos de treino no domínio (0,L)
    # Crie o tensor folha com requires_grad=True
    # E então multiplique-o por L
    x_train_raw = torch.rand(1000, 1) # Não precisa de requires_grad=True aqui
    x_train = (x_train_raw * L).requires_grad_(True) # Crie a nova folha com requires_grad=True
    
    # Alternativa mais simples (e muitas vezes usada):
    # x_train = torch.rand(1000, 1) * L # Isso cria um tensor que não é folha
    # No entanto, se você SEMPRE passa esse x_train para uma função que FAZ x.requires_grad_(True),
    # o PyTorch fará uma cópia e tornará a cópia folha.
    # Mas para evitar o erro, a primeira abordagem é mais explícita.
    # O seu erro original ocorreu porque dentro de physics_loss, você tentou alterar requires_grad
    # de um tensor que já era o resultado de uma operação.

    # Removemos x.requires_grad = True de physics_loss pois x_train já é folha com grad
    loss_pde = physics_loss(model, x_train, EI, q)
    loss_bc = boundary_loss(model, EI, L=L) 
    
    loss = loss_pde + 100 * loss_bc 

    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0: 
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, PDE Loss: {loss_pde.item():.6f}, BC Loss: {loss_bc.item():.6f}")

# ============================================================
# 🔹 6. Plot da solução obtida
# ============================================================

x_plot = torch.linspace(0, L, 100).view(-1,1)
u_plot = model(x_plot).detach().numpy()

def y_calc(x_val): 
    return (q * x_val / (24 * EI)) * (L**3 - 2 * L * x_val**2 + x_val**3)

plt.figure(figsize=(10, 6))
plt.plot(x_plot.numpy(), u_plot, label='PINN Solution', color='blue')
plt.plot(x_plot.numpy(), y_calc(x_plot.numpy()), label='Solução Analítica', linestyle='dashed', color='orange')
plt.title("Deflexão da viga biapoiada - PINN")
plt.xlabel("x [m]")
plt.ylabel("u(x) [m]")
plt.grid(True)
plt.legend()
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src import Loss

class PINNViga(nn.Module):
    def __init__(self):
        super(PINNViga, self).__init__()
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

def create_model():
    model = PINNViga()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, optimizer

def run_model(model, optimizer, EI, q, L, num_epochs=5000):
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        x_train_raw = torch.rand(1000, 1)
        x_train = (x_train_raw * L).requires_grad_(True)
        
        loss_pde = Loss.physics_loss(model, x_train, EI, q)
        loss_bc = Loss.boundary_loss(model, EI, L=L)

        loss = loss_pde + 1000 * loss_bc

        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0: 
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, PDE Loss: {loss_pde.item():.6f}, BC Loss: {loss_bc.item():.6f}")

    x_plot = torch.linspace(0, L, 100).view(-1,1)
    u_plot = model(x_plot).detach().numpy()

    return x_plot.numpy(), u_plot
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


def PICNNtrain(model, dataloader, init_z, lr=0.001, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        sum_loss = 0
        cpt=0

        for x, c, y in dataloader:
            optimizer.zero_grad()

            x.requires_grad_(True)
            c.requires_grad_(True)
            
            output = model(x, c, init_z)

            loss = criterion(output, y) # Compute the loss
            loss.backward() # Backward pass
            optimizer.step() # Update the parameters

            for layers_k in model.layers_z: #model clamping
                for param in layers_k.parameters():
                    param.data.clamp_min_(0)

            sum_loss += loss.item()
            cpt+=1
    
        mean_loss = sum_loss / cpt
        print(f"Epoch {epoch+1}/{epochs} Loss: {mean_loss}")
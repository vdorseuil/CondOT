import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


def PICNNtrain(model, dataloader, init_z, lr=0.001, epochs=50):
    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        sum_loss = 0.0
        cpt=0
        for x_batch, c_batch, y_batch in dataloader:
            x_batch.requires_grad_(True)
            c_batch.requires_grad_(True)
            optimizer.zero_grad() # Zero the gradients
            
            #z_0 = gaussian_transport_data(x_batch.clone().detach().numpy(), y_batch.clone().detach().numpy(), x_batch.clone().detach().numpy())
            #z_0 = init_z(x_batch)
            output = model(x_batch, c_batch, init_z)  # Assuming context c is same as input x

            loss = criterion(output, y_batch) # Compute the loss
            loss.backward() # Backward pass

            #print(y_batch[:,:,0][0], output[:,:,0][0])

            optimizer.step() # Update the parameters

            for layers_k in model.layers_z:
                for param in layers_k.parameters():
                    param.data.clamp_min_(0)

            sum_loss += loss.item()
            cpt+=1
    
        mean_loss = sum_loss / cpt
        print(f"Epoch {epoch+1}/{epochs} Loss: {mean_loss}")
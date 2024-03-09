import torch.nn as nn
import torch.optim as optim
import torch

from geomloss import SamplesLoss

from visualization import compute_grad

def compute_grad_torch(source, context, model, init_z = lambda x: (1/2) * torch.norm(x, dim=-1, keepdim=True)**2):
    source.requires_grad_(True)
    context.requires_grad_(True)
    z_0 = init_z(source)
    output_model = model(source, context, z_0)
    grad_model = torch.autograd.grad(outputs=output_model, inputs=source, grad_outputs=torch.ones_like(output_model), create_graph=True)[0]
    return grad_model

def train_wasserstein(model, dataloader, init_z, lr=0.001, epochs=50):
    # Define the loss function and the optimizer
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_was =  SamplesLoss(loss="sinkhorn", p=2, blur=0.01)

    for epoch in range(epochs):
        sum_loss = 0.0
        cpt=0
        for x_batch, c_batch, y_batch in dataloader:
            x_batch.requires_grad_(True)
            c_batch.requires_grad_(True)
            #y_batch.requires_grad_(True)
            optimizer.zero_grad() # Zero the gradients
            
            z_0 = init_z(x_batch)
            output = model(x_batch, c_batch, z_0)  # Assuming context c is same as input x

            transported_bis = compute_grad_torch(x_batch, c_batch, model)

            #print(transported_bis)
            loss = loss_was(transported_bis, y_batch) # Compute the loss
            loss = loss.mean()
            loss.backward() # Backward pass

            optimizer.step() # Update the parameters

            for layers_k in model.layers_z:
                for param in layers_k.parameters():
                    param.data.clamp_min_(0)

            sum_loss += loss.item()
            cpt+=1
    
        mean_loss = sum_loss / cpt
        print(f"Epoch {epoch+1}/{epochs} Loss: {mean_loss}")
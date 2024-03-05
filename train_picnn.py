import torch.nn as nn
import torch.optim as optim

def PICNNtrain(model, dataloader, epochs=50):
    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        for x_batch, c_batch, y_batch in dataloader:
            x_batch.requires_grad_(True)
            c_batch.requires_grad_(True)

            optimizer.zero_grad() # Zero the gradients
            
            output = model(x_batch, c_batch)  # Assuming context c is same as input x

            loss = criterion(output, y_batch) # Compute the loss
            loss.backward() # Backward pass

            optimizer.step() # Update the parameters
            for layers_k in model.layers_z:
                for param in layers_k.parameters():
                    param.data.clamp_min_(0)

        # for name, parameter in model.named_parameters():
        #     if parameter.requires_grad and parameter.grad is not None:
        #         grad_norm = parameter.grad.norm().item()
        #         print(f"Gradient norm for {name}: {grad_norm}")
        
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")
    print('Finished Training')

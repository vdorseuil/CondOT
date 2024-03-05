import torch
import torch.optim as optim

def train_makkuva(ICNNf, ICNNg, dataloader, epochs = 100, train_freq_g = 10):

    # Define the loss function and the optimizer
    optimizer_f = optim.Adam(ICNNf.parameters(), lr = 0.0001)
    optimizer_g = optim.Adam(ICNNg.parameters(), lr = 0.0001)

    for epoch in range(epochs):
        for freq in range(train_freq_g) :
            for x, c, y in dataloader:
                # Optimizing ICNNg
                optimizer_f.zero_grad() # Zero the gradients
                optimizer_g.zero_grad() # Zero the gradients

                x.requires_grad_(True)
                y.requires_grad_(True)
                c.requires_grad_(True)

                output_g = ICNNg(y, c)
                grad_g = torch.autograd.grad(output_g, y, grad_outputs=torch.ones_like(output_g), create_graph=True)[0]

                loss_g = ICNNf(grad_g, c) - torch.sum(y * grad_g, dim=-1, keepdim=True)
                loss_g = loss_g.mean(dim=(1, 2)).mean()

                #print('min f(grad_g, c)', ICNNf(grad_g, c)[0][0])
                #print('max f(grad_g, c)', torch.sum(y * grad_g, dim=-1, keepdim=True)[0][0])

                loss_g.backward() # Backward pass
                optimizer_g.step() # Update the parameters

                for layers_k in ICNNg.layers_z:
                    for param in layers_k.parameters():
                        param.data.clamp_min_(0)
                        
                # for name, parameter in ICNNg.named_parameters():
                #     if parameter.requires_grad and parameter.grad is not None:
                #         grad_norm = parameter.grad.norm().item()
                #         print(f"Gradient norm for {name}: {grad_norm}")
                        
                #print(f"training g {freq+1}/{train_freq_g} loss_g: {loss_g.item()}")

        for x, c, y in dataloader:
            optimizer_f.zero_grad() # Zero the gradients
            optimizer_g.zero_grad()

            x.requires_grad_(True)
            y.requires_grad_(True)
            c.requires_grad_(True)
        
            output_g = ICNNg(y, c)
            grad_g = torch.autograd.grad(outputs=output_g, inputs=y, grad_outputs=torch.ones_like(output_g), create_graph=True)[0]

            #print('ICNNf(grad_g, c)', ICNNf(grad_g, c))

            loss_f = ICNNf(x, c) - ICNNf(grad_g, c)
            #loss_f =  torch.mean(loss_f)
            loss_f =   torch.mean(loss_f) #page 24, f is updated by fixing g and maximizing (15) with a single iteration

            #print('max f(grad_g, c)', ICNNf(grad_g, c)[0][0])

            loss_f.backward() # Backward pass
            optimizer_f.step()
            
            for layers_k in ICNNf.layers_z:
                    for param in layers_k.parameters():
                        param.data.clamp_min_(0)

            # for name, parameter in ICNNf.named_parameters():
            #     if parameter.requires_grad and parameter.grad is not None:
            #         grad_norm = parameter.grad.norm().item()
            #         print(f"Gradient norm for {name}: {grad_norm}")
            
        print(f"Epoch {epoch+1}/{epochs} loss_g: {loss_g.item()}, loss_f: {loss_f.item()}")
    
def train_makkuva_epoch(ICNNf, ICNNg, dataloader, train_freq_g = 10) :
    # Define the loss function and the optimizer
    optimizer_f = optim.Adam(ICNNf.parameters(), lr = 0.0001)
    optimizer_g = optim.Adam(ICNNg.parameters(), lr = 0.0001)

    sum_loss_g = 0
    cpt_g = 0

    for freq in range(train_freq_g) :
        for x, c, y in dataloader:
            # Optimizing ICNNg
            optimizer_f.zero_grad() # Zero the gradients
            optimizer_g.zero_grad() # Zero the gradients

            x.requires_grad_(True)
            y.requires_grad_(True)
            c.requires_grad_(True)

            output_g = ICNNg(y, c)
            grad_g = torch.autograd.grad(output_g, y, grad_outputs=torch.ones_like(output_g), create_graph=True)[0]

            loss_g = ICNNf(grad_g, c) - torch.sum(y * grad_g, dim=-1, keepdim=True)
            loss_g = loss_g.mean(dim=(1, 2)).mean()

            #print('min f(grad_g, c)', ICNNf(grad_g, c)[0][0])
            #print('max f(grad_g, c)', torch.sum(y * grad_g, dim=-1, keepdim=True)[0][0])

            loss_g.backward() # Backward pass
            optimizer_g.step() # Update the parameters

            for layers_k in ICNNg.layers_z:
                for param in layers_k.parameters():
                    param.data.clamp_min_(0)
                    
            # for name, parameter in ICNNg.named_parameters():
            #     if parameter.requires_grad and parameter.grad is not None:
            #         grad_norm = parameter.grad.norm().item()
            #         print(f"Gradient norm for {name}: {grad_norm}")
                    
            #print(f"training g {freq+1}/{train_freq_g} loss_g: {loss_g.item()}")

        if freq == train_freq_g - 1 :
            cpt_g+=1
            sum_loss_g += loss_g.item()
        
    mean_loss_g = sum_loss_g/cpt_g
    
    sum_loss_f = 0
    cpt_f = 0

    for x, c, y in dataloader:
        optimizer_f.zero_grad() # Zero the gradients
        optimizer_g.zero_grad()

        x.requires_grad_(True)
        y.requires_grad_(True)
        c.requires_grad_(True)
    
        output_g = ICNNg(y, c)
        grad_g = torch.autograd.grad(outputs=output_g, inputs=y, grad_outputs=torch.ones_like(output_g), create_graph=True)[0]

        #print('ICNNf(grad_g, c)', ICNNf(grad_g, c))

        loss_f = ICNNf(x, c) - ICNNf(grad_g, c)
        #loss_f =  torch.mean(loss_f)
        loss_f =   torch.mean(loss_f) #page 24, f is updated by fixing g and maximizing (15) with a single iteration

        #print('max f(grad_g, c)', ICNNf(grad_g, c)[0][0])

        loss_f.backward() # Backward pass
        optimizer_f.step()
        
        for layers_k in ICNNf.layers_z:
                for param in layers_k.parameters():
                    param.data.clamp_min_(0)

        sum_loss_f += loss_f.item()
        cpt_f+=1

        # for name, parameter in ICNNf.named_parameters():
        #     if parameter.requires_grad and parameter.grad is not None:
        #         grad_norm = parameter.grad.norm().item()
        #         print(f"Gradient norm for {name}: {grad_norm}")
    
    mean_loss_f = sum_loss_f/cpt_f
        
    print(f"loss_g: {mean_loss_g}, loss_f: {mean_loss_f}")
import torch
import torch.optim as optim
import numpy as np
from gaussian_transport import gaussian_transport_data
from visualization import compute_grad

def train_makkuva(ICNNf, ICNNg, dataloader, epochs = 100, train_freq_g = 10, regularize_g = False):

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

                if regularize_g == True :
                    R_g = 0
                    for layers_k in ICNNg.layers_z:
                        for param in layers_k.parameters():
                            R_g+= torch.norm(torch.max(-param, torch.zeros_like(param)), p=2)
                    
                    loss_g = loss_g + 0.1*R_g
                
                loss_g.backward() # Backward pass
                optimizer_g.step() # Update the parameters

                if regularize_g == False :
                    for layers_k in ICNNg.layers_z:
                        for param in layers_k.parameters():
                            param.data.clamp_min_(0)

        for x, c, y in dataloader:
            optimizer_f.zero_grad() # Zero the gradients
            optimizer_g.zero_grad()

            x.requires_grad_(True)
            y.requires_grad_(True)
            c.requires_grad_(True)
        
            output_g = ICNNg(y, c)
            grad_g = torch.autograd.grad(outputs=output_g, inputs=y, grad_outputs=torch.ones_like(output_g), create_graph=True)[0]

            loss_f = ICNNf(x, c) - ICNNf(grad_g, c)
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
    
def train_makkuva_epoch(ICNNf, ICNNg, prev_param_f, prev_param_g, dataloader, init_z_f, init_z_g, lr=0.0001, train_freq_g = 10, train_freq_f = 1, regularize_g = False, regularize_f = False, lambda_proximal = 0) :
    # Define the loss function and the optimizer
    optimizer_f = optim.SGD(ICNNf.parameters(), lr = lr)
    optimizer_g = optim.SGD(ICNNg.parameters(), lr = lr)

    sum_loss_g = 0
    cpt_g = 0

    #train_freq_g = np.random.randint(1, train_freq_g+1)
    #train_freq_f = np.random.randint(1, train_freq_f+1)

    for freq in range(train_freq_g) :
        for x, c, y in dataloader:
            # Optimizing ICNNg
            optimizer_f.zero_grad() # Zero the gradients
            optimizer_g.zero_grad() # Zero the gradients

            x.requires_grad_(True)
            y.requires_grad_(True)
            c.requires_grad_(True)

            #z_0 = init_z_g(y.clone().detach().numpy())
            output_g = ICNNg(y, c, init_z_g)
            grad_g = torch.autograd.grad(output_g, y, grad_outputs=torch.ones_like(output_g), create_graph=True)[0]
           
            #y_trans = compute_grad(source = x, context = c, model = ICNNf, init_z=init_z_f).detach()

            loss_g = 0
            loss_g = ICNNf(grad_g, c, init_z_f) - torch.sum(y * grad_g, dim=-1, keepdim=True)
            #loss_g = ICNNf(grad_g, c, init_z_f) - torch.sum(y_trans * grad_g, dim=-1, keepdim=True)
            #loss_g = (ICNNf(grad_g, c, z_0_g)+old_ICNNf(grad_g, c, z_0_g))/2 - torch.sum(y * grad_g, dim=-1, keepdim=True)
            loss_g = loss_g.mean(dim=(1, 2)).mean()
            
            loss_g.backward(retain_graph=True) # Backward pass

            simulated_parameters = {}
            with torch.no_grad():
                for name, param in ICNNg.named_parameters():
                    if param.grad is not None and not torch.isnan(param.grad).any():
                        learning_rate = lr
                        # This simulates the updated parameter without actually updating the mode
                        param_simulated = param - learning_rate * param.grad
                        param_simulated.requires_grad_(True)
                        simulated_parameters[name] = param_simulated
                    else :
                        print('name', name)
                        simulated_parameters[name] = param + 1e-6*torch.randn_like(param)
                
            #computing regularizer
            if regularize_g == True :
                R_g = 0
                # for layers_k in simulated_parameters.layers_z:
                #     for param in layers_k.parameters():
                for name, param in ICNNg.named_parameters():
                    if name[:9]=='layers_z.' :
                        R_g += torch.norm(torch.max(-simulated_parameters[name], torch.zeros_like(simulated_parameters[name])), p=2)
                        #R_f += torch.norm(torch.max(-param, torch.zeros_like(param)), p=2)
                #print('R_g', R_g.item())
                loss_g = loss_g + R_g

            #computing proximal term
            proximal_term = 0
            if lambda_proximal > 0 and prev_param_f is not None:
                for sim_param, prev_param in zip(param_simulated, prev_param_g):
                        proximal_term += (lambda_proximal / 2) * torch.norm(sim_param - prev_param)**2
                #print('proximal_term', proximal_term.item())
                loss_g = loss_g + proximal_term

            optimizer_g.zero_grad() # Zero the gradients
            loss_g.backward() # Backward pass
            optimizer_g.step()


            if regularize_g == False :
                for layers_k in ICNNg.layers_z:
                    for param in layers_k.parameters():
                        param.data.clamp_min_(0)

            #print('loss_g', loss_g.item())

        if freq == train_freq_g - 1 :
            cpt_g+=1
            sum_loss_g += loss_g.item()
        
    mean_loss_g = sum_loss_g/cpt_g
    
    sum_loss_f = 0
    cpt_f = 0

    for freq in range(train_freq_f) :
        for x, c, y in dataloader:
            optimizer_f.zero_grad() # Zero the gradients
            optimizer_g.zero_grad()

            x.requires_grad_(True)
            y.requires_grad_(True)
            c.requires_grad_(True)
        

            output_g = ICNNg(y, c, init_z_g)
            grad_g = torch.autograd.grad(outputs=output_g, inputs=y, grad_outputs=torch.ones_like(output_g), create_graph=True)[0]

            # old_output_g = old_ICNNg(y, c, z_0_y)
            # old_grad_g = torch.autograd.grad(outputs=old_output_g, inputs=y, grad_outputs=torch.ones_like(output_g), create_graph=True)[0]
            # old_z_0_g = init_z(grad_g)
            # loss_f = ICNNf(x, c, z_0_x) - (ICNNf(grad_g, c, z_0_g)+old_ICNNf(old_grad_g, c, old_z_0_g))/2

            loss_f = 0
            loss_f = ICNNf(x, c, init_z_f) - ICNNf(grad_g, c, init_z_f)
            loss_f = torch.mean(loss_f) #page 24, f is updated by fixing g and maximizing (15) with a single iteration

            # Computing expected parameters
            loss_f.backward(retain_graph=True) # Backward pass

            simulated_parameters = {}
            with torch.no_grad():
                for name, param in ICNNf.named_parameters():
                    if param.grad is not None and not torch.isnan(param.grad).any():
                        learning_rate = lr
                        # This simulates the updated parameter without actually updating the mode
                        param_simulated = param - learning_rate * param.grad
                        param_simulated.requires_grad_(True)
                        simulated_parameters[name] = param_simulated
                    else :
                        print('name f ', name)
                        simulated_parameters[name] = param + 1e-6*torch.randn_like(param)
                
            #computing regularizer
            if regularize_f == True :
                R_f = 0
                # for layers_k in simulated_parameters.layers_z:
                #     for param in layers_k.parameters():
                for name, param in ICNNf.named_parameters():
                    if name[:9]=='layers_z.' :
                        R_f += torch.norm(torch.max(-simulated_parameters[name], torch.zeros_like(simulated_parameters[name])), p=2)

                        #R_f += torch.norm(torch.max(-param, torch.zeros_like(param)), p=2)
                print('R_f', R_f.item())
                loss_f = loss_f + R_f

            #computing proximal term
            proximal_term = 0
            if lambda_proximal > 0 and prev_param_f is not None:
                for sim_param, prev_param in zip(param_simulated, prev_param_f):
                        proximal_term += (lambda_proximal / 2) * torch.norm(sim_param - prev_param)**2
                print('proximal_term', proximal_term.item())
                loss_f = loss_f + proximal_term

            optimizer_f.zero_grad() # Zero the gradients
            loss_f.backward() # Backward pass
            optimizer_f.step()


            # print('prev_param_g', prev_param_g)
            # for name, param in ICNNg.named_parameters():
            #     print('ICNNG', name, param)

            # # print('prev_param_f', prev_param_f)
            # for name, param in ICNNf.named_parameters():
            #     print('ICNNF', name, param)
            #     print('simulated_parameters', name,  simulated_parameters[name])



            if regularize_f == False :
                for layers_k in ICNNf.layers_z:
                        for param in layers_k.parameters():
                            param.data.clamp_min_(0)

            #print('loss f', loss_f.item())

        if freq == train_freq_f - 1 :
            sum_loss_f += loss_f.item()
            cpt_f+=1
        
    mean_loss_f = sum_loss_f/cpt_f
        
    print(f"loss_g: {mean_loss_g}, loss_f: {mean_loss_f}")
    param_f = [param.clone().detach() for param in ICNNf.parameters()]
    param_g = [param.clone().detach() for param in ICNNg.parameters()]
    return((mean_loss_f, mean_loss_g, param_f, param_g))
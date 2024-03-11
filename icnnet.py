import torch.nn as nn
from torch.nn import init
import torch
from gaussian_transport import gaussian_transport_data

class ICNNet(nn.Module):
    def __init__(self, z_0_size = 1, layer_sizes = [1,4,1], context_layer_sizes=[1,2,1], init_bunne = True):
        super(ICNNet, self).__init__()
        self.n_layers = len(layer_sizes)

        self.layers_activation = nn.ModuleList([nn.LeakyReLU() for _ in range(self.n_layers-1)])
        self.layers_z = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False) for i in range(self.n_layers-1)]) #non zero entries in the matrix ?
        self.layers_zu = nn.ModuleList([nn.Sequential(nn.Linear(context_layer_sizes[i], layer_sizes[i]), nn.ReLU()) for i in range(self.n_layers-1)])
        self.layers_x = nn.ModuleList([nn.Linear(layer_sizes[0], layer_sizes[i+1], bias=False) for i in range(self.n_layers-1)])
        self.layers_xu = nn.ModuleList([nn.Linear(context_layer_sizes[i], z_0_size) for i in range(self.n_layers-1)])
        self.layers_u = nn.ModuleList([nn.Linear(context_layer_sizes[i], layer_sizes[i+1]) for i in range(self.n_layers-1)])
        self.layers_v = nn.ModuleList([nn.Sequential(nn.Linear(context_layer_sizes[i], context_layer_sizes[i+1]), nn.LeakyReLU()) for i in range(self.n_layers-1)])
        
        if init_bunne == True :
            for layer in self.layers_z:
                nn.init.xavier_uniform_(layer.weight)

            for seq_layer in self.layers_zu:
                for layer in seq_layer:
                    if isinstance(layer, nn.Linear):  # check if the layer is Linear
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)

            for layer in self.layers_x:
                nn.init.xavier_uniform_(layer.weight)
            
            for layer in self.layers_xu:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            
            for layer in self.layers_u:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            
            for seq_layer in self.layers_v:
                for layer in seq_layer:
                    if isinstance(layer, nn.Linear):  # check if the layer is Linear
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
        
        eps = 1e-6
        if init_bunne == 'TR':
            for i in range(self.n_layers-1):
                init.constant_(self.layers_z[i].weight, 1.0 / layer_sizes[i])

                #init.constant_(self.layers_x[i].weight, 1)
                #init.constant_(self.layers_xu[i].weight, 1)

                init.constant_(self.layers_xu[i].bias, eps)

                init.constant_(self.layers_u[i].bias, eps)
                init.constant_(self.layers_zu[i][0].bias, 1)

                init.constant_(self.layers_v[i][0].bias, 0) #because of sequential, select nn.linear and not activation
                init.eye_(self.layers_v[i][0].weight) #because of sequential, select nn.linear and not activation



            init.eye_(self.layers_zu[0][0].weight) #because of sequential, select nn.linear and not activation
            init.constant_(self.layers_zu[0][0].bias, 0) #because of sequential, select nn.linear and not activation
            
    def forward(self, x, c, z_0):
        input = x
        z = z_0
        u = c #it's embedding
        for i in range(self.n_layers-1):
            if i == self.n_layers-2:
                z = self.layers_z[i](z * self.layers_zu[i](u)) + self.layers_x[i](input * self.layers_xu[i](u)) + self.layers_u[i](u)
                u = self.layers_v[i](u)
            else : 
                z = self.layers_activation[i](self.layers_z[i](z * self.layers_zu[i](u)) + self.layers_x[i](input * self.layers_xu[i](u)) + self.layers_u[i](u))
                u = self.layers_v[i](u)
        return z
    
def compute_grad(source, target, context, model):
    source.requires_grad_(True)
    context.requires_grad_(True)
    z_0 = gaussian_transport_data(source = source.clone().detach().numpy(), target = target.clone().detach().numpy(), data = source.clone().detach().numpy())
    output_model = model(source, context, z_0)
    grad_model = torch.autograd.grad(outputs=output_model, inputs=source, grad_outputs=torch.ones_like(output_model), create_graph=True)[0].detach().numpy()
    return grad_model
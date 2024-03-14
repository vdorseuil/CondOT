import torch
import torch.nn as nn
from torch.nn import init

class ICNNet(nn.Module):
    def __init__(self, layer_sizes, context_layer_sizes, init_bunne = True):
        super(ICNNet, self).__init__()
        self.n_layers = len(layer_sizes)

        self.layers_activation = nn.ModuleList([nn.LeakyReLU() for _ in range(self.n_layers-1)])
        self.layers_z = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False) for i in range(self.n_layers-1)]) #non zero entries in the matrix ?
        self.layers_zu = nn.ModuleList([nn.Sequential(nn.Linear(context_layer_sizes[i], layer_sizes[i]), nn.LeakyReLU()) for i in range(self.n_layers-1)])
        self.layers_x = nn.ModuleList([nn.Linear(layer_sizes[0], layer_sizes[i+1], bias=False) for i in range(self.n_layers-1)])
        self.layers_xu = nn.ModuleList([nn.Linear(context_layer_sizes[i], layer_sizes[0]) for i in range(self.n_layers-1)])
        self.layers_u = nn.ModuleList([nn.Linear(context_layer_sizes[i], layer_sizes[i+1]) for i in range(self.n_layers-1)])
        self.layers_v = nn.ModuleList([nn.Sequential(nn.Linear(context_layer_sizes[i], context_layer_sizes[i+1]), nn.LeakyReLU()) for i in range(self.n_layers-2)])
    
        eps = 1e-6
        if init_bunne : #more or less to propagate identity
            for i in range(self.n_layers-1):
                init.constant_(self.layers_z[i].weight, 1.0 / layer_sizes[i])
                init.constant_(self.layers_xu[i].bias, eps)
                init.constant_(self.layers_u[i].bias, eps)
                init.constant_(self.layers_zu[i][0].bias, 1)

                if i < self.n_layers-2:
                    init.constant_(self.layers_v[i][0].bias, eps) #because of sequential, select nn.linear and not activation
                    init.eye_(self.layers_v[i][0].weight) #because of sequential, select nn.linear and not activation

            init.eye_(self.layers_zu[0][0].weight) #because of sequential, select nn.linear and not activation
            init.constant_(self.layers_zu[0][0].bias, 1) #because of sequential, select nn.linear and not activation
            
    def forward(self, x, c, init_z):
        input = x
        z = init_z(x)
        u = c #or it's embedding if necessary
        for i in range(self.n_layers-1):
            if i == self.n_layers-2:
                z = self.layers_z[i](z * self.layers_zu[i](u)) + self.layers_x[i](input * self.layers_xu[i](u)) + self.layers_u[i](u)
                #u = self.layers_v[i](u) #not necessary as we are only interested in the output z
            else : 
                z = self.layers_activation[i](self.layers_z[i](z * self.layers_zu[i](u)) + self.layers_x[i](input * self.layers_xu[i](u)) + self.layers_u[i](u))
                u = self.layers_v[i](u)
        return z
    
def compute_grad(source, context, model, init_z):
    source.retain_grad()
    z_0 = init_z(source)
    output_model = model(source, context, init_z)
    grad_model = torch.autograd.grad(outputs=output_model, inputs=source, grad_outputs=torch.ones_like(output_model), create_graph=True)[0]
    return grad_model
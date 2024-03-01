import torch.nn as nn
import torch

class ICNNet(nn.Module):
    def __init__(self, input_size = 1, layer_sizes = [1,4,1], context_layer_sizes=[1,2,1]):
        super(ICNNet, self).__init__()
        self.n_layers = len(layer_sizes)

        self.layers_activation = nn.ModuleList([nn.Softplus() for _ in range(self.n_layers-1)])

        self.layers_z = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False) for i in range(self.n_layers-1)]) #non zero entries in the matrix ?

        self.layers_zu = nn.ModuleList([nn.Sequential(nn.Linear(context_layer_sizes[i], layer_sizes[i]), nn.ReLU()) for i in range(self.n_layers-1)])

        self.layers_x = nn.ModuleList([nn.Linear(input_size, layer_sizes[i+1], bias=False) for i in range(self.n_layers-1)])

        self.layers_xu = nn.ModuleList([nn.Linear(context_layer_sizes[i], input_size) for i in range(self.n_layers-1)])

        self.layers_u = nn.ModuleList([nn.Linear(context_layer_sizes[i], layer_sizes[i+1]) for i in range(self.n_layers-1)])

        self.layers_v = nn.ModuleList([nn.Sequential(nn.Linear(context_layer_sizes[i], context_layer_sizes[i+1]), nn.ReLU()) for i in range(self.n_layers-1)])
        
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

    def forward(self, x, c):
        input = x
        z = torch.zeros_like(x)
        u = c
        for i in range(self.n_layers-1):
            if i == self.n_layers-2:
                z = self.layers_z[i](z * self.layers_zu[i](u)) + self.layers_x[i](input * self.layers_xu[i](u)) + self.layers_u[i](u)
                u = self.layers_v[i](u)
            else : 
                z = self.layers_activation[i](self.layers_z[i](z * self.layers_zu[i](u)) + self.layers_x[i](input * self.layers_xu[i](u)) + self.layers_u[i](u))
                u = self.layers_v[i](u)
        return z
#How are the parameters initialize ?

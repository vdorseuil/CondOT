import torch
import matplotlib.pyplot as plt

def compute_grad(source, context, model, init_z = lambda x: (1/2) * torch.norm(x, dim=-1, keepdim=True)**2):
    source.requires_grad_(True)
    context.requires_grad_(True)
    z_0 = init_z(source)
    output_model = model(source, context, z_0)
    grad_model = torch.autograd.grad(outputs=output_model, inputs=source, grad_outputs=torch.ones_like(output_model), create_graph=True)[0].detach().numpy()
    return grad_model

def plot_distribution(source_distribution, target_distribution, transported_distribution, filename):
    plt.scatter(source_distribution[:, 0], source_distribution[:, 1], alpha=0.3, label='source distribution', edgecolors='none')
    plt.scatter(target_distribution[:, 0], target_distribution[:, 1], color='orange', alpha=0.3, label='target distribution', edgecolors='none')
    plt.scatter(transported_distribution[:, 0], transported_distribution[:, 1], alpha=0.3, color='green', label='transported distribution', edgecolors='none')

    for i in range(source_distribution[:, 0].shape[0]):
        t=1
        plt.arrow(source_distribution[i, 0], source_distribution[i, 1], transported_distribution[i, 0] - source_distribution[i, 0], transported_distribution[i, 1] - source_distribution[i, 1], color='green', alpha=0.05, head_width=0.1, head_length=0.1, length_includes_head=True)
        #plt.arrow(transported_distribution[i, 0], transported_distribution[i, 1], target_distribution[i, 0] - transported_distribution[i, 0], target_distribution[i, 1] - transported_distribution[i, 1], color='orange', alpha=0.25, head_width=0.1, head_length=0.1, length_includes_head=True)

        
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Trained transport map using Makkuva\'s method')
    plt.legend(loc='lower center', ncol=3, fontsize='small')

    plt.xlim(-1, 5)
    plt.ylim(-1, 5) 
    plt.grid(True)

    plt.tight_layout()
    plt.rcParams['figure.dpi'] = 200
    plt.savefig(filename)
    plt.clf()
    #plt.show()

def plot_transport(dataset, test, model_f, model_g, filename_f, filename_g, n_points=1000) :
    x_i, y_i, c_i = dataset.X[test, :n_points, :], dataset.Y[test, :n_points, :], dataset.C[test, :n_points, :]

    grad_model_f = compute_grad(x_i, c_i, model_f)
    grad_model_g = compute_grad(y_i, c_i, model_g)

    x_i = x_i.detach().numpy()
    y_i = y_i.detach().numpy()

    plot_distribution(source_distribution=x_i, target_distribution=y_i, transported_distribution=grad_model_f, filename=filename_f)
    plot_distribution(source_distribution=y_i, target_distribution=x_i, transported_distribution=grad_model_g, filename=filename_g)
import torch
import torch.nn as nn
import torchvision
from model import DataDepthTwinsModel
from transforms import Transform


TUKEY_DEPTH_STEPS = 20

device = 'cuda:0'  # TODO: Adjust


def soft_tukey_depth(X_, X, Z, temp):
    X_new = X.repeat(X_.size(dim=0), 1, 1)
    X_new_tr = X_.repeat(X.size(dim=0), 1, 1).transpose(0, 1)
    X_diff = X_new - X_new_tr
    dot_products = X_diff.mul(Z.repeat(X.size(dim=0), 1, 1).transpose(0, 1)).sum(dim=2)
    dot_products_normalized = dot_products.transpose(0, 1).divide(temp * Z.norm(dim=1))
    return torch.sigmoid(dot_products_normalized).sum(dim=0).divide(X.size(dim=0))


def get_kl_divergence(soft_tukey_depths, f, kernel_bandwidth, epsilon=0.0):
    DELTA = 0.005
    kl_divergence = torch.tensor(0)
    for x in torch.arange(0, 0.5, DELTA):
        val = torch.exp(torch.square(soft_tukey_depths - x).divide(torch.tensor(-2 * kernel_bandwidth * kernel_bandwidth))).mean()
        f_val = f(x)
        kl_divergence = kl_divergence.subtract(torch.multiply(torch.tensor(f_val * DELTA), torch.log(val.divide(f_val + epsilon))))
    return kl_divergence


train_data = torchvision.datasets.CIFAR10()
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=128)

model = DataDepthTwinsModel()
optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3)

for x1, x2 in train_dataloader:
    x1, x2 = x1.to(device), x2.to(device)
    y1, y2 = model(x1), model(x2)
    y1_detached, y2_detached = y1.detach(), y2.detach()

    optimizer_model.zero_grad()

    sim_loss = torch.square(y1 - y2)

    z = nn.Parameter(2 * torch.rand(y1.shape[0], y1.shape[1], device=device) - 1)
    optimizer_z = torch.optim.SGD([z], lr=1e+2)

    for i in range(TUKEY_DEPTH_STEPS):
        optimizer_z.zero_grad()
        tukey_depths = soft_tukey_depth(y1_detached, y2_detached, z, 1)
        tukey_depths.sum().backward()
        optimizer_z.step()

    tukey_depths = soft_tukey_depth(y1, y2, z.detach(), 1)
    td_loss = get_kl_divergence(tukey_depths, lambda x: 2, 0.05, 1e-5)

    total_loss = sim_loss + td_loss

    print(f'Sim loss: {sim_loss.item()}')
    print(f'TD loss: {td_loss.item()}')
    print(f'Total loss : {total_loss.item()}')

    total_loss.backward()
    optimizer_model.step()




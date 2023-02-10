import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from dataset import NormalCIFAR10Dataset
from model import DataDepthTwinsModel
from transforms import Transform


TUKEY_DEPTH_STEPS = 40
TEMP = 10
EPOCHS = 100
LEARNING_RATE = 1e-4

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


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


def get_description(epoch, sim_loss, td_loss, total_loss):
    return f'Epoch {epoch} - Loss: {total_loss:.4f} (Sim: {sim_loss:.4f}, TD: {td_loss:.4f})'


train_data = NormalCIFAR10Dataset(normal_class=0, train=True, transform=Transform())
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)

model = DataDepthTwinsModel().to(device)
optimizer_model = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, EPOCHS+1):
    iterator = tqdm(train_dataloader, desc=get_description(epoch, 0, 0, 0), unit='batch')

    summed_sim_loss = torch.tensor(0, device=device, dtype=torch.float)
    summed_td_loss = torch.tensor(0, device=device, dtype=torch.float)
    summed_total_loss = torch.tensor(0, device=device, dtype=torch.float)

    batches = 0

    for x1, x2 in iterator:
        x1, x2 = x1.to(device), x2.to(device)
        y1, y2 = model(x1), model(x2)
        y1_detached, y2_detached = y1.detach(), y2.detach()

        optimizer_model.zero_grad()

        sim_loss = 1e-2 * torch.square(y1 - y2).sum(dim=1).mean()

        z = nn.Parameter(torch.rand(y1.shape[0], y1.shape[1], device=device).multiply(2).subtract(1))
        optimizer_z = torch.optim.SGD([z], lr=1e+2)

        for j in range(TUKEY_DEPTH_STEPS):
            optimizer_z.zero_grad()
            tukey_depths = soft_tukey_depth(y1_detached, y2_detached, z, 1)
            tukey_depths.sum().backward()
            optimizer_z.step()

        tukey_depths = soft_tukey_depth(y1, y2, z.detach(), TEMP)
        # print(tukey_depths.mean().item())
        td_loss = get_kl_divergence(tukey_depths, lambda x: 2, 0.05, 1e-5)


        total_loss = sim_loss + td_loss
        total_loss.backward()
        optimizer_model.step()

        summed_sim_loss += sim_loss
        summed_td_loss += td_loss
        summed_total_loss += total_loss

        batches += 1

        iterator.set_description(
            get_description(
                epoch,
                summed_sim_loss.item() / batches,
                summed_td_loss.item() / batches,
                summed_total_loss.item() / batches
            )
        )

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_model.state_dict(),
            'loss': summed_total_loss.item() / batches
        }

        torch.save(checkpoint, 'checkpoint.pth')
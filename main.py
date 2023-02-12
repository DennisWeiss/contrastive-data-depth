import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np

from common import soft_tukey_depth, get_kl_divergence, evaluate_by_linear_probing, soft_tukey_depth_thru_origin
from dataset import NormalCIFAR10Dataset
from model import DataDepthTwinsModel
from transforms import Transform


BATCH_SIZE = 256
TUKEY_DEPTH_STEPS = 40
TEMP = 1
EPOCHS = 100
LEARNING_RATE = 1e-2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_description(epoch, sim_loss, td_loss, total_loss, avg_tukey_depth):
    return f'Epoch {epoch} - Loss: {total_loss:.4f} (Sim: {sim_loss:.4f}, TD: {td_loss:.4f}), Avg. Tukey Depth: {avg_tukey_depth:.4f}'


def nt_xent(x, t=0.5):
    x = nn.functional.normalize(x, dim=1)
    x_scores = (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return nn.functional.cross_entropy(x_scale, targets.long().to(x_scale.device))


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# train_data = NormalCIFAR10Dataset(normal_class=0, train=True, transform=Transform())
# train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=Transform())
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

model = DataDepthTwinsModel().to(device)
optimizer_model = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-6, nesterov=True)

# scheduler = LambdaLR(
#         optimizer_model,
#         lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
#             step,
#             EPOCHS * len(train_dataloader),
#             LEARNING_RATE,  # lr_lambda computes multiplicative factor
#             1e-3))

for epoch in range(1, EPOCHS+1):
    iterator = tqdm(train_dataloader, desc=get_description(epoch, 0, 0, 0, 0), unit='batch')

    summed_sim_loss = torch.tensor(0, device=device, dtype=torch.float)
    summed_td_loss = torch.tensor(0, device=device, dtype=torch.float)
    summed_total_loss = torch.tensor(0, device=device, dtype=torch.float)
    summed_avg_tukey_depth = torch.tensor(0, device=device, dtype=torch.float)

    batches = 0

    for (x1, x2), _ in iterator:
        x1, x2 = x1.to(device), x2.to(device)
        y1, y2 = model(x1), model(x2)
        y1_detached, y2_detached = y1.detach(), y2.detach()

        # x = x.to(device)
        # sizes = x.size()
        # x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4])
        # y = model(x)
        # y_detached = y.detach()

        optimizer_model.zero_grad()

        sim_loss = torch.square(y1 - y2).sum(dim=1).mean()

        z = nn.Parameter(torch.rand(y1.shape[0], y1.shape[1], device=device).multiply(2).subtract(1))
        optimizer_z = torch.optim.SGD([z], lr=1e+2)

        for j in range(TUKEY_DEPTH_STEPS):
            optimizer_z.zero_grad()
            tukey_depths = soft_tukey_depth_thru_origin(y1_detached, y1_detached, z, TEMP)
            tukey_depths.sum().backward()
            optimizer_z.step()

        tukey_depths = soft_tukey_depth_thru_origin(y1, y1, z.detach(), TEMP)

        if epoch % 5 == 0:
            plt.hist(tukey_depths.cpu().detach().numpy(), bins=30)
            plt.show()

        # print(tukey_depths.mean().item())
        td_loss = get_kl_divergence(tukey_depths, lambda x: 2, 0.05, 1e-5)
        # var_loss = 10 * -tukey_depths.var()

        # dist_loss = torch.square(y2 - y2.mean(dim=0)).sum(dim=1).mean()

        total_loss = sim_loss + td_loss
        # total_loss = nt_xent(y)
        total_loss.backward()
        optimizer_model.step()
        # scheduler.step()

        summed_sim_loss += sim_loss
        summed_td_loss += td_loss
        summed_total_loss += total_loss
        summed_avg_tukey_depth += tukey_depths.mean()

        batches += 1

        iterator.set_description(
            get_description(
                epoch,
                summed_sim_loss.item() / batches,
                summed_td_loss.item() / batches,
                summed_total_loss.item() / batches,
                summed_avg_tukey_depth.item() / batches,
            )
        )

    print(f'Linar probe acc.: {evaluate_by_linear_probing(test_dataloader, model, 128, device)}')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_model.state_dict(),
        'loss': summed_total_loss.item() / batches
    }

    torch.save(checkpoint, 'checkpoint.pth')
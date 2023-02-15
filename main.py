import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets
from pyod.models.knn import KNN
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import numpy as np
from pyod.models.kde import KDE
from sklearn.metrics import roc_auc_score

from common import soft_tukey_depth, get_kl_divergence, evaluate_by_linear_probing, soft_tukey_depth_thru_origin, norm_of_kde
from dataset import NormalCIFAR10Dataset, AnomalousCIFAR10Dataset
from model import DataDepthTwinsModel
from transforms import Transform


LOAD_FROM_CHECKPOINT = True

NORMAL_CLASS = 5
BATCH_SIZE = 256
TUKEY_DEPTH_STEPS = 40
TEMP = 1
EPOCHS = 200
LEARNING_RATE = 3e-4

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


def evaluate_tukey_depth_auroc(model, train_loader, test_normal_loader, test_anomalous_loader):
    y_test = np.zeros(0)
    anomaly_scores = np.zeros(0)

    for x_train in train_loader:
        x_train = x_train.to(device)
        x_train = model(x_train)
        x_train_detached = x_train.detach()

        for test_dataloader, type in [(test_normal_loader, 'normal'), (test_anomalous_loader, 'anomalous')]:
            for x_test in test_dataloader:
                x_test = x_test.to(device)
                x_test = model(x_test)
                x_test_detached = x_test.detach()

                z = nn.Parameter(2 * torch.rand(x_test.shape[0], x_test.shape[1], device=device) - 1)
                optimizer_z = torch.optim.SGD([z], lr=1e+2)

                for i in range(2 * TUKEY_DEPTH_STEPS):
                    optimizer_z.zero_grad()
                    tukey_depths = soft_tukey_depth(x_test_detached, x_train_detached, z, TEMP)
                    tukey_depths.sum().backward()
                    optimizer_z.step()

                with torch.no_grad():
                    tukey_depths = soft_tukey_depth(x_test_detached, x_train_detached, z, TEMP)

                anomaly_scores = np.concatenate((anomaly_scores, (0.5 - tukey_depths).detach().cpu().numpy()), axis=0)
                y_test = np.concatenate((y_test, np.zeros(x_test.shape[0]) if type == 'normal' else np.ones(x_test.shape[0])), axis=0)

    return roc_auc_score(y_test, anomaly_scores)


def evaluate_auroc_anomaly_detection(model, projection_size, train_loader, test_normal_loader, test_anomalous_loader):
    x_train = np.zeros((0, projection_size))
    for x in train_loader:
        x = x.to(device)
        x = model(x)
        x = x.detach().cpu().numpy()
        x_train = np.concatenate((x_train, x), axis=0)

    x_test = np.zeros((0, projection_size))
    y_test = np.zeros(0)

    for x in test_normal_loader:
        x = x.to(device)
        x = model(x)
        x = x.detach().cpu().numpy()
        x_test = np.concatenate((x_test, x), axis=0)
        y_test = np.concatenate((y_test, np.zeros(x.shape[0])), axis=0)

    for x in test_anomalous_loader:
        x = x.to(device)
        x = model(x)
        x = x.detach().cpu().numpy()
        x_test = np.concatenate((x_test, x), axis=0)
        y_test = np.concatenate((y_test, np.ones(x.shape[0])), axis=0)

    # clf = KDE(contamination=0.1, bandwidth=1, metric='l2')
    clf = KNN(n_neighbors=5)
    clf.fit(x_train)

    anomaly_scores = clf.decision_function(x_test)

    return roc_auc_score(y_test, anomaly_scores)


# CIFAR10 1 vs. rest Anomaly Detection

train_data = NormalCIFAR10Dataset(normal_class=NORMAL_CLASS, train=True, transform=Transform())
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

train_data_eval = torch.utils.data.Subset(
    NormalCIFAR10Dataset(normal_class=NORMAL_CLASS, train=True, transform=torchvision.transforms.ToTensor()),
    list(range(1000))
)
train_data_eval_dataloader = torch.utils.data.DataLoader(train_data_eval, batch_size=len(train_data_eval), shuffle=True, drop_last=True)

test_normal_data = NormalCIFAR10Dataset(
    normal_class=NORMAL_CLASS, train=False, transform=torchvision.transforms.ToTensor()
)
test_normal_dataloader = torch.utils.data.DataLoader(test_normal_data, batch_size=1, shuffle=False, drop_last=False)

test_anomalous_data = torch.utils.data.Subset(
    AnomalousCIFAR10Dataset(normal_class=NORMAL_CLASS, train=False, transform=torchvision.transforms.ToTensor()),
    list(range(len(test_normal_data)))
)
test_anomalous_dataloader = torch.utils.data.DataLoader(test_anomalous_data, batch_size=1, shuffle=False, drop_last=False)


train_data_eval_2 = NormalCIFAR10Dataset(normal_class=NORMAL_CLASS, train=True, transform=torchvision.transforms.ToTensor())
train_data_eval_dataloader_2 = torch.utils.data.DataLoader(train_data_eval_2, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

test_normal_data_2 = NormalCIFAR10Dataset(
    normal_class=NORMAL_CLASS, train=False, transform=torchvision.transforms.ToTensor()
)
test_normal_dataloader_2 = torch.utils.data.DataLoader(test_normal_data_2, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

test_anomalous_data_2 = torch.utils.data.Subset(
    AnomalousCIFAR10Dataset(normal_class=NORMAL_CLASS, train=False, transform=torchvision.transforms.ToTensor()),
    list(range(len(test_normal_data)))
)
test_anomalous_dataloader_2 = torch.utils.data.DataLoader(test_anomalous_data_2, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


# CIFAR10 Classification

# train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=Transform())
# train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#
# test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
# test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

model = DataDepthTwinsModel().to(device)
# optimizer_model = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)
# optimizer_model = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
optimizer_model = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

if LOAD_FROM_CHECKPOINT:
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_model.load_state_dict(checkpoint['optimizer_state_dict'])

# scheduler = LambdaLR(
#         optimizer_model,
#         lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
#             step,
#             EPOCHS * len(train_dataloader),
#             LEARNING_RATE,  # lr_lambda computes multiplicative factor
#             1e-3))

for epoch in range(checkpoint['epoch'] + 1 if LOAD_FROM_CHECKPOINT else 1, EPOCHS + 1):
    iterator = tqdm(train_dataloader, desc=get_description(epoch, 0, 0, 0, 0), unit='batch')

    summed_sim_loss = torch.tensor(0, device=device, dtype=torch.float)
    summed_td_loss = torch.tensor(0, device=device, dtype=torch.float)
    summed_total_loss = torch.tensor(0, device=device, dtype=torch.float)
    summed_avg_tukey_depth = torch.tensor(0, device=device, dtype=torch.float)

    batches = 0

    if epoch % 1 == 0:
        # print(f'AUROC: {evaluate_tukey_depth_auroc(model.backbone, train_data_eval_dataloader, test_normal_dataloader, test_anomalous_dataloader)}')
        print(f'KNN AUROC: {evaluate_auroc_anomaly_detection(model.backbone, 512, train_data_eval_dataloader_2, test_normal_dataloader_2, test_anomalous_dataloader_2)}')
    # print(f'Linar probe acc.: {evaluate_by_linear_probing(test_dataloader, model.backbone, 512, device)}')

    for (x1, x2) in iterator:
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
        # sim_loss = 10 * (1 - ((y1 * y2).sum(dim=1) / torch.sqrt((y1 ** 2).sum(dim=1) * (y2 ** 2).sum(dim=1)).clamp(min=1e-7)).mean())

        z = nn.Parameter(torch.rand(y1.shape[0], y1.shape[1], device=device).multiply(2).subtract(1))
        optimizer_z = torch.optim.SGD([z], lr=1e+2)

        for j in range(TUKEY_DEPTH_STEPS):
            optimizer_z.zero_grad()
            tukey_depths = soft_tukey_depth(y1_detached, y1_detached, z, TEMP)
            tukey_depths.sum().backward()
            optimizer_z.step()

        tukey_depths = soft_tukey_depth(y1, y1, z.detach(), TEMP)

        # if epoch % 5 == 0:
        #     plt.hist(tukey_depths.cpu().detach().numpy(), bins=30)
        #     plt.show()

        # print(tukey_depths.mean().item())
        # td_loss = get_kl_divergence(tukey_depths, lambda x: 2, 0.05, 1e-5)
        td_loss = 3 * norm_of_kde(tukey_depths.reshape(-1, 1), 0.1)

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

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_model.state_dict(),
        'loss': summed_total_loss.item() / batches
    }

    torch.save(checkpoint, 'checkpoint.pth')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from common import soft_tukey_depth
from dataset import NormalCIFAR10Dataset, AnomalousCIFAR10Dataset
from model import DataDepthTwinsModel


CLASS = 0
TEST_BATCH_SIZE = 1
TUKEY_DEPTH_STEPS = 40
TEMP = 0.1

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_true_positive_rate(anomalous_tukey_depths, threshold):
    true_positives = 0
    for anomalous_tukey_depth in anomalous_tukey_depths:
        if anomalous_tukey_depth < threshold:
            true_positives += 1
    return true_positives / len(anomalous_tukey_depths)


def get_false_positive_rate(nominal_tukey_depths, threshold):
    false_positive = 0
    for nominal_tukey_depth in nominal_tukey_depths:
        if nominal_tukey_depth < threshold:
            false_positive += 1
    return false_positive / len(nominal_tukey_depths)


def compute_auroc(true_positive_rates, false_positive_rates):
    auroc = 0
    for i in range(len(true_positive_rates)):
        if i == 0:
            auroc += (1 - 0.5 * false_positive_rates[i]) * true_positive_rates[i]
        else:
            auroc += (1 - 0.5 * false_positive_rates[i] - 0.5 * false_positive_rates[i-1]) * (true_positive_rates[i] - true_positive_rates[i-1])
    return auroc


train_data = NormalCIFAR10Dataset(normal_class=CLASS, train=True)
test_data_normal = NormalCIFAR10Dataset(normal_class=CLASS, train=False)
test_data_anomalous = torch.utils.data.Subset(AnomalousCIFAR10Dataset(normal_class=CLASS, train=False), list(range(len(test_data_normal))))

train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
test_dataloader_normal = DataLoader(test_data_normal, batch_size=TEST_BATCH_SIZE, shuffle=False)
test_dataloader_anomalous = DataLoader(test_data_anomalous, batch_size=TEST_BATCH_SIZE, shuffle=False)


model = DataDepthTwinsModel().to(device)

checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()


tukey_depth_results = {
    'normal': [],
    'anomalous': []
}

for x_train in train_dataloader:
    x_train = x_train.to(device)
    with torch.no_grad():
        y_train = model(x_train)
    y_train_detached = y_train.detach()
    y_train_mean = y_train_detached.mean(dim=0)

    for test_dataloader, test_type in [(test_dataloader_normal, 'normal'), (test_dataloader_anomalous, 'anomalous')]:
        for x_test in tqdm(test_dataloader, unit='sample', desc=test_type):
            x_test = x_test.to(device)
            with torch.no_grad():
                y_test = model(x_test)
            y_test_detached = y_test.detach()

            # z = nn.Parameter(torch.rand(y_test.shape[0], y_test.shape[1], device=device).multiply(2).subtract(1))
            # optimizer_z = torch.optim.SGD([z], lr=1e+2)
            #
            # for j in range(TUKEY_DEPTH_STEPS):
            #     optimizer_z.zero_grad()
            #     tukey_depths = soft_tukey_depth(y_test_detached, y_train_detached, z, TEMP)
            #     tukey_depths.sum().backward()
            #     optimizer_z.step()
            #
            # with torch.no_grad():
            #     tukey_depths = soft_tukey_depth(y_test_detached, y_train_detached, z, TEMP)
            #     tukey_depth_results[test_type] += tukey_depths.detach().cpu().tolist()

            tukey_depth_results[test_type] += (-torch.square(y_test_detached - y_train_mean).sum(dim=1).sqrt()).cpu().tolist()

print(np.asarray(tukey_depth_results['normal']).mean())
print(np.asarray(tukey_depth_results['anomalous']).mean())


true_positive_rates = []
false_positive_rates = []

for threshold in tqdm(np.arange(-50, 0, 1e-3)):
    true_positive_rates.append(get_true_positive_rate(tukey_depth_results['anomalous'], threshold))
    false_positive_rates.append(get_false_positive_rate(tukey_depth_results['normal'], threshold))


# fig0_nominal = plt.figure()
# plt.hist(tukey_depth_results['normal'], bins=50)
# plt.xlabel('soft Tukey depth')
# plt.ylabel('count')
# plt.title(f'Histogram of soft Tukey depths of test {CLASS} class w.r.t. train {CLASS} class')
# fig0_nominal.show()
#
# fig0_anomalous = plt.figure()
# plt.hist(tukey_depth_results['anomalous'], bins=50, color='orange')
# plt.xlabel('soft Tukey depth')
# plt.ylabel('count')
# plt.title(f'Histogram of soft Tukey depths of test non-{CLASS} classes w.r.t. train {CLASS} class')
# fig0_anomalous.show()


fig2 = plt.figure()
plt.plot(false_positive_rates, true_positive_rates)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
fig2.show()


auroc = compute_auroc(true_positive_rates, false_positive_rates)

print(f'AUROC: {auroc}')
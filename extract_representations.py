import torch
import torchvision
import numpy as np
from tqdm import tqdm

from model import DataDepthTwinsModel


NORMAL_CLASS = 5
REPRESENTATION_SIZE = 512

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = DataDepthTwinsModel().to(device)

checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

x_data = np.zeros((0, REPRESENTATION_SIZE))
y_data = np.zeros(0)

count_normal = 0
count_anomalous = 0

for x, y in tqdm(loader):
    x = x.to(device)
    if y[0] == NORMAL_CLASS:
        if count_normal >= 3000:
            continue
        count_normal += 1
    else:
        if count_anomalous >= 3000:
            continue
        count_anomalous += 1

    with torch.no_grad():
        z = model.backbone(x)
        x_data = np.append(x_data, z.cpu().numpy(), axis=0)
        y_data = np.append(y_data, np.asarray([int(y[0] != NORMAL_CLASS)]), axis=0)

print(np.linalg.norm(x_data[y_data == 0] - x_data.mean(axis=0), axis=1).mean())
print(np.linalg.norm(x_data[y_data == 1] - x_data.mean(axis=0), axis=1).mean())

np.savez_compressed(f'CV_by_CDD/CIFAR10_{NORMAL_CLASS}.npz', X=x_data, y=y_data)
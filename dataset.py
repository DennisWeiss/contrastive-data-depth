import torchvision
from torch.utils.data import Dataset


class NormalCIFAR10Dataset(Dataset):
    def __init__(self, normal_class, train=True, transform=torchvision.transforms.ToTensor()):
        self.normal_class = normal_class
        self.transform = transform

        data = torchvision.datasets.CIFAR10(root='./data', train=train, download=True)
        self.data = [x[0] for x in data if x[1] == normal_class]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

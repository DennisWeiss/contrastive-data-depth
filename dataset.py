import torchvision
from torch.utils.data import Dataset


default_transform = torchvision.transforms.Compose([
    # torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
])


class NormalCIFAR10Dataset(Dataset):
    def __init__(self, normal_class, train=True, transform=default_transform):
        self.normal_class = normal_class
        self.transform = transform

        data = torchvision.datasets.CIFAR10(root='./data', train=train, download=True)
        self.data = [x[0] for x in data if x[1] == normal_class]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])


class AnomalousCIFAR10Dataset(Dataset):
    def __init__(self, normal_class, train=True, transform=default_transform):
        self.normal_class = normal_class
        self.transform = transform

        data = torchvision.datasets.CIFAR10(root='./data', train=train, download=True)
        self.data = [x[0] for x in data if x[1] != normal_class]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])
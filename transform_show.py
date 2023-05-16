from transforms import Transform
import torchvision.datasets
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


index = 174

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=Transform())


# plt.imshow(dataset.__getitem__(index)[0][0])
# dataset.__getitem__(index)[0][0].save('orig.png', 'PNG')
# plt.show()

plt.imshow(dataset.__getitem__(index)[0][1])
# dataset.__getitem__(index)[0][1].save('transformed8.png', 'PNG')
plt.show()

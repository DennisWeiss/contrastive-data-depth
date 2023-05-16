import random

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageOps, ImageFilter


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def get_color_distortion(scale=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = torchvision.transforms.ColorJitter(0.8 * scale, 0.8 * scale, 0.8 * scale, 0.2 * scale)
    rnd_color_jitter = torchvision.transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = torchvision.transforms.RandomGrayscale(p=0.2)
    color_distort = torchvision.transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class Transform:
    def __init__(self):
        # self.transform = torchvision.transforms.Compose([
        #     # torchvision.transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
        #     torchvision.transforms.RandomHorizontalFlip(p=0.5),
        #     torchvision.transforms.RandomApply(
        #         [torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
        #                                 saturation=0.2, hue=0.1)],
        #         p=0.8
        #     ),
        #     torchvision.transforms.RandomGrayscale(p=0.2),
        #     GaussianBlur(p=1.0),
        #     Solarization(p=0.0),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        # self.transform_prime = torchvision.transforms.Compose([
        #     # torchvision.transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
        #     torchvision.transforms.RandomHorizontalFlip(p=0.5),
        #     torchvision.transforms.RandomApply(
        #         [torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
        #                                 saturation=0.2, hue=0.1)],
        #         p=0.8
        #     ),
        #     torchvision.transforms.RandomGrayscale(p=0.2),
        #     GaussianBlur(p=0.1),
        #     Solarization(p=0.2),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])

        self.transform = torchvision.transforms.Compose([
              torchvision.transforms.RandomResizedCrop(128),
              torchvision.transforms.RandomHorizontalFlip(p=0.5),
              get_color_distortion(scale=0.5),
              # torchvision.transforms.RandomRotation((0, 360)),
              torchvision.transforms.ToTensor()
          ])

    def __call__(self, x):
        y1 = self.transform(x)
        # y2 = self.transform_prime(x)
        y2 = self.transform(x)
        return y1, y2
        # return torch.stack([y1, y2])


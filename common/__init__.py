import torch
import numpy as np
from sklearn.linear_model import LogisticRegression


def soft_tukey_depth(X_, X, Z, temp):
    # X_ = X_ / X_.norm(dim=1).reshape(-1, 1)
    # X = X / X.norm(dim=1).reshape(-1, 1)
    X_new = X.repeat(X_.size(dim=0), 1, 1)
    X_new_tr = X_.repeat(X.size(dim=0), 1, 1).transpose(0, 1)
    X_diff = X_new - X_new_tr
    dot_products = X_diff.mul(Z.repeat(X.size(dim=0), 1, 1).transpose(0, 1)).sum(dim=2)
    dot_products_normalized = dot_products.transpose(0, 1).divide(temp * Z.norm(dim=1))
    return torch.sigmoid(dot_products_normalized).sum(dim=0).divide(X.size(dim=0))


def soft_tukey_depth_thru_origin(X_, X, Z, temp):
    X_ = X_ / X_.norm(dim=1).reshape(-1, 1)
    X = X / X.norm(dim=1).reshape(-1, 1)
    return soft_tukey_depth(X_, X, Z - ((X_ * Z).sum(dim=1)).reshape(-1, 1) * X_, temp)


def get_kl_divergence(soft_tukey_depths, f, kernel_bandwidth, epsilon=0.0):
    DELTA = 0.005
    kl_divergence = torch.tensor(0)
    for x in torch.arange(0, 0.5, DELTA):
        val = torch.exp(torch.square(soft_tukey_depths - x).divide(torch.tensor(-2 * kernel_bandwidth * kernel_bandwidth))).mean()
        f_val = f(x)
        kl_divergence = kl_divergence.subtract(torch.multiply(torch.tensor(f_val * DELTA), torch.log(val.divide(f_val + epsilon))))
    return kl_divergence


def evaluate_by_linear_probing(loader, model, projection_size, device):
    X = np.zeros((0, projection_size))
    y = np.zeros(0)
    for images, target in loader:
        images = images.to(device)
        with torch.no_grad():
            X = np.concatenate((X, model(images).cpu().numpy()), axis=0)
        y = np.concatenate((y, target.numpy()), axis=0)
    clf = LogisticRegression(max_iter=2000)
    # clf = KNeighborsClassifier(n_neighbors=20)

    # normalize data
    X = X - X.mean(axis=0)
    X = X / (X.std(axis=0) + 1e-7)

    clf.fit(X, y)
    return clf.score(X, y)


def norm_of_kde(X, kernel_bandwidth):
    X_new = X.repeat(X.size(dim=0), 1, 1)
    X_new_tr = X.repeat(X.size(dim=0), 1, 1).transpose(0, 1)
    X_diff = X_new - X_new_tr
    return torch.exp(-(X_diff ** 2).sum(dim=2) / (4 * kernel_bandwidth * kernel_bandwidth)).mean()

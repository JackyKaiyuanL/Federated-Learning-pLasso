"""Data generation / partitioning.

* ``generate_synthetic`` + ``build_clients`` – the FedProx synthetic(alpha,beta)
  benchmark from the original notebooks (linear model).
* ``load_mnist`` + ``dirichlet_partition`` + ``build_mnist_clients`` – MNIST with
  a Dirichlet(alpha) label-skew non-IID split (the standard NIID-Bench protocol),
  used with the CNN model.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .models import Client

DIMENSION = 60
NUM_CLASS = 10


# --------------------------------------------------------------------------- #
# Synthetic FedProx benchmark
# --------------------------------------------------------------------------- #
def _softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def generate_synthetic(alpha, beta, iid=False, num_users=30, seed=0):
    """FedProx synthetic(alpha, beta). Returns (X_split, y_split, n_per_user).

    Each user gets ``2*n`` points (first half train, second half test); the
    train/test split happens in :func:`build_clients`.
    """
    rng = np.random.default_rng(seed)
    samples_per_user = rng.lognormal(4, 2, num_users).astype(int) + 50
    X_split = [[] for _ in range(num_users)]
    y_split = [[] for _ in range(num_users)]

    diagonal = np.power(np.arange(1, DIMENSION + 1), -1.2)
    cov_x = np.diag(diagonal)

    if not iid:
        mean_W = rng.normal(0, alpha, num_users)
        B = rng.normal(0, beta, num_users)
        mean_x = np.vstack([rng.normal(B[i], 1, DIMENSION) for i in range(num_users)])
    else:
        w_const = rng.normal(0, alpha)
        mean_W = np.full(num_users, w_const)
        shared_mean_x = rng.normal(rng.normal(0, beta), 1, DIMENSION)
        mean_x = np.tile(shared_mean_x, (num_users, 1))
        W_shared = rng.normal(w_const, 1, (DIMENSION, NUM_CLASS))
        b_shared = rng.normal(w_const, 1, NUM_CLASS)

    for i in range(num_users):
        if iid:
            W, b = W_shared, b_shared
        else:
            W = rng.normal(mean_W[i], 1, (DIMENSION, NUM_CLASS))
            b = rng.normal(mean_W[i], 1, NUM_CLASS)
        n = samples_per_user[i] * 2
        xx = rng.multivariate_normal(mean_x[i], cov_x, n)
        yy = np.array([np.argmax(_softmax(xx[j] @ W + b)) for j in range(n)])
        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

    return X_split, y_split, samples_per_user


def build_clients(X_split, y_split, samples_per_user, num_users):
    """Synthetic split -> Client list + pooled global/test clients (linear model)."""
    client_list, tr_x, tr_y, te_x, te_y = [], [], [], [], []
    for i in range(num_users):
        x = np.asarray(X_split[i], dtype=np.float64)
        y = np.asarray(y_split[i], dtype=np.int64)
        x = np.c_[x, np.ones(len(y))]
        k = samples_per_user[i]
        xtr = torch.tensor(x[:k], dtype=torch.float32)
        ytr = torch.tensor(y[:k], dtype=torch.long)
        client_list.append(Client(xtr, ytr))
        tr_x.append(xtr); tr_y.append(ytr)
        te_x.append(torch.tensor(x[k:], dtype=torch.float32))
        te_y.append(torch.tensor(y[k:], dtype=torch.long))
    global_client = Client(torch.cat(tr_x), torch.cat(tr_y))
    test_client = Client(torch.cat(te_x), torch.cat(te_y))
    return client_list, global_client, test_client


# --------------------------------------------------------------------------- #
# MNIST + Dirichlet non-IID
# --------------------------------------------------------------------------- #
def load_mnist(root="./data"):
    """Download MNIST and return (x_train, y_train, x_test, y_test) as tensors.

    Images are normalised with the standard MNIST mean/std and kept on CPU as
    float32 tensors of shape (N, 1, 28, 28).
    """
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    te = datasets.MNIST(root, train=False, download=True, transform=tf)

    def stack(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))])
        ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
        return xs, ys

    x_train, y_train = stack(tr)
    x_test, y_test = stack(te)
    return x_train, y_train, x_test, y_test


def dirichlet_partition(labels, num_clients, alpha, seed=0, min_size=10):
    """Partition sample indices across clients by a Dirichlet(alpha) label skew.

    Lower ``alpha`` => more heterogeneous (each client sees fewer classes).
    Returns a list of index arrays, one per client. Retries until every client
    has at least ``min_size`` samples (standard NIID-Bench behaviour).
    """
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1
    rng = np.random.default_rng(seed)

    while True:
        idx_per_client = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx_c, cuts)):
                idx_per_client[k].extend(part.tolist())
        sizes = [len(ix) for ix in idx_per_client]
        if min(sizes) >= min_size:
            break
    return [np.array(sorted(ix)) for ix in idx_per_client]


def build_mnist_clients(num_clients=100, alpha=0.5, seed=0, root="./data"):
    """Build MNIST Client list + pooled global(train) / test clients (CNN model)."""
    x_train, y_train, x_test, y_test = load_mnist(root)
    parts = dirichlet_partition(y_train.numpy(), num_clients, alpha, seed)
    client_list = [Client(x_train[ix], y_train[ix]) for ix in parts]
    global_client = Client(x_train, y_train)        # pooled train (for loss curve)
    test_client = Client(x_test, y_test)
    return client_list, global_client, test_client


# --------------------------------------------------------------------------- #
# CIFAR-10 + Dirichlet non-IID (single-label, ResNet model)
# --------------------------------------------------------------------------- #
def load_cifar10(root="./data"):
    """Download CIFAR-10, return (x_train,y_train,x_test,y_test) as tensors.
    Images normalised with CIFAR-10 mean/std, shape (N,3,32,32) float32."""
    from torchvision import datasets, transforms
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean, std)])
    tr = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    te = datasets.CIFAR10(root, train=False, download=True, transform=tf)

    def stack(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))])
        ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
        return xs, ys
    x_train, y_train = stack(tr)
    x_test, y_test = stack(te)
    return x_train, y_train, x_test, y_test


def build_cifar_clients(num_clients=100, alpha=0.5, seed=0, root="./data"):
    """CIFAR-10 Client list + pooled global(train) / test clients (ResNet model)."""
    x_train, y_train, x_test, y_test = load_cifar10(root)
    parts = dirichlet_partition(y_train.numpy(), num_clients, alpha, seed)
    client_list = [Client(x_train[ix], y_train[ix]) for ix in parts]
    global_client = Client(x_train, y_train)
    test_client = Client(x_test, y_test)
    return client_list, global_client, test_client


def load_cifar100(root="./data"):
    """CIFAR-100 (100 classes) -> tensors, CIFAR-100 mean/std normalisation."""
    from torchvision import datasets, transforms
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean, std)])
    tr = datasets.CIFAR100(root, train=True, download=True, transform=tf)
    te = datasets.CIFAR100(root, train=False, download=True, transform=tf)

    def stack(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))])
        ys = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
        return xs, ys
    return (*stack(tr), *stack(te))


def build_cifar100_clients(num_clients=100, alpha=0.5, seed=0, root="./data"):
    """CIFAR-100 Client list + pooled global / test clients (ResNet, 100 classes)."""
    x_train, y_train, x_test, y_test = load_cifar100(root)
    parts = dirichlet_partition(y_train.numpy(), num_clients, alpha, seed)
    client_list = [Client(x_train[ix], y_train[ix]) for ix in parts]
    return client_list, Client(x_train, y_train), Client(x_test, y_test)


# --------------------------------------------------------------------------- #
# Pascal VOC 2007 multi-label
# --------------------------------------------------------------------------- #
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def load_voc(root="./data", image_set="trainval", size=128):
    """VOC2007 -> (x, y) with x (N,3,size,size) float, y (N,20) multi-label float."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    ds = datasets.VOCDetection(root, year="2007", image_set=image_set,
                               download=True, transform=tf)
    cls_idx = {c: i for i, c in enumerate(VOC_CLASSES)}
    xs, ys = [], []
    for img, target in ds:
        objs = target["annotation"]["object"]
        objs = objs if isinstance(objs, list) else [objs]
        y = torch.zeros(len(VOC_CLASSES))
        for o in objs:
            y[cls_idx[o["name"]]] = 1.0
        xs.append(img); ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def load_coco(root="./data", size=128, split="val2017"):
    """MS-COCO multi-label (80 classes) from the val2017 subset (~5k images).

    Parses ``instances_<split>.json`` directly (no pycocotools dependency):
    each image -> 80-dim multi-label vector of present categories.
    """
    import json as _json
    from PIL import Image
    from torchvision import transforms
    base = os.path.join(root, "coco")
    ann = _json.load(open(os.path.join(base, "annotations", f"instances_{split}.json")))
    cat_ids = sorted(c["id"] for c in ann["categories"])      # 80 ids (with gaps)
    cat_idx = {cid: i for i, cid in enumerate(cat_ids)}
    fname = {im["id"]: im["file_name"] for im in ann["images"]}
    labels = {}
    for a in ann["annotations"]:
        labels.setdefault(a["image_id"], set()).add(cat_idx[a["category_id"]])

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_dir = os.path.join(base, split)
    ids = sorted(labels)                                       # only annotated images
    xs, ys = [], []
    for iid in ids:
        img = Image.open(os.path.join(img_dir, fname[iid])).convert("RGB")
        xs.append(tf(img))
        y = torch.zeros(len(cat_ids));
        for c in labels[iid]:
            y[c] = 1.0
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def build_coco_clients(num_clients=40, alpha=0.5, seed=0, root="./data", size=128):
    """COCO multi-label Client list + global / test (resnet_ml, 80 classes).
    80/20 train/test split; Dirichlet partition over each image's rarest label."""
    x, y = load_coco(root, size)
    rng = np.random.default_rng(123)                           # fixed train/test split
    perm = rng.permutation(len(y))
    n_te = len(y) // 5
    te, tr = perm[:n_te], perm[n_te:]
    x_train, y_train, x_test, y_test = x[tr], y[tr], x[te], y[te]
    gfreq = y_train.sum(0)
    primary = [int(p[gfreq[p].argmin()]) for p in
               (torch.nonzero(row).flatten() for row in y_train)]
    parts = dirichlet_partition(np.array(primary), num_clients, alpha, seed, min_size=8)
    client_list = [Client(x_train[ix], y_train[ix]) for ix in parts]
    return client_list, Client(x_train, y_train), Client(x_test, y_test)


def build_voc_clients(num_clients=40, alpha=0.5, seed=0, root="./data", size=128):
    """VOC multi-label Client list + pooled global / test clients (resnet_ml).

    Non-IID partition: each image is assigned a 'primary' label = its rarest
    present class (spreads rare classes), then a Dirichlet(alpha) split over
    those primary labels induces label skew across clients.
    """
    x_train, y_train = load_voc(root, "trainval", size)
    x_test, y_test = load_voc(root, "test", size)
    # primary label = rarest present class (by global frequency)
    global_freq = y_train.sum(0)
    primary = []
    for y in y_train:
        present = torch.nonzero(y).flatten()
        primary.append(int(present[global_freq[present].argmin()]))
    parts = dirichlet_partition(np.array(primary), num_clients, alpha, seed, min_size=8)
    client_list = [Client(x_train[ix], y_train[ix]) for ix in parts]
    global_client = Client(x_train, y_train)
    test_client = Client(x_test, y_test)
    return client_list, global_client, test_client

import os

import numpy as np

DATASETS = {
    "Caltech-3V": "caltech3v",
    "Caltech-4V": "caltech4v",
    "Caltech-5V": "caltech5v",
    "Caltech-6V": "caltech6v",
    "YouTube": "youtube",
    "MS-COCO": "coco",
    "VOC2007": "voc2007",
}


def _load_mat(path):
    try:
        import scipy.io as sio
        m = sio.loadmat(path)
        get = lambda k: np.asarray(m[k])
        keys = set(m.keys())
    except NotImplementedError:
        import h5py
        f = h5py.File(path, "r")
        get = lambda k: np.asarray(f[k]).T
        keys = set(f.keys())
    views = []
    v = 1
    while f"X{v}" in keys:
        views.append(get(f"X{v}").astype(np.float32))
        v += 1
    y = get("Y").astype(np.int64).ravel()
    return views, y


def load_dataset(name, root="data", norm="minmax"):
    d = DATASETS.get(name, name)
    views, y = _load_mat(os.path.join(root, f"{d}.mat"))
    _, y = np.unique(y, return_inverse=True)
    if norm == "minmax":
        views = [(x - x.min(0)) / np.maximum(x.max(0) - x.min(0), 1e-12) for x in views]
    elif norm == "zscore":
        views = [(x - x.mean(0)) / np.maximum(x.std(0), 1e-12) for x in views]
    elif norm != "none":
        raise ValueError(norm)
    views = [np.ascontiguousarray(x, dtype=np.float32) for x in views]
    return views, y.astype(np.int64)

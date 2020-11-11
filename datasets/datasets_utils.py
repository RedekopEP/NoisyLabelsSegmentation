import numpy as np


def load_image(path: str, bbox_percent: list[float]) -> np.array:
    img = np.load(str(path))
    x1 = img.shape[0] * bbox_percent[0] // 100
    x2 = img.shape[0] * bbox_percent[1] // 100
    if (x2 - x1) % 2 != 0:
        x2 = x2 + 1
    y1 = img.shape[1] * bbox_percent[2] // 100
    y2 = img.shape[1] * bbox_percent[3] // 100
    if (y2 - y1) % 2 != 0:
        y2 = y2 + 1
    img = img[x1: x2, y1: y2]
    return img


def load_mask(path: str, flag_binary: bool, bbox_percent: list[float]) -> np.array:
    mask = np.load(str(path).replace('images', 'masks'))
    if flag_binary:
        mask[mask > 0] = 1
    x1 = mask.shape[0] * bbox_percent[0] // 100
    x2 = mask.shape[0] * bbox_percent[1] // 100
    if (x2 - x1) % 2 != 0:
        x2 = x2 + 1
    y1 = mask.shape[1] * bbox_percent[2] // 100
    y2 = mask.shape[1] * bbox_percent[3] // 100

    if (y2 - y1) % 2 != 0:
        y2 = y2 + 1
    mask = mask[x1: x2, y1: y2]
    return mask.astype(np.uint8)

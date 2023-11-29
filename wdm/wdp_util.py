import numpy as np
from PIL import Image
import blobfile as bf
from wdm import dist_util, logger
import torch
SELECTED = 0
RANDOM = 1


def generate_wdp_trigger(
    wdp_key, wdp_trigger_type, wdp_trigger_shape, wdp_trigger_kwargs=None
):
    assert wdp_trigger_shape[2] == wdp_trigger_shape[3]
    resolution = wdp_trigger_shape[2]

    if wdp_trigger_type == SELECTED:
        wdp_trigger_path = wdp_trigger_kwargs["wdp_trigger_path"]
        assert wdp_trigger_path is not None

    elif wdp_trigger_type == RANDOM:
        wdp_trigger_path = wdp_trigger_kwargs["wdp_trigger_path"]
        assert wdp_trigger_path is not None
    else:
        logger.error("Unknown trigger type")
        
    with bf.BlobFile(wdp_trigger_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()

    while min(*pil_image.size) >= 2 * resolution:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = resolution / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - resolution) // 2
    crop_x = (arr.shape[1] - resolution) // 2
    arr = arr[crop_y: crop_y + resolution, crop_x: crop_x + resolution]
    arr = arr.astype(np.float32) / 127.5 - 1

    wdp_trigger = np.transpose(arr, [2, 0, 1])

    return wdp_trigger

def weight_perturb(model, std=1e-3):
    for name, param in model.named_parameters():
            if 'bias' in name or 'bn' in name:
                continue
            noise = torch.normal(0, std, size=param.size())
            param.data.add_(noise)
    return True
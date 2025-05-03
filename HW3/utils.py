import numpy as np
import skimage.io as sio
from pycocotools import mask as mask_utils
import torchvision.transforms.functional as TF
import torch
import random

def decode_maskobj(mask_obj):
    if isinstance(mask_obj["counts"], str):
        mask_obj = mask_obj.copy()
        mask_obj["counts"] = mask_obj["counts"].encode("utf-8")
    return mask_utils.decode(mask_obj)


def encode_mask(binary_mask):
    
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle["counts"] = rle["counts"].decode("utf-8")  # 為了 JSON 儲存
    return rle


def read_maskfile(filepath):
    
    mask_array = sio.imread(filepath)
    return mask_array


def encode_all_masks(masks, scores=None, labels=None, image_id=None):
   
    results = []
    N = masks.shape[0]

    for i in range(N):
        rle = encode_mask(masks[i])
        instance = {
            "image_id": image_id,
            "category_id": int(labels[i]) if labels is not None else 1,
            "segmentation": rle,
            "score": float(scores[i]) if scores is not None else 1.0,
        }
        results.append(instance)

    return results

class ResizeTransform:
    def __init__(self, size=(384, 384)):
        self.size = size  # (H, W)

    def __call__(self, image, target):
        orig_h, orig_w = image.shape[1:]  # image: [C, H, W]
        new_h, new_w = self.size

        # Resize image
        image = TF.resize(image, self.size)

        # Resize masks
        if "masks" in target:
            masks = target["masks"]
            resized_masks = torch.stack([
                TF.resize(mask.unsqueeze(0), self.size).squeeze(0)
                for mask in masks
            ])
            target["masks"] = resized_masks

        # Scale boxes
        if "boxes" in target:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes = target["boxes"]
            boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y], device=boxes.device)
            target["boxes"] = boxes

        return image, target
# inference.py
import json
import numpy as np
from pycocotools import mask as coco_mask
from tqdm import tqdm

def encode_rle(mask):
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def run_inference(model, data_loader, device, output_file='test-results.json'):
    model.eval()
    results = []

    with torch.no_grad():
        for images, image_ids in tqdm(data_loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, out in enumerate(outputs):
                for j in range(len(out["scores"])):
                    score = out["scores"][j].item()
                    if score < 0.5:
                        continue
                    mask = out["masks"][j, 0].cpu().numpy() > 0.5
                    rle = encode_rle(mask)
                    result = {
                        "image_id": image_ids[i].replace(".tif", ""),
                        "category_id": int(out["labels"][j]),
                        "segmentation": rle,
                        "score": score
                    }
                    results.append(result)

    with open(output_file, 'w') as f:
        json.dump(results, f)
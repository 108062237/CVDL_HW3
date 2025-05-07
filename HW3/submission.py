import os
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from dataloader import get_test_loader
from model import build_model


def encode_mask(binary_mask):
    """Encode a binary mask to COCO RLE format."""
    arr = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def main():
    # Load configuration
    cfg = yaml.safe_load(open('configs/config.yaml', 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and load weights
    model = build_model(cfg)
    ckpt_path = os.path.join('checkpoints', 'best_model.pth')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Prepare test DataLoader
    data_dir = cfg['DATA']['ROOT_DIR']
    batch_size = int(cfg['TEST']['BATCH_SIZE'])
    num_workers = int(cfg['DATA']['NUM_WORKERS'])
    test_loader = get_test_loader(data_dir, batch_size, num_workers)

    # Load image name to COCO image_id mapping
    mapping_path = os.path.join(data_dir, 'test_image_name_to_ids.json')
    with open(mapping_path, 'r') as f:
        raw_mapping = json.load(f)
    imgname_to_id = {}
    # If dict, use directly
    if isinstance(raw_mapping, dict):
        imgname_to_id = raw_mapping
    # If list, try converting
    elif isinstance(raw_mapping, list):
        for item in raw_mapping:
            if isinstance(item, dict) and 'file_name' in item and 'id' in item:
                imgname_to_id[item['file_name']] = item['id']
            elif isinstance(item, dict) and len(item) == 1:
                k, v = next(iter(item.items()))
                imgname_to_id[k] = v
    # Else leave empty

    results = []
    # Inference
    for images, img_names in tqdm(test_loader, desc='Testing'):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)
        for name, output in zip(img_names, outputs):
            # Determine image_id by trying candidate keys
            candidates = [
                name,
                f"{name}.tif",
                os.path.join('test', f"{name}.tif"),
                os.path.join(data_dir, 'test', f"{name}.tif"),
                os.path.join(data_dir, f"{name}.tif")
            ]
            image_id = None
            for key in candidates:
                if key in imgname_to_id:
                    image_id = imgname_to_id[key]
                    break
            if image_id is None:
                raise KeyError(f"Cannot find image_id for '{name}' in mapping.")

            # Extract model outputs
            boxes = output.get('boxes', torch.empty((0, 4)))
            labels = output.get('labels', torch.empty((0,), dtype=torch.int64))
            scores = output.get('scores', torch.empty((0,)))
            masks = output.get('masks', torch.empty((0, 1, 1, 1)))

            boxes = boxes.cpu().numpy()
            labels = labels.cpu().numpy()
            scores = scores.cpu().numpy()
            masks = masks.cpu().numpy()

            # Process each detected instance
            for i in range(boxes.shape[0]):
                xmin, ymin, xmax, ymax = boxes[i]
                w = float(xmax - xmin)
                h = float(ymax - ymin)
                bbox = [float(xmin), float(ymin), w, h]
                score = float(scores[i])
                category_id = int(labels[i])

                # Encode mask
                binary_mask = masks[i, 0] > 0.5
                rle = encode_mask(binary_mask)

                results.append({
                    'image_id': int(image_id),
                    'bbox': bbox,
                    'score': score,
                    'category_id': category_id,
                    'segmentation': rle
                })

    # Save to JSON
    out_path = 'test-results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved {len(results)} instances to {out_path}")


if __name__ == '__main__':
    main()

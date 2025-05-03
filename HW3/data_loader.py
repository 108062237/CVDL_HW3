# dataloader.py
import os
import cv2
import imageio.v3 as sio
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MedicalDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.image_ids = sorted(os.listdir(os.path.join(root, 'train' if train else 'test')))

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.root, 'train' if self.train else 'test', image_id, 'image.tif')
        image = cv2.imread(str(image_path))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        if self.train:
            masks = []
            labels = []
            for i in range(1, 5):
                mask_path = os.path.join(self.root, 'train', image_id, f'class{i}.tif')
                if not os.path.exists(mask_path):
                    continue
                try:
                    mask = sio.imread(mask_path)
                except Exception as e:
                    print(f"[Warning] Skip invalid mask {mask_path}: {e}")
                    continue
                instances = np.unique(mask)[1:]
                for inst_id in instances:
                    binary_mask = (mask == inst_id).astype(np.uint8)
                    masks.append(torch.tensor(binary_mask))
                    labels.append(i)
            target = {
                'boxes': self._get_boxes(masks),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'masks': torch.stack(masks) if masks else torch.zeros((0, *image.shape[1:])),
                'image_id': torch.tensor([idx])
            }
            return image, target
        else:
            return image, image_id

    def __len__(self):
        return len(self.image_ids)

    def _get_boxes(self, masks):
        boxes = []
        for mask in masks:
            pos = torch.nonzero(mask)
            xmin = pos[:, 1].min().item()
            xmax = pos[:, 1].max().item()
            ymin = pos[:, 0].min().item()
            ymax = pos[:, 0].max().item()
            boxes.append([xmin, ymin, xmax, ymax])
        return torch.tensor(boxes, dtype=torch.float32)


def get_dataloaders(data_root):
    train_dataset = MedicalDataset(data_root, train=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    test_dataset = MedicalDataset(data_root, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    data_root = "data"
    train_loader, test_loader = get_dataloaders(data_root)
    for images, targets in train_loader:
        print(f"Batch size: {len(images)}")
        print(f"Image shape: {images[0].shape}")
        print(f"Target boxes: {targets[0]['boxes'].shape}")
        print(f"Target labels: {targets[0]['labels'].shape}")
        break
import os
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import skimage.io as sio


def load_image(path):
    """Load an image (.tif) and convert to RGB PIL image."""
    img = Image.open(path)
    return img.convert("RGB")


def get_train_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
    ])


def get_val_transforms():
    return T.Compose([
        T.ToTensor(),
    ])


class CellDataset(Dataset):
    """
    Dataset for cell instance segmentation.
    - train/val: data/train/[id]/image.tif + class{1..4}.tif
    - test:    data/test/[id].tif
    """
    def __init__(self, root_dir, transforms=None, is_test=False):
        self.root = root_dir
        self.transforms = transforms
        self.is_test = is_test
        if is_test:
            # test: all tif files in root_dir
            self.ids = [os.path.splitext(f)[0] for f in os.listdir(root_dir) if f.endswith('.tif')]
        else:
            # train/val: each subfolder is an image id
            self.ids = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        if self.is_test:
            img = load_image(os.path.join(self.root, f"{img_id}.tif"))
            if self.transforms:
                img = self.transforms(img)
            return img, img_id

        # train/val mode
        folder = os.path.join(self.root, img_id)
        image = load_image(os.path.join(folder, 'image.tif'))
        masks = []
        labels = []
        # extract binary masks per instance and labelS
        for cls in range(1, 5):
            mask_path = os.path.join(folder, f'class{cls}.tif')
            if not os.path.exists(mask_path):
                continue
            mask_arr = sio.imread(mask_path)
            for inst_id in np.unique(mask_arr):
                if inst_id == 0:
                    continue
                binary_mask = (mask_arr == inst_id)
                masks.append(torch.tensor(binary_mask, dtype=torch.uint8))
                labels.append(cls)

        # stack masks and labels
        if len(masks) > 0:
            masks = torch.stack(masks)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # no instances
            H, W = image.height, image.width
            masks = torch.zeros((0, H, W), dtype=torch.uint8)
            labels = torch.tensor([], dtype=torch.int64)

        # compute bboxes [xmin, ymin, xmax, ymax]
        boxes = []
        for m in masks:
            pos = torch.nonzero(m)
            ymin = torch.min(pos[:, 0]).item()
            xmin = torch.min(pos[:, 1]).item()
            ymax = torch.max(pos[:, 0]).item()
            xmax = torch.max(pos[:, 1]).item()
            boxes.append([xmin, ymin, xmax, ymax])
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx])
        }

        if target['boxes'].shape[0] == 0:
            return None

        # apply transforms (supporting torchvision-style or custom)
        if self.transforms:
            try:
                image, target = self.transforms(image, target)
            except Exception:
                image = self.transforms(image)
        return image, target


def collate_fn(batch):
    """Collate function for DataLoader."""
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))


def get_train_val_loaders(data_dir, batch_size, val_split=0.2, num_workers=4):
    train_root = os.path.join(data_dir, 'train')
    dataset = CellDataset(train_root, transforms=get_train_transforms(), is_test=False)
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    return train_loader, val_loader


def get_test_loader(data_dir, batch_size, num_workers=4):
    test_root = os.path.join(data_dir, 'test')
    dataset = CellDataset(test_root, transforms=get_val_transforms(), is_test=True)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    train_loader, val_loader = get_train_val_loaders(
        args.data_dir, args.batch_size, args.val_split, args.num_workers
    )
    print('Train batches:', len(train_loader), 'Val batches:', len(val_loader))
    for imgs, targets in train_loader:
        print('Train batch images:', len(imgs), 'Targets keys:', targets[0].keys())
        print('Train image shape:', imgs[0].shape)
        print('Train target boxes:', targets[0]['boxes'].shape)
        print('Train target masks:', targets[0]['masks'].shape)
        print('Train target labels:', targets[0]['labels'].shape)
        print('Train target image_id:', targets[0]['image_id'].shape)
        break

    test_loader = get_test_loader(
        args.data_dir, args.batch_size, args.num_workers
    )
    print('Test batches:', len(test_loader))
    for imgs, ids in test_loader:
        print('Test image shape:', imgs[0].shape, 'Sample IDs:', ids[:3])
        break

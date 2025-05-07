import os
import yaml
import torch
import argparse
from tqdm import tqdm
from dataloader import get_train_val_loaders
from model import build_model


def train_one_epoch(model, optimizer, loader, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc='Training', leave=False)
    for images, targets in loop:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        running_loss += loss_value

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loop.set_postfix(loss=f"{loss_value:.4f}")
    avg_loss = running_loss / len(loader)
    return avg_loss


def evaluate(model, loader, device):
    # Keep model in train mode to get loss values
    model.train()
    running_loss = 0.0
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation', leave=False)
        for images, targets in loop:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            running_loss += loss_value
            loop.set_postfix(val_loss=f"{loss_value:.4f}")
    avg_loss = running_loss / len(loader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--optimizer', type=str, default='',choices=['sgd', 'adam', 'adamw'], help='Override optimizer from config')
    args = parser.parse_args()

    # Load configuration
    cfg = yaml.safe_load(open(args.config, 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loaders
    train_loader, val_loader = get_train_val_loaders(
        cfg['DATA']['ROOT_DIR'],
        int(cfg['TRAIN']['BATCH_SIZE']),
        float(cfg['DATA']['VAL_SPLIT']),
        int(cfg['DATA']['NUM_WORKERS'])
    )

    # Model
    model = build_model(cfg)
    model.to(device)

    # Optimizer selection
    opt_cfg = cfg.get('OPTIMIZER', {})
    opt_name = args.optimizer if args.optimizer else opt_cfg.get('NAME', 'sgd')
    lr = float(cfg['TRAIN']['LR'])
    weight_decay = float(cfg['TRAIN']['WEIGHT_DECAY'])
    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name.lower() == 'sgd':
        momentum = float(opt_cfg.get('MOMENTUM', 0.9))
        nesterov = bool(opt_cfg.get('NESTEROV', False))
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    elif opt_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
    elif opt_name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer '{opt_name}'")

    print(f"Using optimizer: {opt_name}")

    # Prepare checkpoint directory
    ckpt_dir = 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float('inf')
    total_epochs = int(cfg['TRAIN']['EPOCHS'])
    for epoch in range(total_epochs):
        print(f"Epoch {epoch+1}/{total_epochs}")
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"  [Train] Avg Loss: {train_loss:.4f}")

        val_loss = evaluate(model, val_loader, device)
        print(f"  [Val]   Avg Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best model: {ckpt_path}")

    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()

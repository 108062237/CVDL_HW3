from tqdm import tqdm

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    for images, targets in tqdm(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
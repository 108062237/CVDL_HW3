# main.py
import torch
from model import get_model
from data_loader import get_dataloaders
from train import train_one_epoch
from inference import run_inference


def main():
    data_root = './data'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load data
    train_loader, test_loader = get_dataloaders(data_root)

    # Load model
    model = get_model()
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_one_epoch(model, train_loader, optimizer, device)

    # Inference and save result
    run_inference(model, test_loader, device, output_file='test-results.json')


if __name__ == '__main__':
    main()

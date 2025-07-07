import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from ResNet import ResNet, ResBlock


CONFIG = {
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "batch_size": 512,
    "num_epochs": 500,
    "learning_rate": 0.1,
    "weight_decay": 1e-2,
    "model_save_path": "best_resnet_model2.pth"
}

print(f"Using device: {CONFIG['device']}")

def train_model(model, num_epochs, criterion, optimizer, scheduler, device, save_path):
    model.to(device)
    best_val_loss = float('inf') 
    
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        
        model.train()
        train_losses = []
        
        loop = tqdm(train_loader, desc="Training") 
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)
            train_losses.append(loss.item())
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
        
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        model.eval()
        val_losses = []
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Validating"): 
                data, targets = data.to(device), targets.to(device)
                scores = model(data)
                loss = criterion(scores, targets)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
    
        print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

def check_accuracy(data_loader, model, device):
    print(f"Checking accuracy on {'train' if data_loader.dataset.train else 'test'} dataset...")
    num_correct = 0 
    num_samples = 0 
    model.to(device)
    model.eval()

    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            scores = model(data)
            _, predicted = scores.max(1)
            num_correct += (predicted == targets).sum()
            num_samples += targets.size(0)
            
        acc = float(num_correct) / float(num_samples) * 100
        print(f"Accuracy: {num_correct}/{num_samples} ({acc:.2f}%)")

    model.train() 

batch_size = 256

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),     
    transforms.RandomHorizontalFlip(),       
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=test_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


model = ResNet(9)
    
    
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.SGD(
    model.parameters(), 
    lr=CONFIG['learning_rate'], 
    momentum=0.9,
    weight_decay=CONFIG['weight_decay']
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=0.01, 
    steps_per_epoch=len(train_loader), 
    epochs=CONFIG['num_epochs']
)

train_model(
    model=model, 
    num_epochs=CONFIG['num_epochs'], 
    criterion=criterion, 
    optimizer=optimizer, 
    scheduler=scheduler,
    device=CONFIG['device'],
    save_path=CONFIG['model_save_path']
)

print("\n--- Final Performance of Best Model ---")
model.load_state_dict(torch.load(CONFIG['model_save_path']))
check_accuracy(train_loader, model, CONFIG['device'])
check_accuracy(test_loader, model, CONFIG['device'])

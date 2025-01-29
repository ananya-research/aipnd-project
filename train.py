# train.py
import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import json

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define enhanced transforms for better generalization
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    # Define dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, validloader, testloader, train_data

def build_model(arch='vgg16', hidden_units=1024):
    if arch == 'vgg16':
        from torchvision.models import VGG16_Weights
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Unsupported architecture")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 102),
        nn.LogSoftmax(dim=1)
    )

    return model

def train_model(model, trainloader, validloader, criterion, optimizer, epochs=30, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        model.eval()
        valid_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")

def test_model(model, testloader, criterion, device='cuda'):
    model.eval()
    model.to(device)
    
    test_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    print(f"Test Loss: {test_loss/len(testloader):.3f}.. "
          f"Test Accuracy: {accuracy/len(testloader):.3f}")

def save_checkpoint(model, train_data, path='checkpoint.pth'):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
    }
    torch.save(checkpoint, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    trainloader, validloader, testloader, train_data = load_data(args.data_dir)
    model = build_model(arch=args.arch, hidden_units=args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    train_model(model, trainloader, validloader, criterion, optimizer, epochs=args.epochs, device=device)

    print("Testing the model on test dataset...")
    test_model(model, testloader, criterion, device=device)

    print("Saving model checkpoint...")
    save_checkpoint(model, train_data, path=f"{args.save_dir}/checkpoint.pth")

if __name__ == "__main__":
    main()

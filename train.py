import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from modules.model import ImageClassifier
from modules.dataset import ImageDataset
from torch.utils.data import DataLoader, random_split
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Script arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs="?", default='MNIST', help='dataset to use')
    parser.add_argument('--images', type=str, nargs="?", default=None, help='path to images')
    parser.add_argument('--labels', type=str, nargs="?", default=None, help='path to labels')
    parser.add_argument('--batch_size', type=int, nargs="?", default=64, help='batch size for training')
    parser.add_argument('--n_epochs', type=int, nargs="?", default=5, help='number of epochs to train over')
   
    args = parser.parse_args()
    return args

def train_func(model, optimizer, criterion, dataloader):
    model.train()
    epoch_loss = 0.0
    for image, label in tqdm(dataloader):
        if torch.cuda.is_available():
            image, label = image.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(dataloader)
    return model, epoch_loss

def test_func(model, criterion, dataloader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for image, label in tqdm(dataloader):
            image, label = image.to(DEVICE), label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).sum()
    test_loss /= len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return test_loss, accuracy

def get_mnist_dataloaders(transform, batch_size, download=False, shuffle=True):
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/data/', train=True, 
                                download=download, 
                                transform=transform),
        batch_size=batch_size, shuffle=shuffle, drop_last=True)#, collate_fn=generate_batch)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/data/', train=False, 
                                download=download,
                                transform=transform),
        batch_size=batch_size, shuffle=shuffle, drop_last=True)#, collate_fn=generate_batch)
    return trainloader, testloader

def get_custom_dataloaders(dataset, split, batch_size):
    train_len = int(split * len(dataset))
    lens = [train_len, len(dataset)-train_len]
    trainset, testset = random_split(dataset, lens)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader

def main(args):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0], [1])
    ])

    if args.dataset == 'MNIST':
        trainloader, testloader = get_mnist_dataloaders(transform, batch_size=args.batch_size)
    elif args.dataset == 'custom':
        dataset = ImageDataset(args.images, args.labels, transform)
        trainloader, testloader = get_custom_dataloaders(dataset, 0.8, args.batch_size)
    else:
        pass

    model = ImageClassifier(10, 64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    for epoch in range(args.n_epochs):
        model, train_loss = train_func(model, optimizer, criterion, trainloader)
        valid_loss, accuracy = test_func(model, criterion, testloader)
        scheduler.step()
        print(f'Epoch: {epoch}, train: {train_loss}, valid: {valid_loss}, accuracy: {accuracy}')
        model.save_checkpoint(optimizer, epoch, train_loss, 'checkpoints/checkpoint.pt')
        
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
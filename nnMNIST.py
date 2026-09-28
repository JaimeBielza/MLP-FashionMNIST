import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time

class NeuralNetwork(nn.Module):
    def __init__(self, n_input, num_clases=10, hidden_size=512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            #1st later
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            #2nd later
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),

            #3rd layer -> outputlayer
            nn.Linear(hidden_size, num_clases)

        )
    def forward(self, x):
        flatten_input = self.flatten(x)
        logits = self.layers(flatten_input)
        return logits



def config_device():
    device = (
        "mps" if torch.mps.is_available() else "cpu"
    )
    print(f'Using device: {device}')
    return device

def get_data():
    train_ds = datasets.FashionMNIST(
        root="data", 
        train=True,
        download=True,
        transform = ToTensor()
    )
    test_ds = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor()
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset = test_ds,
        batch_size = 64
    )

    return train_loader, test_loader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 200 == 0:
            current = (batch + 1)*len(X)
            print(f" Loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test -> Accuracy: {100 * accuracy:>0.1f}%, Loss medio: {test_loss:>8f}\n")
    return accuracy


if __name__ == '__main__':
    #1 Config device (we use MacBook Pro M5)
    device = config_device()
    classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]   

    #Get data
    [train_loader, test_loader] = get_data()
    model = NeuralNetwork(28*28).to(device)
    print(f"Neural Network config: \n{model}")

    #Config perdida y Config optimizador
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #Entrenamos
    start = time.time()
    epochs = 50
    for t in range(epochs):
        print(f"Epoca {t+1}\n---------------------- ")
        train(train_loader, model, loss, optimizer)
        #Testeamos
        test(test_loader, model, loss)

    print(f"Entrenamoento terminado en: {(time.time()-start)/60:0.2f} minutos")
    

    #Save model
    torch.save(model.state_dict(), "fashion_mnist_model.pth")

    #Example
    test_ds = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor()
    )
    x, y = test_ds[0]
    with torch.no_grad():
        x = x.unsqueeze(0).to(device)
        pred = model(x)
        predicted, actual = classes[pred.argmax(1).item()], classes[y]
        print(f"Prediccion: {predicted}, Real: {actual}")






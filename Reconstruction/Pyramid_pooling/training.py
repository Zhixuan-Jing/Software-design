import torch
from torch import nn
from torch.nn.modules.activation import Softmax
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import math

def train(model,optimizer,epoch,dataloader):
    size = len(dataloader.dataset)
    for i in range(epoch):
        for batch,(X,y) in enumerate(dataloader):
            X=X.cuda()
            y=y.cuda()
            pred=model.forward(X)
            loss=model.loss(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("Epoch %d/%d finished"%(i,epoch))

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X=X.cuda()
            y=y.cuda()            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
if __name__ == "__main__":
    device = torch.device("cuda:0")
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    network=DARNN(14,28,1).to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    train(network,optimizer,epochs,train_dataloader)
    test_loop(test_dataloader,network,network.loss)
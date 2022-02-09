import torch
from torch import nn
from torch.nn.modules.activation import Softmax
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
class pyramid(nn.Module):
    def __init__(self):
        super(pyramid,self).__init__()
        self.flatten = nn.Flatten()
        self.loss=nn.CrossEntropyLoss()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,4,5),
            # 20*20*4
            nn.MaxPool2d(2,2),
            #10*10*4
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(4,8,5),
            # 6*6*8
            nn.MaxPool2d(2,2),
            # 3*3*8
            nn.ReLU()
        )
        self.fcon=nn.Linear(4*4*8,10)
        # pooling4d = nn.MaxPool2d(4,4)
        # pooling2d = nn.MaxPool2d(2,2)


    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        # p4d = nn.MaxPool2d()
        print(x.shape)
        x=self.flatten(x)
        x=self.fcon(x)
        return x
    

def train(model,optimizer,epoch,dataloader):
    res = []
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
                res.append(loss)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return res
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

    network=pyramid().cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.1)
    result = train(network,optimizer,10,train_dataloader)
    test_loop(test_dataloader,network,network.loss)
    plt.plot(list(x for x in range(len(result))),result)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
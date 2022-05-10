import torch
from torch import nn
from torch.nn.modules.activation import Softmax
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from tqdm import trange
class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.flatten = nn.Flatten()
        self.loss=nn.CrossEntropyLoss()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,4,5),
            # 24*24*4
            nn.MaxPool2d(2,2),
            #12*12*4
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(4,8,5),
            # 8*8*8
            nn.MaxPool2d(2,2),
            # 4*4*8
            nn.ReLU()
        )
        self.fcon=nn.Sequential(
            nn.Linear(5*5*8,100),
            nn.Linear(100,50),
            nn.Linear(50,10)
        )

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.flatten(x)
        x=self.fcon(x)
        return x
    

def train(model,optimizer,epoch,dataloader):
    res = []
    size = len(dataloader.dataset)
    for i in trange(epoch):
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
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
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
    epochs = 100
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    network=cnn().cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    result = train(network,optimizer,epochs,train_dataloader)
    test_loop(test_dataloader,network,network.loss)
    plt.plot(list(x for x in range(len(result))),result)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
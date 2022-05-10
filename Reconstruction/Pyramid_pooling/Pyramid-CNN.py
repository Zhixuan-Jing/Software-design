import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import trange

class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()
        self.flatten = nn.Flatten(1,3)
        self.loss = nn.CrossEntropyLoss()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 4, 3),

            # nn.MaxPool2d(2, 2),

            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 1, 3),

            # nn.MaxPool2d(2, 2),

            nn.ReLU()
        )
        self.fcon = nn.Linear(21, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        dim = x.shape[-1]

        pad = (4-dim%4)%4
        # print(dim,pad)
        p4d = nn.MaxPool2d((dim+pad)//4,(dim+pad)//4,padding=pad)
        
        p2d = nn.MaxPool2d((dim+pad)//2,(dim+pad)//2,padding=pad)
        
        p1d = nn.MaxPool2d(dim,dim)

        x = torch.cat([self.flatten(p4d(x)),self.flatten(p2d(x)),self.flatten(p1d(x))],axis=1)

        x = self.fcon(x)
        return x


def train(model, optimizer, epoch, dataloader):
    res = []
    for i in trange(epoch):
        for batch, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()
            pred = model.forward(X)
            loss = model.loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                res.append(loss)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]",end="\r")
        pass
    return res


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.cuda()
            y = y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n",end="\r")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10
    training_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.CIFAR100(
        root="data",
        train=False,
        download=False,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    network = Pyramid().cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    result = train(network, optimizer, epochs, train_dataloader)
    test_loop(test_dataloader, network, network.loss)
    print(len(result))
    plt.plot(list(x for x in range(len(result))), result)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

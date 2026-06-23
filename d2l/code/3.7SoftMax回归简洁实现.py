import torch
from torch import nn
import torchvision
from torchvision import transforms
from d2l import torch as d2l

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=0),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=0))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)     


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)


net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(init_weights)

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
lr = 0.1
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
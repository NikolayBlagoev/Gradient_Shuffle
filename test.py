import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import numpy as np
net = resnet18().to("cuda")
net.train()
optimizer = SGD(net.parameters(),lr = 0.01,momentum=0,dampening=0,weight_decay=0,nesterov=False)
criterion = CrossEntropyLoss()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)
i = 0
arr = []
for images, labels in iter(trainloader):
    
    optimizer.zero_grad()
    images  = images.to("cuda")
    labels  = labels.to("cuda")
    ret = net(images)
    loss = criterion(ret,labels)
    loss.backward()
    optimizer.step()
    arr.append(loss.item())
    if i % 100 == 0:
        print(np.mean(np.array(arr)))
        arr = []
    i += 1

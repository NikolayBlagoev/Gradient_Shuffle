import random
import torch

layer = torch.nn.Linear(5,2,bias=False)
op = torch.optim.SGD(layer.parameters())

criterion = torch.nn.MSELoss()
inp = torch.tensor([4.0,2.0,1.0,3.0,5.0])
trgt1 = torch.tensor([1.0,1.0])
inp2 = torch.tensor([8.0,4.0,1.0,2.0,1.0])
trgt2 = torch.tensor([2.0,1.0])
op.zero_grad()
out = layer(inp)
loss = criterion(out,trgt1)
loss.backward()
for p in layer.parameters():
    print(p.grad)


op.zero_grad()
out = layer(inp2)
loss = criterion(out,trgt2)
loss.backward()
for p in layer.parameters():
    print(p.grad)


op.zero_grad()
out = layer(inp2)
loss = criterion(out,trgt2)
loss.backward()
out = layer(inp)
loss = criterion(out,trgt1)
loss.backward()
for p in layer.parameters():
    print(p.grad)
# print( next(layer.parameters()).grad)
exit()
us = 1
world_size = 8
k = 3
for itr in range(8):
    for i in range(world_size):
        if i ==  int(us):
                                continue
        pb = i
        group = pb - int(us) - itr
        if i > int(us):
            group -= 1
        grpoup = group % (world_size - 1)
        
        while group < 0:
            group = world_size - 1 + group 
        group = group // k
        group = group % 3
        print(f"TO {pb} goes group {group}")
    print("---")
        
                            
from dataclasses import dataclass
from multiprocessing import Lock, Process, Queue, current_process
import pickle
from torch import manual_seed
from typing import Callable
from contextlib import redirect_stdout
import time
import torch
from torchvision.models import resnet34
from torch import optim, stack, mean, split, cat, tensor
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import cuda
from torch.nn import CrossEntropyLoss, Linear
# Messages Exchanged by the processes
@dataclass
class Gradients:
    seq_id: bytes
    data: bytes
    split_start: str
    split_end: str
    node_id: int

@dataclass
class GetGradients:
    seq_id: bytes
    split_start: str
    split_end: str
    node_id: int

@dataclass
class Start:
    iteration: bytes

@dataclass
class Aggregate:
    iteration: bytes


def run_p(queue_in: Queue, queue_out: Queue, world_size = 4, node_id: int = 0, 
                    device = "cuda"):
    
    manual_seed(0)
    net = resnet34()
    # If necessary half the params to fit more models
    # net.half()    
    optimizer = optim.SGD(net.parameters())
    with open(f'log{node_id}.txt', 'a') as file, redirect_stdout(file):
        loc =  ResNetSubP(queue_in,queue_out,net,optimizer,node_id,world_size,device=device)
        loc.start()
    
class ResNetSubP(object):
    def __init__(self,queue_in: Queue, queue_out: Queue, net, optimizer, node_id = 0, world_size = 4,
                    device = "cuda") -> None:
        self.net = net
        self.net.fc = Linear(512,100,bias=True)
        self.net.to(device)
        self.device = device
        self.queue_in: Queue = queue_in
        self.queue_out: Queue = queue_out
        self.optimizer = optimizer
        self.node_id = node_id
        self.aggregation = []
        self.peer_parameters = dict()
        self.prev_aggregation = 0
        self.started = True
        self.world_size = world_size
        self.sizes = []
        self.len_sizes = []
        self.ttl_l = 0
        self.epoch = 0
        self.model_description = ["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4","avgpool","fc"]
        transform = transforms.Compose(
                        [
                            
                            transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dataset = CIFAR100("../data",transform=transform,train=True,download=True)
        self.trnds = CIFAR100("../data",transform=transform,train=False,download=True)
        self.trnds = DataLoader(self.trnds,
                                          batch_size=16, 
                                          shuffle=True)
        self.loss_fn = CrossEntropyLoss()
        
        self.dl = iter(DataLoader(self.dataset,
                                          batch_size=16, # x world size gives actual batch size
                                          shuffle=True))
        
        for _ in range(self.node_id):
            next(self.dl)
        self.str_ends = dict()
        strt = 0
        for i,md in enumerate(self.model_description):
            tmp = 0
            for param in self.net.__getattr__(md).parameters():
                self.sizes.append(param.shape)
                
                self.len_sizes.append(len(param.view(-1)))
                self.ttl_l += self.len_sizes[-1]
                tmp += self.len_sizes[-1]
            self.str_ends[md] = (strt,strt+tmp)
            strt += tmp
        pass


    def start(self):
        while self.started:
            while self.queue_in.empty() and self.started:
                
                continue
            if not self.started:
                break
            task = self.queue_in.get(True)
            if isinstance(task, Start):
                self.optimizer.zero_grad()
                try:
                    for _ in range(self.world_size):
                        next(self.dl)
                    
                    dt, lbl = next(self.dl)
                except StopIteration:
                    
                    self.epoch += 1
                    if self.epoch == 10:
                        exit()
                    tl = iter(self.trnds)
                    correct = 0
                    total = 0
                    
                    for _ in range(8):
                        img,lbl = next(tl)
                        img = img.to(self.device)
                        lbl = lbl.to(self.device)
                        
                        
                        out = self.net(img)
                        _, predicted = torch.max(out, 1)
                        total += lbl.size(0)
                        correct += (predicted == lbl).sum().item()
                        del img, lbl, out
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"ACC:{correct/total}\n")
                    self.dl = iter(DataLoader(self.dataset,
                                          batch_size=16, # x world size gives actual batch size
                                          shuffle=True))
                    for _ in range(self.node_id):
                        next(self.dl)
                    dt, lbl = next(self.dl)
                
                dt = dt.to(self.device)
                lbl = lbl.to(self.device)
                logits = self.net(dt)
                loss = self.loss_fn(logits, lbl)
                with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                    log.write(f"LOSS:{loss.item()}\n")
                loss.backward()
                self.optimizer.step()
                tmp = []
        
                for param in self.net.parameters():
                    tmp.append(param.data.view(-1))
                prev_grad = cat(tmp).to("cpu")
                # self.aggregation.append(prev_grad)
                self.prev_aggregation = prev_grad.clone().detach()
                self.queue_out.put(Start(b'\x01'))
                # TODO: GET GRADIENTS
                # self.optimizer.step()
                continue
            elif isinstance(task, Gradients):
                ret = pickle.loads(task.data)
                
                self.aggregation.append((ret,task.split_start,task.split_end))
                continue
            elif isinstance(task, GetGradients):
                tmp = self.prev_aggregation.clone().detach()
                strt = 0
                end = 0
                
                for k,v in self.str_ends.items():
                    if k == task.split_start:
                        strt = v[0]
                    if k == task.split_end:
                        end = v[1]
                        break
                tmp[0:strt] = 0
                if end < self.str_ends.get("avgpool")[1]:
                    tmp[end:self.str_ends.get("avgpool")[1]] = 0  
                else:
                    end = self.ttl_l
                self.queue_out.put(Gradients(int(strt).to_bytes(4,byteorder = "big") + int(end).to_bytes(4,byteorder = "big") + task.seq_id, pickle.dumps(tmp),"","",self.node_id),True)
            elif isinstance(task, Aggregate):
                i = 0
                while i < len(self.aggregation):
                    # print("AGGREGATING ",self.aggregation[i][1],self.aggregation[i][2])
                    self.aggregation[i][0].data[0:self.aggregation[i][1]] = self.prev_aggregation.data[0:self.aggregation[i][1]]
                    self.aggregation[i][0].data[self.aggregation[i][2]:] = self.prev_aggregation.data[self.aggregation[i][2]:]
                    self.aggregation[i] = self.aggregation[i][0]
                    i += 1
                # print("MINE ",0,self.ttl_l)
                # for k,v in self.str_ends.items():
                #     print(k,v[0],v[1])
                self.aggregation.append(self.prev_aggregation)
                ret = self.custom_avg(self.aggregation)
                print(ret.shape)
                ret = split(ret, self.len_sizes)
                for i, param in enumerate(self.net.parameters()):
                    param.data = ret[i].view(self.sizes[i]).to(self.device)
                cuda.empty_cache()



    def custom_avg(self,list_of_tensors):
        tmp = mean(stack(list_of_tensors), dim = 0)
        i = 0
        while i < len(list_of_tensors):
            del list_of_tensors[i]
        
        return tmp



                


                
        

    
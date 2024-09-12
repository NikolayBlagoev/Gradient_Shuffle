from dataclasses import dataclass
from multiprocessing import Lock, Process, Queue, current_process
import pickle
from torch import manual_seed
from typing import Callable
from contextlib import redirect_stdout
import time
import torch
from torch import optim, stack, mean, split, cat, tensor,save
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.llama import LLamaClassification,LLamaEmbedding,precompute_freqs_cis,TransformerBlock,RMSNorm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch import cuda
import traceback

from torch.nn import CrossEntropyLoss, Linear
# Messages Exchanged by the processes
@dataclass
class Gradients:
    seq_id: bytes
    data: bytes
    split_start: str
    split_end: str
    node_id: int
    deferred: int

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

class LLama(torch.nn.Module):
    def __init__(self, vocab_size, dmodel = 4096, num_heads = 32, multiple_of = 256, norm_eps = 1e-5, dropout_prob = 1e2, ctx_size = 2048, padding_idx = None, device = "cuda", n_layers = 4, ffn_dim_multiplier = None) -> None:
        super().__init__()
        self.embedding = LLamaEmbedding(vocab_size,dmodel,padding_idx=padding_idx,device=device)
        self.freqs_cis = precompute_freqs_cis(dmodel // num_heads, ctx_size * 2).to(device)
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self,f"transformer_{i}",TransformerBlock(
                    dmodel=dmodel,
                    num_heads=num_heads,
                    freq_cis=self.freqs_cis,
                    multiple_of=multiple_of,
                    norm_eps=norm_eps,
                    ffn_dim_multiplier=ffn_dim_multiplier, 
                    device = device
                ))
        self.norm = RMSNorm(dmodel, eps=norm_eps,device=device)
        self.ln = torch.nn.Linear(dmodel, vocab_size, bias=False,device=device)
    def forward(self, x):
        _, seq_l = x.shape
        h = self.embedding(x)
        
        for i in range(self.n_layers):
            h = getattr(self,f"transformer_{i}")(h)
        
        h = self.norm(h)
        output = self.ln(h).float()
        return output
def run_p(queue_in: Queue, queue_out: Queue, world_size = 4, node_id: int = 0, 
                    device = "cuda"):
    
    manual_seed(0)
    seq_l = 128
    tkns = SPTokenizer()
    ts = TinyStories(tkns,batch_size = 64 // world_size, seq_l=seq_l)
    net = LLama(tkns.vocab_size,dmodel=256,num_heads=8,multiple_of=256,ctx_size=seq_l,n_layers=16)
    # If necessary half the params to fit more models
    # net.half()    
    optimizer = optim.SGD(net.parameters(),lr=4e-3,momentum=0,dampening=0,weight_decay=0,nesterov=False)
    with open(f'log{node_id}.txt', 'a') as file, redirect_stdout(file):
        loc =  ResNetSubP(queue_in,queue_out,net,optimizer,node_id,world_size,ts,device=device)
        loc.start()
    
class ResNetSubP(object):
    def __init__(self,queue_in: Queue, queue_out: Queue, net, optimizer, node_id = 0, world_size = 4, ds = None, lr = 4e-3,
                    device = "cuda") -> None:
        self.net = net
        self.net.to(device)
        self.device = device
        self.lr = lr
        self.queue_in: Queue = queue_in
        self.queue_out: Queue = queue_out
        self.optimizer = optimizer
        self.node_id = node_id
        self.aggregation = []
        self.iteration = 0
        self.peer_parameters = dict()
        self.prev_aggregation = 0
        self.started = True
        self.world_size = world_size
        self.sizes = []
        self.len_sizes = []
        self.ttl_l = 0
        self.next_gradients = dict()
        self.epoch = 0
        self.model_description = ["embedding"]
        self.model_description += [f"transformer_{i}" for i in range(net.n_layers)]
        self.model_description += ["norm","ln"]
        self.ds = ds
        self.dl = iter(ds)
        
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
        try:
            while self.started:
                while self.queue_in.empty() and self.started:
                    
                    continue
                if not self.started:
                    break
                task = self.queue_in.get(True)
                if isinstance(task, Start):
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"=======NEW ITERATION:========\n")
                    self.optimizer.zero_grad()
                    try:
                        for _ in range(self.world_size):
                            next(self.dl)
                        
                        x, y = next(self.dl)
                    except StopIteration:
                        
                        self.epoch += 1
                        if self.iteration >= 80000:
                            save(net.state_dict(), f"gw4p50k1.pth")
                            exit()
                        
                        self.dl = iter(self.ds)
                        for _ in range(self.node_id):
                            next(self.dl)
                        x, y = next(self.dl)
                    
                    x = x.to(self.device)
                    y = y.to(self.device)
                    x = self.net(x)
                    B, T, C = x.shape
                    x = x.view(B*T,C)
                    y = y.view(B*T)
                    loss = F.cross_entropy(x,y)
                    
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"LOSS:{loss.item()}\n")
                    loss.backward()
                    # self.optimizer.step()
                    tmp = []
            
                    for param in self.net.parameters():
                        if param.grad == None:
                            tmp.append(torch.zeros_like(param.data).view(-1))
                            
                        else:
                            tmp.append(param.grad.data.view(-1))
                    prev_grad = cat(tmp).to("cpu")
                    
                    self.prev_aggregation = prev_grad.clone().detach()
                    self.queue_out.put(Start(b'\x01'))
                    # TODO: GET GRADIENTS
                    # self.optimizer.step()
                    continue
                elif isinstance(task, Gradients):
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"ADDING GRADIENT from {task.node_id} {self.iteration} {task.deferred}\n")
                    ret = pickle.loads(task.data)
                    if task.deferred == self.iteration:
                        self.aggregation.append((ret,task.split_start,task.split_end))
                        
                    else:
                        if self.next_gradients.get(task.deferred) == None:
                            self.next_gradients[task.deferred] = []
                        self.next_gradients[task.deferred].append((ret,task.split_start,task.split_end))
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
                    tmp = tmp[strt:end]
                    self.queue_out.put(Gradients(int(strt).to_bytes(4,byteorder = "big") + int(end).to_bytes(4,byteorder = "big") + task.seq_id, pickle.dumps(tmp),"","",self.node_id,self.iteration),True)
                elif isinstance(task, Aggregate):
                    self.iteration += 1
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"===AGGEGATING==== {len(self.aggregation)}\n")
                    i = 0
                    tmp = {}
                    while i < len(self.aggregation):
                        # print("AGGREGATING ",self.aggregation[i][1],self.aggregation[i][2])
                        k = self.aggregation[i][1]
                        if tmp.get(k) == None:
                            tmp[k] = ([],self.aggregation[i][2])
                        tmp[k][0].append(self.aggregation[i][0])
                        # self.aggregation[i][0].data[0:self.aggregation[i][1]] = self.prev_aggregation.data[0:self.aggregation[i][1]]
                        # .data[self.aggregation[i][2]:] = self.prev_aggregation.data[self.aggregation[i][2]:]
                        # self.aggregation[i] = self.aggregation[i][0]
                        i += 1
                    for k,v in tmp.items():
                        strt = k
                        fn = v[1]
                        v[0].append(self.prev_aggregation[strt:fn].detach().clone())
                        tmp[k] = (self.custom_avg(v[0]),fn)
                    ret = []
                    i = 0
                    ks = list(tmp.keys())
                    ks.sort()
                    mrkr = 0
                    while i < self.ttl_l:
                        
                        if len(ks) <= mrkr or ks[mrkr] > i:
                            to_move = -1
                            if len(ks) > mrkr:
                                to_move = ks[mrkr]
                                ret.append(self.prev_aggregation[i:to_move].detach().clone()) 
                            else:
                                ret.append(self.prev_aggregation[i:].detach().clone()) 
                            if to_move == -1:
                                break
                            i =  ks[mrkr]
                        
                        
                        ret.append(tmp[i][0])
                        i = tmp[i][1]
                        mrkr += 1
                        # del tmp[k][0]
                    tmp.clear()
                    ret = cat(ret)
                    while i < len(self.aggregation):
                        del self.aggregation[i]
                    self.aggregation.clear()
                    if self.next_gradients.get(self.iteration) != None:
                        self.aggregation = self.next_gradients[self.iteration]
                        del self.next_gradients[self.iteration]
                    else:
                        self.aggregation = []
                    # print("MINE ",0,self.ttl_l)
                    # for k,v in self.str_ends.items():
                    #     print(k,v[0],v[1])
                    # self.aggregation.append(self.prev_aggregation)
                    
                    
                    ret = split(ret, self.len_sizes)
                    
                    for i, param in enumerate(self.net.parameters()):
                        param.data -= self.lr*ret[i].view(self.sizes[i]).to(self.device)
                    
                    cuda.empty_cache()
        except Exception:
            with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                log.write(f"{traceback.format_exc()}\n")
            
            exit()




    def custom_avg(self,list_of_tensors):
        tmp = mean(stack(list_of_tensors), dim = 0)
        i = 0
        while i < len(list_of_tensors):
            del list_of_tensors[i]
        
        return tmp



                


                
        

    
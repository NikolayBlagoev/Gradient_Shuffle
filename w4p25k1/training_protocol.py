from dataclasses import dataclass

import os
import random
from typing import Callable
from deccom.peers.peer import Peer
from deccom.protocols.abstractprotocol import AbstractProtocol
from deccom.protocols.wrappers import *
from datetime import datetime
import asyncio
from traceback import print_exception, format_exc
from llm_subp import *
from deccom.cryptofuncs.hash import SHA256

    

class TrainingProtocol(AbstractProtocol):
    required_lower = AbstractProtocol.required_lower + \
        ["find_peer", "get_peer", "get_peers", "connected_callback","disconnected_callback"]
    
    GRADIENT = int.from_bytes(b'\x17',byteorder="big")
    def __init__(self, world_size: int, k: int, queue_in: Queue, queue_out: Queue, subprocess:Process, submodule=None, callback: Callable[[tuple[str, int], bytes], None] = lambda : ...):
        
        super().__init__(submodule, callback)
        self.group = []
        self.k = k
        self.tag = 0
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.subprocess = subprocess
        self.disconnected_callback = lambda *args : ...
        self.connected_callback = lambda *args : ...
        self.world_size = world_size
        self.gradients_received = dict()
        self.queue_reader = None
        self.iteration = 0
        self.can_acrue = False
        self.model_description = ["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4","avgpool","fc"]
    
    
        
    @bindto("open_connection")
    async def _lower_open_connection(self, remote_ip, remote_port, node_id: bytes):
        return
    
    @bindto("get_peer")
    def _lower_get_peer(self, node_id)->Peer:
        return None
    
    @bindto("find_peer")
    async def _lower_find_peer(self, id: bytes) -> Peer:
        return None


    async def start_iteration(self):
        if self.peer.pub_key != "0":
            await asyncio.sleep(3)
        self.queue_out.put(Start(b'\x01'), True)

    async def start(self, p: Peer):
        await super().start(p)
        
        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
            log.write(f"===={self.peer.id_node} {self.peer.tcp} STARTING===\n")
        
        loop = asyncio.get_event_loop() 
        self.queue_reader = loop.create_task(self.read_from_queue())
        loop.create_task(self.start_iteration())
        

        
            

    async def read_from_queue(self):
        while self.started:
            while self.queue_in.empty() and self.started:
                await asyncio.sleep(0.5)
            if not self.started:
                with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    log.write(f"====CLOSING???===\n")
                return
            task = self.queue_in.get(True)
            try:
                if isinstance(task,SendGradient):
                    pr = await self._lower_find_peer(SHA256(str(task.to)))
                    await self.send_datagram(int(task.iteration).to_bytes(4,byteorder="big") + int(task.tag).to_bytes(1,byteorder="big")+int(task.start).to_bytes(8,byteorder="big")+int(task.end).to_bytes(8,byteorder="big")+self.peer.id_node, pr.addr)
                    continue
                elif isinstance(task,Start):
                    
                    for i in range(self.world_size):
                        if i ==  int(self.peer.pub_key):
                            continue
                        pb = i
                        group = pb - self.iteration - int(self.peer.pub_key)
                        if i > int(self.peer.pub_key):
                            group -= 1
                        grpoup = group % (self.world_size - 1)
                        while group < 0:
                            group =  4 + group 
                        group = group // self.k
                        group = group % 4
                        pr = await self._lower_find_peer(bytes(SHA256(str(pb))))
                        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                            log.write(f"TO {pb} {pr.pub_key} goes group {group}\n")
                        
                        if group == 0:
                            self.queue_out.put(GetGradients(self.iteration, "transformer_12", "ln",pb,self.tag),True)
                        elif group == 1:
                            self.queue_out.put(GetGradients(self.iteration, "embedding", "transformer_3",pb,self.tag),True)
                            
                        elif group == 2:
                            self.queue_out.put(GetGradients(self.iteration, "transformer_4", "transformer_7",pb,self.tag),True)
                        elif group == 3:
                            self.queue_out.put(GetGradients(self.iteration, "transformer_8", "transformer_11",pb,self.tag),True)

                        self.tag += 1
                        self.tag = self.tag % 253
                            # self.queue_out.put(GetGradients(pr.id_node, "layer2", "layer2",0),True)
                        # elif group == 3:
                        #     self.queue_out.put(GetGradients(pr.id_node, "layer3", "layer3",0),True)
                    self.can_acrue = True
                    self.aggregate()
                    continue

                continue
            except Exception as e:
                with open(f'log{self.peer.pub_key}.txt', 'a') as f:
                    f.write(str(e))
                    f.write("!!!!!!!!!!!!!!!\n")
                    f.write(format_exc())
                    


    @bindfrom("connected_callback")
    def peer_connected(self, nodeid, peer: Peer):
        # print("NEW PEER")
        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"CONNECTED WITH {peer.pub_key}\n")
        self.group.append(peer)
        self.connected_callback(nodeid,peer)
        return
    
    def aggregate(self):
        cl = 0
        for k,v in self.gradients_received.items():
            if v.get(self.iteration) != None:
                cl +=1
        if cl != self.world_size - 1:
            return
        if not self.can_acrue:
            return
        self.can_acrue = False
        for k,v in self.gradients_received.items():
            del v[self.iteration]
        self.iteration += 1
        self.queue_out.put(Aggregate(b'\x01'),True)
        
        self.queue_out.put(Start(b'\x01'), True)
    
    def process_datagram(self, addr: tuple[str, int], data: bytes):
        itr = int.from_bytes(data[:4],byteorder="big")
        tag = data[4]
        strt = int.from_bytes(data[5:13],byteorder="big")
        end = int.from_bytes(data[13:21],byteorder="big")
        nodeid = data[21:]
        pr = self._lower_get_peer(nodeid)
        if strt == end:
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"NO GRADIENT FROM {pr.pub_key} {itr}\n")
        else:
            self.queue_out.put(Gradients(itr,strt,end,int(pr.pub_key),tag), True)

        if self.gradients_received.get(nodeid) == None:
            self.gradients_received[nodeid] = dict()
        self.gradients_received[nodeid][itr] = 1
        cl = 0
        for k,v in self.gradients_received.items():
            if v.get(self.iteration) != None:
                cl +=1
        if cl == self.world_size - 1:
            self.aggregate()

    
    
      
    
   

    

    
    async def stop(self):
        
        await super().stop()
        
        self.queue_in.close()
        self.queue_out.close()
                
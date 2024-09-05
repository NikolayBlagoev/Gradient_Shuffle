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
from resnet_trainer import *
from deccom.cryptofuncs.hash import SHA256

    

class TrainingProtocol(AbstractProtocol):
    required_lower = AbstractProtocol.required_lower + \
        ["find_peer", "set_stream_callback",
            "open_connection", "send_stream", "get_peer", "get_peers", "connected_callback","disconnected_callback"]
    
    GRADIENT = int.from_bytes(b'\x17',byteorder="big")
    def __init__(self, world_size: int, queue_in: Queue, queue_out: Queue, subprocess:Process, submodule=None, callback: Callable[[tuple[str, int], bytes], None] = lambda : ...):
        
        super().__init__(submodule, callback)
        self.group = []
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.subprocess = subprocess
        self.disconnected_callback = lambda *args : ...
        self.connected_callback = lambda *args : ...
        self.world_size = world_size
        self.gradients_received = dict()
        self.queue_reader = None
        self.iteration = 0
        self.model_description = ["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4","avgpool","fc"]
    
    
        
    @bindto("open_connection")
    async def _lower_open_connection(self, remote_ip, remote_port, node_id: bytes):
        return
    @bindto("send_stream")
    async def _lower_send_stream(self, node_id, data, ignore_sz = 0):
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
            log.write(f"===={self.peer.id_node} STARTING===\n")
        
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
                if isinstance(task,Gradients):
                    await self.send_stream(task.seq_id[8:], task.seq_id[:8] + task.data)
                    continue
                elif isinstance(task,Start):
                    # Distribute Gradients
                    # with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    #     log.write(f"DISTRIBUTING GRADIENTS\n")
                    for i in range(self.world_size):
                        if i ==  int(self.peer.pub_key):
                            continue
                        pb = i
                        group = pb + self.iteration + int(self.peer.pub_key)
                        if i > int(self.peer.pub_key):
                            group -= 1
                        
                        group = group % 4
                        pr = await self._lower_find_peer(bytes(SHA256(str(pb))))
                        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                            log.write(f"TO {pb} goes group {group}\n")
                        
                        if group == 0:
                            self.queue_out.put(GetGradients(pr.id_node, "layer4", "fc",0),True)
                        elif group == 1:
                            self.queue_out.put(GetGradients(pr.id_node, self.model_description[0], "layer1",0),True)
                        elif group == 2:
                            self.queue_out.put(GetGradients(pr.id_node, "layer2", "layer2",0),True)
                        elif group == 3:
                            self.queue_out.put(GetGradients(pr.id_node, "layer3", "layer3",0),True)
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
        self.iteration += 1
        for k,v in self.gradients_received.items():
            v.pop()

        self.queue_out.put(Aggregate(b'\x01'),True)
        self.queue_out.put(Start(b'\x01'), True)
    
    def process_datagram(self, addr: tuple[str, int], data: bytes):
        nodeid = data

        if self.gradients_received.get(nodeid) == None:
            self.gradients_received[nodeid] = []
        self.gradients_received[nodeid].append(1)
        cl = 0
        for k,v in self.gradients_received.items():
            if len(v) > 0:
                cl +=1
        if cl == self.world_size - 1:
            self.aggregate()

    @bindfrom("stream_callback")
    def process_data(self, data:bytes, nodeid, addr, retry = False):
        try:

            self.queue_out.put(Gradients(b'\x01',data[8:],int.from_bytes(data[0:4],byteorder="big"),int.from_bytes(data[4:8],byteorder="big"),0), True)
            if self.gradients_received.get(nodeid) == None:
                self.gradients_received[nodeid] = []
            self.gradients_received[nodeid].append(1)
            cl = 0
            for k,v in self.gradients_received.items():
                if len(v) > 0:
                    cl +=1
            if cl == self.world_size - 1:
                self.aggregate()
            
        except Exception as e:
                with open(f'log{self.peer.pub_key}.txt', 'a') as f:
                    f.write(str(e))
                    f.write("!!!!!!!!!!!!!!!\n")
                    f.write(format_exc())
    
      
    async def send_stream(self, node_id, data):
        # print("SENDING TO")
        
        try:
                
            await asyncio.wait_for(self._lower_find_peer(bytes(node_id)), timeout=10)
            p: Peer = self._lower_get_peer(node_id)
            if p == None:
                raise asyncio.TimeoutError()
                
        except asyncio.TimeoutError:
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"FAILED TO FIND PEER:{node_id}\n")

            
 
        ret = await self._lower_open_connection(p.addr[0], p.tcp, p.id_node, port_listen = 0)
        if not ret:
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"FAILED TO OPEN CONNECTION:{datetime.now()},{seqdata},{p.pub_key}\n")
                
            
        ret = await self._lower_send_stream(node_id, data)
   

    def get_lowest_stream(self):
        submodule = self.submodule
        while submodule != None and not hasattr(submodule, "get_lowest_stream") and hasattr(submodule, "submodule") :
            submodule = submodule.submodule
        if submodule != None and hasattr(submodule, "get_lowest_stream"):
            ret = submodule.get_lowest_stream()
            if ret == None:
                return self
            else:
                return ret
        else:
            
            return self

    
    async def stop(self):
        
        await super().stop()
        
        self.queue_in.close()
        self.queue_out.close()
                
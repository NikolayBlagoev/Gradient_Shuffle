import random
import time
from deccom.protocols.peerdiscovery.kademliadiscovery import KademliaDiscovery
from sys import argv
import asyncio
from deccom.cryptofuncs.hash import SHA256
from deccom.nodes import StreamNode, Node
from deccom.protocols.defaultprotocol import DefaultProtocol
from deccom.peers import Peer
from deccom.protocols.streamprotocol import StreamProtocol
from resnet_trainer import *
from multiprocessing import Lock, Process, Queue, current_process
import json
from pprint import pprint
from training_protocol import TrainingProtocol

if __name__ == '__main__':
    world_size = 8
    k = 3
    curr_id = int(argv[1])
    loop = asyncio.new_event_loop()

    while True:
        my_peer  = Peer(None, pub_key=str(curr_id))
        
        port = None

        protocol = DefaultProtocol()
        gossip = KademliaDiscovery([],interval=12, always_split = True)
        gossip.set_lower(protocol)
        stream = StreamProtocol(False)
        stream.set_lower(gossip)
        n = Peer(("127.0.0.1", 10015))
        if argv[1] != "0":
            gossip.bootstrap_peers.append(n)
            time.sleep(1)
        



        queue_in = Queue(1024)
        queue_out = Queue(1024)
        subprocess = Process(target=run_p,args=(queue_out,queue_in,world_size,curr_id,"cuda")) 
        trainingp = TrainingProtocol(world_size,k, queue_in, queue_out,subprocess)
        trainingp.set_lower(stream)
        subprocess.start()

        me = StreamNode(my_peer , trainingp,ip_addr="0.0.0.0", port = 10015 if curr_id == 0 else port)
        print( "TCP", me.tcp_port)

        
        print("run...")

        loop.run_until_complete(me.listen())
        loop.run_forever()
        
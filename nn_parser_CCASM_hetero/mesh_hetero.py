import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from run_configs import *


F = {} # fitness value for each link
bw_scales = {}
# construct noc nodes
for nop_id in range (NOP_SIZE): 
    for src_local_id in range (NOC_NODE_NUM):  
        for dst_local_id in range (NOC_NODE_NUM):
            src = src_local_id + NOC_NODE_NUM * nop_id
            dst = dst_local_id + NOC_NODE_NUM * nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = 1
            bw_scales[(src,local)] = 1
            src_x = src_local_id %  NoC_w
            src_y = int(src_local_id / NoC_w)
            dst_x = dst_local_id %  NoC_w
            dst_y = int(dst_local_id / NoC_w)
            if (src_x == dst_x) :
                if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = 1
            elif (src_y == dst_y) :
                if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = 1
# construct NoP nodes
for src_nop_id in range (NOP_SIZE):
    for dst_nop_id in range (NOP_SIZE):
        src =  NOC_NODE_NUM * NoP_w * NoP_w + src_nop_id
        dst =  NOC_NODE_NUM * NoP_w * NoP_w + dst_nop_id
        local = src + 1000
        F[(local,src)] = 0
        F[(src,local)] = 0
        bw_scales[(local,src)] = nop_scale_ratio
        bw_scales[(src,local)] = nop_scale_ratio
        src_x = src_nop_id %  NoP_w
        src_y = int(src_nop_id / NoP_w)
        dst_x = dst_nop_id %  NoP_w
        dst_y = int(dst_nop_id / NoP_w)
        if (src_x == dst_x) :
            if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                F[(src,dst)] = 0
                bw_scales[(src,dst)] = nop_scale_ratio
        elif (src_y == dst_y) :
            if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                F[(src,dst)] = 0
                bw_scales[(src,dst)] = nop_scale_ratio

# construct noc and nop connection
for nop_id in range (NOP_SIZE):
    nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
    noc_router_id = nop_id * NOC_NODE_NUM
    F[(noc_router_id,nop_router_id)] = 0
    F[(nop_router_id,noc_router_id)] = 0
    bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
    bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio

print ("F",F)
print ("bw_scales",bw_scales)
print ("len bw_scales",len(bw_scales))
# print ("F",F)

noc_route_table = {}
hops = {}

def noc_id (real_id):
    return (real_id % NOC_NODE_NUM)

def chip_id (real_id):
    return (int (real_id /NOC_NODE_NUM) )

def comm_id (real_id): # the communication router id in one chip
    return (chip_id(real_id)*NOC_NODE_NUM)
def nop_id (real_id):
    return (real_id - NOC_NODE_NUM*NOP_SIZE)
    
for src in range (0,NOC_NODE_NUM*NOP_SIZE):
    for dst in range (0,NOC_NODE_NUM*NOP_SIZE):
        print ("src = ",src,"dst = ",dst)
        noc_route_table[(src,dst)] = []
        cur_src = src
        cur_dst = src
        if chip_id(cur_src) != chip_id(dst):
            while cur_src != comm_id(src): # go to the first noc node
                src_noc_x = noc_id(cur_src) %  NoC_w
                src_noc_y = int(noc_id(cur_src) / NoC_w)
                dst_noc_x = noc_id(comm_id(src)) % NoC_w
                dst_noc_y = int(noc_id(comm_id(src))/ NoC_w)
                print (comm_id(src),src_noc_x,src_noc_y,dst_noc_x,dst_noc_y)
                if (src_noc_x > dst_noc_x):  # go west
                    cur_noc_dst = src_noc_x-1 +  src_noc_y * NoC_w
                elif (src_noc_x < dst_noc_x): # go east
                    cur_noc_dst = src_noc_x+1 +  src_noc_y * NoC_w
                elif (src_noc_y < dst_noc_y): # go north
                    cur_noc_dst = src_noc_x + (src_noc_y+1) * NoC_w
                elif (src_noc_y > dst_noc_y): # go south
                    cur_noc_dst = src_noc_x + (src_noc_y-1) * NoC_w
                cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst
                
            # go to the nop node
            cur_dst = chip_id(cur_src) + NOC_NODE_NUM * NOP_SIZE
            noc_route_table[(src,dst)].append((cur_src,cur_dst))
            cur_src = cur_dst

            while cur_src != chip_id(dst) + NOC_NODE_NUM * NOP_SIZE : # nop router of the destination node
                src_nop_x = nop_id(cur_src) % NoP_w
                src_nop_y = int (nop_id(cur_src) / NoP_w)
                dst_nop_x = chip_id(dst) % NoP_w
                dst_nop_y = int(chip_id(dst) / NoP_w)
                if (src_nop_x > dst_nop_x):  # go west
                    cur_nop_dst = src_nop_x-1 +  src_nop_y * NoP_w
                elif (src_nop_x < dst_nop_x): # go east
                    cur_nop_dst = src_nop_x+1 +  src_nop_y * NoP_w
                elif (src_nop_y < dst_nop_y): # go north
                    cur_nop_dst = src_nop_x + (src_nop_y+1) * NoP_w
                elif (src_nop_y > dst_nop_y): # go south
                    cur_nop_dst = src_nop_x + (src_nop_y-1) * NoP_w
                cur_dst = cur_nop_dst+NOC_NODE_NUM*NOP_SIZE
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst
            
            # go to the communication id 
            cur_dst = chip_id(dst) * NOC_NODE_NUM
            noc_route_table[(src,dst)].append((cur_src,cur_dst))
            cur_src = cur_dst

        while cur_src != dst:
            src_noc_x = noc_id(cur_src)  %  NoC_w
            src_noc_y = int(noc_id(cur_src)  / NoC_w)
            dst_noc_x = noc_id(dst) %  NoC_w
            dst_noc_y = int(noc_id(dst)  / NoC_w)
            # print ("src_x",src_x,"src_y",src_y,"dst_x",dst_x,"dst_y",dst_y)
            if (src_noc_x > dst_noc_x): # go west
                cur_noc_dst = src_noc_x-1 +  src_noc_y * NoC_w
            elif (src_noc_x < dst_noc_x): # go east
                cur_noc_dst = src_noc_x+1 +  src_noc_y * NoC_w
            elif (src_noc_y < dst_noc_y): # go north
                cur_noc_dst = src_noc_x + (src_noc_y+1) * NoC_w
            elif (src_noc_y > dst_noc_y): # go south
                cur_noc_dst = src_noc_x + (src_noc_y-1) * NoC_w
            cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
            noc_route_table[(src,dst)].append((cur_src,cur_dst))
            cur_src = cur_dst
        
print ("----noc_route_table------")
for route_item in noc_route_table:
    print (route_item,noc_route_table[route_item])

route_table = {}
for src in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
    for dst in range (1000,1000+NOC_NODE_NUM*NOP_SIZE):
        route_table[(src,dst)] = []
        noc_src = src - 1000
        noc_dst = dst -1000
        route_table[(src,dst)] = noc_route_table[(noc_src,noc_dst)].copy()
        if (src!=dst):
            route_table[(src,dst)].append((noc_dst,dst))
            route_table[(src,dst)].insert(0,(src,noc_src))


for item in route_table:
    hops[item] = len(route_table[item])
    # print (item,route_table[item])

print ("hops==========",sum(hops.values())/NOC_NODE_NUM/NOC_NODE_NUM/NOP_SIZE/NOP_SIZE)
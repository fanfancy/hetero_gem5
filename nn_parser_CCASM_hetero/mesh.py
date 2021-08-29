import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from run_configs import *

# mesh 4*4
F = {}
for src in range (NOC_NODE_NUM):
    for dst in range (NOC_NODE_NUM):
        local = src + 1000
        F[(local,src)] = 0
        F[(src,local)] = 0
        src_x = src %  NoC_w
        src_y = int(src / NoC_w)
        dst_x = dst %  NoC_w
        dst_y = int(dst / NoC_w)
        if (src_x == dst_x) :
            if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                F[(src,dst)] = 0
        elif (src_y == dst_y) :
            if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                F[(src,dst)] = 0

# print ("F",F)

noc_route_table = {}
hops = {}
noc_route_ids = {}

for src in range (0,NOC_NODE_NUM):
    for dst in range (0,NOC_NODE_NUM):
        cur_dst = src
        cur_src = src
        noc_route_table[(src,dst)] = []
        noc_route_ids[(src,dst)] = []
        while cur_dst != dst:
            src_x = cur_src %  NoC_w
            src_y = int(cur_src / NoC_w)
            dst_x = dst %  NoC_w
            dst_y = int(dst / NoC_w)
            # print ("src_x",src_x,"src_y",src_y,"dst_x",dst_x,"dst_y",dst_y)
            if (src_x > dst_x): # go west
                cur_dst = src_x-1 +  src_y * NoC_w
            elif (src_x < dst_x): # go east
                cur_dst = src_x+1 +  src_y * NoC_w
            elif (src_y < dst_y): # go north
                cur_dst = src_x + (src_y+1) * NoC_w
            elif (src_y > dst_y): # go south
                cur_dst = src_x + (src_y-1) * NoC_w
        # print ("cur_dst",cur_dst)
            noc_route_table[(src,dst)].append((cur_src,cur_dst))
            cur_src = cur_dst
            noc_route_ids[(src,dst)].append(cur_dst)
        # print ("\nnoc_route_table:",src,dst,noc_route_table[(src,dst)])
# print ("noc_route_table",noc_route_table)

route_table = {}
for src in range (1000,1000+NOC_NODE_NUM):
    for dst in range (1000,1000+NOC_NODE_NUM):
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

print ("hops==========",sum(hops.values())/NOC_NODE_NUM/NOC_NODE_NUM)
    




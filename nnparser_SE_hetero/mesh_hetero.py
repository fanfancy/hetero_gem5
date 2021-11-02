import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from config import *



def construct_noc_nop_Routerless(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio):
    noc_dict = {}
    CORE_NUM = NOC_NODE_NUM*NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE
    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}
    NoC_h = NOC_NODE_NUM / NoC_w

    # construct noc nodes
    for nop_id in range (NOP_SIZE): 
        for src_local_id in range (NOC_NODE_NUM):
            src = src_local_id + NOC_NODE_NUM * nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = 1
            bw_scales[(src,local)] = 1
            energy_ratio[(local,src)] = NOC_energy_ratio
            energy_ratio[(src,local)] = NOC_energy_ratio
            src_x = src_local_id %  NoC_w
            src_y = int(src_local_id / NoC_w)

            for dst_local_id in range (NOC_NODE_NUM):
                dst = dst_local_id + NOC_NODE_NUM * nop_id
                dst_x = dst_local_id %  NoC_w
                dst_y = int(dst_local_id / NoC_w)
                if (src_y == dst_y) and (src_y == 0):
                    if (src_x - dst_x == 1):
                        F[(src, dst)] = 0
                        bw_scales[(src, dst)] = NoC_h - 1 + NoC_w - src_x
                        energy_ratio[(src,dst)] = NOC_energy_ratio

                elif (src_y == dst_y) and (src_y != 0):
                    if (dst_x - src_x == 1):
                        F[(src, dst)] = 0
                        bw_scales[(src, dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio

                elif (src_x == dst_x) and (src_x == 0):
                    if (dst_y - src_y == 1):
                        F[(src,dst)] = 0
                        bw_scales[(src,dst)] = NoC_w - 1 + NoC_h - dst_y
                        energy_ratio[(src,dst)] = NOC_energy_ratio

                elif (src_x == dst_x) and (src_x != 0):
                    if (src_y - dst_y == 1):
                        F[(src,dst)] = 0
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio

    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = nop_id * NOC_NODE_NUM
        F[(noc_router_id,nop_router_id)] = 0
        F[(nop_router_id,noc_router_id)] = 0
        bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))

    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous mesh ---- \n\n")

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

    for src in range (0, NOC_NODE_NUM*NOP_SIZE):
        for dst in range (0, NOC_NODE_NUM*NOP_SIZE):
            noc_route_table[(src,dst)] = []
            cur_src = src
            cur_dst = src

            if chip_id(cur_src) != chip_id(dst):
                if (src != 0) or ((nop_id(dst) % NoC_w) != 0):
                    continue
                
                while cur_src != comm_id(src): # go to the first noc node
                    src_noc_x = noc_id(cur_src) %  NoC_w
                    src_noc_y = int(noc_id(cur_src) / NoC_w)
                    dst_noc_x = noc_id(comm_id(src)) % NoC_w
                    dst_noc_y = int(noc_id(comm_id(src))/ NoC_w)

                    if (src_noc_y > dst_noc_y): # go south
                        cur_noc_dst = src_noc_x + (src_noc_y-1) * NoC_w
                    elif (src_noc_x > dst_noc_x):  # go west
                        cur_noc_dst = src_noc_x-1 +  src_noc_y * NoC_w
                    else:
                        raise NotImplementedError
                    
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
                dst_noc_y = int(noc_id(dst) / NoC_w)

                if (noc_id(src) // NoC_w == 1) and (noc_id(src) % NoC_w == 0): # input
                    if (dst_noc_y == 0):
                        while(noc_id(cur_src) % NoC_w != NoC_w - 1): # east
                            noc_route_table[(src, dst)].append((cur_src, cur_src + 1))
                            cur_src = cur_src + 1
                        while(noc_id(cur_src) // NoC_w != dst_noc_y): # sorth
                            noc_route_table[(src, dst)].append((cur_src, cur_src - NoC_w))
                            cur_src = cur_src - NoC_w
                        while(noc_id(cur_src) % NoC_w != dst_noc_x): # west
                            noc_route_table[(src, dst)].append((cur_src, cur_src - 1))
                            cur_src = cur_src - 1
                    else:
                        while(noc_id(cur_src) // NoC_w != dst_noc_y): # north
                            noc_route_table[(src, dst)].append((cur_src, cur_src + NoC_w))
                            cur_src = cur_src + NoC_w
                        while(noc_id(cur_src) % NoC_w != dst_noc_x): # east
                            noc_route_table[(src, dst)].append((cur_src, cur_src + 1))
                            cur_src = cur_src + 1

                elif (noc_id(src) // NoC_w == 2) and (noc_id(src) % NoC_w == 0): # wgt
                    while(noc_id(cur_src) // NoC_w != NoC_h - 1): # north
                        noc_route_table[(src, dst)].append((cur_src, cur_src + NoC_w))
                        cur_src = cur_src + NoC_w

                    while(noc_id(cur_src) % NoC_w != dst_noc_x): # east
                        noc_route_table[(src, dst)].append((cur_src, cur_src + 1))
                        cur_src = cur_src + 1

                    while(noc_id(cur_src) // NoC_w != dst_noc_y): # sorth
                        noc_route_table[(src, dst)].append((cur_src, cur_src - NoC_w))
                        cur_src = cur_src - NoC_w

                elif (noc_id(dst) // NoC_w == 0) and (noc_id(dst) % NoC_w == 0): # out
                    while(noc_id(cur_src) // NoC_w != dst_noc_y): # sorth
                        noc_route_table[(src, dst)].append((cur_src, cur_src - NoC_w))
                        cur_src = cur_src - NoC_w

                    while(noc_id(cur_src) % NoC_w != dst_noc_x): # west
                        noc_route_table[(src, dst)].append((cur_src, cur_src - 1))
                        cur_src = cur_src - 1

                elif (src == 0) and (nop_id(dst) % NoC_w == 0):
                    while(noc_id(cur_src) // NoC_w != dst_noc_y): # north
                        noc_route_table[(src, dst)].append((cur_src, cur_src + NoC_w))
                        cur_src = cur_src + NoC_w
                else:
                    break

    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item,noc_route_table[route_item])

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
    noc_dict["route_table"]= route_table
    noc_dict["F"]= F
    noc_dict["bw_scales"]= bw_scales
    noc_dict["energy_ratio"] = energy_ratio
    return noc_dict, ALL_SIM_NODE_NUM    


def construct_noc_nop_Torus(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio):
    noc_dict = {}
    CORE_NUM = NOC_NODE_NUM*NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE
    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}
    NoC_h = NOC_NODE_NUM / NoC_w

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
                energy_ratio[(local,src)] = NOC_energy_ratio
                energy_ratio[(src,local)] = NOC_energy_ratio
                src_x = src_local_id %  NoC_w
                src_y = int(src_local_id / NoC_w)
                dst_x = dst_local_id %  NoC_w
                dst_y = int(dst_local_id / NoC_w)
                if (src_x == dst_x) :
                    if (abs(src_y - dst_y) == 1) or (abs(src_y - dst_y) == NoC_h - 1) :
                        F[(src,dst)] = 0
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio
                elif (src_y == dst_y) :
                    if (abs(src_x - dst_x) == 1) or (abs(src_x - dst_x) == NoC_w - 1):
                        F[(src,dst)] = 0
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio

    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (abs(src_y - dst_y) == 1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (abs(src_x - dst_x) == 1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = nop_id * NOC_NODE_NUM
        F[(noc_router_id,nop_router_id)] = 0
        F[(nop_router_id,noc_router_id)] = 0
        bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))

    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous Torus ---- \n\n")

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
            # print ("src = ",src,"dst = ",dst)
            noc_route_table[(src,dst)] = []
            cur_src = src
            cur_dst = src
            if chip_id(cur_src) != chip_id(dst):
                while cur_src != comm_id(src): # go to the first noc node
                    src_noc_x = noc_id(cur_src) %  NoC_w
                    src_noc_y = int(noc_id(cur_src) / NoC_w)
                    dst_noc_x = noc_id(comm_id(src)) % NoC_w
                    dst_noc_y = int(noc_id(comm_id(src))/ NoC_w)
                    # print (comm_id(src),src_noc_x,src_noc_y,dst_noc_x,dst_noc_y)
                    if (src_noc_x > dst_noc_x) and (abs(dst_noc_x - src_noc_x) <= NoC_w / 2):  # go west
                        cur_noc_dst = (src_noc_x - 1) % NoC_w +  src_noc_y * NoC_w
                    elif (src_noc_x > dst_noc_x):
                        cur_noc_dst = (src_noc_x + 1) % NoC_w +  src_noc_y * NoC_w
                    elif (src_noc_x < dst_noc_x) and (abs(dst_noc_x - src_noc_x) <= NoC_w / 2): # go east
                        cur_noc_dst = (src_noc_x + 1) % NoC_w +  src_noc_y * NoC_w
                    elif (src_noc_x < dst_noc_x):
                        cur_noc_dst = (src_noc_x - 1) % NoC_w +  src_noc_y * NoC_w
                    elif (src_noc_y < dst_noc_y) and (abs(src_noc_y - dst_noc_y) <= NoC_h / 2): # go north
                        cur_noc_dst = src_noc_x + ((src_noc_y + 1) % NoC_h) * NoC_w
                    elif (src_noc_y < dst_noc_y):
                        cur_noc_dst = src_noc_x + ((src_noc_y - 1) % NoC_h) * NoC_w
                    elif (src_noc_y > dst_noc_y) and (abs(src_noc_y - dst_noc_y) <= NoC_h / 2): # go south
                        cur_noc_dst = src_noc_x + ((src_noc_y - 1) % NoC_h) * NoC_w
                    elif (src_noc_y > dst_noc_y):
                        cur_noc_dst = src_noc_x + ((src_noc_y + 1) % NoC_h) * NoC_w

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
                if (src_noc_x > dst_noc_x) and (abs(dst_noc_x - src_noc_x) <= NoC_w / 2): # go west
                    cur_noc_dst = (src_noc_x - 1) % NoC_w +  src_noc_y * NoC_w
                elif (src_noc_x > dst_noc_x):
                    cur_noc_dst = (src_noc_x + 1) % NoC_w +  src_noc_y * NoC_w
                elif (src_noc_x < dst_noc_x) and (abs(dst_noc_x - src_noc_x) <= NoC_w / 2): # go east
                    cur_noc_dst = (src_noc_x + 1) % NoC_w +  src_noc_y * NoC_w
                elif (src_noc_x < dst_noc_x):
                    cur_noc_dst = (src_noc_x - 1) % NoC_w +  src_noc_y * NoC_w

                elif (src_noc_y < dst_noc_y) and (abs(src_noc_y - dst_noc_y) <= NoC_h / 2): # go north
                    cur_noc_dst = src_noc_x + ((src_noc_y + 1) % NoC_h) * NoC_w
                elif (src_noc_y < dst_noc_y):
                    cur_noc_dst = src_noc_x + ((src_noc_y - 1) % NoC_h) * NoC_w
                elif (src_noc_y > dst_noc_y) and (abs(src_noc_y - dst_noc_y) <= NoC_h / 2): # go south
                    cur_noc_dst = src_noc_x + ((src_noc_y - 1) % NoC_h) * NoC_w
                else:
                    cur_noc_dst = src_noc_x + ((src_noc_y + 1) % NoC_h) * NoC_w

                cur_dst = chip_id(cur_src) * NOC_NODE_NUM + cur_noc_dst
                noc_route_table[(src,dst)].append((cur_src,cur_dst))
                cur_src = cur_dst
            
    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item,noc_route_table[route_item])

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
    noc_dict["route_table"]= route_table
    noc_dict["F"]= F
    noc_dict["bw_scales"]= bw_scales
    noc_dict["energy_ratio"]= energy_ratio 
    return noc_dict,ALL_SIM_NODE_NUM    

def construct_noc_nop_Mesh(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio):
    noc_dict = {}
    CORE_NUM = NOC_NODE_NUM*NOP_SIZE
    ALL_SIM_NODE_NUM = CORE_NUM + NOP_SIZE
    F = {} # fitness value for each link
    bw_scales = {}
    energy_ratio = {}
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
                energy_ratio[(local,src)] = NOC_energy_ratio
                energy_ratio[(src,local)] = NOC_energy_ratio
                src_x = src_local_id %  NoC_w
                src_y = int(src_local_id / NoC_w)
                dst_x = dst_local_id %  NoC_w
                dst_y = int(dst_local_id / NoC_w)
                if (src_x == dst_x) :
                    if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                        F[(src,dst)] = 0
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio
                elif (src_y == dst_y) :
                    if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                        F[(src,dst)] = 0
                        bw_scales[(src,dst)] = 1
                        energy_ratio[(src,dst)] = NOC_energy_ratio

    # construct NoP nodes
    for src_nop_id in range (NOP_SIZE):
        for dst_nop_id in range (NOP_SIZE):
            src =  NOC_NODE_NUM * NOP_SIZE + src_nop_id
            dst =  NOC_NODE_NUM * NOP_SIZE + dst_nop_id
            local = src + 1000
            F[(local,src)] = 0
            F[(src,local)] = 0
            bw_scales[(local,src)] = nop_scale_ratio
            bw_scales[(src,local)] = nop_scale_ratio
            energy_ratio[(local,src)] = DIE2DIE_energy_ratio
            energy_ratio[(src,local)] = DIE2DIE_energy_ratio
            src_x = src_nop_id %  NoP_w
            src_y = int(src_nop_id / NoP_w)
            dst_x = dst_nop_id %  NoP_w
            dst_y = int(dst_nop_id / NoP_w)
            if (src_x == dst_x) :
                if (src_y - dst_y == 1) or (src_y- dst_y == -1) :
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio
                    
            elif (src_y == dst_y) :
                if (src_x - dst_x == 1) or (src_x - dst_x == -1):
                    F[(src,dst)] = 0
                    bw_scales[(src,dst)] = nop_scale_ratio
                    energy_ratio[(src,dst)] = DIE2DIE_energy_ratio

    # construct noc and nop connection
    for nop_id in range (NOP_SIZE):
        nop_router_id = nop_id + NOC_NODE_NUM * NOP_SIZE 
        noc_router_id = nop_id * NOC_NODE_NUM
        F[(noc_router_id,nop_router_id)] = 0
        F[(nop_router_id,noc_router_id)] = 0
        bw_scales[(noc_router_id,nop_router_id)] = nop_scale_ratio
        bw_scales[(nop_router_id,noc_router_id)] = nop_scale_ratio
        energy_ratio[(noc_router_id,nop_router_id)] = DIE2DIE_energy_ratio
        energy_ratio[(nop_router_id,noc_router_id)] = DIE2DIE_energy_ratio
        # print ("(nop_router_id,noc_router_id)", (nop_router_id,noc_router_id))


    
    print ("len(F)", len(F))
    print ("F",F)
    print ("bw_scales",bw_scales)
    print ("len bw_scales",len(bw_scales))
    print ("----- finish construct the heterogeneous mesh ---- \n\n")

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
            # print ("src = ",src,"dst = ",dst)
            noc_route_table[(src,dst)] = []
            cur_src = src
            cur_dst = src
            if chip_id(cur_src) != chip_id(dst):
                while cur_src != comm_id(src): # go to the first noc node
                    src_noc_x = noc_id(cur_src) %  NoC_w
                    src_noc_y = int(noc_id(cur_src) / NoC_w)
                    dst_noc_x = noc_id(comm_id(src)) % NoC_w
                    dst_noc_y = int(noc_id(comm_id(src))/ NoC_w)
                    # print (comm_id(src),src_noc_x,src_noc_y,dst_noc_x,dst_noc_y)
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
            
    # print ("----noc_route_table------")
    # for route_item in noc_route_table:
    #     print (route_item,noc_route_table[route_item])

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
    noc_dict["route_table"]= route_table
    noc_dict["F"]= F
    noc_dict["bw_scales"]= bw_scales
    noc_dict["energy_ratio"] = energy_ratio
    return noc_dict, ALL_SIM_NODE_NUM    


def construct_noc_nop_topo(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio, topology = 'Mesh'):
    if topology == 'Mesh':
        noc_dict, ALL_SIM_NODE_NUM = construct_noc_nop_Mesh(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio)
    elif topology == 'Torus':
        noc_dict, ALL_SIM_NODE_NUM = construct_noc_nop_Torus(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio)
    elif topology == 'Routerless':
        noc_dict, ALL_SIM_NODE_NUM = construct_noc_nop_Routerless(NOC_NODE_NUM, NoC_w, NOP_SIZE, NoP_w, nop_scale_ratio)
    else:
        raise NotImplementedError

    assert len(noc_dict["route_table"]) == (NOC_NODE_NUM * NOP_SIZE) ** 2, "len(noc_dict['route_table']) = %d which is wrong" % len(noc_dict['route_table'])
    assert len(noc_dict['F']) == len(noc_dict['bw_scales']), "len(noc_dict['F']) != len(noc_dict['bw_scales'])"

    return noc_dict, ALL_SIM_NODE_NUM

if __name__ == '__main__':
    TOPO_param = {"NoC_w":5, "NOC_NODE_NUM": 20, "NoP_w": 3, "NOP_SIZE": 6, "nop_scale_ratio":0.5, "topology": 'Routerless'}
    NoC_param, all_sim_node_num = construct_noc_nop_topo(TOPO_param["NOC_NODE_NUM"],TOPO_param["NoC_w"], TOPO_param["NOP_SIZE"],TOPO_param["NoP_w"], TOPO_param["nop_scale_ratio"], topology = TOPO_param["topology"])
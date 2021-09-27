import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from numpy.core.fromnumeric import mean
import re
np.set_printoptions(threshold=sys.maxsize)
from DNN import *
from pathlib import Path

wgt_tag =  (int(1001))
act_tag =  (int(1002))
out_tag =  (int(1003))

task = "VGG16"  
if_hetero = 0

if if_hetero == 0:
    from mesh import *
else:
    from mesh_hetero import *


task_name = "single_engine_example"
output_folder_name = "./task/"+task_name+"_"+str(NoC_w)
output_folder_name_start = output_folder_name+"_start"
output_folder_name_pipe = output_folder_name+"_pipe"

# 建立DNN模型
print ("task = ",task)
DNN1 = DNNModel("DNN1")
DNN_input(task, DNN1)
print (DNN1.layer_list[0].i_H)

#### a simple example #########
P = Q = 224; K=64; C=3; R=S=3
# dataflow parameter
P2=Q2=16; P1=Q1=14; 
K2 = 16; K1=1;  K0 = 4
C1=1; C0=3
R0=S0=3
assert(P2*P1==P)
assert(Q2*Q1==Q)
assert(K0*K1*K2==K)
compuation_num = P*Q*K*C*R*S
compuation_cycles = compuation_num/K2/K0/C0
print ("compuation_num=",compuation_num)
print ("compuation_cycles=",compuation_cycles)


# storage size
neuron_width  = 16 # bit
OL1 = 2; AL1 = 2; WL1 = 0.8 # KByte
AL1_mem = AL1*8*1024/neuron_width/2 # /2是因为ping-pong
OL1_mem = OL1*8*1024/neuron_width/2 
WL1_mem = WL1*8*1024/neuron_width/2 

L2=999999

data_flow = ['Q1', 'P1',   'C1',   'K1',  'S0',    'R0',   'Q2',   'P2',    'top']
ol1_ratio = [Q1,    P1,     1,      K1,     1,      1,      Q2,     P2,     1]
al1_ratio = [Q1,    P1,     C1,     1,      1,      1,      Q2,     P2,     1] 
wl1_ratio = [1,     1,      C1,     K1,     S0,     R0,     1,      1 ,     1]
all_param = [Q1,    P1,     C1,     K1,     S0,     R0,     Q2,     P2,     1]
out_final = [0,     0,      0,      0,      0,      1,      1,      1,      1]
if_act_share = 1 # depend on the parallel dimension
if_wgt_share = 0

OL1_need = {}
AL1_need = {}
WL1_need = {}
L1_need = {}
cal_cycles = {}
if_out_final = {}

ol1_need = K0; al1_need= C0; wl1_need = K0*C0; cal =1

# ------------------ 计算3个buffer存储需求&每级for循环循环次数 ------------------

for id in range(len(data_flow)):
    param = data_flow[id]
    ol1_need = ol1_need * ol1_ratio[id] # 单位:neuron
    al1_need = al1_need * al1_ratio[id]
    wl1_need = wl1_need * wl1_ratio[id]
    cal = cal * all_param[id]
    cal_cycles[param] = cal
    OL1_need[param] = ol1_need
    AL1_need[param] = al1_need
    WL1_need[param] = wl1_need
    L1_need[param] = wl1_need + al1_need + ol1_need
    if_out_final[param] = out_final[id]

repeat = 1
repeat_num = {}
    
for id in range(len(data_flow)):
    real_id = len(data_flow) - id -1
    param = data_flow[real_id] 
    repeat = repeat * all_param[real_id]
    repeat_num[param] = repeat


# ------------------ 决定存储临界点 ------------------
for id in range(len(data_flow)):
    param = data_flow[id]
    if OL1_need[param] > OL1_mem: 
        ol1_cp = param
        ol1_cp_id = id
        break
    ol1_cp = "top"
    ol1_cp_id = id

for id in range(len(data_flow)):
    param = data_flow[id]
    if WL1_need[param] > WL1_mem: 
        wl1_cp = param
        wl1_cp_id = id
        break
    wl1_cp = "top"
    wl1_cp_id = id

for id in range(len(data_flow)):
    param = data_flow[id]
    if AL1_need[param] > AL1_mem: 
        al1_cp = param
        al1_cp_id = id
        break
    al1_cp = "top"
    al1_cp_id = id

print ("OL1_need",OL1_need)
print ("AL1_need",AL1_need)
print ("WL1_need",WL1_need)
print ("repeat_num",repeat_num)
print ("cal_cycles",cal_cycles)
print ("ol1_cp=",ol1_cp,"al1_cp=",al1_cp,"wl1_cp=",wl1_cp)

# ------------------ 性能预测：计算整层所有计算和通信数据的数目 ------------------

pkt_num_wr_opt = 0
pkt_num_rd_opt = 0
pkt_num_rd_wgt_same = 0
pkt_num_rd_wgt_uniq = 0
pkt_num_rd_act_same = 0
pkt_num_rd_act_uniq = 0

cur = data_flow[ol1_cp_id]; inner = data_flow[ol1_cp_id-1]  
if (if_out_final[cur]!=1): 
    print("read opt mem ", OL1_need[inner],"repeat ",repeat_num[cur]) 
    pkt_num_rd_opt = pkt_num_rd_opt + int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit)) * repeat_num[cur]
    rd_out_data_num = OL1_need[inner]
else:
    pkt_num_rd_opt = 0
    rd_out_data_num = 0
print("write opt mem ", OL1_need[inner],"repeat ",repeat_num[cur])
pkt_num_wr_opt =  pkt_num_wr_opt + int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit)) *repeat_num[cur]
out_data_num = OL1_need[inner] # 用于生成仿真指令
    
cur = data_flow[al1_cp_id]; inner = data_flow[al1_cp_id-1]  
print("read act mem ",AL1_need[inner],if_act_share,"repeat ",repeat_num[cur])
if if_act_share==1: 
    pkt_num_rd_act_same = pkt_num_rd_act_same + int(math.ceil(AL1_need[inner]/flit_per_pkt/neu_per_flit))*repeat_num[cur]
else:
    pkt_num_rd_act_uniq = pkt_num_rd_act_uniq + int(math.ceil(AL1_need[inner]/flit_per_pkt/neu_per_flit))*repeat_num[cur]
act_data_num = AL1_need[inner] # 用于生成仿真指令

cur = data_flow[wl1_cp_id]; inner = data_flow[wl1_cp_id-1]  
print("read wgt mem ",WL1_need[inner],if_wgt_share,"repeat ",repeat_num[cur]) 
if if_wgt_share==1:
    pkt_num_rd_wgt_same = pkt_num_rd_wgt_same + int(math.ceil(WL1_need[inner]/flit_per_pkt/neu_per_flit)) *repeat_num[cur]
else: 
    pkt_num_rd_wgt_uniq = pkt_num_rd_wgt_uniq + int(math.ceil(WL1_need[inner]/flit_per_pkt/neu_per_flit)) *repeat_num[cur]
wgt_data_num = WL1_need[inner] # 用于生成仿真指令

ol1_node = 0; al1_node = 5; wl1_node = 10
mem_node_list = [ol1_node,al1_node,wl1_node,15]
cc_node_list = [1,2,3,4, 6,7,8,9, 11,12,13,14, 16,17,18,19]
cc_node_info = {}
F_cur=F.copy()

def all_full(dict):
    for item in dict:
        if dict[item] == True: return False
    return True

# 对每个cc node 构建其通信需求
for cc_id in cc_node_list: #单播
    ## from wgt mem to cc node 
    bw_needed = (pkt_num_rd_wgt_uniq) * flit_per_pkt  / compuation_cycles # 单位是flits/cycle
    for link in route_table[(wl1_node + 1000, cc_id + 1000)]:
        F_cur[link] += ( bw_needed / bw_scales[link] )

# 广播操作 from wgt mem to cc node 
wgt_multicast = []
if pkt_num_rd_wgt_same != 0:
    hungry_status = {}
    for hung_cc_id in cc_node_list:
        hungry_status[hung_cc_id] = True
    cur_src_list = [wl1_node]
    bw_needed = pkt_num_rd_wgt_same * flit_per_pkt  / compuation_cycles
    while all_full(hungry_status) == False:
        next_src_list = []
        for item in route_table:
            src = item[0]-1000
            if (src in cur_src_list) and (len(route_table[item])==3): # 可以一跳到达
                if ((item[1] - 1000) in hungry_status) and hungry_status[item[1]-1000] == True:  # 需要接收
                    hungry_status[item[1]-1000] = False
                    for link in route_table[item]: 
                        F_cur[link] += ( bw_needed / bw_scales[link] )
                    next_src_list.append(item[1] - 1000)
                    wgt_multicast.append((src,item[1] - 1000))
                    # print ("增加传输：",item)
        cur_src_list = next_src_list


for cc_id in cc_node_list: #单播
    ## from act mem to cc node 
    bw_needed = (pkt_num_rd_act_uniq) * flit_per_pkt  / compuation_cycles  
    for link in route_table[(al1_node + 1000, cc_id + 1000)]:
        F_cur[link] += ( bw_needed / bw_scales[link] )
# 广播操作 from act mem to cc node 
act_multicast = []
if pkt_num_rd_act_same != 0:
    hungry_status = {}
    for hung_cc_id in cc_node_list:
        hungry_status[hung_cc_id] = True
    cur_src_list = [al1_node]
    bw_needed = pkt_num_rd_act_same * flit_per_pkt  / compuation_cycles
    debug_id = 0
    while all_full(hungry_status) == False:
        # debug_id+=1; print("debug_id = ",debug_id)
        next_src_list = []
        for item in route_table:
            src = item[0]-1000
            if (src in cur_src_list) and (len(route_table[item])==3): # 可以一跳到达
                if ((item[1] - 1000) in hungry_status) and hungry_status[item[1]-1000] == True:  # 需要接收
                    hungry_status[item[1]-1000] = False
                    for link in route_table[item]: 
                        F_cur[link] += ( bw_needed / bw_scales[link] )
                    next_src_list.append(item[1] - 1000)
                    act_multicast.append((src,item[1] - 1000))
                    # print ("增加传输：",item)
        cur_src_list = next_src_list


for cc_id in cc_node_list: #单播
    ## from cc node to output mem
    bw_needed = pkt_num_wr_opt * flit_per_pkt / compuation_cycles 
    for link in route_table[(cc_id + 1000, ol1_node + 1000)]:
        F_cur[link] += ( bw_needed / bw_scales[link] )
    
    ## from output mem to cc node
    bw_needed = pkt_num_rd_opt * flit_per_pkt / compuation_cycles 
    for link in route_table[(ol1_node + 1000, cc_id + 1000)]:
        F_cur[link] += ( bw_needed / bw_scales[link] )

if (max(F_cur.values()) < 1):
        degrade_ratio = 1
else:
    degrade_ratio = max(F_cur.values()) 
print ("F_cur",F_cur)

# --------------------- 生成用于仿真的指令 ---------------------

# 寻找最小的循环单位
inner_cp_id = min(al1_cp_id,wl1_cp_id,ol1_cp_id)
inner_cp = data_flow[inner_cp_id]
ol1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[ol1_cp]) # 1
al1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[al1_cp]) # 1
wl1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[wl1_cp]) # 14
cal_cycle_per_run = cal_cycles[data_flow[inner_cp_id-1]]
print ("ol1_repeat_interval",ol1_repeat_interval, \
    "al1_repeat_interval",al1_repeat_interval,"wl1_repeat_interval",wl1_repeat_interval)

# 计算最小循环单位中的packet传输数目
out_packet = int(math.ceil(out_data_num/flit_per_pkt/neu_per_flit))
act_packet = int(math.ceil(act_data_num/flit_per_pkt/neu_per_flit))
wgt_packet = int(math.ceil(wgt_data_num/flit_per_pkt/neu_per_flit))
rd_out_packet = int (math.ceil(rd_out_data_num/flit_per_pkt/neu_per_flit))

# 计算插空packet数目
small_wgt_packet =  round ( (wgt_packet  / wl1_repeat_interval) ) # 四舍五入
small_act_packet =  round ( (act_packet  / al1_repeat_interval) ) 
small_out_packet =  round ( (out_packet  / ol1_repeat_interval) ) 
small_rd_out_packet =  round ( (rd_out_packet / ol1_repeat_interval) )

# 清空文件夹
out_folder = Path(output_folder_name_pipe)
if out_folder.is_dir():
    os.system('rm '+output_folder_name_pipe+'/*')
else:
    os.system('mkdir '+output_folder_name_pipe)

out_folder = Path(output_folder_name_start)
if out_folder.is_dir():
    os.system('rm '+output_folder_name_start+'/*')
else:
    os.system('mkdir '+output_folder_name_start)

mem_wait_packet = {}

# cal cycle不再进行处理
# if cal_cycle_per_run > small_out_packet: 
#     cal_cycle_per_run = cal_cycle_per_run - small_out_packet - 1
# else:
#     cal_cycle_per_run = 0

for mem_id in mem_node_list:
    mem_wait_packet[mem_id] = 0

# pipeline仿真 指令
for cal_core_id in cc_node_list:  
    with open (output_folder_name_pipe+'/'+str(cal_core_id)+'.txt','a') as core_file:
        if small_out_packet!= 0: print ("send "+str(ol1_node)+" "+str(small_out_packet)+" "+str(out_tag), file= core_file)
        
        if (1):
            if if_wgt_share == 1 and small_wgt_packet != 0:
                for item in wgt_multicast:
                    if item[0] == cal_core_id:    
                        print ("send "+str(item[1])+" "+str(small_wgt_packet)+" "+str(wgt_tag), file = core_file)
                print ("wait "+str(small_wgt_packet) +" "+str(wgt_tag),file = core_file)
            if if_act_share == 1 and small_act_packet !=0 :
                for item in act_multicast:
                    if item[0] == cal_core_id:    
                        print ("send "+str(item[1])+" "+str(small_act_packet)+" "+str(act_tag), file = core_file)
                print ("wait "+str(small_act_packet) +" "+str(act_tag),file = core_file)

        print ("wait "+str(small_rd_out_packet) +" "+str(out_tag),file = core_file)
        print ("cal",cal_cycle_per_run, file = core_file)
        print ("finish", file= core_file)
    
        mem_wait_packet[ol1_node] += small_out_packet

    with open (output_folder_name_pipe+'/'+str(wl1_node)+'.txt','a') as mem_file:
        if small_wgt_packet!= 0 and if_wgt_share != 1: print ("send "+str(cal_core_id)+" "+str(small_wgt_packet)+" "+str(wgt_tag), file= mem_file)
        
    with open (output_folder_name_pipe+'/'+str(al1_node)+'.txt','a') as mem_file:
        if small_act_packet!= 0 and if_act_share != 1: print ("send "+str(cal_core_id)+" "+str(small_act_packet)+" "+str(act_tag), file= mem_file)
       
    with open (output_folder_name_pipe+'/'+str(ol1_node)+'.txt','a') as mem_file:
        if small_rd_out_packet!= 0: print ("send "+str(cal_core_id)+" "+str(small_rd_out_packet)+" "+str(out_tag), file= mem_file)

if small_wgt_packet!= 0 and if_wgt_share == 1: 
    with open (output_folder_name_pipe+'/'+str(wl1_node)+'.txt','a') as mem_file:
        for item in wgt_multicast:
            if item[0] == wl1_node: print ("send "+str(item[1])+" "+str(small_wgt_packet)+" "+str(wgt_tag), file = mem_file)

if small_act_packet!= 0 and if_act_share == 1: 
    with open (output_folder_name_pipe+'/'+str(al1_node)+'.txt','a') as mem_file:
        for item in act_multicast:
            if item[0] == al1_node: print ("send "+str(item[1])+" "+str(small_act_packet)+" "+str(act_tag), file = mem_file)

# ol1 node task
with open (output_folder_name_pipe+'/'+str(ol1_node)+'.txt','a') as mem_file:
    if mem_wait_packet[ol1_node] != 0: print ("wait "+str(mem_wait_packet[ol1_node])+" "+str(out_tag), file= mem_file)
    

for mem_id in mem_node_list:
    with open (output_folder_name_pipe+'/'+str(mem_id)+'.txt','a') as mem_file:
        print ("finish", file= mem_file)

# 启动延迟仿真 指令
for cal_core_id in cc_node_list:     
    with open (output_folder_name_start+'/'+str(cal_core_id)+'.txt','a') as core_file:
        if if_act_share == 1 and act_packet !=0 :
            for item in act_multicast:
                if item[0] == cal_core_id:
                    print ("wait "+str(act_packet)+" "+str(act_tag), file = core_file)
                    break
            for item in act_multicast:
                if item[0] == cal_core_id:    
                    print ("send "+str(item[1])+" "+str(act_packet)+" "+str(act_tag), file = core_file)
        if if_wgt_share == 1 and wgt_packet !=0 :
            for item in wgt_multicast:
                if item[0] == cal_core_id:
                    print ("wait "+str(wgt_packet)+" "+str(wgt_tag), file = core_file)
                    break
            for item in wgt_multicast:
                if item[0] == cal_core_id:    
                    print ("send "+str(item[1])+" "+str(wgt_packet)+" "+str(wgt_tag), file = core_file)

        if if_act_share == 0: print ("wait "+str(act_packet)+" "+str(act_tag), file = core_file)
        if if_wgt_share == 0: print ("wait "+str(wgt_packet)+" "+str(wgt_tag), file = core_file)

        print ("finish", file= core_file)
    with open (output_folder_name_start+'/'+str(wl1_node)+'.txt','a') as mem_file:
        if if_wgt_share == 0: print ("send "+str(cal_core_id)+" "+str(wgt_packet)+" "+str(wgt_tag), file= mem_file)

    with open (output_folder_name_start+'/'+str(al1_node)+'.txt','a') as mem_file:
        if if_act_share == 0: print ("send "+str(cal_core_id)+" "+str(act_packet)+" "+str(act_tag), file= mem_file)

if wgt_packet!= 0 and if_wgt_share == 1: 
    with open (output_folder_name_start+'/'+str(wl1_node)+'.txt','a') as mem_file:
        for item in wgt_multicast:
            if item[0] == wl1_node: print ("send "+str(item[1])+" "+str(wgt_packet)+" "+str(wgt_tag), file = mem_file)

if act_packet!= 0 and if_act_share == 1: 
    with open (output_folder_name_start+'/'+str(al1_node)+'.txt','a') as mem_file:
        for item in act_multicast:
            if item[0] == al1_node: print ("send "+str(item[1])+" "+str(act_packet)+" "+str(act_tag), file = mem_file)


for mem_id in mem_node_list:
    with open (output_folder_name_start+'/'+str(mem_id)+'.txt','a') as mem_file:
        print ("finish", file= mem_file)
        
## summary 
print ("\n------------summary------------")
print ("repeat times = ", repeat_num[inner_cp])
print ("prediced latency = ", compuation_cycles*degrade_ratio, "degrade_ratio = ",degrade_ratio)
# TODO 广播的处理方式
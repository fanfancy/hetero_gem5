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

NoC_w = 4
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
L1_mem_size = 2*1024 # Byte
L1_mem = L1_mem_size/2 # neuron num

L2=999999

data_flow = ['Q1', 'P1',   'C1',   'K1',  'S0',    'R0',   'Q2',   'P2',    'top']
ol1_ratio = [Q1,    P1,     1,      K1,     1,      1,      Q2,     P2,     1]
al1_ratio = [Q1,    P1,     C1,     1,      1,      1,      Q2,     P2,     1] 
wl1_ratio = [1,     1,      C1,     K1,     S0,     R0,     1,      1 ,     1]
all_param = [Q1,    P1,     C1,     K1,     S0,     R0,     Q2,     P2,     1]
out_final = [0,     0,      0,      0,      0,      1,      1,      1,      1]
act_share = 1 # depend on the parallel dimension
wgt_share = 0 

OL1_need = {}
AL1_need = {}
WL1_need = {}
L1_need = {}
cal_cycles = {}
if_out_final = {}

ol1_need = K0; al1_need= C0; wl1_need = K0*C0; cal =1


for id in range(len(data_flow)):
    param = data_flow[id]
    ol1_need = ol1_need * ol1_ratio[id]
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
def new_func(param, repeat, repeat_num):
    repeat_num[param] = repeat

for id in range(len(data_flow)):
    real_id = len(data_flow) - id -1
    param = data_flow[real_id] 
    repeat = repeat * all_param[real_id]
    new_func(param, repeat, repeat_num)


# 决定存储临界点
for id in range(len(data_flow)):
    param = data_flow[id]
    if L1_need[param] > L1_mem: 
        l1_cp = param
        l1_cp_id = id
        break
    l1_cp = "top"
    l1_cp_id = id

print ("l1_cp",l1_cp,"l1_cp_id",l1_cp_id)
for id in range(l1_cp_id, len(data_flow)):
    if (ol1_ratio[id] != 1):
        ol1_cp = data_flow[id]
        ol1_cp_id = id
        break
    ol1_cp = "top"
    ol1_cp_id = id
for id in range(l1_cp_id, len(data_flow)):
    if (al1_ratio[id] != 1):
        al1_cp = data_flow[id]
        al1_cp_id = id
        break
    al1_cp = "top"
    al1_cp_id = id
for id in range(l1_cp_id, len(data_flow)):
    if (wl1_ratio[id] != 1):
        wl1_cp = data_flow[id]
        wl1_cp_id = id
        break
    wl1_cp = "top"
    wl1_cp_id = id


print ("OL1_need",OL1_need)
print ("AL1_need",AL1_need)
print ("WL1_need",WL1_need)
print ("repeat_num",repeat_num)
print ("cal_cycles",cal_cycles)
print ("l1_cp",l1_cp,"ol1_cp=",ol1_cp,"al1_cp=",al1_cp,"wl1_cp=",wl1_cp)


pkt_num_wr = 0
pkt_num_rd_same = 0
pkt_num_rd_uniq = 0

cur = data_flow[ol1_cp_id]; inner = data_flow[ol1_cp_id-1]  
#  if (if_out_final[cur]!=1): TODO !
#      print("read opt mem ", OL1_need[inner],"repeat ",repeat_num[cur]) 
#      pkt_num_rd_uniq = pkt_num_rd_uniq + OL1_need[inner]*repeat_num[cur]
print("write opt mem ", OL1_need[inner],"repeat ",repeat_num[cur])
pkt_num_wr =  pkt_num_wr + int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit)) *repeat_num[cur]
out_data_num = OL1_need[inner]
    
cur = data_flow[al1_cp_id]; inner = data_flow[al1_cp_id-1]  
print("read act mem ",AL1_need[inner],act_share,"repeat ",repeat_num[cur])
if act_share==1: 
    pkt_num_rd_same = pkt_num_rd_same + int(math.ceil(AL1_need[inner]/flit_per_pkt/neu_per_flit))*repeat_num[cur]
else:
    pkt_num_rd_uniq = pkt_num_rd_uniq + int(math.ceil(AL1_need[inner]/flit_per_pkt/neu_per_flit))*repeat_num[cur]
act_data_num = AL1_need[inner]

cur = data_flow[wl1_cp_id]; inner = data_flow[wl1_cp_id-1]  
print("read wgt mem ",WL1_need[inner],wgt_share,"repeat ",repeat_num[cur]) 
if wgt_share==1:
    pkt_num_rd_same = pkt_num_rd_same + int(math.ceil(WL1_need[inner]/flit_per_pkt/neu_per_flit)) *repeat_num[cur]
else: 
    pkt_num_rd_uniq = pkt_num_rd_uniq + int(math.ceil(WL1_need[inner]/flit_per_pkt/neu_per_flit)) *repeat_num[cur]
wgt_data_num = WL1_need[inner]

mem_node_list = [0,5,10,15]
cc_node_list = [1,2,3,4, 6,7,8,9, 11,12,13,14, 16,17,18,19]
cc_node_info = {}
F_cur=F.copy()

# 对每个cc node 构建其通信需求
out_folder = Path(output_folder_name+'overall')
if out_folder.is_dir():
    os.system('rm '+output_folder_name+'overall/*')
else:
    os.system('mkdir '+output_folder_name+'overall')


wr_packet_all = pkt_num_wr
rd_packet_all = pkt_num_rd_uniq + pkt_num_rd_same

mem_wait_all = {}
for mem_id in mem_node_list:
    mem_wait_all[mem_id] = 0

for cc_id in cc_node_list:
    mem_id = int(cc_id /NoC_w)*NoC_w

    ## form mem to cc node 
    bw_needed = rd_packet_all * flit_per_pkt  / compuation_cycles # TODO link bandwidth & data width
    for link in route_table[(mem_id + 1000, cc_id + 1000)]:
        F_cur[link] += ( bw_needed / bw_scales[link] )

    ## form cc node to mem
    bw_needed = wr_packet_all * flit_per_pkt / compuation_cycles # TODO link bandwidth & data width
    for link in route_table[(cc_id + 1000, mem_id + 1000)]:
        F_cur[link] += ( bw_needed / bw_scales[link] )

    if (max(F_cur.values()) < 1):
            degrade_ratio = 1
    else:
        degrade_ratio = max(F_cur.values()) 

# --------------------- generate the detail task files ---------------------
# all_param = [Q1,    P1,     C1,     K1,     S0,     R0,     Q2,     P2,     1]

# 最小和最大的循环单位
inner_cp_id = min(al1_cp_id,wl1_cp_id,ol1_cp_id)
outer_cp_id = max(al1_cp_id,wl1_cp_id,ol1_cp_id)
inner_cp = data_flow[inner_cp_id]
outer_cp = data_flow[outer_cp_id]
full_task_num = int(repeat_num[inner_cp]/repeat_num[outer_cp])       # 14
ol1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[ol1_cp]) # 1
al1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[al1_cp]) # 1
wl1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[wl1_cp]) # 14
cal_cycle_per_run = cal_cycles[data_flow[inner_cp_id-1]]
print ("full_task_num",full_task_num,"ol1_repeat_interval",ol1_repeat_interval, \
    "al1_repeat_interval",al1_repeat_interval,"wl1_repeat_interval",wl1_repeat_interval)

# 计算最小研究单位中的packet传输数目
out_packet = int(math.ceil(out_data_num/flit_per_pkt/neu_per_flit))
act_packet = int(math.ceil(act_data_num/flit_per_pkt/neu_per_flit))
wgt_packet = int(math.ceil(wgt_data_num/flit_per_pkt/neu_per_flit))

# 计算插空packet数目
small_wgt_packet =  int ( (wgt_packet  / wl1_repeat_interval) )
small_act_packet =  int ( (act_packet  / al1_repeat_interval) ) 
small_out_packet =  int ( (out_packet  / ol1_repeat_interval) ) 

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

if cal_cycle_per_run > small_out_packet: 
    cal_cycle_per_run = cal_cycle_per_run - small_out_packet - 1
else:
    cal_cycle_per_run = 0

for mem_id in mem_node_list:
    mem_wait_packet[mem_id] = 0

# ************************ 旧的方法2 ************************
#  for i in range (full_task_num): 
#      id = i + 1
#      for cal_core_id in cc_node_list:
#          mem_id = int(cal_core_id /NoC_w)*NoC_w
#          with open (output_folder_name+'/'+str(cal_core_id)+'.txt','a') as core_file:
#              if i % ol1_repeat_interval == 0:
#                  print ("send "+str(mem_id)+" "+str(out_packet), file= core_file)
#              print ("cal",cal_cycle_per_run, file = core_file)
#              if id % al1_repeat_interval == 0:
#                  print ("wait "+str(act_packet), file = core_file)
#              if id % wl1_repeat_interval == 0:
#                  print ("wait "+str(wgt_packet), file = core_file)
#          with open (output_folder_name+'/'+str(mem_id)+'.txt','a') as mem_file:
#              if i % al1_repeat_interval == 0:
#                  print ("send "+str(cal_core_id)+" "+str(act_packet), file = mem_file)
#              if i % wl1_repeat_interval == 0:
#                  print ("send "+str(cal_core_id)+" "+str(wgt_packet), file = mem_file)
#              if id % ol1_repeat_interval == 0:
#                  mem_wait_packet[mem_id] += out_packet

# pipeline 后的task file
for cal_core_id in cc_node_list:
    mem_id = int(cal_core_id /NoC_w)*NoC_w        
    with open (output_folder_name_pipe+'/'+str(cal_core_id)+'.txt','a') as core_file:
        print ("send "+str(mem_id)+" "+str(small_out_packet), file= core_file)
        print ("cal",cal_cycle_per_run, file = core_file)
        print ("wait "+str(small_act_packet + small_wgt_packet), file = core_file)
        print ("finish", file= core_file)
    with open (output_folder_name_pipe+'/'+str(mem_id)+'.txt','a') as mem_file:
        print ("send "+str(cal_core_id)+" "+str(small_act_packet + small_wgt_packet), file= mem_file)
        mem_wait_packet[mem_id] += small_out_packet

for mem_id in mem_node_list:
    with open (output_folder_name_pipe+'/'+str(mem_id)+'.txt','a') as mem_file:
        print ("wait "+str(mem_wait_packet[mem_id]), file= mem_file)
        print ("finish", file= mem_file)


for cal_core_id in cc_node_list:
    mem_id = int(cal_core_id /NoC_w)*NoC_w        
    with open (output_folder_name_start+'/'+str(cal_core_id)+'.txt','a') as core_file:
        print ("wait "+str(act_packet + wgt_packet), file = core_file)
        print ("finish", file= core_file)
    with open (output_folder_name_start+'/'+str(mem_id)+'.txt','a') as mem_file:
        print ("send "+str(cal_core_id)+" "+str(act_packet + wgt_packet), file= mem_file)

for mem_id in mem_node_list:
    with open (output_folder_name_start+'/'+str(mem_id)+'.txt','a') as mem_file:
        print ("finish", file= mem_file)
        
## summary 
print ("------------summary------------")
print ("remain repeat times = ", repeat_num[inner_cp])
print ("prediced latency = ", compuation_cycles*degrade_ratio, "degrade_ratio = ",degrade_ratio)
# TODO 广播的处理方式
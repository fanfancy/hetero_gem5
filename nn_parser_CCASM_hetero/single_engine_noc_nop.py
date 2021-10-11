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
if_hetero = 1

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

#### a NoP+NoC example #########
# 硬件信息
CoreNum = 16; ChipNum = 16; 
neuron_width  = 16 # bit
OL1 = 1; AL1 = 1; WL1 = 1 # KByte
OL2 = 64; AL2 = 64; WL2 = 64 # KByte
# 卷积配置
P = Q = 224; K=256; C=64; R=S=3
# 映射方案 (目前只实现了K维度有并行度)
# P3 Q3 PP3  PQ3  PK3 | P2 Q2  PP2  PQ2  PK2 |R  S  K1  C1  P1  Q1 | PC0 PK0
PP3, P3, PP2, P2, P1 = 1, 14, 1, 4, 4
PQ3, Q3, PQ2, Q2, Q1 = 1, 14, 1, 4, 4
PK3, PK2, K1, PK0 = 4, 16, 1, 4
C1, PC0 = 16 ,4
R0, S0 = 3, 3

runtimeP = PP3*P3*PP2*P2*P1
runtimeQ = PQ3*Q3*PQ2*Q2*Q1
runtimeK = PK3*PK2*K1*PK0
runtimeC = C1*PC0
runtimeR = R
runtimeS = S
runtimeCoreNum = PK2*PQ2*PP2
runtimeChipNum = PP3*PQ3*PK3

assert(runtimeP>=P);assert(runtimeQ>=Q);assert(runtimeK>=K);assert(runtimeC>=C)
assert(runtimeCoreNum <= CoreNum);assert(runtimeChipNum <= ChipNum)

compuation_num = runtimeP*runtimeQ*runtimeK*runtimeC*runtimeR*runtimeS
compuation_cycles = compuation_num/runtimeCoreNum/runtimeChipNum/PC0/PK0
print ("compuation_num=",compuation_num)
print ("compuation_cycles=",compuation_cycles)

# storage size
AL1_mem = AL1*8*1024/neuron_width/2 # /2是因为ping-pong
OL1_mem = OL1*8*1024/neuron_width/2 
WL1_mem = WL1*8*1024/neuron_width/2
AL2_mem = AL2*8*1024/neuron_width/2
OL2_mem = OL2*8*1024/neuron_width/2
WL2_mem = WL2*8*1024/neuron_width/2 

# dataflow中只用写明时序for参数
data_flow = ['Q1', 'P1',   'C1',   'K1',  'S0',    'R0',   'Q2',   'P2',    'Q3',   'P3',  'top']
ol1_ratio = [Q1,    P1,     1,      K1,     1,      1,      Q2,     P2,     Q3,     P3,     1]
al1_ratio = [Q1,    P1,     C1,     1,      1,      1,      Q2,     P2,     Q3,     P3,     1] 
wl1_ratio = [1,     1,      C1,     K1,     S0,     R0,     1,      1 ,     1,      1,      1]
all_param = [Q1,    P1,     C1,     K1,     S0,     R0,     Q2,     P2,     Q3,     P3,     1]
out_final = [0,     0,      0,      0,      0,      1,      1,      1,      1,      1,      1]
if_act_share = 1 # depend on the parallel dimension TODO 如果pq提供并行度之后 这个参数应该是分组的
if_wgt_share = 0

OL1_need = {}; AL1_need = {}; WL1_need = {}; L1_need = {}
OL2_need = {}; AL2_need = {}; WL2_need = {}; L2_need = {}

cal_cycles = {}
if_out_final = {}

ol1_need = PK0; al1_need= PC0; wl1_need = PK0*PC0; cal =1

# ------------------ 计算6个buffer存储需求&每级for循环循环次数 ------------------

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
    # L2
    OL2_need[param] = ol1_need * PK2 * PQ2 * PP2
    AL2_need[param] = al1_need * PQ2 * PP2 
    WL2_need[param] = wl1_need * PK2 

repeat = 1
repeat_num = {}
    
for id in range(len(data_flow)):
    real_id = len(data_flow) - id -1
    param = data_flow[real_id] 
    repeat = repeat * all_param[real_id]
    repeat_num[param] = repeat

# ------------------ 决定存储临界点 ------------------

def find_cp(the_data_flow,storage_need,storage_size):
    for id in range(len(data_flow)):
        param = the_data_flow[id]
        if storage_need[param] > storage_size: 
            the_cp = param
            the_cp_id = id
            break
        the_cp = "top"
        the_cp_id = id
    return the_cp,the_cp_id

ol1_cp,ol1_cp_id = find_cp(data_flow,OL1_need,OL1_mem)
al1_cp,al1_cp_id = find_cp(data_flow,AL1_need,AL1_mem)
wl1_cp,wl1_cp_id = find_cp(data_flow,WL1_need,WL1_mem)
ol2_cp,ol2_cp_id = find_cp(data_flow,OL2_need,OL2_mem)
al2_cp,al2_cp_id = find_cp(data_flow,AL2_need,AL2_mem)
wl2_cp,wl2_cp_id = find_cp(data_flow,WL2_need,WL2_mem)

print ("OL1_need",OL1_need); print ("OL2_need",OL2_need)
print ("AL1_need",AL1_need); print ("AL2_need",AL2_need)
print ("WL1_need",WL1_need); print ("WL2_need",WL2_need)

print ("repeat_num",repeat_num)
print ("cal_cycles",cal_cycles)
print ("ol1_cp=",ol1_cp,"al1_cp=",al1_cp,"wl1_cp=",wl1_cp)
print ("ol2_cp=",ol2_cp,"al2_cp=",al2_cp,"wl2_cp=",wl2_cp)

# ------------------ 构建mem cal core 位置和属性等 ------------------
# 从wxy import进来

ol2_node = 20; al2_node = 25; wl2_node = 30

act_core_dict = {}
act_core_dict[0] = [21,22,23,24, 26,27,28,29, 31,32,33,34, 36,37,38,39]

wgt_core_dict = {} 
wgt_core_dict[0] = [21];  wgt_core_dict[1] = [22];  wgt_core_dict[2] = [23];  wgt_core_dict[3] = [24] 
wgt_core_dict[4] = [26];  wgt_core_dict[5] = [27];  wgt_core_dict[6] = [28];  wgt_core_dict[7] = [29] 
wgt_core_dict[8] = [31];  wgt_core_dict[9] = [32];  wgt_core_dict[10] = [33]; wgt_core_dict[11] = [34] 
wgt_core_dict[12] = [36]; wgt_core_dict[13] = [37]; wgt_core_dict[14] = [38]; wgt_core_dict[15] = [39] 

out_core_dict = {} 
out_core_dict[0] = [21];  out_core_dict[1] = [22];  out_core_dict[2] = [23];  out_core_dict[3] = [24] 
out_core_dict[4] = [26];  out_core_dict[5] = [27];  out_core_dict[6] = [28];  out_core_dict[7] = [29] 
out_core_dict[8] = [31];  out_core_dict[9] = [32];  out_core_dict[10] = [33]; out_core_dict[11] = [34] 
out_core_dict[12] = [36]; out_core_dict[13] = [37]; out_core_dict[14] = [38]; out_core_dict[15] = [39] 

dram_node  = 0
act_chip_dict = {}; out_chip_dict = {}; wgt_chip_dict = {}
out_chip_dict[0] = [20,40,80,100]
act_chip_dict[0] = [25,45,85,105]
wgt_chip_dict[0] = [30,50,90,110]

# 依据信息构建 mem_node_list 和 cc_node_list 
mem_node_list = [ol2_node,al2_node,wl2_node,dram_node]
cc_node_list = []
for item in wgt_core_dict:
    core_id_list = wgt_core_dict[item]
    for core_id in core_id_list:
        if core_id not in cc_node_list:
            cc_node_list.append(core_id)

all_sim_node_num = CORE_NUM

# ------------------ 性能预测：计算整层所有计算和通信数据的数目 ------------------
# L1 用于统计通信总量 & prediction
core_pkt_num_wr_opt = 0
core_pkt_num_rd_opt = 0
core_pkt_num_rd_wgt = 0
core_pkt_num_rd_act = 0

# L1 用于生成task file的变量
core_rd_out_data_num = 0
core_out_data_num = 0 
core_act_data_num = 0
core_wgt_data_num = 0

cur = data_flow[ol1_cp_id]; inner = data_flow[ol1_cp_id-1]  
if (if_out_final[cur]!=1): 
    print("CORE: read opt mem ", OL1_need[inner],"repeat ",repeat_num[cur]) 
    core_pkt_num_rd_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit)) * repeat_num[cur]
    core_rd_out_data_num += OL1_need[inner]
else:
    core_pkt_num_rd_opt += 0
    core_rd_out_data_num += 0
print("CORE: write opt mem ", OL1_need[inner],"repeat ",repeat_num[cur])
core_pkt_num_wr_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit)) *repeat_num[cur]
core_out_data_num += OL1_need[inner] # 用于生成仿真指令
    
cur = data_flow[al1_cp_id]; inner = data_flow[al1_cp_id-1]  
print("CORE: read act mem ",AL1_need[inner],"repeat ",repeat_num[cur])
core_pkt_num_rd_act +=  int(math.ceil(AL1_need[inner]/flit_per_pkt/neu_per_flit))*repeat_num[cur]
core_act_data_num += AL1_need[inner] # 用于生成仿真指令

cur = data_flow[wl1_cp_id]; inner = data_flow[wl1_cp_id-1]  
print("CORE: read wgt mem ",WL1_need[inner],"repeat ",repeat_num[cur]) 
core_pkt_num_rd_wgt += int(math.ceil(WL1_need[inner]/flit_per_pkt/neu_per_flit)) *repeat_num[cur]
core_wgt_data_num += WL1_need[inner] # 用于生成仿真指令

# L2 用于统计通信总量 & prediction
chip_pkt_num_wr_opt = 0
chip_pkt_num_rd_opt = 0
chip_pkt_num_rd_wgt = 0
chip_pkt_num_rd_act = 0

# L2 用于生成task file的变量
chip_rd_out_data_num = 0
chip_out_data_num = 0 
chip_act_data_num = 0
chip_wgt_data_num = 0

cur = data_flow[ol2_cp_id]; inner = data_flow[ol2_cp_id-1]  
if (if_out_final[cur]!=1): 
    print("Chip: read opt mem ", OL2_need[inner],"repeat ",repeat_num[cur]) 
    chip_pkt_num_rd_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit)) * repeat_num[cur]
    chip_rd_out_data_num += OL2_need[inner]
else:
    chip_pkt_num_rd_opt += 0
    chip_rd_out_data_num += 0
print("Chip: write opt mem ", OL2_need[inner],"repeat ",repeat_num[cur])
chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit)) *repeat_num[cur]
chip_out_data_num += OL2_need[inner] # 用于生成仿真指令
    
cur = data_flow[al2_cp_id]; inner = data_flow[al2_cp_id-1]  
print("Chip: read act mem ",AL2_need[inner],"repeat ",repeat_num[cur])
chip_pkt_num_rd_act +=  int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit))*repeat_num[cur]
chip_act_data_num += AL2_need[inner] # 用于生成仿真指令

cur = data_flow[wl2_cp_id]; inner = data_flow[wl2_cp_id-1]  
print("Chip: read wgt mem ",WL2_need[inner],"repeat ",repeat_num[cur]) 
chip_pkt_num_rd_wgt += int(math.ceil(WL2_need[inner]/flit_per_pkt/neu_per_flit)) *repeat_num[cur]
chip_wgt_data_num += WL2_need[inner] # 用于生成仿真指令


F_cur=F.copy()

def all_full(dict): # 广播时候用的
    for item in dict:
        if dict[item] == True: return False
    return True

# 对core构建通信需求
# 用到的信息: core_pkt_num_wr_opt; core_pkt_num_rd_opt; core_pkt_num_rd_wgt; core_pkt_num_rd_act
bw_needed = (core_pkt_num_rd_act) * flit_per_pkt  / compuation_cycles # act 带宽需求,单位是flits/cycle 
for item in act_core_dict:
    dst_list = act_core_dict[item]
    for dst in dst_list:
        for link in route_table[(al2_node + 1000, dst + 1000)]:
            F_cur[link] += ( bw_needed / bw_scales[link] )

bw_needed = (core_pkt_num_rd_wgt) * flit_per_pkt  / compuation_cycles # wgt 带宽需求,单位是flits/cycle 
for item in wgt_core_dict:
    dst_list = wgt_core_dict[item]
    for dst in dst_list:
        for link in route_table[(wl2_node + 1000, dst + 1000)]:
            F_cur[link] += ( bw_needed / bw_scales[link] )

bw_needed = (core_pkt_num_rd_opt) * flit_per_pkt  / compuation_cycles # out read带宽需求,单位是flits/cycle 
for item in out_core_dict:
    dst_list = out_core_dict[item]
    for dst in dst_list:
        for link in route_table[(ol2_node + 1000, dst + 1000)]:
            F_cur[link] += ( bw_needed / bw_scales[link] )

bw_needed = (core_pkt_num_wr_opt) * flit_per_pkt  / compuation_cycles # out write带宽需求,单位是flits/cycle 
for item in out_core_dict:
    dst_list = out_core_dict[item]
    for dst in dst_list:
        for link in route_table[(dst + 1000, ol2_node+1000)]:
            F_cur[link] += ( bw_needed / bw_scales[link] )

# 对chip构建通信需求
# 用到的信息: chip_pkt_num_wr_opt; chip_pkt_num_rd_opt; chip_pkt_num_rd_wgt; chip_pkt_num_rd_act
bw_needed = (chip_pkt_num_rd_act) * flit_per_pkt  / compuation_cycles # act 带宽需求,单位是flits/cycle 
for item in act_core_dict:
    dst_list = act_core_dict[item]
    for dst in dst_list:
        for link in route_table[(dram_node + 1000, dst + 1000)]:
            F_cur[link] += ( bw_needed / bw_scales[link] )

bw_needed = (chip_pkt_num_rd_wgt) * flit_per_pkt  / compuation_cycles # wgt 带宽需求,单位是flits/cycle 
for item in wgt_core_dict:
    dst_list = wgt_core_dict[item]
    for dst in dst_list:
        for link in route_table[(dram_node + 1000, dst + 1000)]:
            F_cur[link] += ( bw_needed / bw_scales[link] )

bw_needed = (chip_pkt_num_rd_opt) * flit_per_pkt  / compuation_cycles # out read带宽需求,单位是flits/cycle 
for item in out_core_dict:
    dst_list = out_core_dict[item]
    for dst in dst_list:
        for link in route_table[(dram_node + 1000, dst + 1000)]:
            F_cur[link] += ( bw_needed / bw_scales[link] )

bw_needed = (chip_pkt_num_wr_opt) * flit_per_pkt  / compuation_cycles # out write带宽需求,单位是flits/cycle 
for item in out_core_dict:
    dst_list = out_core_dict[item]
    for dst in dst_list:
        for link in route_table[(dst + 1000, dram_node+1000)]:
            F_cur[link] += ( bw_needed / bw_scales[link] )


if (max(F_cur.values()) < 1):
        degrade_ratio = 1
else:
    degrade_ratio = max(F_cur.values()) 
print ("F_cur",F_cur)
print ("degrade_ratio",degrade_ratio)

# --------------------- 生成用于仿真的指令 ---------------------

# 寻找最小的循环单位
inner_cp_id = min(al1_cp_id,wl1_cp_id,ol1_cp_id,al2_cp_id,wl2_cp_id,ol2_cp_id)
inner_cp = data_flow[inner_cp_id]
ol1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[ol1_cp]) 
al1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[al1_cp]) 
wl1_repeat_interval = int(repeat_num[inner_cp] / repeat_num[wl1_cp]) 
ol2_repeat_interval = int(repeat_num[inner_cp] / repeat_num[ol2_cp]) 
al2_repeat_interval = int(repeat_num[inner_cp] / repeat_num[al2_cp]) 
wl2_repeat_interval = int(repeat_num[inner_cp] / repeat_num[wl2_cp]) 

cal_cycle_per_run = cal_cycles[data_flow[inner_cp_id-1]]
print ("ol1_repeat_interval",ol1_repeat_interval, "al1_repeat_interval",al1_repeat_interval,"wl1_repeat_interval",wl1_repeat_interval)
print ("ol2_repeat_interval",ol2_repeat_interval, "al2_repeat_interval",al2_repeat_interval,"wl2_repeat_interval",wl2_repeat_interval)

# 计算最小循环单位中的packet传输数目
core_out_packet = int(math.ceil(core_out_data_num/flit_per_pkt/neu_per_flit))
core_act_packet = int(math.ceil(core_act_data_num/flit_per_pkt/neu_per_flit))
core_wgt_packet = int(math.ceil(core_wgt_data_num/flit_per_pkt/neu_per_flit))
core_rd_out_packet = int (math.ceil(core_rd_out_data_num/flit_per_pkt/neu_per_flit))

chip_out_packet = int(math.ceil(chip_out_data_num/flit_per_pkt/neu_per_flit))
chip_act_packet = int(math.ceil(chip_act_data_num/flit_per_pkt/neu_per_flit))
chip_wgt_packet = int(math.ceil(chip_wgt_data_num/flit_per_pkt/neu_per_flit))
chip_rd_out_packet = int (math.ceil(chip_rd_out_data_num/flit_per_pkt/neu_per_flit))

# 计算插空packet数目
core_small_wgt_packet =  round ( (core_wgt_packet  / wl1_repeat_interval) ) # 四舍五入
core_small_act_packet =  round ( (core_act_packet  / al1_repeat_interval) ) 
core_small_out_packet =  round ( (core_out_packet  / ol1_repeat_interval) ) 
core_small_rd_out_packet =  round ( (core_rd_out_packet / ol1_repeat_interval) )

chip_small_wgt_packet =  round ( (chip_wgt_packet  / wl2_repeat_interval) ) # 四舍五入
chip_small_act_packet =  round ( (chip_act_packet  / al2_repeat_interval) ) 
chip_small_out_packet =  round ( (chip_out_packet  / ol2_repeat_interval) ) 
chip_small_rd_out_packet =  round ( (chip_rd_out_packet / ol2_repeat_interval) )

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

for mem_id in mem_node_list:
    mem_wait_packet[mem_id] = 0

# pipeline仿真 指令

# cc节点：wait3种数据;  cal; send output
for cal_core_id in cc_node_list:  
    with open (output_folder_name_pipe+'/'+str(cal_core_id)+'.txt','a') as core_file:
        if core_small_out_packet!= 0: 
            print ("send "+str(ol2_node)+" "+str(core_small_out_packet)+" "+str(out_tag), file= core_file)
        print ("cal",cal_cycle_per_run, file = core_file)
        print ("wait "+str(core_small_act_packet) +" "+str(act_tag),file = core_file)
        print ("wait "+str(core_small_wgt_packet) +" "+str(wgt_tag),file = core_file)
        print ("wait "+str(core_small_rd_out_packet) +" "+str(out_tag),file = core_file)
        mem_wait_packet[ol2_node] += core_small_out_packet


    with open (output_folder_name_pipe+'/'+str(wl2_node)+'.txt','a') as mem_file:
        if core_small_wgt_packet!= 0: print ("send "+str(cal_core_id)+" "+str(core_small_wgt_packet)+" "+str(wgt_tag), file= mem_file)
        
    with open (output_folder_name_pipe+'/'+str(al2_node)+'.txt','a') as mem_file:
        if core_small_act_packet!= 0: print ("send "+str(cal_core_id)+" "+str(core_small_act_packet)+" "+str(act_tag), file= mem_file)
       
    with open (output_folder_name_pipe+'/'+str(ol2_node)+'.txt','a') as mem_file:
        if core_small_rd_out_packet!= 0: print ("send "+str(cal_core_id)+" "+str(core_small_rd_out_packet)+" "+str(out_tag), file= mem_file)

# ol1 node task
with open (output_folder_name_pipe+'/'+str(ol2_node)+'.txt','a') as mem_file:
    if mem_wait_packet[ol2_node] != 0: print ("wait "+str(mem_wait_packet[ol2_node])+" "+str(out_tag), file= mem_file)
    

# chiplet traffic: dram -> act L2
for item in act_chip_dict:
    dst_list = act_chip_dict[item]
    for dst in dst_list:
        with open (output_folder_name_pipe+'/'+str(dst)+'.txt','a') as al2_file:
            print ("wait "+str(chip_small_act_packet) +" "+str(act_tag),file = al2_file)
        with open (output_folder_name_pipe+'/'+str(dram_node)+'.txt','a') as dram_file:
            print ("send "+str(dst)+" "+str(chip_small_act_packet)+" "+str(act_tag), file= dram_file)

# chiplet traffic: dram -> wgt L2
for item in wgt_chip_dict:
    dst_list = wgt_chip_dict[item]
    for dst in dst_list:
        with open (output_folder_name_pipe+'/'+str(dst)+'.txt','a') as wl2_file:
            print ("wait "+str(chip_small_wgt_packet) +" "+str(wgt_tag),file = wl2_file)
        with open (output_folder_name_pipe+'/'+str(dram_node)+'.txt','a') as dram_file:
            print ("send "+str(dst)+" "+str(chip_small_wgt_packet)+" "+str(wgt_tag), file= dram_file)

# chiplet traffic: dram -> ol2 L2
for item in out_chip_dict:
    dst_list = out_chip_dict[item]
    for dst in dst_list:
        with open (output_folder_name_pipe+'/'+str(dst)+'.txt','a') as ol2_file:
            print ("wait "+str(chip_small_out_packet) +" "+str(out_tag),file = ol2_file)
        with open (output_folder_name_pipe+'/'+str(dram_node)+'.txt','a') as dram_file:
            print ("send "+str(dst)+" "+str(chip_small_out_packet)+" "+str(out_tag), file= dram_file)

for sim_node in range (all_sim_node_num):   
    with open (output_folder_name_pipe+'/'+str(sim_node)+'.txt','a') as node_file:
        print ("finish",file = node_file)
# 启动延迟仿真 指令

# todo 增加额外的nop router的task file，否则gem5无法运行

## summary 
print ("\n------------summary------------")
print ("repeat times = ", repeat_num[inner_cp])
print ("prediced latency = ", compuation_cycles*degrade_ratio, "degrade_ratio = ",degrade_ratio)
# TODO 广播的处理方式
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
from pathlib import Path
from GaEncode import *
from config import *
from multicast_method import *

wgt_tag =  (int(1001))
act_tag =  (int(1002))
out_tag =  (int(1003))

# 建立DNN模型
#print ("task = ",task)
#print (DNN1.layer_list[0].i_H)


#### a NoP+NoC example #########
# 硬件信息
# memory_param = {"OL1":,"OL2":,"AL1":,"AL2":,"WL1":,"WL2":}
# 卷积配置 从basicParam_noc_nop中import进来

def calFitness_granu(for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, network_param, HW_param, memory_param, if_multicast):
	# 映射方案 (目前只实现了K维度有并行度)
	# Ga Encode
	#for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list = GaGetChild()
	CoreNum = HW_param["PE"][0] * HW_param["PE"][1]
	PE_lenth = HW_param["PE"][1]
	PE_height = HW_param["PE"][0]
	ChipNum = HW_param["Chiplet"][0] * HW_param["Chiplet"][1]
	OL1 = memory_param["OL1"]
	OL2 = memory_param["OL2"]
	AL1 = memory_param["AL1"]
	AL2 = memory_param["AL2"]
	WL1 = memory_param["WL1"]
	WL2 = memory_param["WL2"]
	data_flow = for_list[0]
	ol1_ratio = for_list[1]
	al1_ratio = for_list[2]
	wl1_ratio = for_list[3]
	all_param = for_list[4]
	out_final = for_list[5]
	if_act_share_PE = for_list[6]
	if_wgt_share_PE = for_list[7]
	if_act_share_Chiplet = for_list[8]
	if_wgt_share_Chiplet = for_list[9]

	# mapping parameter
	P1,P2,P3 = partition_list["P"][0],partition_list["P"][1],partition_list["P"][2]
	Q1,Q2,Q3 = partition_list["Q"][0],partition_list["Q"][1],partition_list["Q"][2]
	C1,C2,C3 = partition_list["C"][0],partition_list["C"][1],partition_list["C"][2]
	K1,K2,K3 = partition_list["K"][0],partition_list["K"][1],partition_list["K"][2]
	PP2,PQ2,PK2 = parallel_dim_list[0][0],parallel_dim_list[0][1],parallel_dim_list[0][2]
	PP3,PQ3,PK3 = parallel_dim_list[1][0],parallel_dim_list[1][1],parallel_dim_list[1][2]
	PK0 = HW_param["intra_PE"]["K"]
	PC0 = HW_param["intra_PE"]["C"]

	# network parameter
	P = network_param["P"]
	Q = network_param["Q"]
	K = network_param["K"]
	C = network_param["C"]
	R = network_param["R"]
	S = network_param["S"]
	stride = network_param["stride"]
	# memory node id
	ol2_node = PE_height * (PE_lenth+1)
	if PE_height == 2:
		al2_node = ol2_node + PE_lenth + 1
		wl2_node = ol2_node + PE_lenth + 1
	else:
		assert(PE_height > 1)
		al2_node = ol2_node + PE_lenth + 1
		wl2_node = ol2_node + (PE_lenth + 1) * 2
	dram_node  = 0
	#print(route_table)


	runtimeP = PP3*P3*PP2*P2*P1
	runtimeQ = PQ3*Q3*PQ2*Q2*Q1
	runtimeK = PK3*K3*PK2*K2*K1*PK0
	runtimeC = C3*C2*C1*PC0
	runtimeR = R # R S不拆分,在PE level的时序for参数里
	runtimeS = S
	runtimeCoreNum = PK2*PQ2*PP2
	runtimeChipNum = PP3*PQ3*PK3

	assert(runtimeP>=P);assert(runtimeQ>=Q);assert(runtimeK>=K);assert(runtimeC>=C)
	assert(runtimeCoreNum <= CoreNum);assert(runtimeChipNum <= ChipNum)

	energy_MAC = P*Q*K*C*R*S * MAC_energy_ratio
	compuation_num = runtimeP*runtimeQ*runtimeK*runtimeC*runtimeR*runtimeS
	compuation_cycles = compuation_num/runtimeCoreNum/runtimeChipNum/PC0/PK0
	#print ("compuation_num=",compuation_num)
	#print ("compuation_cycles=",compuation_cycles)

	# storage size
	AL1_mem = AL1*8*1024/act_wgt_width/2 # /2是因为ping-pong
	OL1_mem = OL1*8*1024/psum_width/2 
	WL1_mem = WL1*8*1024/act_wgt_width/2
	AL2_mem = AL2*8*1024/act_wgt_width/2
	OL2_mem = OL2*8*1024/psum_width/2
	WL2_mem = WL2*8*1024/act_wgt_width/2 
	A_PE_mem = PC0
	W_PE_mem = PC0*PK0	

	OL1_need = {}; AL1_need = {}; WL1_need = {}; L1_need = {}
	OL2_need = {}; AL2_need = {}; WL2_need = {}; L2_need = {}

	cal_cycles = {}
	if_out_final = {}


	ol1_need = PK0; al1_need_CKpart= PC0; wl1_need = PK0*PC0; cal =1
	al1_need_Qpart = 1; al1_need_Ppart = 1; al1_need_Rpart = 1; al1_need_Spart = 1
	# ------------------ 计算6个buffer存储需求&每级for循环循环次数 ------------------

	for id in range(len(data_flow)):
		param = data_flow[id]
		ol1_need = ol1_need * ol1_ratio[id] # 单位:neuron

		# al1 need calculation
		if "K" == param[0] or "C" == param[0]:
			al1_need_CKpart = al1_need_CKpart * all_param[id]
		elif "Q" == param[0]:
			al1_need_Qpart = al1_need_Qpart * all_param[id]
		elif "P" == param[0]:
			al1_need_Ppart = al1_need_Ppart * all_param[id]
		elif "R" == param[0]:
			al1_need_Rpart = al1_need_Rpart * all_param[id]
		elif "S" == param[0]:
			al1_need_Spart = al1_need_Spart * all_param[id]

		al1_need_Q_final = al1_need_Qpart * stride + al1_need_Spart - stride
		al1_need_P_final = al1_need_Ppart * stride + al1_need_Rpart - stride
		al1_need = al1_need_CKpart * al1_need_Q_final * al1_need_P_final

		
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
		AL2_need[param] = al1_need * PQ2 * PP2 #这里有点问题
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
		for id in range(len(the_data_flow)):
			param = the_data_flow[id]
			if storage_need[param] > storage_size: 
				the_cp = param
				the_cp_id = id
				break
			the_cp = "top"
			the_cp_id = id
		utilization_ratio = storage_need[the_data_flow[the_cp_id-1]] / storage_size
		return the_cp,the_cp_id,utilization_ratio

	ol1_cp,ol1_cp_id,ol1_utilization_ratio = find_cp(data_flow,OL1_need,OL1_mem)
	al1_cp,al1_cp_id,al1_utilization_ratio = find_cp(data_flow,AL1_need,AL1_mem)
	wl1_cp,wl1_cp_id,wl1_utilization_ratio = find_cp(data_flow,WL1_need,WL1_mem)
	ol2_cp,ol2_cp_id,ol2_utilization_ratio = find_cp(data_flow,OL2_need,OL2_mem)
	al2_cp,al2_cp_id,al2_utilization_ratio = find_cp(data_flow,AL2_need,AL2_mem)
	wl2_cp,wl2_cp_id,wl2_utilization_ratio = find_cp(data_flow,WL2_need,WL2_mem)
	ape_cp,ape_cp_id,ape_utilization_ratio = find_cp(data_flow,AL1_need,A_PE_mem)
	wpe_cp,wpe_cp_id,wpe_utilization_ratio = find_cp(data_flow,WL1_need,W_PE_mem)


	#print ("OL1_need",OL1_need); print ("OL2_need",OL2_need)
	#print ("AL1_need",AL1_need); print ("AL2_need",AL2_need)
	#print ("WL1_need",WL1_need); print ("WL2_need",WL2_need)

	#print ("repeat_num",repeat_num)
	#print ("cal_cycles",cal_cycles)
	#print ("ol1_cp=",ol1_cp,"al1_cp=",al1_cp,"wl1_cp=",wl1_cp)
	#print ("ol2_cp=",ol2_cp,"al2_cp=",al2_cp,"wl2_cp=",wl2_cp)

	# ------------------ 构建mem cal core 位置和属性等 ------------------
	# 从wxy import进来

	act_core_dict = act_wgt_dict["act_core"][0]["recv"]
	wgt_core_dict = act_wgt_dict["wgt_core"][0]["recv"]
	act_chip_dict = act_wgt_dict["act_chiplet"]["recv"]
	wgt_chip_dict = act_wgt_dict["wgt_chiplet"]["recv"]
	out_core_dict = out_dict["rd_core"][0]["recv"]
	out_chip_dict = out_dict["rd_chip"]["recv"]

	# 依据信息构建 mem_node_list 和 cc_node_list 
	mem_node_list = [ol2_node,al2_node,wl2_node,dram_node]
	cc_node_list = []
	for item in wgt_core_dict:
		core_id_list = wgt_core_dict[item]
		for core_id in core_id_list:
			if core_id not in cc_node_list:
				cc_node_list.append(core_id)

	# ------------------ 性能预测：计算整层所有计算和通信数据的数目 ------------------
	# REG <-> L1 用于统计通信总量 & prediction
	pe_neu_num_rd_wgt = 0 # 单位 neuron数目 
	pe_neu_num_rd_act = 0

	cur = data_flow[ape_cp_id]; inner = data_flow[ape_cp_id-1]  
	if ape_cp == "top":
		pe_neu_num_rd_act += AL1_need[data_flow[ape_cp_id]] * 1
	else:
		pe_neu_num_rd_act += AL1_need[data_flow[ape_cp_id]] * repeat_num[data_flow[ape_cp_id+1]]

	cur = data_flow[wl1_cp_id]; inner = data_flow[wl1_cp_id-1]  
	pe_neu_num_rd_wgt += WL1_need[inner] * repeat_num[cur]

	pe_neu_num_rd_wgt = pe_neu_num_rd_wgt * CoreNum * ChipNum # 考虑到片上有CoreNum * ChipNum个PE
	pe_neu_num_rd_act = pe_neu_num_rd_act * CoreNum * ChipNum # 考虑到片上有CoreNum * ChipNum个PE
	energy_rd_wgt_L1 = pe_neu_num_rd_wgt * SRAM_energy(WL1) * act_wgt_width 
	energy_rd_act_L1 = pe_neu_num_rd_act * SRAM_energy(AL1) * act_wgt_width  

	# L1 用于统计通信总量 & prediction
	core_pkt_num_wr_opt = 0; core_neu_num_wr_opt = 0  # 单位分别是 packet | neuron数目 
	core_pkt_num_rd_opt = 0; core_neu_num_rd_opt = 0
	core_pkt_num_rd_wgt = 0; core_neu_num_rd_wgt = 0
	core_pkt_num_rd_act = 0; core_neu_num_rd_act = 0

	# L1 用于生成task file的变量
	core_rd_out_data_num = 0
	core_out_data_num = 0 
	core_act_data_num = 0
	core_wgt_data_num = 0

	cur = data_flow[ol1_cp_id]; inner = data_flow[ol1_cp_id-1]  
	if (if_out_final[cur]!=1): 
		#print("CORE: read opt mem ", OL1_need[inner],"repeat ",repeat_num[cur]) 
		core_pkt_num_rd_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit_psum)) * repeat_num[cur]
		core_neu_num_rd_opt += OL1_need[inner] * repeat_num[cur]
		core_rd_out_data_num += OL1_need[inner]
	else:
		core_pkt_num_rd_opt += 0
		core_neu_num_rd_opt += 0
		core_rd_out_data_num += 0
	#print("CORE: write opt mem ", OL1_need[inner],"repeat ",repeat_num[cur])
	if (if_out_final[cur]!=1):
		core_pkt_num_wr_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit_psum)) *repeat_num[cur]
	else:
		core_pkt_num_wr_opt += int(math.ceil(OL1_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
	core_out_data_num += OL1_need[inner] # 用于生成仿真指令
	core_neu_num_wr_opt += OL1_need[inner] * repeat_num[cur]
		
	cur = data_flow[al1_cp_id]; inner = data_flow[al1_cp_id-1]  
	#print("CORE: read act mem ",AL1_need[inner],"repeat ",repeat_num[cur])
	core_pkt_num_rd_act +=  int(math.ceil(AL1_need[inner]/flit_per_pkt/neu_per_flit_act_wgt))*repeat_num[cur]
	core_act_data_num += AL1_need[inner] # 用于生成仿真指令
	if al1_cp == "top":
		core_neu_num_rd_act += AL1_need[data_flow[al1_cp_id]] * 1
	else:
		core_neu_num_rd_act += AL1_need[data_flow[al1_cp_id]] * repeat_num[data_flow[al1_cp_id+1]]

	cur = data_flow[wl1_cp_id]; inner = data_flow[wl1_cp_id-1]  
	#print("CORE: read wgt mem ",WL1_need[inner],"repeat ",repeat_num[cur]) 
	core_pkt_num_rd_wgt += int(math.ceil(WL1_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
	core_wgt_data_num += WL1_need[inner] # 用于生成仿真指令
	core_neu_num_rd_wgt += WL1_need[inner] * repeat_num[cur]

	# 考虑上并行度带来的数据复用机会 (多播)
	if if_multicast == 1:
		core_neu_num_wr_opt = core_neu_num_wr_opt * CoreNum * ChipNum  # 没有机会复用
		core_neu_num_rd_opt = core_neu_num_rd_opt * CoreNum * ChipNum 
		core_neu_num_rd_wgt = core_neu_num_rd_wgt * CoreNum * ChipNum /PP2 / PQ2
		core_neu_num_rd_act = core_neu_num_rd_act * CoreNum * ChipNum /PK2 
	elif if_multicast == 0:
		core_neu_num_wr_opt = core_neu_num_wr_opt * CoreNum * ChipNum  # 没有机会复用
		core_neu_num_rd_opt = core_neu_num_rd_opt * CoreNum * ChipNum 
		core_neu_num_rd_wgt = core_neu_num_rd_wgt * CoreNum * ChipNum  
		core_neu_num_rd_act = core_neu_num_rd_act * CoreNum * ChipNum  

	if (if_out_final[data_flow[ol1_cp_id]]!=1):
		energy_wr_opt_L2 = core_neu_num_wr_opt * SRAM_energy(OL2) * psum_width 
	else: 
		energy_wr_opt_L2 = core_neu_num_wr_opt * SRAM_energy(OL2) * act_wgt_width 
	energy_rd_opt_L2 = core_neu_num_rd_opt * SRAM_energy(OL2) * psum_width
	energy_rd_wgt_L2 = core_neu_num_rd_wgt * SRAM_energy(WL2) * act_wgt_width
	energy_rd_act_L2 = core_neu_num_rd_act * SRAM_energy(AL2) * act_wgt_width

	# L2 用于统计通信总量 & prediction
	chip_pkt_num_wr_opt = 0; chip_neu_num_wr_opt = 0
	chip_pkt_num_rd_opt = 0; chip_neu_num_rd_opt = 0
	chip_pkt_num_rd_wgt = 0; chip_neu_num_rd_wgt = 0
	chip_pkt_num_rd_act = 0; chip_neu_num_rd_act = 0

	# L2 用于生成task file的变量
	chip_rd_out_data_num = 0
	chip_out_data_num = 0 
	chip_act_data_num = 0
	chip_wgt_data_num = 0

	cur = data_flow[ol2_cp_id]; inner = data_flow[ol2_cp_id-1]  
	if (if_out_final[cur]!=1): 
		#print("Chip: read opt mem ", OL2_need[inner],"repeat ",repeat_num[cur]) 
		chip_pkt_num_rd_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_psum)) * repeat_num[cur]
		chip_rd_out_data_num += OL2_need[inner]
		chip_neu_num_rd_opt += OL2_need[inner] * repeat_num[cur]
	else:
		chip_pkt_num_rd_opt += 0
		chip_rd_out_data_num += 0
		chip_neu_num_rd_opt += 0
	#print("Chip: write opt mem ", OL2_need[inner],"repeat ",repeat_num[cur])
	if (if_out_final[cur]!=1): 
		chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_psum)) *repeat_num[cur]
	else:
		chip_pkt_num_wr_opt += int(math.ceil(OL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
	chip_out_data_num += OL2_need[inner] # 用于生成仿真指令
	chip_neu_num_wr_opt += OL2_need[inner] * repeat_num[cur]
		
	cur = data_flow[al2_cp_id]; inner = data_flow[al2_cp_id-1]  
	#print("Chip: read act mem ",AL2_need[inner],"repeat ",repeat_num[cur])
	chip_pkt_num_rd_act +=  int(math.ceil(AL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt))*repeat_num[cur]
	chip_act_data_num += AL2_need[inner] # 用于生成仿真指令
	if al2_cp == "top":
		chip_neu_num_rd_act += AL2_need[data_flow[al2_cp_id]] * 1
	else:
		chip_neu_num_rd_act += AL2_need[data_flow[al2_cp_id]] * repeat_num[data_flow[al2_cp_id+1]]

	cur = data_flow[wl2_cp_id]; inner = data_flow[wl2_cp_id-1]  
	#print("Chip: read wgt mem ",WL2_need[inner],"repeat ",repeat_num[cur]) 
	chip_pkt_num_rd_wgt += int(math.ceil(WL2_need[inner]/flit_per_pkt/neu_per_flit_act_wgt)) *repeat_num[cur]
	chip_wgt_data_num += WL2_need[inner] # 用于生成仿真指令
	chip_neu_num_rd_wgt += WL2_need[inner] * repeat_num[cur]

	# 考虑上并行度带来的数据复用机会
	if if_multicast == 1:
		chip_neu_num_wr_opt = chip_neu_num_wr_opt * ChipNum  # 没有机会复用
		chip_neu_num_rd_opt = chip_neu_num_rd_opt * ChipNum 
		chip_neu_num_rd_wgt = chip_neu_num_rd_wgt * ChipNum /PP3 / PQ3
		chip_neu_num_rd_act = chip_neu_num_rd_act * ChipNum /PK3 
	elif if_multicast == 0:
		chip_neu_num_wr_opt = chip_neu_num_wr_opt * ChipNum  # 没有机会复用
		chip_neu_num_rd_opt = chip_neu_num_rd_opt * ChipNum 
		chip_neu_num_rd_wgt = chip_neu_num_rd_wgt * ChipNum  
		chip_neu_num_rd_act = chip_neu_num_rd_act * ChipNum 
	
	if (if_out_final[cur]!=1): 
		energy_wr_opt_dram = chip_neu_num_wr_opt * DRAM_energy_ratio * psum_width 
	else:
		energy_wr_opt_dram = chip_neu_num_wr_opt * DRAM_energy_ratio * act_wgt_width 
	energy_rd_opt_dram = chip_neu_num_rd_opt * DRAM_energy_ratio * psum_width
	energy_rd_wgt_dram = chip_neu_num_rd_wgt * DRAM_energy_ratio * act_wgt_width
	energy_rd_act_dram = chip_neu_num_rd_act * DRAM_energy_ratio * act_wgt_width

	runtime_calNum = runtimeP*runtimeQ*runtimeR*runtimeS*runtimeC*runtimeK
	runtime_list = [runtimeP, runtimeQ, runtimeC, runtimeK, runtimeChipNum, runtimeCoreNum,runtime_calNum]
	cp_list = [ol1_cp_id, al1_cp_id, wl1_cp_id, ol2_cp_id, al2_cp_id, wl2_cp_id]
	utilization_ratio_list = [ol1_utilization_ratio,al1_utilization_ratio,wl1_utilization_ratio, \
						      ol2_utilization_ratio,al2_utilization_ratio,wl2_utilization_ratio]
	energy_L1_list = [energy_rd_wgt_L1, energy_rd_act_L1]
	energy_dram_list = [energy_wr_opt_dram, energy_rd_opt_dram, energy_rd_wgt_dram, energy_rd_act_dram]
	energy_L2_list = [energy_wr_opt_L2, energy_rd_opt_L2, energy_rd_wgt_L2, energy_rd_act_L2]
	energy_die2die = 0;	energy_core2core = 0
	assert(DIE2DIE_energy_ratio!=NOC_energy_ratio)
	

	return(compuation_cycles,runtime_list,cp_list,utilization_ratio_list, \
		energy_dram_list, energy_L2_list,energy_L1_list, energy_MAC)


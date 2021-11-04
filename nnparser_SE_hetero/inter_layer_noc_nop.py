import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from matplotlib import pyplot as plt
import openpyxl

def getChipletID(p,q,k,dim_mapping_seq, debug_flag = 0):
	if debug_flag == 0:
		chiplet_id = p * dim_mapping_seq["P"] + q * dim_mapping_seq["Q"] + k * dim_mapping_seq["K"]
	else:
		chiplet_id = str([p,q,k])
	return chiplet_id

def extendChipletID(pre_id, k, k_offset, debug_flag = 0):
	if debug_flag == 0:
		chiplet_id = pre_id + k * k_offset
	else:
		chiplet_id = pre_id[:-2] + str(k) + "]"
	return chiplet_id

#获得包含的维度量
def getIndex(addr, offset, index_dict):
	dict_out = {}
	addr_start = addr
	addr_end = addr+offset-1
	for i in range(len(index_dict)-1):
		start = index_dict[i]
		end = index_dict[i+1] - 1
		if addr_start < start and addr_end >= start and addr_end <= end:
			dict_out[i] = addr_end - start + 1
		elif addr_start < start and addr_end > end:
			dict_out[i] = end - start + 1
		elif addr_start >= start and addr_end <= end:
			dict_out[i] = addr_end - addr_start + 1
		elif addr_start >= start and addr_start <=end and addr_end > end:
			dict_out[i] = end - addr_start + 1
	return dict_out

#获得当前chiplet所需之前chiplet的通信量
def getCommDict(addr, size_dict, dict, dim_mapping_seq, debug_flag):
	p = addr[0]
	q = addr[1]
	p_size = size_dict["P"]
	q_size = size_dict["Q"]
	k_size = size_dict["K"]

	p_dict = getIndex(p, p_size,dict["P"])
	q_dict = getIndex(q, q_size,dict["Q"])
	k_dict = getIndex(0, k_size,dict["K"])

	comm_dict = {}
	for q in q_dict:
		for p in p_dict:
			for k in k_dict:
				chiplet_id = getChipletID(p,q,k,dim_mapping_seq,debug_flag)
				comm_num = q_dict[q] * p_dict[p] * k_dict[k]
				comm_dict[chiplet_id] = comm_num
	return comm_dict

#计算层间通信
# network_param_1 = layer i : {"P":P, "Q":Q, "K":K, "C":C, "R":R, "S":S}
# network_param_2 = layer i+1 : {"P":P, "Q":Q, "K":K, "C":C, "R":R, "S":S, "stride":stride, "padding":padding}
# parallel_1 = layer i : {"P":PP,"Q":PQ,"K":PK}
# parallel_2 = layer i+1 : {"P":PP,"Q":PQ,"K":PK}
# chiplet_size = [chiplet_h, chiplet_w]
# dim_seq_1 = ["P","Q","K"]，相当于考量chiplet_id是先按哪个维度排列的
# debug_flag = 0 :正常运行，1 :方便看结果
def getInterLayerComm(dim_seq_1, dim_seq_2, network_param_1, parallel_1, network_param_2, parallel_2, debug_flag = 0):
	#---上一网络层的output act每个chiplet所拥有的大小

	P_i = math.ceil(network_param_1["P"]/parallel_1["P"])
	Q_i = math.ceil(network_param_1["Q"]/parallel_1["Q"])
	K_i = math.ceil(network_param_1["K"]/parallel_1["K"])
	single_chip_act_size_1 = {"P":P_i, "Q":Q_i, "K":K_i}
	
	#---获得有关chiplet id编号的信息
	dim_mapping_seq_1 = {"P":1,"Q":1,"K":1}
	par = 1
	for dim in dim_seq_1:
		dim_mapping_seq_1[dim] = par
		par *= parallel_1[dim]
	dim_mapping_seq_2 = {"P":1,"Q":1,"K":1}
	par = 1
	for dim in dim_seq_2:
		dim_mapping_seq_2[dim] = par
		par *= parallel_2[dim]

	#---当前层的output act与input act每个chiplet所拥有的大小
	stride = network_param_2["stride"]
	padding = network_param_2["padding"]
	P_output = math.ceil(network_param_2["P"]/parallel_2["P"])
	Q_output = math.ceil(network_param_2["Q"]/parallel_2["Q"])
	P_input = network_param_2["R"] + (P_output-1) * stride
	Q_input = network_param_2["S"] + (Q_output-1) * stride
	K_input = network_param_2["C"]
	single_chip_act_size_2 = {"P":P_input, "Q":Q_input, "K":K_input}

	#---记录每一块chiplet所需要的input act的起始块地址offset
	chiplet_act_index_dict = {}
	chiplet_act_index_dict_single = {}
	for q in range(parallel_2["Q"]):
		for p in range(parallel_2["P"]):
			chiplet_index = getChipletID(p,q,0,dim_mapping_seq_2,debug_flag)
			offset_p = P_output * p *stride - padding
			offset_q = Q_output * q *stride - padding
			offset = [offset_p, offset_q]
			chiplet_act_index_dict[chiplet_index] = offset
	
	#---记录前一层chiplet的output act分配情况
	chiplet_act_index_dict_pre = {"P":{},"Q":{},"K":{}}
	for p in range(parallel_1["P"]):
		offset_p = P_i * p
		chiplet_act_index_dict_pre["P"][p] = offset_p
	chiplet_act_index_dict_pre["P"][parallel_1["P"]] = network_param_1["P"]

	for q in range(parallel_1["Q"]):
		offset_q = Q_i * q
		chiplet_act_index_dict_pre["Q"][q] = offset_q
	chiplet_act_index_dict_pre["Q"][parallel_1["Q"]] = network_param_1["Q"]

	for k in range(parallel_1["K"]):
		offset_k = K_i * k
		chiplet_act_index_dict_pre["K"][k] = offset_k
	chiplet_act_index_dict_pre["K"][parallel_1["K"]] = network_param_1["K"]

	#---记录当前chiplet与上一层chiplet的通信情况
	#---comm_inter_layer_dict_o_i[layer(i+1)_chiplet_id] = {[layer(i)_chiplet_id]:comm_num,...}
	#---comm_inter_layer_dict_i_o[layer(i)_chiplet_id] = {[layer(i+1)_chiplet_id]:comm_num,...}
	comm_inter_layer_dict_o_i = {}
	comm_inter_layer_dict_i_o = {}
	for chip_id in chiplet_act_index_dict:
		index = chiplet_act_index_dict[chip_id]
		comm_dict = getCommDict(index, single_chip_act_size_2, chiplet_act_index_dict_pre, dim_mapping_seq_1, debug_flag)
		comm_inter_layer_dict_o_i[chip_id] = comm_dict
		for chip_id_i in comm_dict:
			if chip_id_i not in comm_inter_layer_dict_i_o:
				comm_inter_layer_dict_i_o[chip_id_i] = {}
			comm_inter_layer_dict_i_o[chip_id_i][chip_id] = comm_dict[chip_id_i]
	
	comm_dict = {}
	for chip_id_i in comm_inter_layer_dict_i_o:
		comm_dict[chip_id_i] = {}
		num = 0
		for chip_id_o in comm_inter_layer_dict_i_o[chip_id_i]:
			chip_id_list = []
			for k in range(parallel_2["K"]):
				chip_id_o_ex = extendChipletID(chip_id_o, k, dim_mapping_seq_2["K"], debug_flag)
				if debug_flag == 1:
					chip_id_list.append(chip_id_o_ex)
				else:
					if chip_id_o_ex != chip_id_i:
						chip_id_list.append(chip_id_o_ex)
			comm_dict[chip_id_i][num] = [comm_inter_layer_dict_i_o[chip_id_i][chip_id_o], chip_id_list]
			num += 1

	return comm_dict

if __name__ == '__main__':
	# 1表示前一层，2表示当前层
	network_param_1 = {"P":64,"Q":64,"C":4,"K":8,"R":3,"S":3}
	network_param_2 = {"P":32,"Q":32,"C":8,"K":16,"R":3,"S":3, "stride":2, "padding":1}
	parallel_1 = {"P":4,"Q":4,"K":1}
	parallel_2 = {"P":1,"Q":4,"K":4}
	dim_seq_1 = ["Q","P","K"]
	dim_seq_2 = ["P","Q","K"]
	dict = getInterLayerComm(dim_seq_1, dim_seq_2, network_param_1, parallel_1, network_param_2, parallel_2, 0)
	for id in dict:
		print(id, " : ",dict[id])
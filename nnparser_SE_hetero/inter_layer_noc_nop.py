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
			p_index = P_output * p
			q_index = Q_output * q
			if p_index >= network_param_2["P"]:
				break
			chiplet_index = getChipletID(p,q,0,dim_mapping_seq_2,debug_flag)
			offset_p = P_output * p *stride - padding
			offset_q = Q_output * q *stride - padding
			offset = [offset_p, offset_q]
			chiplet_act_index_dict[chiplet_index] = offset
	#print("chiplet_act_index_dict")
	#print(chiplet_act_index_dict)
	#---记录前一层chiplet的output act分配情况
	chiplet_act_index_dict_pre = {"P":{},"Q":{},"K":{}}
	for p in range(parallel_1["P"]):
		offset_p = P_i * p
		if offset_p < network_param_1["P"]:
			chiplet_act_index_dict_pre["P"][p] = offset_p
		else:
			#chiplet_act_index_dict_pre["P"][p] = network_param_1["P"]
			break
	if p in chiplet_act_index_dict_pre["P"]:
		chiplet_act_index_dict_pre["P"][p+1] = network_param_1["P"]
	else:
		chiplet_act_index_dict_pre["P"][p] = network_param_1["P"]

	for q in range(parallel_1["Q"]):
		offset_q = Q_i * q
		chiplet_act_index_dict_pre["Q"][q] = offset_q
		if offset_q < network_param_1["Q"]:
			chiplet_act_index_dict_pre["Q"][q] = offset_q
		else:
			break
	if q in chiplet_act_index_dict_pre["Q"]:
		chiplet_act_index_dict_pre["Q"][q+1] = network_param_1["Q"]
	else:
		chiplet_act_index_dict_pre["Q"][q] = network_param_1["Q"]

	for k in range(parallel_1["K"]):
		offset_k = K_i * k
		chiplet_act_index_dict_pre["K"][k] = offset_k
		if offset_k < network_param_1["K"]:
			chiplet_act_index_dict_pre["K"][k] = offset_k
		else:
			break
	if k in chiplet_act_index_dict_pre["K"]:
		chiplet_act_index_dict_pre["K"][k+1] = network_param_1["K"]
	else:
		chiplet_act_index_dict_pre["K"][k] = network_param_1["K"]

	#print("chiplet_act_index_dict_pre")
	#print(chiplet_act_index_dict_pre)

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
	#print("comm_inter_layer_dict_o_i")
	#print(comm_inter_layer_dict_o_i)
	#print("comm_inter_layer_dict_i_o")
	#print(comm_inter_layer_dict_i_o)

	comm_num_dict = {}
	comm_type_dict = {"uni-cast":0, "multi-cast":0, "broadcast":0}
	comm_type_times_dict = {"uni-cast":0, "multi-cast":0, "broadcast":0}
	for chip_id_i in comm_inter_layer_dict_i_o:
		comm_num_dict[chip_id_i] = {}
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

			commNum = comm_inter_layer_dict_i_o[chip_id_i][chip_id_o]
			if len(chip_id_list) == 1:
				comm_type_dict["uni-cast"] += commNum
				comm_type_times_dict["uni-cast"] += 1
			elif len(chip_id_list) + 1 < (parallel_2["P"]*parallel_2["Q"]*parallel_2["K"]) and len(chip_id_list) > 1:
				comm_type_dict["multi-cast"] += commNum
				comm_type_times_dict["multi-cast"] += 1
			elif len(chip_id_list) + 1 == (parallel_2["P"]*parallel_2["Q"]*parallel_2["K"]):
				comm_type_dict["broadcast"] += commNum
				comm_type_times_dict["broadcast"] += 1
			comm_num_dict[chip_id_i][num] = [commNum, chip_id_list]
			num += 1
	#print("comm_num_dict")
	#print(comm_num_dict)
	#print("comm_type_dict")
	#print(comm_type_dict)
	#print("comm_type_times_dict")
	#print(comm_type_times_dict)

	return comm_num_dict, comm_type_dict,comm_type_times_dict

if __name__ == '__main__':

	excel_datas = []

	# 1表示前一层，2表示当前层
	network_param_1 = {"P":224,"Q":224,"C":3,"K":64,"R":3,"S":3}
	network_param_2 = {"P":224,"Q":224,"C":64,"K":64,"R":3,"S":3, "stride":1, "padding":1}
	
	# parallel_type_1 = {"P":{"P":16,"Q":1,"K":1},'Q':{"P":1,"Q":16,"K":1},'K':{"P":1,"Q":1,"K":16},"PQ":{"P":4,"Q":4,"K":1},'PK':{"P":4,"Q":1,"K":4},'QK':{"P":1,"Q":4,"K":4}}
	# parallel_type_2 = {"P":{"P":16,"Q":1,"K":1},'Q':{"P":1,"Q":16,"K":1},'K':{"P":1,"Q":1,"K":16},"PQ":{"P":4,"Q":4,"K":1},'PK':{"P":4,"Q":1,"K":4},'QK':{"P":1,"Q":4,"K":4}}
	parallel_type_1 = {"P":{"P":16,"Q":1,"K":1},'K':{"P":1,"Q":1,"K":16},'PK':{"P":4,"Q":1,"K":4}}
	parallel_type_2 = {"P":{"P":16,"Q":1,"K":1},'K':{"P":1,"Q":1,"K":16},'PK':{"P":4,"Q":1,"K":4}}

	times = 0
	for type2 in parallel_type_2:
		for type1 in parallel_type_1:
			parallel_1 = parallel_type_1[type1]
			parallel_2 = parallel_type_2[type2]
			dim_seq_1 = ["K","P","Q"]
			dim_seq_2 = ["K","P","Q"]
			print("----------- times = ", times)
			print("parallel_1")
			print(parallel_1)
			print("parallel_2")
			print(parallel_2)

			dict1, type_dict1, type_dict2 = getInterLayerComm(dim_seq_1, dim_seq_2, network_param_1, parallel_1, network_param_2, parallel_2, 0)
			
			excel_datas.append([type1+"-"+type2, type1, type2, type_dict2["uni-cast"], type_dict2["multi-cast"], type_dict2["broadcast"], type_dict1["uni-cast"], type_dict1["multi-cast"], type_dict1["broadcast"] ])
			print(dict1)
			times += 1
	
	workbook = openpyxl.Workbook()
	sheet = workbook.get_sheet_by_name('Sheet') 
	# 写入标题
	column_tite = ["parallel-type","parallel-type-i","parallel-type-i+1", "times-uni-cast","times-multi-cast","times-broadcast","num-uni-cast","num-multi-cast","num-broadcast"]
	for col,column in enumerate(column_tite):
		sheet.cell(1, col+1, column)
	# 写入每一行
	for row, data in enumerate(excel_datas):
		for col, column_data in enumerate(data):
			sheet.cell(row+2, col+1, column_data)

	setname = str(network_param_1["P"]) + "_" + str(network_param_1["Q"]) + "_" + str(network_param_2["P"]) + "_" + str(network_param_2["Q"]) + "_" + str(network_param_2["K"]) + "_" + str(network_param_2["C"])

	workbook.save("inter_layer_output_"+setname+".xls")
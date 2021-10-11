import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from basicParam import *

# 质因数分解
def getZhiyinShu(num, list):
	isZhishu = True
	i = 2
	square = int(math.sqrt(num)) + 1
	while i <= square:
		if num % i == 0:
			list.append(i)
			isZhishu = False
			getZhiyinShu(num / i , list)
			i += 1
			break
		i += 1
	if isZhishu:
		list.append(int(num))

# 将一个数拆成三个整数相乘
def setPartition_1(num, dim):
	par_list = []
	getZhiyinShu(num, par_list)
	print(par_list)
	par_dim = [1, 1, 1]
	for i in par_list:
		ran = random.randint(1, dim)
		par_dim[ran-1] *= i
	return par_dim


# 将num拆分为dim个整数相乘（大于等于）
def setPartition(num, dim):
	list = []
	num_i = num
	for i in range(0, dim - 1):
		ran = random.randint(1, math.ceil(num_i))
		list.append(ran)
		num_i = num_i / ran

	list.append(math.ceil(num_i))
	random.shuffle(list)
	return list

# 将P_num个并行量进行拆分，随机选两个维度d1、d2，以及对应的并行量p1、p2.
def setParallel(P_num):
	dim = len(parallel_select)
	index = math.ceil(math.log(P_num, 2))
	p1 = random.randint(0,index)
	p2 = index - p1
	p1 = pow(2,p1)
	p2 = pow(2,p2)
	d1 = random.randint(0,dim-1)
	d2 = random.randint(0,dim-1)
	while d2 == d1:
		d2 = random.randint(0,dim-1)
	d1 = parallel_select[d1]
	d2 = parallel_select[d2]
	return d1, d2, p1, p2

# 只对并行量P_num拆分为p1、p2，不决定其维度
def setParallelNum(P_num):
	index = math.ceil(math.log(P_num, 2))
	p1 = random.randint(0,index)
	p2 = index - p1
	p1 = pow(2,p1)
	p2 = pow(2,p2)
	return p1, p2

# 获得for拆分的code
def getPartitionChild():
	size_i = []
	parallel_num_set = []
	parallel_dim_set = []
	for i in range(4):
		size_i.append(size[i])

	#---并行度拆分（空间并行）---
	if Par_type == 4:
		d1, d2, p1, p2 = setParallel(PEs)
		parallel_dim_set.append(d1)
		parallel_dim_set.append(d2)
		parallel_num_set.append(p1)
		parallel_num_set.append(p2)
		size_i[d1] = math.ceil(size_i[d1]/p1)
		size_i[d2] = math.ceil(size_i[d2]/p2)
		d1, d2, p1, p2 = setParallel(Chiplets)
		parallel_dim_set.append(d1)
		parallel_dim_set.append(d2)
		parallel_num_set.append(p1)
		parallel_num_set.append(p2)
		size_i[d1] = math.ceil(size_i[d1]/p1)
		size_i[d2] = math.ceil(size_i[d2]/p2)
	elif Par_type == 3:
		parallel_dim_set.append(3)
		parallel_dim_set.append(2)
		parallel_dim_set.append(3)
		parallel_dim_set.append(2)
		p1, p2 = setParallelNum(PEs)
		parallel_num_set.append(p1)
		parallel_num_set.append(p2)
		size_i[3] = math.ceil(size_i[3]/p1)
		size_i[2] = math.ceil(size_i[2]/p2)
		p1, p2 = setParallelNum(Chiplets)
		parallel_num_set.append(p1)
		parallel_num_set.append(p2)
		size_i[3] = math.ceil(size_i[3]/p1)
		size_i[2] = math.ceil(size_i[2]/p2)

	#---维度拆分（时间并行）---
	ran1 = random.randint(0,2)
	print("random1 = ", ran1)
	if ran1 == 0:
		Pset = setPartition_1(size_i[0], 3)
		Qset = setPartition_1(size_i[1], 3)
		Cset = setPartition_1(size_i[2], 3)
		Kset = setPartition_1(size_i[3], 3)
	else:
		Pset = setPartition(size_i[0], 3)
		Qset = setPartition(size_i[1], 3)
		Cset = setPartition(size_i[2], 3)
		Kset = setPartition(size_i[3], 3)
	code = parallel_dim_set + parallel_num_set + Pset + Qset + Cset + Kset # [Kp1,CP1,Kp2,Cp2,C1~C3,K1~K3,P1~P3,Q1~Q3]
	return code

# 获得for循环顺序的code
def getOrderCode():
	list1 = list(range(6))
	list2 = list(range(4))
	list3 = list(range(4))

	random.shuffle(list1)
	random.shuffle(list2)
	random.shuffle(list3)
	code = list1 + list2 + list3
	return code

# 获得GA code（for拆分+for_loop顺序）
def getChild():
	code1 = getPartitionChild()
	code2 = getOrderCode()
	code = code1 + code2
	return code

# 把GA code进行解析，解析成{[for_type, dim, for_num]}
def codeParse(code):
	parse_dict = {}
	loop_order = code[20:]
	parallel_code = code[4:8]
	parallel_dim = code[0:4]
	for_code = {}
	for_code[0] = code[8:11]
	for_code[1] = code[11:14]
	for_code[2] = code[14:17]
	for_code[3] = code[17:20]
	line_num = 0
	for i in range(3):
		ii = 3 - i
		if i == 0:
			for j in range(6):
				dim = loop_order[j]
				if dim == 4:
					num = R
				elif dim == 5:
					num = S
				else:
					num = for_code[dim][i]
				
				parse_dict[line_num] = [0,dim,num]
				line_num += 1
		else:
			for j in range(4):
				dim = loop_order[i*4+2 + j]
				num = for_code[dim][i]

				parse_dict[line_num] = [0,dim,num]
				line_num += 1

		if i == 0:
			if parallel_code[0] > 1:
				parse_dict[line_num] = [1,parallel_dim[0],parallel_code[0]]
				line_num += 1
			if parallel_code[1] > 1:
				parse_dict[line_num] = [1,parallel_dim[1],parallel_code[1]]
				line_num += 1
		if i == 1:
			if parallel_code[2] > 1:
				parse_dict[line_num] = [2,parallel_dim[2],parallel_code[2]]
				line_num += 1
			if parallel_code[3] > 1:
				parse_dict[line_num] = [2,parallel_dim[3],parallel_code[3]]
				line_num += 1
	
	#parse_dict[line_num] = [0,3,K0]
	#line_num += 1
	#parse_dict[line_num] = [0,2,C0]

	return parse_dict

# 输出基本参数
def printBasicSet():
	print(" ")
	print("basic set :")
	print("P = " + str(P))
	print("Q = " + str(Q))
	print("C = " + str(C))
	print("K = " + str(K))
	print("Chiplets = " + str(Chiplets))
	print("PEs = " + str(PEs))
	print("C0 = " + str(C0))
	print("K0 = " + str(K0))
	print("")

# 输出For循环
def printParseDict(dict):
	for i in reversed(range(len(dict))):
		type = dict[i][0]
		dim = dict[i][1]
		num = dict[i][2]
		if num > 1:
			line = type_list[type] + " " + dim_list[dim] + " in [0 : " + str(num) + "):"
			print(line)

# 获得计算核心对于act与wgt的共享情况
# para1是activation的组数，para2是weight的组数
def getPEDistribution16(para1, para2):
	assert(para1*para2 == 16)
	act_PE_dict = {}
	wgt_PE_dict = {}

	#---act + wgt send节点列表---
	act_PE_dict["send"] = {0:[mem_16["a"]]}
	wgt_PE_dict["send"] = {0:[mem_16["w"]]}

	#---act recv节点列表---
	if para1 == 1:
		act_PE_dict["recv"] = set_1_e_16[0]
	elif para1 == 2:
		ran = random.randint(0,len(set_2_e_8)-1)
		act_PE_dict["recv"] = set_2_e_8[ran]
	elif para1 == 4:
		ran = random.randint(0,len(set_4_e_4_1)-1)
		act_PE_dict["recv"] = set_4_e_4_1[ran]
	elif para1 == 8:
		ran = random.randint(0,len(set_8_e_2)-1)
		act_PE_dict["recv"] = set_8_e_2[ran]
	elif para1 == 16:
		act_PE_dict["recv"] = set_16_e_1[0]
	else:
		print("act parallel error!!!")
	
	#---wgt recv节点列表---
	if para2 == 1:
		wgt_PE_dict["recv"] = set_1_e_16[0]
	elif para2 == 2:
		wgt_PE_dict["recv"] = set_2_e_8[ran]
	elif para2 == 4:
		wgt_PE_dict["recv"] = set_4_e_4_2[ran]
	elif para2 == 8:
		wgt_PE_dict["recv"] = set_8_e_2[ran]
	elif para2 == 16:
		wgt_PE_dict["recv"] = set_16_e_1[0]
	else:
		print("act parallel error!!!")
	return act_PE_dict, wgt_PE_dict

def getPEDistribution4(para1, para2):
	assert(para1*para2 == 4)
	act_PE_dict = {}
	wgt_PE_dict = {}

	#---act + wgt send节点列表---
	act_PE_dict["send"] = {0:[mem_4["a"]]}
	wgt_PE_dict["send"] = {0:[mem_4["w"]]}

	#---act recv节点列表---
	if para1 == 1:
		act_PE_dict["recv"] = set_1_e_4[0]
	elif para1 == 2:
		ran = random.randint(0,len(set_2_e_2_1)-1)
		act_PE_dict["recv"] = set_2_e_2_1[ran]
	elif para1 == 4:
		ran = random.randint(0,len(set_4_e_1)-1)
		act_PE_dict["recv"] = set_4_e_1[0]
	else:
		print("act parallel error!!!")
	
	#---wgt recv节点列表---
	if para2 == 1:
		wgt_PE_dict["recv"] = set_1_e_4[0]
	elif para2 == 2:
		wgt_PE_dict["recv"] = set_2_e_2_2[ran]
	elif para2 == 4:
		wgt_PE_dict["recv"] = set_4_e_1[0]
	else:
		print("act parallel error!!!")
	return act_PE_dict, wgt_PE_dict

# 给dict的每个元素加上固定偏移量num
# dict = {"send":{0:[],...},"recv":{0:[]...}}
def dictAddInt(dict, num):
	send = dict["send"]
	recv = dict["recv"]
	for i in send:
		for x in range(len(send[i])):
			send[i][x] += num

	for i in recv:
		for x in range(len(recv[i])):
			recv[i][x] +=  num
	dict1 = copy.deepcopy({"send":send,"recv":recv})
	#print(send)
	return dict1

# 转换为chiplet
def dictChipletChange(dict, flag1, flag2):
	send = dict["send"]
	recv = dict["recv"]
	if Chiplets == 16:
		NoP2NoCnode = NoP2NoCnode_16
	elif Chiplets == 4:
		NoP2NoCnode = NoP2NoCnode_4
	for i in send:
		for x in range(len(send[i])):
			num = send[i][x]
			print(num)
			send[i][x] = NoP2NoCnode[num] + A_W_offset[flag1]
	for i in recv:
		for x in range(len(recv[i])):
			num = recv[i][x]
			recv[i][x] = NoP2NoCnode[num] + A_W_offset[flag2]
	dict1 = copy.deepcopy({"send":send,"recv":recv})
	#print(send)
	return dict1

# type = 0 : NoC ; type = 1 : NoP
def getPEExtent(dict, type=0, flag1 = 0, flag2 = 0):
	list = []
	#list.append(dict)
	if Chiplets == 16:
		NoC_node_offset = NoC_node_offset_16
	elif Chiplets == 4:
		NoC_node_offset = NoC_node_offset_4
	if type == 0:
		for i in NoC_node_offset:
			dict1 = copy.deepcopy(dict)
			dict1 = dictAddInt(dict1, i)		
			list.append(dict1)
	else:
		dict1 = copy.deepcopy(dict)
		dict1 = dictChipletChange(dict1,flag1,flag2)
		return dict1
	return list

# 获得输出特征图的数据节点通信关系
def getOutputDict():
	if PEs == 16:
		rd_out_PE_dict_temp = {"send":{0:[0]},"recv":set_16_e_1[0]}
		wr_out_PE_dict_temp = {"send":set_16_e_1[0],"recv":{0:[0]}}
	elif PEs == 4:
		rd_out_PE_dict_temp = {"send":{0:[0]},"recv":set_4_e_1[0]}
		wr_out_PE_dict_temp = {"send":set_4_e_1[0],"recv":{0:[0]}}
	if Chiplets == 16:
		rd_out_Chiplet_dict_temp = {"send":{0:[0]},"recv":set_16_e_1[0]}
		wr_out_Chiplet_dict_temp = {"send":set_16_e_1[0],"recv":{0:[0]}}
	elif Chiplets == 4:
		rd_out_Chiplet_dict_temp = {"send":{0:[0]},"recv":set_4_e_1[0]}
		wr_out_Chiplet_dict_temp = {"send":set_4_e_1[0],"recv":{0:[0]}}
	rd_out_PE_dict = getPEExtent(rd_out_PE_dict_temp)
	wr_out_PE_dict = getPEExtent(wr_out_PE_dict_temp)
	rd_out_Chiplet_dict = getPEExtent(rd_out_Chiplet_dict_temp,1,"o","o")
	wr_out_Chiplet_dict = getPEExtent(wr_out_Chiplet_dict_temp,1,"o","o")
	return rd_out_PE_dict, wr_out_PE_dict, rd_out_Chiplet_dict, wr_out_Chiplet_dict

# 解析parse，得到对应的输出
def parseChange(parse):
	data_flow = []
	data_flow_dim = []
	ol1_ratio = []
	al1_ratio = []
	wl1_ratio = []
	all_param = []
	out_final = []
	if_act_share_PE = 0
	if_wgt_share_PE = 0
	if_act_share_Chiplet = 0
	if_wgt_share_Chiplet = 0
	act_share_PE = [1,1] # [组数, 组内元素数目]
	wgt_share_PE = [1,1]
	act_share_Chiplet = [1,1]
	wgt_share_Chiplet = [1,1]
	parallel_dim_list = {0:[1,1,1],1:[1,1,1]}
	index_PQKC = [1,1,1,1]
	
	for i in range(len(parse)):
		for_type = parse[i][0]
		dim = parse[i][1]
		num = parse[i][2]
		if for_type == 0:
			if dim < 4:
				data_flow.append(dim_list[dim]+str(index_PQKC[dim]))
				index_PQKC[dim] += 1
			else:
				data_flow.append(dim_list[dim])
			data_flow_dim.append(dim)
			
			if O_correlation[dim] == 1:
				ol1_ratio.append(num)
			else:
				ol1_ratio.append(1)
			
			if A_correlation[dim] == 1:
				al1_ratio.append(num)
			else:
				al1_ratio.append(1)

			if W_correlation[dim] == 1:
				wl1_ratio.append(num)
			else:
				wl1_ratio.append(1)
			all_param.append(num)
		elif for_type == 1:
			if dim == 0:
				parallel_dim_list[0][0] = num
			elif dim == 1:
				parallel_dim_list[0][1] = num
			elif dim == 3:
				parallel_dim_list[0][2] = num

			if A_correlation[dim] == 0:
				if_act_share_PE = 1
				act_share_PE[1] *= num
			else:
				act_share_PE[0] *= num
			
			if W_correlation[dim] == 0:
				if_wgt_share_PE = 1
				wgt_share_PE[1] *= num
			else:
				wgt_share_PE[0] *= num
		elif for_type == 2:
			if dim == 0:
				parallel_dim_list[1][0] = num
			elif dim == 1:
				parallel_dim_list[1][1] = num
			elif dim == 3:
				parallel_dim_list[1][2] = num

			if A_correlation[dim] == 0:
				if_act_share_Chiplet = 1
				act_share_Chiplet[1] *= num
			else:
				act_share_Chiplet[0] *= num

			if W_correlation[dim] == 0:
				if_wgt_share_Chiplet = 1
				wgt_share_Chiplet[1] *= num
			else:
				wgt_share_Chiplet[0] *= num

	out_id = 0		
	for i in reversed(range(len(ol1_ratio))):
		if ol1_ratio[i] == 1:
			out_id = i
			break
	for i in range(len(ol1_ratio)):
		if i < out_id:
			out_final.append(0)
		else:
			out_final.append(1)
	data_flow.append("top")
	ol1_ratio.append(1)
	al1_ratio.append(1)
	wl1_ratio.append(1)
	all_param.append(1)
	out_final.append(1)
	if PEs == 16:
		act_PE_dict_temp, wgt_PE_dict_temp = getPEDistribution16(act_share_PE[0],wgt_share_PE[0])
	elif PEs == 4:
		act_PE_dict_temp, wgt_PE_dict_temp = getPEDistribution4(act_share_PE[0],wgt_share_PE[0])
	if Chiplets == 16:
		act_Chiplet_dict_temp, wgt_Chiplet_dict_temp = getPEDistribution16(act_share_Chiplet[0],wgt_share_Chiplet[0])
	elif Chiplets == 4:
		act_Chiplet_dict_temp, wgt_Chiplet_dict_temp = getPEDistribution4(act_share_Chiplet[0],wgt_share_Chiplet[0])
	act_PE_dict = getPEExtent(act_PE_dict_temp)
	wgt_PE_dict = getPEExtent(wgt_PE_dict_temp)
	act_Chiplet_dict = getPEExtent(act_Chiplet_dict_temp,1,"o","a")
	wgt_Chiplet_dict = getPEExtent(wgt_Chiplet_dict_temp,1,"o","w")
	return data_flow, ol1_ratio, al1_ratio, wl1_ratio, all_param, out_final, if_act_share_PE, if_wgt_share_PE, if_act_share_Chiplet, if_wgt_share_Chiplet, act_PE_dict, wgt_PE_dict, act_Chiplet_dict, wgt_Chiplet_dict, parallel_dim_list

def printOut(for_list, act_wgt_dict, parallel_dim_list, out_dict):
	print("data_flow = ",for_list[0])
	print("ol1_ratio = ",for_list[1])
	print("al1_ratio = ",for_list[2])
	print("wl1_ratio = ",for_list[3])
	print("all_param = ",for_list[4])
	print("out_final = ",for_list[5])
	print("if_act_share_PE = ",for_list[6])
	print("if_wgt_share_PE = ",for_list[7])
	print("if_act_share_Chiplet = ",for_list[8])
	print("if_wgt_share_Chiplet = ",for_list[9])
	print("parallel_dim_list = ",parallel_dim_list)
	print("")
	print("act_PE_dict = ",act_wgt_dict["act_core"])
	print("")
	print("wgt_PE_dict = ",act_wgt_dict["wgt_core"])
	print("")
	print("act_Chiplet_dict = ",act_wgt_dict["act_chiplet"])
	print("")
	print("wgt_Chiplet_dict = ",act_wgt_dict["wgt_chiplet"])
	print("")
	print("rd_out_PE_dict = ",out_dict["rd_core"])
	print(" ")
	print("wr_out_PE_dict = ",out_dict["wr_core"])
	print(" ")
	print("rd_out_Chiplet_dict = ",out_dict["rd_chip"])
	print(" ")
	print("wr_out_Chiplet_dict = ",out_dict["wr_chip"])

# act_PE_dict = {0:[],1:[]}，列表内为act复用的PE list， 0、1代表不同的组
def GaGetChild():
	printBasicSet()
	code = getChild()
	parse = codeParse(code)
	printParseDict(parse)

	#---获得PQKC的拆分---
	partition_list = {"P":[],"Q":[],"K":[],"C":[]}
	partition_list["P"] = code[8:11]
	partition_list["Q"] = code[11:14]
	partition_list["C"] = code[14:17]
	partition_list["K"] = code[17:20]

	#---data_flow, ol1_ratio, al1_ratio, wl1_ratio, all_param, out_final, if_act_share_PE, if_wgt_share_PE, if_act_share_Chiplet, if_wgt_share_Chiplet---
	for_list = {}
	#---act_PE_dict, wgt_PE_dict, act_Chiplet_dict, wgt_Chiplet_dict---
	act_wgt_dict = {}
	for_list[0], for_list[1], for_list[2], for_list[3], for_list[4], for_list[5], for_list[6], for_list[7], for_list[8], for_list[9], act_wgt_dict["act_core"], act_wgt_dict["wgt_core"], act_wgt_dict["act_chiplet"], act_wgt_dict["wgt_chiplet"], parallel_dim_list = parseChange(parse)
	#data_flow, ol1_ratio, al1_ratio, wl1_ratio, all_param, out_final, if_act_share_PE, if_wgt_share_PE, if_act_share_Chiplet, if_wgt_share_Chiplet, act_PE_dict, wgt_PE_dict, act_Chiplet_dict, wgt_Chiplet_dict, parallel_dim_list = parseChange(parse)
	
	#---rd_out_PE_dict, wr_out_PE_dict, rd_out_Chiplet_dict, wr_out_Chiplet_dict---
	out_dict = {}
	out_dict["rd_core"], out_dict["wr_core"], out_dict["rd_chip"], out_dict["wr_chip"] = getOutputDict()
	#rd_out_PE_dict, wr_out_PE_dict, rd_out_Chiplet_dict, wr_out_Chiplet_dict = getOutputDict()

	printOut(for_list, act_wgt_dict, parallel_dim_list, out_dict)

	return for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list
	#return data_flow, ol1_ratio, al1_ratio, wl1_ratio, all_param, out_final, if_act_share_PE, if_wgt_share_PE, if_act_share_Chiplet, if_wgt_share_Chiplet, act_PE_dict, wgt_PE_dict, act_Chiplet_dict, wgt_Chiplet_dict, rd_out_PE_dict, wr_out_PE_dict, rd_out_Chiplet_dict, wr_out_Chiplet_dict

#for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list = GaGetChild()
#print(parallel_dim_list)

import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from basicParam_noc_nop import *

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
def setParallel(P_num, level):
	list = setPartition_1(P_num, 2)
	p1 = list[0]
	p2 = list[1]
	dim = len(parallel_select[level])
	d1 = random.randint(0,dim-1)
	d2 = random.randint(0,dim-1)	
	d1 = parallel_select[level][d1]
	d2 = parallel_select[level][d2]
	return d1, d2, p1, p2

# 只对并行量P_num拆分为p1、p2，不决定其维度
def setParallelNum(P_num):
	index = math.ceil(math.log(P_num, 2))
	p1 = random.randint(0,index)
	p2 = index - p1
	p1 = pow(2,p1)
	p2 = pow(2,p2)
	return p1, p2


class GaEncode:
	def __init__(self, network_param, HW_param, debug=0, parallel_level=3, debug_file="./random_test_record.txt"):
		self.HW_param = HW_param
		self.Chiplets = HW_param["Chiplet"]
		self.PEs = HW_param["PE"]
		self.intra_PE = HW_param["intra_PE"]		#["C":4,"K":4]
		self.parallel_level = parallel_level

		self.network_param = network_param
		self.size = {}
		self.getSize()
		self.P = self.size["P"]
		self.Q = self.size["Q"]
		self.C = self.size["C"]
		self.K = self.size["K"]
		self.R = self.size["R"]
		self.S = self.size["S"]

		self.NoC_node_offset = []
		self.NoP2NoCnode = []
		self.A_W_offset = {}
		self.setNodeID()

		#self.PEmappingSet = {}
		#self.Chipmappingset = {}
		#self.setmapppingSet("Chiplet")
		#self.setmapppingSet("PE")


		self.debug = debug
		self.debug_file = debug_file

		#Dim_result = {"P":[],"Q":[],"K":[],"C":[]}
		#loop_order = {"PE":[],"Chiplet":[],"Pakage":[]}

	def getSize(self):
		for i in self.intra_PE:
			num = math.ceil(self.network_param[i] / self.intra_PE[i])
			self.size[i] = num
		for i in self.network_param:
			if i not in self.size:
				self.size[i] = self.network_param[i]

	def setNodeID(self):
		Chiplet_lenth = int(self.Chiplets ** 0.5)
		PE_lenth = int(self.PEs ** 0.5)
		assert(Chiplet_lenth*Chiplet_lenth == self.Chiplets)
		assert(PE_lenth*PE_lenth == self.PEs)

		if PE_lenth == 2:
			self.A_W_offset["o"] = 0
			self.A_W_offset["a"] = PE_lenth + 1
			self.A_W_offset["w"] = PE_lenth + 1
			self.A_W_offset["noc-chiplet"] = 0
		else:
			assert(PE_lenth > 1)
			self.A_W_offset["o"] = 0
			self.A_W_offset["a"] = PE_lenth + 1
			self.A_W_offset["w"] = (PE_lenth + 1) * 2
			self.A_W_offset["noc-chiplet"] = 0
		PE_num = PE_lenth * (PE_lenth+1)

		num = 0
		for i in range(self.Chiplets):
			if i % Chiplet_lenth == 0: 
				self.NoP2NoCnode.append(0)
			num += PE_num
			self.NoC_node_offset.append(num)
			self.NoP2NoCnode.append(num)

			if (i+1) % Chiplet_lenth == 0:
				num += PE_num

	def setmappingSet(self, num, set1, set2):
		assert(num == set1*set2)
		list1 = {}
		list2 = {}
		lenth = int(num ** 0.5)
		node_list = []
		ID = 0
		for i in range(num):
			if i % lenth == 0:
				ID += 1
			node_list.append(ID)
			ID += 1
		for i in range(num):
			set1_id = i // set2
			if set1_id not in list1:
				list1[set1_id] = []
			list1[set1_id].append(node_list[i])

		for i in range(num):
			set2_id = i // set1
			if set2_id not in list2:
				list2[set2_id] = []
			list2[set2_id].append(list1[i % set1][set2_id])
		return list1, list2

	# 获得for拆分的code
	def getPartitionChild(self):
		size_i = []
		parallel_num_set = []
		parallel_dim_set = []
		for i in range(4):
			dim_num = dim_list[i]
			size_i.append(self.size[dim_num])

		#---并行度拆分（空间并行）---
		if Par_type == 4:
			d1, d2, p1, p2 = setParallel(self.PEs, "PE")
			parallel_dim_set.append(d1)
			parallel_dim_set.append(d2)
			parallel_num_set.append(p1)
			parallel_num_set.append(p2)
			size_i[d1] = math.ceil(size_i[d1]/p1)
			size_i[d2] = math.ceil(size_i[d2]/p2)
			d1, d2, p1, p2 = setParallel(self.Chiplets,"Chiplet")
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
			p1, p2 = setParallelNum(self.PEs)
			parallel_num_set.append(p1)
			parallel_num_set.append(p2)
			size_i[3] = math.ceil(size_i[3]/p1)
			size_i[2] = math.ceil(size_i[2]/p2)
			p1, p2 = setParallelNum(self.Chiplets)
			parallel_num_set.append(p1)
			parallel_num_set.append(p2)
			size_i[3] = math.ceil(size_i[3]/p1)
			size_i[2] = math.ceil(size_i[2]/p2)

		#---维度拆分（时间并行）---
		ran1 = random.randint(0,2)
		if ran1 == 0:
			Pset = setPartition_1(size_i[0], self.parallel_level)
			Qset = setPartition_1(size_i[1], self.parallel_level)
			Cset = setPartition_1(size_i[2], self.parallel_level)
			Kset = setPartition_1(size_i[3], self.parallel_level)
		else:
			Pset = setPartition(size_i[0], self.parallel_level)
			Qset = setPartition(size_i[1], self.parallel_level)
			Cset = setPartition(size_i[2], self.parallel_level)
			Kset = setPartition(size_i[3], self.parallel_level)
		for i in range(self.parallel_level,3):
			Pset.append(1)
			Qset.append(1)
			Kset.append(1)
			Cset.append(1)
		code = parallel_dim_set + parallel_num_set + Pset + Qset + Cset + Kset # [Kp1,CP1,Kp2,Cp2,C1~C3,K1~K3,P1~P3,Q1~Q3]
		return code

	# 获得for循环顺序的code
	def getOrderCode(self):
		code = []
		for i in range(3):
			if i == 0:
				list1 = list(range(6))
			else:
				list1 = list(range(4))
			random.shuffle(list1)
			code += list1
		return code

	# 获得GA code（for拆分+for_loop顺序）
	def getChild(self):
		code1 = self.getPartitionChild()
		code2 = self.getOrderCode()
		code = code1 + code2
		return code

	# 把GA code进行解析，解析成{[for_type, dim, for_num]}
	def codeParse(self, code):
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
						num = self.R
					elif dim == 5:
						num = self.S
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
	def printBasicSet(self):
		print(" ")
		print("basic set :")
		print("P = " + str(self.network_param["P"]))
		print("Q = " + str(self.network_param["Q"]))
		print("C = " + str(self.network_param["C"]))
		print("K = " + str(self.network_param["K"]))
		print("R = " + str(self.network_param["R"]))
		print("S = " + str(self.network_param["S"]))
		print("Chiplets = " + str(self.HW_param["Chiplet"]))
		print("PEs = " + str(self.HW_param["PE"]))
		print("intra PE parallel : ", self.HW_param["intra_PE"])
		print("")

	def printBasicSetFile(self, file1):
		print(" ", file = file1)
		print("basic set :", file = file1)
		print("P = " + str(self.network_param["P"]), file = file1)
		print("Q = " + str(self.network_param["Q"]), file = file1)
		print("C = " + str(self.network_param["C"]), file = file1)
		print("K = " + str(self.network_param["K"]), file = file1)
		print("R = " + str(self.network_param["R"]), file = file1)
		print("S = " + str(self.network_param["S"]), file = file1)
		print("Chiplets = " + str(self.HW_param["Chiplet"]), file = file1)
		print("PEs = " + str(self.HW_param["PE"]), file = file1)
		print("intra PE parallel : ", self.HW_param["intra_PE"], file = file1)
		print("", file = file1)

	# 输出For循环
	def printParseDict(self, dict):
		for i in reversed(range(len(dict))):
			type = dict[i][0]
			dim = dict[i][1]
			num = dict[i][2]
			if num > 1:
				line = type_list[type] + " " + dim_list[dim] + " in [0 : " + str(num) + "):"
				print(line)

	# 获得计算核心对于act与wgt的共享情况
	# para1是activation的组数，para2是weight的组数
	def getPEDistribution(self, flag, num, set1, set2):
		act_PE_dict = {}
		wgt_PE_dict = {}
		act_set_type = "0"
		wgt_set_type = "0"

		#---act + wgt send节点列表---
		if flag == 0:
			act_PE_dict["send"] = {0:[A_W_offset["a"]]}
			wgt_PE_dict["send"] = {0:[A_W_offset["w"]]}
		else:
			act_PE_dict["send"] = {0:[0]}
			wgt_PE_dict["send"] = {0:[0]}

		#---act recv节点列表---
		act_PE_dict["recv"], wgt_PE_dict["recv"]= self.setmappingSet(num, set1, set2)
		
		act_set_type = "set_"+str(set1)+"e_"+str(set2)
		wgt_set_type = "set_"+str(set2)+"e_"+str(set1)

		return act_PE_dict, wgt_PE_dict, act_set_type, wgt_set_type

	def getPEDistribution16(self, para1, para2):
		assert(para1*para2 == 16)
		act_PE_dict = {}
		wgt_PE_dict = {}
		act_set_type = "0"
		wgt_set_type = "0"

		#---act + wgt send节点列表---
		act_PE_dict["send"] = {0:[mem_16["a"]]}
		wgt_PE_dict["send"] = {0:[mem_16["w"]]}

		#---act recv节点列表---
		ran = random.randint(0,len(set16[para1])-1)
		if para1 == 4:
			act_PE_dict["recv"] = set16[para1][ran]
			wgt_PE_dict["recv"] = set16[para1][len(set16[4])-ran-1]
		else:
			act_PE_dict["recv"] = set16[para1][ran]
			wgt_PE_dict["recv"] = set16[para2][ran]
		
		act_set_type = "set_"+str(para1)+"e_"+str(para2)
		wgt_set_type = "set_"+str(para2)+"e_"+str(para1)

		return act_PE_dict, wgt_PE_dict, act_set_type, wgt_set_type

	def getPEDistribution4(self, para1, para2):
		assert(para1*para2 == 4)
		act_PE_dict = {}
		wgt_PE_dict = {}
		act_set_type = "0"
		wgt_set_type = "0"

		#---act + wgt send节点列表---
		act_PE_dict["send"] = {0:[mem_4["a"]]}
		wgt_PE_dict["send"] = {0:[mem_4["w"]]}

		ran = random.randint(0,len(set4[para1])-1)
		if para1 == 2:
			act_PE_dict["recv"] = set4[para1][ran]
			wgt_PE_dict["recv"] = set4[para1][len(set4[2])-ran-1]
		else:
			act_PE_dict["recv"] = set4[para1][ran]
			wgt_PE_dict["recv"] = set4[para2][ran]
		
		act_set_type = "set_"+str(para1)+"e_"+str(para2)
		wgt_set_type = "set_"+str(para2)+"e_"+str(para1)

		return act_PE_dict, wgt_PE_dict, act_set_type, wgt_set_type

	# 给dict的每个元素加上固定偏移量num
	# dict = {"send":{0:[],...},"recv":{0:[]...}}
	def dictAddInt(self, dict, num):
		send = dict["send"]
		recv = dict["recv"]
		for i in send:
			for x in range(len(send[i])):
				send[i][x] += num

		for i in recv:
			for x in range(len(recv[i])):
				recv[i][x] +=  num
		dict1 = copy.deepcopy({"send":send,"recv":recv})
		return dict1

	# 转换为chiplet
	def dictChipletChange(self, dict, flag1, flag2):
		send = dict["send"]
		recv = dict["recv"]
		for i in send:
			for x in range(len(send[i])):
				num = send[i][x]
				send[i][x] = self.NoP2NoCnode[num] + self.A_W_offset[flag1]
		for i in recv:
			for x in range(len(recv[i])):
				num = recv[i][x]
				recv[i][x] = self.NoP2NoCnode[num] + self.A_W_offset[flag2]
		dict1 = copy.deepcopy({"send":send,"recv":recv})
		return dict1

	# type = 0 : NoC ; type = 1 : NoP
	def getPEExtent(self, dict, type=0, flag1 = 0, flag2 = 0):
		list = []
		#list.append(dict)
		if type == 0:
			for i in self.NoC_node_offset:
				dict1 = copy.deepcopy(dict)
				dict1 = self.dictAddInt(dict1, i)		
				list.append(dict1)
		else:
			dict1 = copy.deepcopy(dict)
			dict1 = self.dictChipletChange(dict1,flag1,flag2)
			return dict1
		return list

	# 获得输出特征图的数据节点通信关系
	def getOutputDict(self):
		rd_out_PE_dict_temp,a1,b1,c1 = self.getPEDistribution(1, self.PEs, self.PEs, 1)
		wr_out_PE_dict_temp = {"send":rd_out_PE_dict_temp["recv"],"recv":{0:[0]}}
		rd_out_Chiplet_dict_temp,a1,b1,c1 = self.getPEDistribution(1, self.Chiplets, self.Chiplets, 1)
		wr_out_Chiplet_dict_temp = {"send":rd_out_Chiplet_dict_temp["recv"],"recv":{0:[0]}}

		#if self.PEs == 16:
		#	rd_out_PE_dict_temp = {"send":{0:[0]},"recv":set_16_e_1[0]}
		#	wr_out_PE_dict_temp = {"send":set_16_e_1[0],"recv":{0:[0]}}
		#elif self.PEs == 4:
		#	rd_out_PE_dict_temp = {"send":{0:[0]},"recv":set_4_e_1[0]}
		#	wr_out_PE_dict_temp = {"send":set_4_e_1[0],"recv":{0:[0]}}
		#if self.Chiplets == 16:
		#	rd_out_Chiplet_dict_temp = {"send":{0:[0]},"recv":set_16_e_1[0]}
		#	wr_out_Chiplet_dict_temp = {"send":set_16_e_1[0],"recv":{0:[0]}}
		#elif self.Chiplets == 4:
		#	rd_out_Chiplet_dict_temp = {"send":{0:[0]},"recv":set_4_e_1[0]}
		#	wr_out_Chiplet_dict_temp = {"send":set_4_e_1[0],"recv":{0:[0]}}
		rd_out_PE_dict = self.getPEExtent(rd_out_PE_dict_temp)
		wr_out_PE_dict = self.getPEExtent(wr_out_PE_dict_temp)
		rd_out_Chiplet_dict = self.getPEExtent(rd_out_Chiplet_dict_temp,1,"o","o")
		wr_out_Chiplet_dict = self.getPEExtent(wr_out_Chiplet_dict_temp,1,"o","o")
		return rd_out_PE_dict, wr_out_PE_dict, rd_out_Chiplet_dict, wr_out_Chiplet_dict

	# 解析parse，得到对应的输出
	def parseChange(self, parse):
		data_flow = []
		data_flow_dim = []
		ol1_ratio = []
		al1_ratio = []
		wl1_ratio = []
		all_param = []
		out_final = []
		if_act_share_PE = {0:0,1:"0"}
		if_wgt_share_PE = {0:0,1:"0"}
		if_act_share_Chiplet = {0:0,1:"0"}
		if_wgt_share_Chiplet = {0:0,1:"0"}
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
					parallel_dim_list[0][0] *= num
				elif dim == 1:
					parallel_dim_list[0][1] *= num
				elif dim == 3:
					parallel_dim_list[0][2] *= num

				if A_correlation[dim] == 0:
					if_act_share_PE[0] = 1
					act_share_PE[1] *= num
				else:
					act_share_PE[0] *= num
				
				if W_correlation[dim] == 0:
					if_wgt_share_PE[0] = 1
					wgt_share_PE[1] *= num
				else:
					wgt_share_PE[0] *= num
			elif for_type == 2:
				if dim == 0:
					parallel_dim_list[1][0] *= num
				elif dim == 1:
					parallel_dim_list[1][1] *= num
				elif dim == 3:
					parallel_dim_list[1][2] *= num

				if A_correlation[dim] == 0:
					if_act_share_Chiplet[0] = 1
					act_share_Chiplet[1] *= num
				else:
					act_share_Chiplet[0] *= num

				if W_correlation[dim] == 0:
					if_wgt_share_Chiplet[0] = 1
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
		act_PE_dict_temp, wgt_PE_dict_temp, if_act_share_PE[1], if_wgt_share_PE[1] = self.getPEDistribution(0, self.PEs, act_share_PE[0],wgt_share_PE[0])
		act_Chiplet_dict_temp, wgt_Chiplet_dict_temp, if_act_share_Chiplet[1], if_wgt_share_Chiplet[1] = self.getPEDistribution(1, self.Chiplets, act_share_Chiplet[0],wgt_share_Chiplet[0])
		#if self.PEs == 16:
		#	act_PE_dict_temp, wgt_PE_dict_temp, if_act_share_PE[1], if_wgt_share_PE[1] = self.getPEDistribution16(act_share_PE[0],wgt_share_PE[0])
		#elif self.PEs == 4:
		#	act_PE_dict_temp, wgt_PE_dict_temp, if_act_share_PE[1], if_wgt_share_PE[1] = self.getPEDistribution4(act_share_PE[0],wgt_share_PE[0])
		#if self.Chiplets == 16:
		#	act_Chiplet_dict_temp, wgt_Chiplet_dict_temp, if_act_share_Chiplet[1], if_wgt_share_Chiplet[1] = self.getPEDistribution16(act_share_Chiplet[0],wgt_share_Chiplet[0])
		#elif self.Chiplets == 4:
		#	act_Chiplet_dict_temp, wgt_Chiplet_dict_temp, if_act_share_Chiplet[1], if_wgt_share_Chiplet[1] = self.getPEDistribution4(act_share_Chiplet[0],wgt_share_Chiplet[0])
		act_PE_dict = self.getPEExtent(act_PE_dict_temp)
		wgt_PE_dict = self.getPEExtent(wgt_PE_dict_temp)
		act_Chiplet_dict = self.getPEExtent(act_Chiplet_dict_temp,1,"o","a")
		wgt_Chiplet_dict = self.getPEExtent(wgt_Chiplet_dict_temp,1,"o","w")
		return data_flow, ol1_ratio, al1_ratio, wl1_ratio, all_param, out_final, if_act_share_PE, if_wgt_share_PE, if_act_share_Chiplet, if_wgt_share_Chiplet, act_PE_dict, wgt_PE_dict, act_Chiplet_dict, wgt_Chiplet_dict, parallel_dim_list

	def printOut(self, for_list, act_wgt_dict, parallel_dim_list, out_dict):
		print("data_flow = ",for_list[0])
		print("ol1_ratio = ",for_list[1])
		print("al1_ratio = ",for_list[2])
		print("wl1_ratio = ",for_list[3])
		print("all_param = ",for_list[4])
		print("out_final = ",for_list[5])
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
		if self.debug == 1:
			f = open(self.debug_file,'a')
			print("if_act_share_PE = ",for_list[6], file = f)
			print("if_wgt_share_PE = ",for_list[7], file = f)
			print("if_act_share_Chiplet = ",for_list[8], file = f)
			print("if_wgt_share_Chiplet = ",for_list[9], file = f)
			f.close()

	# act_PE_dict = {0:[],1:[]}，列表内为act复用的PE list， 0、1代表不同的组
	def GaGetChild(self):
		if self.debug == 1:
			self.printBasicSet()
		code = self.getChild()
		parse = self.codeParse(code)
		if self.debug == 1:
			self.printParseDict(parse)

		#---获得PQKC的拆分---
		partition_list = {"P":[],"Q":[],"K":[],"C":[]}
		partition_list["P"] = code[8:11]
		partition_list["Q"] = code[11:14]
		partition_list["C"] = code[14:17]
		partition_list["K"] = code[17:20]

		for_list = {}
		act_wgt_dict = {}
		for_list[0], for_list[1], for_list[2], for_list[3], for_list[4], for_list[5], for_list[6], for_list[7], for_list[8], for_list[9], act_wgt_dict["act_core"], act_wgt_dict["wgt_core"], act_wgt_dict["act_chiplet"], act_wgt_dict["wgt_chiplet"], parallel_dim_list = self.parseChange(parse)
		
		out_dict = {}
		out_dict["rd_core"], out_dict["wr_core"], out_dict["rd_chip"], out_dict["wr_chip"] = self.getOutputDict()

		if self.debug == 1:
			self.printOut(for_list, act_wgt_dict, parallel_dim_list, out_dict)

		return for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list

	def GaGetChildParse(self, code):
		parse = self.codeParse(code)

		#---获得PQKC的拆分---
		partition_list = {"P":[],"Q":[],"K":[],"C":[]}
		partition_list["P"] = code[8:11]
		partition_list["Q"] = code[11:14]
		partition_list["C"] = code[14:17]
		partition_list["K"] = code[17:20]

		for_list = {}

		act_wgt_dict = {}
		for_list[0], for_list[1], for_list[2], for_list[3], for_list[4], for_list[5], for_list[6], for_list[7], for_list[8], for_list[9], act_wgt_dict["act_core"], act_wgt_dict["wgt_core"], act_wgt_dict["act_chiplet"], act_wgt_dict["wgt_chiplet"], parallel_dim_list = self.parseChange(parse)

		out_dict = {}
		out_dict["rd_core"], out_dict["wr_core"], out_dict["rd_chip"], out_dict["wr_chip"] = self.getOutputDict()
		
		
		return for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list

if __name__ == '__main__':
	network_param = {"P":224,"Q":224,"C":4,"K":64,"R":3,"S":3}
	HW_param = {"Chiplet":4,"PE":16,"intra_PE":{"C":4,"K":4}}
	debug=1
	GATest = GaEncode(network_param, HW_param, debug)
	for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list = GATest.GaGetChild()
	print(parallel_dim_list)
import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from single_engine_predict_noc_nop import *
from matplotlib import pyplot as plt


degrade_ratio_list = []

def randomTest(GATest,iterTime):
	fitness_min_ran = 0
	fitness_list = []
	fitness_min_ran_list = []
	for i in range(iterTime):
		#---生成个代---
		for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list = GATest.GaGetChild()
		#---计算适应度---
		fitness, degrade_ratio, compuation_cycles = calFitness(for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, GATest.network_param)
		#---比较适应度，并记录相关变量---
		if fitness_min_ran == 0 or fitness < fitness_min_ran:
			fitness_min_ran = fitness
			for_list_1 = copy.deepcopy(for_list)
			act_wgt_dict_1 = copy.deepcopy(act_wgt_dict)
			out_dict_1 = copy.deepcopy(out_dict)
			parallel_dim_list_1 = copy.deepcopy(parallel_dim_list)
			partition_list_1 = copy.deepcopy(partition_list)
			compuation_cycles_1 = compuation_cycles
			degrade_ratio_1 = degrade_ratio
		fitness_list.append(fitness)
		fitness_min_ran_list.append(fitness_min_ran)
		degrade_ratio_list.append (degrade_ratio)
		print("######---------Times = ", i)
		print("fitness_min_ran = ",fitness_min_ran)
		print("compuation_cycles_1 = ",compuation_cycles_1)
		print("degrade_ratio_1 = ",degrade_ratio_1)
		print("######---------over")
		print("")
		
		#---生成task file
	GATest.printOut(for_list_1, act_wgt_dict_1, parallel_dim_list_1, out_dict_1)
	resultCheck(parallel_dim_list_1,partition_list_1, GATest.network_param)
	#createTaskFile(for_list_1, act_wgt_dict_1, out_dict_1, parallel_dim_list_1, partition_list_1,GATest.network_param)
	return compuation_cycles_1,degrade_ratio_1, fitness_min_ran_list

def randomTest_1(GATest,iterTime):
	fitness_min_ran = 0
	fitness_list = []
	fitness_min_ran_list = []

	for_list_list = {}
	parallel_dim_list_list = {}
	partition_list_list = {}

	for i in range(iterTime):
		#---生成个代---
		for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list = GATest.GaGetChild()
		for_list_list[i] = for_list
		parallel_dim_list_list[i] = parallel_dim_list
		partition_list_list[i] = partition_list

		#---计算适应度---
		fitness, degrade_ratio, compuation_cycles = calFitness(for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list, GATest.network_param)
		#---比较适应度，并记录相关变量---
		if fitness_min_ran == 0 or fitness < fitness_min_ran:
			fitness_min_ran = fitness
			for_list_1 = copy.deepcopy(for_list)
			act_wgt_dict_1 = copy.deepcopy(act_wgt_dict)
			out_dict_1 = copy.deepcopy(out_dict)
			parallel_dim_list_1 = copy.deepcopy(parallel_dim_list)
			partition_list_1 = copy.deepcopy(partition_list)
			compuation_cycles_1 = compuation_cycles
			degrade_ratio_1 = degrade_ratio
		fitness_list.append(fitness)
		fitness_min_ran_list.append(fitness_min_ran)
		degrade_ratio_list.append (degrade_ratio)
		print("######---------Times = ", i)
		print("fitness_min_ran = ",fitness_min_ran)
		print("compuation_cycles_1 = ",compuation_cycles_1)
		print("degrade_ratio_1 = ",degrade_ratio_1)
		print("######---------over")
		print("")
		
	#---生成task file
	resultOut(fitness_list,for_list_list, parallel_dim_list_list, partition_list_list, GATest.network_param)
	#createTaskFile(for_list_1, act_wgt_dict_1, out_dict_1, parallel_dim_list_1, partition_list_1, GATest.network_param)
	return compuation_cycles_1,degrade_ratio_1, fitness_min_ran_list

def resultOut(fitness_list,for_list_list, parallel_dim_list_list, partition_list_list,network_param):
	fitness_list_a = np.array(fitness_list)
	fitness_sort_index = fitness_list_a.argsort()
	f = open("./random_test_record.txt",'a')
	for i in range(10):
		index = fitness_sort_index[i]
		print("----",i,"th-----", file = f)
		print("fitness: ", fitness_list[index], file = f)
		print("parallel_dim_list = ", parallel_dim_list_list[index], file = f)
		print("if_act_share_PE = ",for_list_list[index][6], file = f)
		print("if_wgt_share_PE = ",for_list_list[index][7], file = f)
		print("if_act_share_Chiplet = ",for_list_list[index][8], file = f)
		print("if_wgt_share_Chiplet = ",for_list_list[index][9], file = f)
		resultCheck(parallel_dim_list_list[index],partition_list_list[index], network_param)
	f.close()


def resultCheck(parallel_dim_list,partition_list, network_param):
	dim_num_list = [1,1,C0,K0]
	
	for i in partition_list["P"]:
		dim_num_list[0] *= i
	for i in partition_list["Q"]:
		dim_num_list[1] *= i
	for i in partition_list["C"]:
		dim_num_list[2] *= i
	for i in partition_list["K"]:
		dim_num_list[3] *= i

	dim_num_list[0] *= parallel_dim_list[0][0] * parallel_dim_list[1][0]
	dim_num_list[1] *= parallel_dim_list[0][1] * parallel_dim_list[1][1]
	dim_num_list[3] *= parallel_dim_list[0][2] * parallel_dim_list[1][2]
	
	result_list = []
	if dim_num_list[0] > network_param["P"]:
		result_list.append(0)
	elif dim_num_list[0] == network_param["P"]:
		result_list.append(1)
	else:
		print("P num error")
	
	if dim_num_list[1] > network_param["Q"]:
		result_list.append(0)
	elif dim_num_list[1] == network_param["Q"]:
		result_list.append(1)
	else:
		print("Q num error")
	
	if dim_num_list[2] > network_param["C"]:
		result_list.append(0)
	elif dim_num_list[2] == network_param["C"]:
		result_list.append(1)
	else:
		print("C num error")
	
	if dim_num_list[3] > network_param["K"]:
		result_list.append(0)
	elif dim_num_list[3] == network_param["K"]:
		result_list.append(1)
	else:
		print("K num error")

	f = open("./random_test_record.txt",'a')
	print("partiton equal result:", file = f)
	print(result_list, file = f)
	f.close()

if __name__ == '__main__':

	network_param = {"P":224,"Q":224,"C":4,"K":64,"R":3,"S":3}
	HW_param = {"Chiplet":4,"PE":16,"intra_PE":{"C":4,"K":4}}
	debug=0
	GATest = GaEncode(network_param, HW_param, debug)

	iterTime = 10000
	fitness_min_ran = 0
	index = range(iterTime)

	random_test_iter = 20
	f = open("./random_test_record.txt",'w')
	GATest.printBasicSetFile(f)
	f.close()

	for i in range(random_test_iter):
		print("###### test iteration = ",i)
		compuation_cycles_1,degrade_ratio_1, fitness_min_ran_list = randomTest_1(GATest, iterTime)
		print(fitness_min_ran_list[len(fitness_min_ran_list)-1])
		f = open("./random_test_record.txt",'a')
		print("###### test iteration = ",i, file = f)
		print("fitness:",fitness_min_ran_list[len(fitness_min_ran_list)-1], file=f)
		f.close()

	#plt.figure(1)
	#plt.scatter(index,degrade_ratio_list)
	#plt.savefig("randomTest.png")


	#degrade_ratio_list.sort()

	#plt.figure(2)
	#plt.scatter(index,degrade_ratio_list)
	#plt.savefig("randomTest2.png")

	#plt.figure(3)
	#plt.scatter(index[:900],degrade_ratio_list[:900])
	#plt.savefig("randomTest3.png")



	

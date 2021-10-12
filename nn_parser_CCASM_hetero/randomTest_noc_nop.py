import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from single_engine_predict_noc_nop import *
from matplotlib import pyplot as plt

iterTime = 1000
fitness_min_ran = 0
index = range(iterTime)
degrade_ratio_list = []

def randomTest():
	fitness_min_ran = 0
	fitness_list = []
	fitness_min_ran_list = []
	for i in range(iterTime):
		#---生成个代---
		for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list = GaGetChild()
		#---计算适应度---
		fitness, degrade_ratio, compuation_cycles = calFitness(for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list)
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
	createTaskFile(for_list_1, act_wgt_dict_1, out_dict_1, parallel_dim_list_1, partition_list_1)
	return compuation_cycles_1,degrade_ratio_1, fitness_min_ran_list

if __name__ == '__main__':
	compuation_cycles_1,degrade_ratio_1, fitness_min_ran_list = randomTest()
	print("fitness_min_ran = ",fitness_min_ran)
	print("compuation_cycles_1 = ",compuation_cycles_1)
	print("degrade_ratio_1 = ",degrade_ratio_1)


	print(fitness_min_ran_list[len(fitness_min_ran_list)-1])

	plt.figure(1)
	plt.scatter(index,degrade_ratio_list)
	plt.savefig("randomTest.png")


	degrade_ratio_list.sort()

	plt.figure(2)
	plt.scatter(index,degrade_ratio_list)
	plt.savefig("randomTest2.png")

	plt.figure(3)
	plt.scatter(index[:900],degrade_ratio_list[:900])
	plt.savefig("randomTest3.png")

import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from single_engine_test import *
from matplotlib import pyplot as plt

iterTime = 1000
fitness_min = 0
fitness_list = []
fitness_min_list = []
index = []

for i in range(iterTime):
  #---生成个代---
	for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list = GaGetChild()
  #---计算适应度---
	fitness, degrade_ratio, compuation_cycles = calFitness(for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list)
  #---比较适应度，并记录相关变量---
	if fitness_min == 0 or fitness < fitness_min:
		fitness_min = fitness
		for_list_1 = copy.deepcopy(for_list)
		act_wgt_dict_1 = copy.deepcopy(act_wgt_dict)
		out_dict_1 = copy.deepcopy(out_dict)
		parallel_dim_list_1 = copy.deepcopy(parallel_dim_list)
		partition_list_1 = copy.deepcopy(partition_list)
		compuation_cycles_1 = compuation_cycles
		degrade_ratio_1 = degrade_ratio
    
	fitness_list.append(fitness)
	fitness_min_list.append(fitness_min)
	index.append(i)
	print("######---------Times = ", i)
	print("fitness_min = ",fitness_min)
	print("compuation_cycles_1 = ",compuation_cycles_1)
	print("degrade_ratio_1 = ",degrade_ratio_1)
	print("######---------over")
	print("")

createTaskFile(for_list_1, act_wgt_dict_1, out_dict_1, parallel_dim_list_1, partition_list_1)

print("fitness_min = ",fitness_min)
print("compuation_cycles_1 = ",compuation_cycles_1)
print("degrade_ratio_1 = ",degrade_ratio_1)
print("parallel_dim_list_1 = ",parallel_dim_list_1)
print("partition_list_1 = ",partition_list_1)

#plt.plot(index,fitness_min_list)
#plt.plot(index,fitness_list)
#plt.show()

import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from single_engine_predict_noc_nop import *
from matplotlib import pyplot as plt
# Parameter
# 
code_index = [0, 4, 7, 10, 13, 16]
#code_dim = [3,2,3,2,2,2,2,3,3,3,0,0,0,1,1,1]
code_lenth = 4 + 4 + 3 + 3 + 3 + 3 + 6 + 4 + 4
code_index = [0, 4, 8, 11, 14, 17, 20, 26, 30, 34]
order_index = [0, 6, 10, 14]
# GA parameter
pm = 0.01
num_children = 100
num_father_left = int(num_children * 0.1)
num_iter = 10
size = [P,Q,C,K]

def calFitnessAll(code):
	for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list = GaGetChildParse(code)
	fitness , a1, a2 = calFitness(for_list, act_wgt_dict, out_dict, parallel_dim_list, partition_list)
	return fitness

def getFirstGeneration():
	generation = {}
	fitness = np.zeros(num_children)
	for i in range(num_children):
		generation[i] = getChild()
		fitness_i = calFitnessAll(generation[i])
		fitness[i] = fitness_i
	return generation, fitness

def partitionMutate(code):
	ran_i = random.randint(code_index[1],code_index[6]-1)
	num =  code[ran_i]
	ran = random.randint(math.ceil(num*0.5),math.ceil(num*2))
	code[ran_i] = ran
	return code

def repairParallel(a,b,num):
	if a >= b:
		a = int(num/b)
		if a < 1:
			a = 1
	else:
		b = int(num/a)
		if b < 1:
			b = 1
	if a*b > num:
		a,b = repairParallel(a,b,num)
	return a,b

def repairParallel2Equal(a,b,num):
	num_log = math.ceil(math.log(num, 2))
	a_log = math.ceil(math.log(a, 2))
	b_log = math.ceil(math.log(b, 2))

	if a_log >= num_log:
		a = num
		b = 1
	elif b_log >= num_log:
		a = 1
		b = num
	else:
		a = pow(2, int(a_log))
		b = int(num / a)
	return a, b


def partitionRepair(code):
	num = [1,1,C0,K0]
	num_dict = {}
	code_dim = [code[0],code[1],code[2],code[3],0,0,0,1,1,1,2,2,2,3,3,3]

	#check Chiplets PEs
	if code[code_index[1]] * code[code_index[1]+1] != PEs:
		code[code_index[1]], code[code_index[1]+1] = repairParallel2Equal(code[code_index[1]],code[code_index[1]+1],PEs)
	if code[code_index[1]+2] * code[code_index[1]+3] != Chiplets:
		code[code_index[1]+2], code[code_index[1]+3] = repairParallel2Equal(code[code_index[1]+2],code[code_index[1]+3],Chiplets)

	for i in range(4):
		num_dict[i] = []
	
	for i in range(code_index[1],code_index[6]):
		dim = code_dim[i-code_index[1]]
		num[dim] *= code[i]
		if i >= code_index[1] + 4:
			num_dict[dim].append(i)

	error_dim = []
	for i in range(4):
		if size[i] > num[i]:
			error_dim.append(i)
	for i in error_dim:
		ran = random.randint(0,2)
		radix = size[i]/num[i]
		index = num_dict[i][ran]
		code[index] = math.ceil(code[index]*radix)
	return code

def partitionCross(code1, code2):
	if debug == 1:
		print("#####GA######")
		print("father code:")
		print(code1)
		print(code2)
		print("")
	list1 = [] 
	list1 = list(range(code_index[1],code_index[6]-1))
	random.shuffle(list1)
	cross_addr1 = list1[0]
	cross_addr2 = list1[1]

	if cross_addr1 > cross_addr2:
		temp = cross_addr1
		cross_addr1 = cross_addr2
		cross_addr2 = temp

	code1_1 = code1[0:cross_addr1]
	code1_2 = code1[cross_addr1:cross_addr2]
	code1_3 = code1[cross_addr2:]
	code2_1 = code2[0:cross_addr1]
	code2_2 = code2[cross_addr1:cross_addr2]
	code2_3 = code2[cross_addr2:]

	codecross1 = code1_1 + code2_2 + code1_3
	codecross2 = code2_1 + code1_2 + code2_3
	if debug == 1:
		print("cross addr:")
		print(cross_addr1)
		print(cross_addr2)
		print("cross code:")
		print(codecross1)
		print(codecross2)
		print("")

	ran1 = random.random()
	if ran1 < pm:
		codecross1 = partitionMutate(codecross1)
	ran2 = random.random()
	if ran2 < pm:
		codecross2 = partitionMutate(codecross2)
	if debug == 1:
		print("")
		print("mutate code:")
		print(ran1)
		print(codecross1)
		print(codecross2)

	codecross1 = partitionRepair(codecross1)
	codecross2 = partitionRepair(codecross2)
	
	if debug == 1:
		print("repair code:")
		print(codecross1)
		print(codecross2)
	return codecross1, codecross2

def crossUnique(list1, list2, addr):
	list1_1 = list1[0:addr]
	list2_1 = list2[0:addr]

	for i in list2:
		if i not in list1_1:
			list1_1.append(i)
	
	for i in list1:
		if i not in list2_1:
			list2_1.append(i)
	
	return list1_1, list2_1

def orderCross(code1, code2):
	ran = random.randint(1,order_index[3]-1)
	loop1_1 = code1[0:6]
	loop2_1 = code1[6:10]
	loop3_1 = code1[10:14]
	
	loop1_2 = code2[0:6]
	loop2_2 = code2[6:10]
	loop3_2 = code2[10:14]

	if ran < 6:
		loop1_1_i, loop1_2_i = crossUnique(loop1_1, loop1_2, ran)
		loop2_1_i, loop2_2_i = loop2_2, loop2_1
		loop3_1_i, loop3_2_i = loop3_2, loop3_1
	elif ran < 10:
		loop2_1_i, loop2_2_i = crossUnique(loop2_1, loop2_2, ran-6)
		loop1_1_i, loop1_2_i = loop1_1, loop1_2
		loop3_1_i, loop3_2_i = loop3_2, loop3_1
	else:
		loop3_1_i, loop3_2_i = crossUnique(loop3_1, loop3_2, ran-10)
		loop1_1_i, loop1_2_i = loop1_1, loop1_2
		loop2_1_i, loop2_2_i = loop2_1, loop2_2
	
	code1_1 = loop1_1_i + loop2_1_i + loop3_1_i
	code2_1 = loop1_2_i + loop2_2_i + loop3_2_i

	return code1_1, code2_1

# 容易造成两个父代一致，现把竞争部分去掉了
def selectGeneration(select_probability, select_table):
	num_ran1 = random.random()
	flag = 0
	time1 = 0
	while flag == 0:
		if num_ran1 < select_table[time1]:
			flag = 1
			break
		time1 += 1
	
	return time1
#	num_ran2 = random.random()
#	flag = 0
#	time2 = 0
#	while flag == 0:
#		if num_ran2 < select_table[time2]:
#			flag = 1
#			break
#		time2 += 1

#	if select_probability[time1] > select_probability[time2]:
#		return time1
#	else:
#		return time2

def getNextGeneration(father_list, select_probability, select_table, fitness):
	child_list = {}
	childNum = int((num_children - num_father_left)/2)
	fitness_father = []
	for i in range(childNum):
		father_id_1 = selectGeneration(select_probability, select_table)
		father_id_2 = selectGeneration(select_probability, select_table)
		while father_id_1 == father_id_2:
			father_id_2 = selectGeneration(select_probability, select_table)

		father1 = father_list[father_id_1]
		father2 = father_list[father_id_2]

		fitness_father.append(fitness[father_id_1])
		fitness_father.append(fitness[father_id_2])

		partition1 = father1[0:code_index[6]]
		partition2 = father2[0:code_index[6]]
		child_partition_1, child_partition_2 = partitionCross(partition1, partition2)

		order1 = father1[code_index[6]:code_index[9]]
		order2 = father2[code_index[6]:code_index[9]]
		child_order_1, child_order_2 = orderCross(order1, order2)

		child1 = child_partition_1 + child_order_1
		child2 = child_partition_2 + child_order_2

		child_list[2*i] = child1
		child_list[2*i+1] = child2
	
	select_id_list = np.argsort(-select_probability)
	for i in range(num_father_left):
		child_id = num_children - num_father_left + i
		select_id = select_id_list[i]
		child_list[child_id] = father_list[select_id]
	
	return child_list, fitness_father

def calProbability(list):
	list_1 = np.zeros(num_children)
	for i in range(len(list)):
		list_1[i] = 1 / list[i]
	total = list_1.sum(axis = 0)
	probability = np.zeros(num_children)
	for i in range(0, num_children):
		probability[i] = list_1[i] / total
	return probability

def calSelectTable(select_probability):
	probability_list = {}
	probability_pre = 0
	for i in range(0, num_children):
		probability_list[i] = probability_pre + select_probability[i]
		probability_pre += select_probability[i]
	return probability_list

def getNpMin(array):
	list = array.tolist()
	min_num = min(list)
	min_index = list.index(min(list))
	return min_num, min_index

def gaIter():
	print("########Begin")
	f = open("./fitness-record.txt",'w')
	generation, fitness = getFirstGeneration()
	fitness_min = []
	code_best = []
	for i in range(num_iter):
		print("iter = ", str(i))
		select_probability = calProbability(fitness)
		select_table = calSelectTable(select_probability)
		generation, fitness_father = getNextGeneration(generation, select_probability, select_table, fitness)
		for j in range(len(generation)):
			code = generation[j]
			fitness[j] = calFitnessAll(code)
		if debug == 1:
			print("########## times = ",str(i+1), file = f)
			print("fitness_father = ", fitness_father, file = f)
			print("fitness = ",fitness, file = f)
			#print("generation = ", generation, file = f)
		min_num, min_index = getNpMin(fitness)
		code_best = generation[min_index]
		fitness_min.append(min_num)

	#---产生task file
	for_list_1, act_wgt_dict_1, out_dict_1, parallel_dim_list_1, partition_list_1 = GaGetChildParse(code_best)
	createTaskFile(for_list_1, act_wgt_dict_1, out_dict_1, parallel_dim_list_1, partition_list_1)
	
	f.close()
	return fitness_min

if __name__ == '__main__':
	fitness_min = gaIter()

	print(fitness_min[len(fitness_min)-1])
	index = range(len(fitness_min))
	plt.figure(1)
	plt.scatter(index,fitness_min)
	plt.savefig("GaTest.png")

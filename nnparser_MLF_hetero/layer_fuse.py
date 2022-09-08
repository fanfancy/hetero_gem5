import math
import os
import sys
import random
import copy
import argparse
from matplotlib import pyplot as plt

# Parameter
layer_list = {"resnet18":[1,2,2,2,2,6,7,7,7,10,11,11,11,14,15,15,15],"resnet50":[1,2,3,4,5,3,4,5,3,4,11,12,13,14,15,13,14,15,13,14,15,13,23,24,25,26,27,25,26,27,25,26,27,25,26,27,25,26,27,25,41,42,43,44,45,43,44,45,43,50],"VGG16":[1,2,3,4,5,6,6,8,9,9,11,11,13],"alexnet":[1,2,3,4,5,6,7,8],"lenet":[1,2,3,4,5]}
ratio = {}
chiplet_parallel_list = ["P_stable", "PK_stable", "K_stable"]
PE_Frequency = 1000 * 1000 * 1000

def txt_extract(result_indir_p, result_indir_pk, result_indir_k, app_name, fuse_flag):
	
	def getLayerParam(app_name):
		layer_num = 0
		layer_dict = {}
		layer_id_list = []
		f = open(result_abs_path + "/nn_input_noc_nop/" + app_name + ".txt")

		lines = f.readlines()
		for line in lines:
			if line.startswith("#") or line.startswith("*"):
				pass
			elif line != "\n":
				line_item = line.split(" ")
				layer_num += 1
				H = int(line_item[1])
				M = int(line_item[2])
				C = int(line_item[3])
				R = int(line_item[4])
				S = int(line_item[4])
				stride = int(line_item[5])
				padding = int(line_item[6])
				K = int(line_item[7])
				P = int(line_item[8])
				Q = int(line_item[9])

				layer_id_list.append(layer_num)
				layer_id = "layer" + str(layer_num)
				layer_dict[layer_id] = {"H":H,"M":M,"P":P,"Q":Q,"C":C,"K":K,"R":R,"S":S, "stride":stride, "padding":padding}
		f.close()
		return layer_dict

	def getScaleRatio(line, layer_param, PEs = 16, K0 = 16):
		line = line.replace("]","[")
		line = line.replace(",","")
		line = line.replace("\'","")
		line = line.replace(":","")
		line = line.replace("{","")
		line = line.replace("}","")
		line_list = line.split("[")
		item_num = 0
		result_dict = {}
		par_real = {}
		for line_item in line_list:
			if item_num % 2 == 1:
				layer_id = line_list[item_num-1]
				layer_id = layer_id.replace(" ","")
				code_list = line_item.split(" ")
				par_code = code_list[0:4]
				par_num_code = code_list[4:8]
				par_K_num = math.ceil(layer_param[layer_id]["K"]/K0)
				par_P_num = layer_param[layer_id]["P"]
				par_Q_num = layer_param[layer_id]["Q"]
				
				PK2=1;PP2=1
				PK3=1;PP3=1
				PQ2=1
				for index in range(2):
					par = int(par_code[index])
					par_num = int(par_num_code[index])
					if par == 3:
						PK2 *= par_num
					elif par == 0:
						PP2 *= par_num
					elif par == 1:
						PQ2 *= par_num
				for index in range(2):
					par = int(par_code[index+2])
					par_num = int(par_num_code[index+2])
					if par == 3:
						PK3 *= par_num
					elif par == 0:
						PP3 *= par_num

				#---call PE util
				pe_util = PQ2 * PP2 * PK2 / PEs

				#---call Chiplet util
				p_util = math.ceil(par_P_num/PP2) / PP3
				k_util = math.ceil(par_K_num/PK2) / PK3
				if p_util >= 1:
					p_util = 1
				if k_util >= 1:
					k_util = 1
				chiplet_util = p_util * k_util

				util = pe_util * chiplet_util

				result_dict[layer_id] = util
			item_num += 1
		return result_dict

	def lineParse(line, ratio_list = {}):
		line = line.replace(")","(")
		line = line.replace(",","")
		line = line.replace("\'","")
		line = line.replace(":","")
		line = line.replace("{","")
		line = line.replace("}","")
		line_list = line.split(" ")
		item_num = 0
		result_dict = {}
		for line_item in line_list:
			if item_num % 2 == 1:
				layer_id = line_list[item_num-1]
				layer_id = layer_id.replace(" ","")
				if len(ratio_list) == 0:
					result_dict[layer_id] = float(line_item)
				else:
					result_dict[layer_id] = float(line_item) * ratio_list[layer_id]
			item_num += 1
		return result_dict

	def extractFitness_initial(result_indir):
		# --- read files ---
		result_file_init = os.path.join(result_indir, "final_result_record_initial.txt")
		f_init = open(result_file_init)

		lines_init = f_init.readlines()
		line_init_edp = lines_init[0]
		line_init_energy = lines_init[1]
		line_init_latency = lines_init[2]
		line_init_noc_NR = lines_init[5]
		line_init_L2_to_DRAM_NR = lines_init[6]
		line_init_DRAM_to_L2_NR = lines_init[7]

		# --- extract from lines ---
		result_init_edp = lineParse(line_init_edp)
		result_init_energy = lineParse(line_init_energy)
		result_init_latency = lineParse(line_init_latency)
		result_init_NoC_NR = lineParse(line_init_noc_NR)
		result_init_L2_to_DRAM_NR = lineParse(line_init_L2_to_DRAM_NR)
		result_init_DRAM_to_L2_NR = lineParse(line_init_DRAM_to_L2_NR)
		
		fitness_init_dict = {}

		for layer_id in result_init_edp:
			fitness_init_dict[layer_id] = [result_init_edp[layer_id], result_init_energy[layer_id], result_init_latency[layer_id], \
				result_init_NoC_NR[layer_id], result_init_L2_to_DRAM_NR[layer_id], result_init_DRAM_to_L2_NR[layer_id]]

		return fitness_init_dict

	def extractFitness_head_tail(result_indir,  tile_ratio = 4):
		# --- read files ---
		result_file_init = os.path.join(result_indir, "final_result_record_initial.txt")
		result_file_head = os.path.join(result_indir, "final_result_record_headLayer.txt")
		result_file_tail = os.path.join(result_indir, "final_result_record_tailLayer.txt")

		f_init = open(result_file_init)
		f_head = open(result_file_head)
		f_tail = open(result_file_tail)

		lines_init = f_init.readlines()
		line_init_edp = lines_init[0]

		lines_head = f_head.readlines()
		line_head_edp = lines_head[0]
		line_head_energy = lines_head[1]
		line_head_latency = lines_head[2]
		line_head_noc_NR = lines_head[5]
		line_head_L2_to_DRAM_NR = lines_head[6]
		line_head_DRAM_to_L2_NR = lines_head[7]

		lines_tail = f_tail.readlines()
		line_tail_edp = lines_tail[0]
		line_tail_energy = lines_tail[1]
		line_tail_latency = lines_tail[2]
		line_tail_noc_NR = lines_tail[5]
		line_tail_L2_to_DRAM_NR = lines_tail[6]
		line_tail_DRAM_to_L2_NR = lines_tail[7]

		# --- extract from lines ---
		result_init_edp = lineParse(line_init_edp)

		result_head_edp = lineParse(line_head_edp)
		result_tail_edp = lineParse(line_tail_edp)

		result_head_energy = lineParse(line_head_energy)
		result_tail_energy = lineParse(line_tail_energy)

		result_head_latency = lineParse(line_head_latency)
		result_tail_latency = lineParse(line_tail_latency)

		result_head_NoC_NR = lineParse(line_head_noc_NR)
		result_tail_NoC_NR = lineParse(line_tail_noc_NR)

		result_head_L2_to_DRAM_NR = lineParse(line_head_L2_to_DRAM_NR)
		result_tail_L2_to_DRAM_NR = lineParse(line_tail_L2_to_DRAM_NR)

		result_head_DRAM_to_L2_NR = lineParse(line_head_DRAM_to_L2_NR)
		result_tail_DRAM_to_L2_NR = lineParse(line_tail_DRAM_to_L2_NR)

		fitness_head_dict = {}
		fitness_tail_dict = {}
		
		for layer_id in result_init_edp:
			if layer_id in result_head_edp:
				result_head_edp[layer_id] *= tile_ratio * tile_ratio
				result_head_energy[layer_id] *= tile_ratio
				result_head_latency[layer_id] *= tile_ratio
				result_head_NoC_NR[layer_id] *= 1
				result_head_L2_to_DRAM_NR[layer_id] *= 1
				result_head_DRAM_to_L2_NR[layer_id] *= 1
			else:
				result_head_edp[layer_id] = None
				result_head_energy[layer_id] = None
				result_head_latency[layer_id] = None
				result_head_NoC_NR[layer_id] = None
				result_head_L2_to_DRAM_NR[layer_id] = None
				result_head_DRAM_to_L2_NR[layer_id] = None

			if layer_id in result_tail_edp:
				result_tail_edp[layer_id] *= tile_ratio * tile_ratio
				result_tail_energy[layer_id] *= tile_ratio
				result_tail_latency[layer_id] *= tile_ratio
				result_tail_NoC_NR[layer_id] *= 1
				result_tail_L2_to_DRAM_NR[layer_id] *= 1
				result_tail_DRAM_to_L2_NR[layer_id] *= 1
			else:
				result_tail_edp[layer_id] = None
				result_tail_energy[layer_id] = None
				result_tail_latency[layer_id] = None
				result_tail_NoC_NR[layer_id] = None
				result_tail_L2_to_DRAM_NR[layer_id] = None
				result_tail_DRAM_to_L2_NR[layer_id] = None
			
			fitness_head_dict[layer_id] = [result_head_edp[layer_id], result_head_energy[layer_id], result_head_latency[layer_id], \
				result_head_NoC_NR[layer_id], result_head_L2_to_DRAM_NR[layer_id], result_head_DRAM_to_L2_NR[layer_id]]
			fitness_tail_dict[layer_id] = [result_tail_edp[layer_id], result_tail_energy[layer_id], result_tail_latency[layer_id], \
				result_tail_NoC_NR[layer_id], result_tail_L2_to_DRAM_NR[layer_id], result_tail_DRAM_to_L2_NR[layer_id]]
		
		return fitness_head_dict, fitness_tail_dict

	# Fitness per chiplet_parallel extract and file out
	# 获取每层在三种片间并行度方案下的适应度数值
	fitness_init_dict_p = extractFitness_initial(result_indir_p)
	fitness_init_dict_pk = extractFitness_initial(result_indir_pk)
	fitness_init_dict_k = extractFitness_initial(result_indir_k)
	if fuse_flag == 1:
		fitness_head_dict_p, fitness_tail_dict_p = extractFitness_head_tail(result_indir_p)
		fitness_head_dict_pk, fitness_tail_dict_pk = extractFitness_head_tail(result_indir_pk)
		fitness_head_dict_k, fitness_tail_dict_k = extractFitness_head_tail(result_indir_k)

	def getMinFrom3(a1, a2, a3, dim_list):
		if a1 == a2 == a3 == None:
			return None, 'p'
		def getMinFrom2(a1, a2):
			if a1 <= a2:
				return a1, 0
			else:
				return a2, 1
		num, index = getMinFrom2(a1, a2)
		num2, index2 = getMinFrom2(num, a3)
		if index2 == 1:
			return num2, dim_list[2]
		else:
			return num2, dim_list[index]
	
	def getFitnessMinDict(dict_p, dict_k, dict_pk, app_name):
		assert(len(dict_p) == len(dict_k) == len(dict_pk))
		fitness_dict = {"layer_id":[], "edp":[], "energy":[], "latency":[], "NoC_NR":[], "L2_to_DRAM_NR":[], "DRAM_to_L2_NR":[], "select_dim":[]}
		
		layer_id_list = []
		edp_dict = {}
		energy_dict = {}
		latency_dict = {}
		NoC_NR_dict = {}
		L2_to_DRAM_NR_dict = {}
		DRAM_to_L2_NR_dict = {}
		select_dim_dict = {}

		for layer_id in dict_p:
			edp_p = dict_p[layer_id][0]
			edp_k = dict_k[layer_id][0]
			edp_pk = dict_pk[layer_id][0]
			edp_min, dim = getMinFrom3(edp_p, edp_k, edp_pk, ["p",'k','pk'])

			if dim == "p":
				edp = dict_p[layer_id][0]
				energy = dict_p[layer_id][1]
				latency = dict_p[layer_id][2]
				NoC_NR = dict_p[layer_id][3]
				L2_to_DRAM_NR = dict_p[layer_id][4]
				DRAM_to_L2_NR = dict_p[layer_id][5]
			elif dim == "k":
				edp = dict_k[layer_id][0]
				energy = dict_k[layer_id][1]
				latency = dict_k[layer_id][2]
				NoC_NR = dict_k[layer_id][3]
				L2_to_DRAM_NR = dict_k[layer_id][4]
				DRAM_to_L2_NR = dict_k[layer_id][5]
			elif dim == "pk":
				edp = dict_pk[layer_id][0]
				energy = dict_pk[layer_id][1]
				latency = dict_pk[layer_id][2]
				NoC_NR = dict_pk[layer_id][3]
				L2_to_DRAM_NR = dict_pk[layer_id][4]
				DRAM_to_L2_NR = dict_pk[layer_id][5]
			
			layer_id_list.append(layer_id)

			edp_dict[layer_id] = edp
			energy_dict[layer_id] = energy
			latency_dict[layer_id] = latency
			NoC_NR_dict[layer_id] = NoC_NR
			L2_to_DRAM_NR_dict[layer_id] = L2_to_DRAM_NR
			DRAM_to_L2_NR_dict[layer_id] = DRAM_to_L2_NR
			select_dim_dict[layer_id] = dim
		
		for i in range(len(layer_list[app_name])):
			layer_id = "layer" + str(i)
			layer_index = layer_list[app_name][i]
			layer_id_select = "layer" + str(layer_index)
			fitness_dict["layer_id"].append(layer_id)
			fitness_dict["edp"].append(edp_dict[layer_id_select])
			fitness_dict["energy"].append(energy_dict[layer_id_select])
			fitness_dict["latency"].append(latency_dict[layer_id_select])
			fitness_dict["NoC_NR"].append(NoC_NR_dict[layer_id_select])
			fitness_dict["L2_to_DRAM_NR"].append(L2_to_DRAM_NR_dict[layer_id_select])
			fitness_dict["DRAM_to_L2_NR"].append(DRAM_to_L2_NR_dict[layer_id_select])
			fitness_dict["select_dim"].append(select_dim_dict[layer_id_select])

		return fitness_dict

	fitness_init_dict = getFitnessMinDict(fitness_init_dict_p, fitness_init_dict_k, fitness_init_dict_pk, app_name)
	if fuse_flag == 1:
		fitness_head_dict = getFitnessMinDict(fitness_head_dict_p, fitness_head_dict_k, fitness_head_dict_pk, app_name)
		fitness_tail_dict = getFitnessMinDict(fitness_tail_dict_p, fitness_tail_dict_k, fitness_tail_dict_pk, app_name)
	else:
		fitness_head_dict = {"edp":None, "energy":None, "latency":None}
		fitness_tail_dict = {"edp":None, "energy":None, "latency":None}

	return fitness_init_dict, fitness_head_dict, fitness_tail_dict

def layer_fuse(fitness_init_dict, fitness_head_dict, fitness_tail_dict):
	
	# --- get which layers fuse is useful
	fuse_tag = []
	fitness_fuse_dict = {"edp":[], "energy":[], "latency":[]}
	fitness_layer_dict = {"edp":fitness_init_dict["edp"], "energy":fitness_init_dict["energy"], "latency":fitness_init_dict["latency"]}
	edp_i_list = fitness_init_dict["edp"]
	energy_i_list = fitness_init_dict["energy"]
	latency_i_list = fitness_init_dict["latency"]
	energy_h_list = fitness_head_dict["energy"]
	latency_h_list = fitness_head_dict["latency"]
	energy_t_list = fitness_tail_dict["energy"]
	latency_t_list = fitness_tail_dict["latency"]

	for i in range(len(edp_i_list)-1):
		energy_i_h = energy_i_list[i]
		energy_i_t = energy_i_list[i+1]
		latency_i_h = latency_i_list[i]
		latency_i_t = latency_i_list[i+1]
		energy_h = energy_h_list[i]
		energy_t = energy_t_list[i+1]
		latency_h = latency_h_list[i]
		latency_t = latency_t_list[i+1]

		edp_i_fuse = (energy_i_h + energy_i_t) * (latency_i_h + latency_i_t) / PE_Frequency

		if energy_h != None and energy_t != None:
			energy_sum = energy_h + energy_t
			latency_sum = latency_h + latency_t
			edp_fuse = energy_sum * latency_sum / PE_Frequency
			if edp_fuse <= edp_i_fuse:
				fuse_tag.append(1)
			else:
				fuse_tag.append(0)
			fitness_fuse_dict["edp"].append(edp_fuse)
			fitness_fuse_dict["energy"].append(energy_sum)
			fitness_fuse_dict["latency"].append(latency_sum)
		else:
			fuse_tag.append(0)
			fitness_fuse_dict["edp"].append(None)
			fitness_fuse_dict["energy"].append(None)
			fitness_fuse_dict["latency"].append(None)
	
	# --- get need to decide layer fuse id
	one_num = 0
	fuse_code = [0 for _ in range(len(fuse_tag))]
	no_fuse_code = [0 for _ in range(len(fuse_tag))]
	layer_fuse_select_list = []
	for id in range(len(fuse_tag)):
		tag = fuse_tag[id]
		if tag == 1:
			one_num += 1
		else:
			if one_num > 1:
				layer_fuse_select_list.append([id-one_num, id-1])
			elif one_num == 1:
				fuse_code[id-1] = 1
			one_num = 0
	print("fuse_tag:", fuse_tag)
	print("fuse_code:", fuse_code)
	
	def getFuseFitness(fuse_fitness_dict, layer_fitness_dict, code, final_tag = 0):
		
		print("fuse_fitness_dict:", fuse_fitness_dict)
		print("layer_fitness_dict:", layer_fitness_dict)
		print("code:", code)

		layer_num = len(fuse_fitness_dict["latency"]) + 1
		layer_code = [0 for _ in range(layer_num)]
		layer_tag_list = ['i' for _ in range(layer_num)]
		if final_tag == 1:
			print("final_code: ", code)
			print("code_lenth: ", len(code))
		for id in range(len(code)):
			fuse_num = code[id]
			if fuse_num == 1:
				assert(layer_code[id] == 0 and layer_code[id+1] == 0)
				layer_code[id] = 1
				layer_code[id+1] = 1
				layer_tag_list[id] = 'h'
				layer_tag_list[id+1] = 't'
			else:
				pass
		
		if final_tag == 1:
			print("final_layer_code: ", layer_code)
		
		latency_all = 0
		energy_all = 0
		for id in range(len(code)):
			if code[id] == 1:
				latency_all += code[id] * fuse_fitness_dict["latency"][id]
				energy_all += code[id] * fuse_fitness_dict["energy"][id]

		for id in range(layer_num):
			latency_all += (1-layer_code[id]) * layer_fitness_dict["latency"][id]
			energy_all += (1-layer_code[id]) * layer_fitness_dict["energy"][id]
		
		edp_all = latency_all * energy_all / PE_Frequency 
			
		return edp_all, latency_all, energy_all, layer_tag_list

	def exploreFuse(fuse_fitness_dict, layer_fitness_dict, num):
		assert(num == len(fuse_fitness_dict["latency"]))

		def generateCode(code_lenth):
			max_1_num = math.ceil(code_lenth/2)
			code_list = []
			one_num = max_1_num
			if max_1_num * 2 == num + 1:
				code = [0 for _ in range(code_lenth)]
				for i in range(max_1_num):
					code[i*2] = 1
				code_list.append(code)
				one_num -= 1
			
			add_zero_num = code_lenth - one_num * 2 + 1
			code_init = [1]
			for i in range(one_num-1):
				code_init.append(0)
				code_init.append(1)
			
			if add_zero_num == 1:
				for i in range(max_1_num+1):
					insert_id = i*2
					code = code_init[0:insert_id] + [0] + code_init[insert_id:]
					code_list.append(code)
			else:
				assert(add_zero_num == 2)
				for i1 in range(max_1_num+1):
					for i2 in range(i1, max_1_num+1):
						insert_id1 = i1 * 2
						insert_id2 = i2 * 2
						code = code_init[0:insert_id1] + [0] + code_init[insert_id1:insert_id2] + [0] + code_init[insert_id2:]
						code_list.append(code)
						assert(len(code) == code_lenth)
			return code_list		
		
		code_list = generateCode(num)
		
		fitness_record_list = []
		fitness_best = None
		code_best = None
		for code in code_list:
			fitness, latency_all, energy_all, layer_tag_list = getFuseFitness(fuse_fitness_dict, layer_fitness_dict, code)
			fitness_record_list.append(fitness)
			if fitness_best == None or fitness <= fitness_best:
				fitness_best = fitness
				code_best = code
		
		return fitness_best, code_best

	for [start_id, end_id] in layer_fuse_select_list:
		num = end_id - start_id + 1
		fuse_fitness_dict = {}
		fuse_fitness_dict["latency"] = fitness_fuse_dict["latency"][start_id : end_id + 1]
		fuse_fitness_dict["energy"] = fitness_fuse_dict["energy"][start_id : end_id + 1]
		layer_fitness_dict = {}
		layer_fitness_dict["latency"] = fitness_layer_dict["latency"][start_id : end_id + 2]
		layer_fitness_dict["energy"] = fitness_layer_dict["energy"][start_id : end_id + 2]

		fitness, code = exploreFuse(fuse_fitness_dict, layer_fitness_dict, num)

		fuse_code[start_id: end_id+1] = code[:]

	edp_all, latency_all, energy_all, layer_tag_list = getFuseFitness(fitness_fuse_dict, fitness_layer_dict, fuse_code, final_tag = 1)
	edp_all_no_fuse, latency_all_no_fuse, energy_all_no_fuse, layer_tag_list_no_fuse = getFuseFitness(fitness_fuse_dict, fitness_layer_dict, no_fuse_code)
	print("result_fitness: ", edp_all)
	print("code: ", fuse_code)
	print("result_fitness_initial: ", edp_all_no_fuse)

	layer_fuse_fitness_dict = {"latency":[], "NoC_NR":[], "L2_to_DRAM_NR":[], "DRAM_to_L2_NR":[]}
	for i in range(len(layer_tag_list)):
		tag = layer_tag_list[i]
		if tag == 'i':
			latency = fitness_init_dict["latency"][i]
			NoC_NR = fitness_init_dict["NoC_NR"][i]
			L2_to_DRAM_NR = fitness_init_dict["L2_to_DRAM_NR"][i]
			DRAM_to_L2_NR = fitness_init_dict["DRAM_to_L2_NR"][i]
		elif tag == 'h':
			latency = fitness_head_dict["latency"][i]
			NoC_NR = fitness_head_dict["NoC_NR"][i]
			L2_to_DRAM_NR = fitness_head_dict["L2_to_DRAM_NR"][i]
			DRAM_to_L2_NR = fitness_head_dict["DRAM_to_L2_NR"][i]
		elif tag == 't':
			latency = fitness_tail_dict["latency"][i]
			NoC_NR = fitness_tail_dict["NoC_NR"][i]
			L2_to_DRAM_NR = fitness_tail_dict["L2_to_DRAM_NR"][i]
			DRAM_to_L2_NR = fitness_tail_dict["DRAM_to_L2_NR"][i]
		layer_fuse_fitness_dict["latency"].append(latency)
		layer_fuse_fitness_dict["NoC_NR"].append(NoC_NR)
		layer_fuse_fitness_dict["L2_to_DRAM_NR"].append(L2_to_DRAM_NR)
		layer_fuse_fitness_dict["DRAM_to_L2_NR"].append(DRAM_to_L2_NR)

	return edp_all, latency_all, energy_all, layer_fuse_fitness_dict

def layer_no_fuse(fitness_init_dict):
	edp_all = 0
	latency_all = 0
	energy_all = 0
	edp_i_list = fitness_init_dict["edp"]
	energy_i_list = fitness_init_dict["energy"]
	latency_i_list = fitness_init_dict["latency"]

	for layer_id in range(len(edp_i_list)):
		energy_all += energy_i_list[layer_id]
		latency_all += latency_i_list[layer_id]
	
	edp_all = energy_all * latency_all / PE_Frequency

	print("new --------------------------")
	print("edp_all: ", edp_all)
	print("energy_all: ", energy_all)
	print("latency_all: ", latency_all)
	
	layer_fuse_fitness_dict = {"latency":[], "NoC_NR":[], "L2_to_DRAM_NR":[], "DRAM_to_L2_NR":[]}
	for i in range(len(edp_i_list)):
		latency = fitness_init_dict["latency"][i]
		NoC_NR = fitness_init_dict["NoC_NR"][i]
		L2_to_DRAM_NR = fitness_init_dict["L2_to_DRAM_NR"][i]
		DRAM_to_L2_NR = fitness_init_dict["DRAM_to_L2_NR"][i]
		layer_fuse_fitness_dict["latency"].append(latency)
		layer_fuse_fitness_dict["NoC_NR"].append(NoC_NR)
		layer_fuse_fitness_dict["L2_to_DRAM_NR"].append(L2_to_DRAM_NR)
		layer_fuse_fitness_dict["DRAM_to_L2_NR"].append(DRAM_to_L2_NR)

	return edp_all, latency_all, energy_all, layer_fuse_fitness_dict

def main(app_name, fuse_flag, architecture="ours", alg="GA", encode_type="index", dataflow="ours", PE_parallel="All", debug_open=0, save_all_records=0):
	# Variable
	# --- result_indir: 单网络逐层性能结果文件地址
	# --- result_outdir: 多层融合后整个网络性能结果文件地址
	abs_path = os.path.dirname(os.path.abspath(__file__))
	SE_abs_path = os.path.join(abs_path, "../nnparser_SE_hetero_iodie")
	result_outdir = os.path.join(abs_path, "SE_result")
	os.makedirs(result_outdir, exist_ok=True)
	result_outdir = os.path.join(result_outdir, alg + "_" + encode_type)
	os.makedirs(result_outdir, exist_ok=True)
	result_out_file = os.path.join(result_outdir, architecture + "_" + app_name + ".txt")
	file = open(result_out_file, "w")
	file.close()

	result_outdir = os.path.join(result_outdir, "BW_result")
	os.makedirs(result_outdir, exist_ok=True)
	result_outdir = os.path.join(result_outdir, app_name)
	os.makedirs(result_outdir, exist_ok=True)

	edp_dict = {}
	latency_dict = {}
	energy_dict = {}
	latency_BW_dict = {}

	for i in range(16):
		chiplet_num = i + 1
		result_indir = os.path.join(SE_abs_path, "result")
		result_indir = os.path.join(result_indir, "intraLayer")
		result_indir = os.path.join(result_indir, architecture + "_" + app_name)
		result_indir = os.path.join(result_indir, "chiplet_num_"+str(chiplet_num))
		result_indir = os.path.join(result_indir, alg + "_" + encode_type)
		result_indir_p = os.path.join(result_indir, chiplet_parallel_list[0] + "_and_" + PE_parallel)
		result_indir_pk = os.path.join(result_indir, chiplet_parallel_list[1] + "_and_" + PE_parallel)
		result_indir_k = os.path.join(result_indir, chiplet_parallel_list[2] + "_and_" + PE_parallel)

		fitness_init_dict, fitness_head_dict, fitness_tail_dict = txt_extract(result_indir_p, result_indir_pk, result_indir_k, app_name, fuse_flag)
		if fuse_flag == 1:
			edp_all, latency_all, energy_all, layer_fuse_fitness_dict = layer_fuse(fitness_init_dict, fitness_head_dict, fitness_tail_dict)
		else:
			edp_all, latency_all, energy_all, layer_fuse_fitness_dict = layer_no_fuse(fitness_init_dict)
		
		edp_dict[chiplet_num] = edp_all
		latency_dict[chiplet_num] = latency_all
		energy_dict[chiplet_num] = energy_all
		latency_BW_dict[chiplet_num] = copy.deepcopy(layer_fuse_fitness_dict)

		f = open(result_out_file, 'a')
		line = "chiplet_num\t{}\tedp_all\t{}\tlatency_all\t{}\tenergy_all\t{}".format(chiplet_num, edp_all, latency_all, energy_all)
		print(line, file=f)
		f.close()

		f = open(result_outdir + "/chiplet_num_"+str(chiplet_num) + ".txt", 'w')
		line = "layer_id\tlatency\tNoC_NR\tL2_to_DRAM_NR\tDRAM_to_L2_NR\t"
		print(line, file=f)
		for id in range(len(layer_fuse_fitness_dict["latency"])):
			layer_id = id + 1
			latency = layer_fuse_fitness_dict["latency"][id]
			NoC_NR = layer_fuse_fitness_dict["NoC_NR"][id]
			L2_to_DRAM_NR = layer_fuse_fitness_dict["L2_to_DRAM_NR"][id]
			DRAM_to_L2_NR = layer_fuse_fitness_dict["DRAM_to_L2_NR"][id]
			line = "{}\t{}\t{}\t{}\t{}".format(layer_id, latency, NoC_NR, L2_to_DRAM_NR, DRAM_to_L2_NR)
			print(line, file=f)
		f.close()
	
	return edp_dict, latency_dict, energy_dict, latency_BW_dict

def fitness_plot(fitness, fitness_init, app_name):
	x_f = []
	x_f_i = []
	for i in range(len(fitness)):
		chiplet_num = i+1
		x_f.append(chiplet_num-0.2)
		x_f_i.append(chiplet_num+0.2)
	
	plt.figure("chiplet num fuse result")
	plt.title(app_name, fontsize=12)
	plt.xlabel("Chiplet Num", fontsize=10)
	plt.ylabel("Fitness", fontsize=10)
	plt.bar(x_f, fitness, width=0.4, color="royalblue", label="fitness_with_fuse")
	plt.bar(x_f_i, fitness_init, width=0.4, color="lavender", label="fitness_without_fuse")
	plt.tick_params(labelsize=8)
	plt.legend()
	plt.savefig("./multi_nn_result/plot/" + app_name + ".png", bbox_inches = 'tight')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--app_name_line', type=str, default="resnet50", help='app_name_line,using+as split signal')	# simba , nnbaton
	parser.add_argument('--chiplet_num_max_TH', type=int, default=16, help='max chiplet_num')
	parser.add_argument('--fuse_flag', type=int, default=1, help='use layer fuse or not')
	opt = parser.parse_args()
	app_name_line = opt.app_name_line
	chiplet_num_max_TH = opt.chiplet_num_max_TH
	fuse_flag = opt.fuse_flag
	app_name_line.replace('\n',"")
	app_name_list = app_name_line.split("+")

	
	for app_name in app_name_list:
		edp_dict, latency_dict, energy_dict, latency_BW_dict = main(app_name, fuse_flag)





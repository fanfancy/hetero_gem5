import math
import os
import sys
import random
import argparse
from matplotlib import pyplot as plt

# Parameter
layer_list = {"resnet18":[1,2,2,2,2,6,7,7,7,10,11,11,11,14,15,15,15],"resnet50":[1,2,3,4,5,3,4,5,3,4,11,12,13,14,15,13,14,15,13,14,15,13,23,24,25,26,27,25,26,27,25,26,27,25,26,27,25,26,27,25,41,42,43,44,45,43,44,45,43,50],"VGG16":[1,2,3,4,5,6,6,8,9,9,11,11,13],"alexnet":[1,2,3,4,5,6,7,8],"lenet":[1,2,3,4,5]}
ratio = {}
chiplet_parallel_list = ["P_stable", "PK_stable", "K_stable"]

def txt_extract(app_name, chiplet_num, architecture="ours", alg="GA", encode_type="index", dataflow="ours", PE_parallel="All", debug_open=0, save_all_records=0):
	# Variable
	abs_path = os.path.dirname(os.path.abspath(__file__))
	result_abs_path = os.path.join(abs_path, "../nnparser_SE_hetero_iodie")
	app_result_file = os.path.join(abs_path, "SE_result")
	os.makedirs(app_result_file, exist_ok=True)
	app_result_file = os.path.join(app_result_file, alg + "_" + encode_type)
	os.makedirs(app_result_file, exist_ok=True)
	app_result_file = os.path.join(app_result_file, architecture + "_" + app_name + ".txt")
	if chiplet_num == 1:
		file = open(app_result_file, "w")
		file.close()

	result_indir = os.path.join(result_abs_path, "result")
	result_indir = os.path.join(result_indir, "intraLayer")
	result_indir = os.path.join(result_indir, architecture + "_" + app_name)
	result_indir = os.path.join(result_indir, "chiplet_num_"+str(chiplet_num))
	result_indir = os.path.join(result_indir, alg + "_" + encode_type)
	result_indir_p = os.path.join(result_indir, chiplet_parallel_list[0] + "_and_" + PE_parallel)
	result_indir_pk = os.path.join(result_indir, chiplet_parallel_list[1] + "_and_" + PE_parallel)
	result_indir_k = os.path.join(result_indir, chiplet_parallel_list[2] + "_and_" + PE_parallel)

	result_outdir = os.path.join(result_abs_path, "result")
	os.makedirs(result_outdir, exist_ok=True)
	result_outdir = os.path.join(result_outdir, "intraLayer")
	os.makedirs(result_outdir, exist_ok=True)
	result_outdir = os.path.join(result_outdir, architecture + "_" + app_name)
	os.makedirs(result_outdir, exist_ok=True)	
	result_outdir = os.path.join(result_outdir, "chiplet_num_"+str(chiplet_num))
	os.makedirs(result_outdir, exist_ok=True)
	result_outdir = os.path.join(result_outdir, alg + "_" + encode_type)
	os.makedirs(result_outdir, exist_ok=True)
	result_outdir = os.path.join(result_outdir, "extract_result")
	os.makedirs(result_outdir, exist_ok=True)
	result_outFile_p = os.path.join(result_outdir, "fitness_" + chiplet_parallel_list[0] + ".txt")
	result_outFile_pk = os.path.join(result_outdir, "fitness_" + chiplet_parallel_list[1] + ".txt")
	result_outFile_k = os.path.join(result_outdir, "fitness_" + chiplet_parallel_list[2] + ".txt")

	result_fuse_outFile = os.path.join(result_outdir, "fuse_fitness.txt")
	result_fitness_outFile = os.path.join(result_outdir, "initial_fitness.txt")

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

	def extractFitness(result_indir, result_outFile):
		result_file_init = os.path.join(result_indir, "final_result_record_initial.txt")
		result_file_head = os.path.join(result_indir, "final_result_record_headLayer.txt")
		result_file_tail = os.path.join(result_indir, "final_result_record_tailLayer.txt")

		f_init = open(result_file_init)
		f_head = open(result_file_head)
		f_tail = open(result_file_tail)

		lines_init = f_init.readlines()
		line_init_fitness = lines_init[0]
		line_init_par = lines_init[3]
		lines_head = f_head.readlines()
		line_head_fitness = lines_head[0]
		line_head_par = lines_head[3]
		lines_tail = f_tail.readlines()
		line_tail_fitness = lines_tail[0]
		line_tail_par = lines_tail[3]

		scale_init = getScaleRatio(line_init_par, layer_param_list)
		scale_head = getScaleRatio(line_head_par, layer_param_list)
		scale_tail = getScaleRatio(line_tail_par, layer_param_list)

		result_init = lineParse(line_init_fitness, ratio_list=scale_init)
		result_head = lineParse(line_head_fitness, ratio_list=scale_head)
		result_tail = lineParse(line_tail_fitness, ratio_list=scale_tail)


		fitness_dict = {}
		fitness_head_dict = {}
		fitness_tail_dict = {}
		fitness_init_dict = {}

		for layer_id in result_init:
			result_i = result_init[layer_id]
			fitness_init_dict[layer_id] = result_i
			if layer_id in result_head:
				result_h = 16*result_head[layer_id]
				result_h_sub = result_h - result_i
				fitness_head_dict[layer_id] = result_h
			else:
				result_h = None
				result_h_sub = None
			
			if layer_id in result_tail:
				result_t = 16*result_tail[layer_id]
				result_t_sub = result_t - result_i
				fitness_tail_dict[layer_id] = result_t
			else:
				result_t = None
				result_t_sub = None

			result_i = result_init[layer_id]
			fitness_dict[layer_id] = "\t" + str(result_i) + "\t" + str(result_h) + "\t" +  str(result_t) + "\t" + str(result_h_sub) + "\t" + str(result_t_sub)

		result_file_out = open(result_outFile, 'w')

		line1 = app_name + "\t" + "init" + "\t" + "4*head" + "\t" + "4*tail" + "\t" + "4*head-init" + "\t" + "4*tail-init"
		print(line1, file = result_file_out)

		layer_num = 1
		fitness_head_list = []
		fitness_tail_list = []
		fitness_init_list = []
		for index in layer_list[app_name]:
			layer_id = "layer" + str(index)
			layer_cur = "layer" + str(layer_num)
			
			line_fitness = layer_cur + fitness_dict[layer_id]
			print(line_fitness, file=result_file_out)

			fitness_init_list.append(fitness_init_dict[layer_id])
			
			if layer_id in fitness_head_dict:
				fitness_head_list.append(fitness_head_dict[layer_id])
			
			if layer_id in fitness_tail_dict:
				fitness_tail_list.append(fitness_tail_dict[layer_id])

			layer_num += 1
		result_file_out.close()
		
		return fitness_init_list, fitness_head_list, fitness_tail_list#, latency_init_list, latency_head_list, latency_tail_list

	def getMinFrom3(a1, a2, a3, dim_list):
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

	# Get network param
	# 获得网络参数
	layer_param_list = getLayerParam(app_name)

	# Fitness per chiplet_parallel extract and file out
	# 获取每层在三种片间并行度方案下的适应度数值
	fitness_init_list_p, fitness_head_list_p, fitness_tail_list_p = extractFitness(result_indir_p, result_outFile_p)
	fitness_init_list_pk, fitness_head_list_pk, fitness_tail_list_pk = extractFitness(result_indir_pk, result_outFile_pk)
	fitness_init_list_k, fitness_head_list_k, fitness_tail_list_k = extractFitness(result_indir_k, result_outFile_k)

	# Fitness per Layer extract and file out
	# 获得每层的最优的适应度数值
	result_file_out_initial = open(result_fitness_outFile, 'w')
	fitness_all_initial = 0
	title = "layerID" + "min_result" + "\t" + "min_index"
	print(title, file = result_file_out_initial)
	assert(len(fitness_init_list_pk)==len(fitness_init_list_k))
	assert(len(fitness_init_list_pk)==len(fitness_init_list_p))
	result_layer_list = []
	for id in range(len(fitness_init_list_p)):
		layer_index = "Layer" + str(id+1)
		result_p = fitness_init_list_p[id]
		result_pk = fitness_init_list_pk[id]
		result_k = fitness_init_list_k[id]

		min, min_index = getMinFrom3(result_p, result_pk, result_k, ["P", "PK", "K"])

		result_layer_list.append(min)
		fitness_all_initial += min

		context_line = layer_index + "\t" + str(min) + "\t" + str(min_index)
		print(context_line, file = result_file_out_initial)
	result_file_out_initial.close()

	# Fitness by fuse extract and file out
	result_file_out_fuse = open(result_fuse_outFile, 'w')
	title = "LayerID" + "\t" + "initial_min" + "\t" + "min_result" + "\t" + "min_index" + "\t" + "fuseNeed"
	print(title, file = result_file_out_fuse)
	assert(len(fitness_head_list_p)==len(fitness_tail_list_p))
	assert(len(fitness_head_list_k)==len(fitness_tail_list_k))
	assert(len(fitness_head_list_pk)==len(fitness_tail_list_pk))
	assert(len(fitness_head_list_pk)==len(fitness_head_list_k))
	assert(len(fitness_head_list_pk)==len(fitness_head_list_p))
	result_fuse_list = []
	fuse_tag_list = []
	for id in range(len(fitness_head_list_p)):
		result_initial_min = result_layer_list[id] + result_layer_list[id+1]

		layer_index = "Layer({}-{})".format(id+1, id+2)

		#result_p = fitness_head_list_p[id] + fitness_tail_list_p[id]
		#result_pk = fitness_head_list_pk[id] + fitness_tail_list_pk[id]
		#result_k = fitness_head_list_k[id] + fitness_tail_list_k[id]
		#min, min_index = getMinFrom3(result_p, result_pk, result_k, ["P", "PK", "K"])
		fitness_head_min, id_1 =  getMinFrom3(fitness_head_list_p[id], fitness_head_list_pk[id], fitness_head_list_k[id], ["P", "PK", "K"])
		fitness_tail_min, id_1 =  getMinFrom3(fitness_tail_list_p[id], fitness_tail_list_pk[id], fitness_tail_list_k[id], ["P", "PK", "K"])
		min = fitness_head_min + fitness_tail_min
		result_fuse_list.append(min)

		if min <= result_initial_min:
			fuse_tag = 1
		else:
			fuse_tag = 0
		fuse_tag_list.append(fuse_tag)
		context_line = layer_index + "\t" + str(result_initial_min) + "\t" + str(min) + "\t" + str(min_index) + "\t" +str(fuse_tag)
		print(context_line, file = result_file_out_fuse)
	result_file_out_fuse.close()

	# Get Layer Fuse result
	fuse_select_list = []

	# --- get need to decide layer fuse id
	one_num = 0
	fuse_code = [0 for _ in range(len(fuse_tag_list))]
	print("fuse_tag_list : ", fuse_tag_list)
	print("len(fuse_tag_list): ",len(fuse_tag_list))
	for id in range(len(fuse_tag_list)):
		tag = fuse_tag_list[id]
		if tag == 1:
			one_num += 1
		else:
			if one_num > 1:
				fuse_select_list.append([id-one_num, id-1])
			elif one_num == 1:
				fuse_code[id-1] = 1
			one_num = 0

	def getFuseFitness(fuse_fitness_list, layer_fitness_list, code, final_tag = 0):
		layer_num = len(layer_fitness_list)
		layer_code = [0 for _ in range(layer_num)]
		if final_tag == 1:
			print("final_code: ", code)
			print("code_lenth: ", len(code))
		for id in range(len(code)):
			fuse_num = code[id]
			if fuse_num == 1:
				assert(layer_code[id] == 0 and layer_code[id+1] == 0)
				layer_code[id] = 1
				layer_code[id+1] = 1
			else:
				pass
		
		if final_tag == 1:
			print("final_layer_code: ", layer_code)
		
		fitness_all = 0
		for id in range(len(code)):
			fitness_all += code[id] * fuse_fitness_list[id]

		for id in range(layer_num):
			fitness_all += (1-layer_code[id]) * layer_fitness_list[id]
			
		return fitness_all

	def exploreFuse(fuse_fitness_list, layer_fitness_list, num):
		assert(num == len(fuse_fitness_list))

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
			fitness = getFuseFitness(fuse_fitness_list, layer_fitness_list, code)
			fitness_record_list.append(fitness)
			if fitness_best == None or fitness <= fitness_best:
				fitness_best = fitness
				code_best = code
		
		return fitness_best, code_best


	for [start_id, end_id] in fuse_select_list:
		num = end_id - start_id + 1

		fuse_fitness_list = result_fuse_list[start_id : end_id + 1]
		layer_fitness_list = result_layer_list[start_id : end_id + 2]

		fitness, code = exploreFuse(fuse_fitness_list, layer_fitness_list, num)

		fuse_code[start_id: end_id+1] = code[:]

	fitness_all = getFuseFitness(result_fuse_list, result_layer_list, fuse_code, final_tag = 1)
	print("result_fitness: ", fitness_all)
	print("code: ", fuse_code)
	print("result_fitness_initial: ", fitness_all_initial)

	f = open(app_result_file, 'a')
	line = "chiplet_num\t{}\tfitness_all\t{}".format(chiplet_num, fitness_all)
	print(line, file=f)
	f.close()
	return fitness_all, fitness_all_initial

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
	opt = parser.parse_args()
	app_name_line = opt.app_name_line
	chiplet_num_max_TH = opt.chiplet_num_max_TH
	app_name_line.replace('\n',"")
	app_name_list = app_name_line.split("+")

	
	for app_name in app_name_list:
		outfile = "./multi_nn_result/" + app_name + "_fitness.txt"
		f_o = open(outfile, 'w')
		result_list = []
		result_initial_list = []
		for i in range(chiplet_num_max_TH):
			chiplet_num = i + 1
			result, result_initial = txt_extract(app_name, chiplet_num)
			result_list.append(result)
			result_initial_list.append(result_initial)
			print("chiplet_num\t{}\tno_fuse_result\t{}\tfuse_result\t{}".format(chiplet_num, result, result_initial), file=f_o)
		
		fitness_plot(result_list, result_initial_list, app_name)
from linecache import lazycache
import os
import copy
import random
from matplotlib import pyplot as plt
import argparse
import numpy as np
import datetime
PE_Frequency = 1000 * 1000 * 1000
Optimization_Objective_index = {"edp": 0, "energy": 2, "latency": 1}

# --- debug param
debug_in_getIdealParam = False
debug_in_BWR = False
debug_in_BWR_simple = False
debug_in_evaluation_tp_sp = False
debug_in_evoluation_temporal_spatial = False
debug_in_evoluation_temporal_spatial_simple = False
debug_in_record_fitness_iter = True

cur_dir = os.path.dirname(os.path.abspath(__file__))
SE_evaluation_dir = os.path.join(cur_dir, "../nnparser_SE_hetero_iodie/")
nn_param_dir = os.path.join(SE_evaluation_dir, "nn_input_noc_nop")
SE_result_dir = os.path.join(SE_evaluation_dir,"result/intraLayer")

result_outdir = os.path.join(cur_dir,"multi_nn_result")
os.makedirs(result_outdir, exist_ok=True)
result_plot = os.path.join(result_outdir, "plot")
os.makedirs(result_plot, exist_ok=True)

def getNNParam(nn_name):
	# 获取神经网络的计算量和参数量信息
	# --- 对神经网络性能进行理论分析
	nn_file_path = os.path.join(nn_param_dir, nn_name + ".txt")
	nn_f = open(nn_file_path)

	print("network model ----- " + nn_name + " -------------")

	layer_computation_num_dict = {}
	layer_param_num_dict = {}
	lines = nn_f.readlines()
	nnParam_dict = {}
	layer_id = 1
	for line in lines:
		if line.startswith("#") or line.startswith("*"):
			pass
		else:
			line = line.replace("\n","")
			line_item = line.split(" ")
			layer_name = line_item[0]
			#layer_id = int(layer_name.replace("layer",""))
			H = int(line_item[1])
			M = int(line_item[2])
			P = int(line_item[8])
			Q = int(line_item[9])
			C = int(line_item[3])
			K = int(line_item[7])
			R = int(line_item[4])
			S = int(line_item[4])
			layer_computation_num = P * Q * K * R * S * C
			layer_param_num = H*M*C + P*Q*K + R*S*C*K
			layer_computation_num_dict[layer_name] = layer_computation_num
			layer_param_num_dict[layer_name] = layer_param_num
			nnParam_dict[layer_id] = layer_computation_num * layer_param_num * layer_param_num / 10000000000000000
			layer_id += 1
	nn_f.close()
	return nnParam_dict

class multi_network_DSE:
	def __init__(self, architecture, chiplet_num, workload_dict, Optimization_Objective, tp_TH, sp_TH, BW_tag = 1, debug_tag = 0):
		self.architecture = architecture
		self.chiplet_num = chiplet_num
		self.workload_dict = workload_dict
		self.workload_list = []
		self.debug_tag = debug_tag
		self.BW_Reallocate_tag = BW_tag
		self.Optimization_Objective = Optimization_Objective
		
		# --- TH Param ---
		self.space_tp_TH = tp_TH			# max tp num
		self.space_sp_TH = sp_TH			# max sp num
		self.chiplet_partition_TH = 100		# max chiplet partition num per TP Tile
		self.workload_schedule_TH = 1000	# max workload schedule method num

		self.chiplet_partition_dict = None

		# --- workload fitness variable ---
		self.ideal_param_dict = None
		self.ideal_param_dict_workload = None
		self.workload_fitness_dict = {}
		self.workload_BWNeeded_dict = {}
		self.merge_workload_fitness_dict = None
		self.merge_workload_BWNeeded_dict = None
		self.merge_workload_dict = None

		# --- record variable ---
		self.fitness_best = None
		self.fitness_record = []
		self.schedule_best = None
		self.schedule_record = []
		self.Nchip_partition_best = None
		self.Nchip_par_record = []
		self.tp_sp_space_best = None
		self.tp_sp_space_record = []
		self.sample_num = 0
	
	# initialize: 初始化操作
	def initialize(self):
		self.getIdealParam()
		self.setTotalWorkloadFitness()
		self.best_fitness = None
		self.fitness_record = []
		self.sample_record = []
		self.distance_record = []

	# getIdealParam: 获得理论计算的各任务的参数量信息
	def getIdealParam(self):
		self.ideal_param_dict = {}
		self.ideal_param_dict_workload = {}
		if debug_in_getIdealParam:
			print("DEBUG IN getIdealParam()--------------------------")
			print("---ideal_param_dict_workload: ")
		for nn_name, workload in self.workload_dict.items():
			self.ideal_param_dict[nn_name] = copy.deepcopy(getNNParam(nn_name))
			for w_name, [start_id, end_id] in workload.items():
				workload_name = nn_name + w_name
				self.workload_list.append(workload_name)
				param = 0
				for layer_id in range(start_id, end_id+1):
					param += self.ideal_param_dict[nn_name][layer_id]
				self.ideal_param_dict_workload[workload_name] = param

				if debug_in_getIdealParam:
					print("---{}: {}".format(workload_name, param))
		
		if debug_in_getIdealParam:
			print("END------------------------------------------------")
			exit()
	
	# setTotalWorkloadFitness: 获得每个Workload的性能参数
	def setTotalWorkloadFitness(self):
		for workload in self.workload_list:
			file = "{}/SE_result/{}/{}{}.txt".format(cur_dir, self.architecture, workload, "_startTile")
			f = open(file)

			file_n = "{}/SE_result/{}/{}{}.txt".format(cur_dir, self.architecture, workload, "_midTile")
			f_n = open(file_n)

			lines = f.readlines()
			lines_n = f_n.readlines()
			self.workload_fitness_dict[workload] = {}
			for line_id in range(len(lines)):
				line = lines[line_id]
				line_n = lines_n[line_id]
				if line.startswith("chiplet"):
					line_item = line.replace("\n","").split("\t")
					line_item_n = line_n.replace("\n","").split("\t")
					chiplet_num = int(line_item[1])
					assert(chiplet_num == int(line_item_n[1]))
					edp = float(line_item[3])
					latency = float(line_item[5])
					energy = float(line_item[7])
					edp_n = float(line_item_n[3])
					latency_n = float(line_item_n[5])
					energy_n = float(line_item_n[7])
					self.workload_fitness_dict[workload][chiplet_num] = {}
					self.workload_fitness_dict[workload][chiplet_num]["startTile"] = [edp, latency, energy]
					self.workload_fitness_dict[workload][chiplet_num]["midTile"] = [edp_n, latency_n, energy_n]
			
			self.workload_BWNeeded_dict[workload] = {}
			BW_dir = cur_dir + "/SE_result/" + self.architecture + "/BW_result/" + workload +"_startTile/"
			BW_dir_n = cur_dir + "/SE_result/" + self.architecture + "/BW_result/" + workload + "_midTile/"
			for i in range(self.chiplet_num):
				chiplet_num = i + 1
				self.workload_BWNeeded_dict[workload][chiplet_num] = {}
				BW_file = BW_dir + "chiplet_num_" + str(chiplet_num) + ".txt"
				BW_f = open(BW_file)
				lines = BW_f.readlines()
				BW_file_n = BW_dir_n + "chiplet_num_" + str(chiplet_num) + ".txt"
				BW_f_n = open(BW_file_n)
				lines_n = BW_f_n.readlines()
				for line_id in range(len(lines)):
					line = lines[line_id]
					line_n = lines_n[line_id]
					if line.startswith("layer_id") or line == "":
						pass
					else:
						line_item = line.replace("\n","").split("\t")
						layer_id = int(line_item[0])
						latency = float(line_item[1])
						NoC_NR = float(line_item[2])
						L2_to_DRAM_NR = float(line_item[3])
						DRAM_to_L2_NR = float(line_item[4])
						BW_list = [latency, NoC_NR, L2_to_DRAM_NR, DRAM_to_L2_NR]

						line_item_n = line_n.replace("\n","").split("\t")
						layer_id_n = int(line_item_n[0])
						latency_n = float(line_item_n[1])
						NoC_NR_n = float(line_item_n[2])
						L2_to_DRAM_NR_n = float(line_item_n[3])
						DRAM_to_L2_NR_n = float(line_item_n[4])
						BW_list_n = [latency_n, NoC_NR_n, L2_to_DRAM_NR_n, DRAM_to_L2_NR_n]

						assert(layer_id == layer_id_n)

						self.workload_BWNeeded_dict[workload][chiplet_num][layer_id] = {}
						self.workload_BWNeeded_dict[workload][chiplet_num][layer_id]["startTile"] = BW_list
						self.workload_BWNeeded_dict[workload][chiplet_num][layer_id]["midTile"] = BW_list_n
	
	# calIdealParam: 计算每个[tp_id, sp_id]内任务的理论参数量
	def calIdealParam(self, tp_sp_space):
		tp_sp_idealParam = {}
		for tp_id, sp_space  in tp_sp_space.items():
			tp_sp_idealParam[tp_id] = {}
			for sp_id, workload_list in sp_space.items():
				tp_sp_idealParam[tp_id][sp_id] = 0
				for workload in workload_list:
					tp_sp_idealParam[tp_id][sp_id] += self.ideal_param_dict_workload[workload]
		return tp_sp_idealParam

	# mergeWorkload: 对于一个Tile内的相同网络的子任务进行合并
	def mergeWorkload(self, tp_sp_space):
		self.merge_workload_fitness_dict = {}
		self.merge_workload_BWNeeded_dict = {}
		self.merge_workload_dict = {}
		merge_tp_sp_space = {}
		for tp_id, sp_space in tp_sp_space.items():
			merge_tp_sp_space[tp_id] = {}
			for sp_id, w_list in sp_space.items():
				app_name_list = {}
				merge_tp_sp_space[tp_id][sp_id] = []
				for w_name in w_list:
					w_items = w_name.split("w")
					app_name = w_items[0]
					w_id = int(w_items[1])
					if app_name not in app_name_list:
						app_name_list[app_name] = []
					app_name_list[app_name].append(w_id)

				for app_name, app_w_list in app_name_list.items():
					app_w_list.sort()
					app_w_name_list = []
					w_tag_dict = {}
					merge_workload_name = ""

					if app_name not in self.merge_workload_dict:
						self.merge_workload_dict[app_name] = {}
					end_layer_id = 0
					start_layer_id = 0
					for w_id in app_w_list:
						if w_id == app_w_list[0]:
							tag = "startTile"
							start_layer_id = self.workload_dict[app_name]["w" + str(w_id)][0]
						else:
							tag = "midTile"
						merge_workload_name += "w" + str(w_id)
						w_name = app_name + "w" + str(w_id)
						app_w_name_list.append(w_name)
						end_layer_id = self.workload_dict[app_name]["w" + str(w_id)][1]
						w_tag_dict[w_name] = tag
					
					self.merge_workload_dict[app_name][merge_workload_name] = [start_layer_id, end_layer_id]
					merge_workload_name = app_name + merge_workload_name
					merge_tp_sp_space[tp_id][sp_id].append(merge_workload_name)

					app_workload_fitness_dict = {}
					app_workload_BW_dict = {}
					layer_offset = 0
					for w_name, tag in w_tag_dict.items():
						for chiplet_num, fitness_list in self.workload_fitness_dict[w_name].items():
							if chiplet_num not in app_workload_fitness_dict:
								app_workload_fitness_dict[chiplet_num] = [0,0,0]
								app_workload_BW_dict[chiplet_num] = {}
							app_workload_fitness_dict[chiplet_num][1] += fitness_list[tag][1]
							app_workload_fitness_dict[chiplet_num][2] += fitness_list[tag][2]
							
							for layer_id, BW_list in self.workload_BWNeeded_dict[w_name][chiplet_num].items():
								layer_id_real = layer_id + layer_offset
								assert(layer_id_real not in app_workload_BW_dict[chiplet_num])
								app_workload_BW_dict[chiplet_num][layer_id_real] = BW_list[tag]
						layer_offset = layer_id_real

					self.merge_workload_fitness_dict[merge_workload_name] = copy.deepcopy(app_workload_fitness_dict)
					self.merge_workload_BWNeeded_dict[merge_workload_name] = copy.deepcopy(app_workload_BW_dict)
					for chiplet_num, fitness_list in self.merge_workload_fitness_dict[merge_workload_name].items():
						edp = fitness_list[1] * fitness_list[2] / PE_Frequency
						self.merge_workload_fitness_dict[merge_workload_name][chiplet_num][0] = edp
		
		return merge_tp_sp_space
	
	# getChipletPartionDict: 获得chiplet数目的可能拆分
	# chiplet_partition_dict : 
		## { 1:[[36]],
		##   2:[[1,35],[2,34]...],
		##   ...
		##   max_sp_num:[[],[],...] }
	def getChipletPartionDict(self, max_sp_num):
		par_per_sp_TH = 100
		par_num = 0
		unchange_TH = 50
		unchange_times = 0

		chiplet_partition_dict = {}
		for i in range(max_sp_num):
			sp_num = i + 1
			par_num = 0
			unchange_times = 0
			chiplet_partition_dict[sp_num] = []
			while(1):
				chiplet_partition = []
				Nchip_rest = self.chiplet_num - sp_num
				for i in range(sp_num-1):
					Nchip = random.randint(0, Nchip_rest)
					chiplet_partition.append(Nchip+1)
					Nchip_rest -= Nchip
				chiplet_partition.append(Nchip_rest+1)
				chiplet_partition.sort()
				if chiplet_partition not in chiplet_partition_dict[sp_num]:
					chiplet_partition_dict[sp_num].append(chiplet_partition)
					unchange_times = 0
					par_num += 1
				else:
					unchange_times += 1
					
				if unchange_times == unchange_TH or par_num == par_per_sp_TH:
					break
		return chiplet_partition_dict

	# calSP: SP Tile Fitness Computation
	def calSP(self, fitness_dict, id):
		if id == None:
			sp_name = "("
		else:
			sp_name = "sp" + str(id) + "("
		latency_sp = 0
		energy_sp = 0
		for name, fitness in fitness_dict.items():
			sp_name += name + "_"
			latency_sp += fitness[1]
			energy_sp += fitness[2]
		sp_name = sp_name.strip("_")
		sp_name += ")"

		edp_sp = latency_sp * energy_sp / PE_Frequency
		fitness_list = [edp_sp, latency_sp, energy_sp]
		return sp_name, fitness_list
	
	# calTP: TP Tile Fitness Computation
	def calTP(self, fitness_dict, id):
		tp_name = "tp" + str(id) + "("
		latency_tp = 0
		energy_tp = 0
		for name, fitness in fitness_dict.items():
			tp_name += name + "_"
			if fitness[1] > latency_tp:
				latency_tp = fitness[1]
			energy_tp += fitness[2]
		tp_name = tp_name.strip("_")
		tp_name += ")"

		edp_tp = latency_tp * energy_tp / PE_Frequency
		fitness_list = [edp_tp, latency_tp, energy_tp]
		return tp_name, fitness_list

	# BW_Reallocate: Bandwidth reallocate
	def BW_Reallocate(self, sp_workload_fitness, sp_Nchip):
		sp_workload_fitness_bw = {}
		bw_reallocate_result = {}
		if len(sp_workload_fitness.keys()) == 1:
			for sp_id, workload_fitness in sp_workload_fitness.items():
				sp_workload_fitness_bw[sp_id] = {}
				bw_reallocate_result[sp_id] = {}
				for workload, fitness in workload_fitness.items():
					sp_workload_fitness_bw[sp_id][workload] = fitness
		else:
			"""
			sp_workload_fitness =  
			{
				0: {
					'resnet18': [1793044.2722481424, 514782.6666666666, 3483109258.2400002], 
					'resnet50': [9630241.645624578, 1326631.466666666, 7259168719.872001]
				}, 
				1: {'resnet50same': [15876227.827040443, 1926204.2666666668, 8242234793.984004]}
			}
			sp_Nchip =  {1: 6, 0: 10}
			"""
			#########################################################
			# init
			#########################################################
			delay_min_dict = {}
			degrad_ratio_dict = {}
			dram_to_L2_min_dict = {}
			L2_to_DRAM_min_dict = {}
			total_cycle_dict = {}
			total_cycle_list = []
			eva_nn_chiplet_num_dict = {}

			for sp_id, workload_fitness in sp_workload_fitness.items():
				sp_workload_fitness_bw[sp_id] = {}
				bw_reallocate_result[sp_id] = {}
				Nchip = sp_Nchip[sp_id]
				if debug_in_BWR:
					print('workload_fitness.keys() = ', workload_fitness.keys())
					print('Nchip = ', Nchip)
				delay_min_dict[sp_id] = {}
				degrad_ratio_dict[sp_id] = {}
				L2_to_DRAM_min_dict[sp_id] = {}
				dram_to_L2_min_dict[sp_id] = {}
				eva_nn_chiplet_num_dict[sp_id] = Nchip

				for workload, fitness in workload_fitness.items():
					sp_workload_fitness_bw[sp_id][workload] = fitness
					workload_BW_list = self.merge_workload_BWNeeded_dict[workload][Nchip]
					if debug_in_BWR:
						print('workload_BW_list = ', workload_BW_list)
					
					for layer_id, BW_list in workload_BW_list.items():
						# latency = BW_list[0]
						# NoC_NR = BW_list[1]
						# L2_to_DRAM_NR = BW_list[2]
						# DRAM_to_L2_NR = BW_list[3]
						delay_min_dict[sp_id][workload + '_' + str(layer_id)] = BW_list[0]
						degrad_ratio_dict[sp_id][workload + '_' + str(layer_id)] = max(BW_list[1], BW_list[2], BW_list[3])
						L2_to_DRAM_min_dict[sp_id][workload + '_' + str(layer_id)] = BW_list[2]
						dram_to_L2_min_dict[sp_id][workload + '_' + str(layer_id)] = BW_list[3]

			if debug_in_BWR:
				print('delay_min_dict = ', delay_min_dict)
				print('degrad_ratio_dict = ', degrad_ratio_dict)
				print('dram_to_L2_min_dict = ', dram_to_L2_min_dict)
				print('L2_to_DRAM_min_dict = ', L2_to_DRAM_min_dict)
			
			for nn_name in sp_workload_fitness.keys():
				total_cycle_dict[nn_name] = {}
				cycle = 0
				for layer, item in delay_min_dict[nn_name].items():
					cycle += item
					total_cycle_dict[nn_name][layer] = int(cycle)
					total_cycle_list.append(int(cycle))
			# total_cycle = max(total_cycle_dict.values())
			total_cycle_list.sort()
			if debug_in_BWR:
				print('total_cycle_dict = ', total_cycle_dict)
				print('total_cycle_list = ', total_cycle_list)
			######################## event ########################
			event_dict = {}
			old_tick = 0
			index = 0
			while index < len(total_cycle_list):
				nn_name_list = list(total_cycle_dict.keys())
				tick = total_cycle_list[index]
				state = {}
				for nn_name in total_cycle_dict.keys():
					layer_list = list(total_cycle_dict[nn_name].keys())
					now_layer = layer_list[0]
					for i, layer in enumerate(layer_list):
						if total_cycle_dict[nn_name][layer] == tick:
							now_layer = layer
						elif total_cycle_dict[nn_name][layer] < tick and i < len(layer_list) - 1:
							now_layer = layer_list[i + 1]
						else:
							break
					state[nn_name] = now_layer
				event_dict[tick] = state

				degrad_ratio_list = []
				for nn_name in state.keys():
					degrad_ratio_list.append(degrad_ratio_dict[nn_name][state[nn_name]])
				
				if debug_in_BWR:
					print(tick, state)
				
				if debug_in_BWR_simple:
					print(degrad_ratio_list, sum(np.array(degrad_ratio_list) == 1))

				######################## update ########################
				if sum(np.array(degrad_ratio_list) <= 1) >= 1 and sum(np.array(degrad_ratio_list) <= 1) < len(degrad_ratio_list):
					nn_name = nn_name_list[np.argmin(degrad_ratio_list)]
					improve_ratio = (1 - max(dram_to_L2_min_dict[nn_name][state[nn_name]], L2_to_DRAM_min_dict[nn_name][state[nn_name]])) / (len(degrad_ratio_list) - 1) * (eva_nn_chiplet_num_dict[nn_name] % 4)
					nn_name_list.remove(nn_name)

					if debug_in_BWR_simple:
						print('improve_ratio = ', improve_ratio)
						print('tick - old_tick = ', tick - old_tick)
					for nn in nn_name_list:
						if debug_in_BWR:
							print('int(((tick - old_tick) * improve_ratio) / eva_nn_chiplet_num_dict[nn]) = ', int(((tick - old_tick) * improve_ratio) / eva_nn_chiplet_num_dict[nn]))
						for layer in total_cycle_dict[nn].keys():
							if total_cycle_dict[nn][layer] >= tick:
								total_cycle_dict[nn][layer] -= int(((tick - old_tick) * improve_ratio) / eva_nn_chiplet_num_dict[nn])
					
					total_cycle_list = []
					for nn in eva_nn_chiplet_num_dict.keys():
						cycle = 0
						for layer in total_cycle_dict[nn].keys():
							total_cycle_list.append(total_cycle_dict[nn][layer])
					total_cycle_list.sort()

					tick = total_cycle_list[index]
					if debug_in_BWR:
						print('--------------')
						print('index = ', index)
						print('tick = ', tick)
						print('new_total_cycle_dict = ', total_cycle_dict)
						print('total_cycle_list = ', total_cycle_list)
				old_tick = tick
				index += 1

			#########################################################
			# output
			#########################################################
			for sp_id, workload_fitness in sp_workload_fitness.items():
				for workload, fitness in workload_fitness.items():
					workload_BW_list = self.merge_workload_BWNeeded_dict[workload][Nchip]
					layer_id = list(workload_BW_list.keys())[-1]
					latency_sp = total_cycle_dict[sp_id][workload + '_' + str(layer_id)]
					energy_sp = sp_workload_fitness_bw[sp_id][workload][2]
					edp_sp = latency_sp * energy_sp / PE_Frequency
					sp_workload_fitness_bw[sp_id][workload] = [edp_sp, latency_sp, energy_sp]
		return sp_workload_fitness_bw, bw_reallocate_result

	# getChipletAllocateMethods_random: get chiplet allocate method randomly
	def getChipletAllocateMethods_random(self, sp_idealParam, TH):
		sp_num = len(sp_idealParam)
		method_num = len(self.chiplet_partition_dict[sp_num])

		# id_list: 随机化选取的方案的id
		# --- 随机获取chiplet allocate方案中的min(method_num, TH)个方案
		id_list = list(range(method_num))
		random.shuffle(id_list)

		sp_idealParam_order = sorted(sp_idealParam.items(), key = lambda x: x[1])
		sp_Nchip_list = []

		for i in range(TH):
			if i >= method_num:
				break
			index = id_list[i]
			chip_num_list = self.chiplet_partition_dict[sp_num][index]	# 1. 获取的划分的方案
			
			# 2. 根据SP Tile理论的参数量sp_idealParam_order对比，分配对应的chiplet数目
			sp_Nchip = {}
			for ii in range(sp_num):
				chip_num = chip_num_list[ii]
				sp_id = sp_idealParam_order[ii][0]
				sp_Nchip[sp_id] = chip_num
			assert(sp_Nchip not in sp_Nchip_list)
			sp_Nchip_list.append(sp_Nchip)
		return sp_Nchip_list

	# evaluation_tp_sp: 评估函数，评估任务调度方案
	# ---include: chiplet allocate + bandwidth reallocate + evaluation
	def evaluation_tp_sp(self, tp_sp_space, tp_sp_idealParam):
		
		if debug_in_evaluation_tp_sp:
			print("DEBUG IN evaluation_tp_sp()----------------------")

		tp_sp_fitness = {}
		bw_reallocate_result = {}
		best_Nchip_partition = {}

		# 评估开始：

		# 1. 每个TP Tile内部探索与评估，以TP Tile的fitness最优为指标迭代
		# ---每个TP Tile内部进行chiplet Allocate的探索
		# ---每个SP Tile内部进行BW Reallocate的探索
		for tp_id, sp_space in tp_sp_space.items():
			# Evaluation in the same TP Tile
			
			sp_idealParam = tp_sp_idealParam[tp_id]	
			if debug_in_evaluation_tp_sp:
				print('tp_id = ', tp_id)
				print('---sp_idealParam = ', sp_idealParam)

			# 1.1 Randomize Chiplet Allocate Methods
			sp_Nchip_list = self.getChipletAllocateMethods_random(sp_idealParam, self.chiplet_partition_TH)

			# 1.2 Chiplet Allocate 方案遍历
			best_tp_fitness = None
			best_bw_reallocate = None
			best_tp_name = None
			best_Nchip = None
			for sp_Nchip in sp_Nchip_list:
				# 1.2.1 Get Workload Fitness
				sp_workload_fitness = {}
				for sp_id in sp_Nchip:
					sp_workload_fitness[sp_id] = {}
					Nchip = sp_Nchip[sp_id]
					for workload in sp_space[sp_id]:
						fitness = self.merge_workload_fitness_dict[workload][Nchip]
						sp_workload_fitness[sp_id][workload] = fitness
				if debug_in_evaluation_tp_sp:
					print('sp_Nchip = ', sp_Nchip)
					print('sp_workload_fitness = ', sp_workload_fitness)
				
				# 1.2.2 BW Reallocate
				if self.BW_Reallocate_tag == 1:
					sp_workload_fitness_bw , bw_reallocate = self.BW_Reallocate(sp_workload_fitness, sp_Nchip)
					if debug_in_BWR_simple and debug_in_evaluation_tp_sp:
						print('sp_workload_fitness_bw = ', sp_workload_fitness_bw)
						print('bw_reallocate = ', bw_reallocate)
				else:
					sp_workload_fitness_bw = copy.deepcopy(sp_workload_fitness)
					bw_reallocate = None
				
				# 1.2.3 Cal Fitness
				sp_fitness_dict = {}
				for sp_id, workload_fitness in sp_workload_fitness_bw.items():
					sp_name, fitness = self.calSP(workload_fitness, sp_id)
					sp_fitness_dict[sp_name] = fitness
				tp_name, tp_fitness = self.calTP(sp_fitness_dict, tp_id)

				# 1.2.4 迭代比较
				Objective_id = Optimization_Objective_index[self.Optimization_Objective]
				if best_tp_fitness == None or tp_fitness[Objective_id] < best_tp_fitness[Objective_id]:
					best_tp_fitness = copy.deepcopy(tp_fitness)
					best_bw_reallocate = copy.deepcopy(bw_reallocate)
					best_tp_name = tp_name
					best_Nchip = copy.deepcopy(sp_Nchip)
				
				self.sample_num += 1

			# 1.3 最优方案记录，得到TP Tile内的评估结果	
			tp_sp_fitness[best_tp_name] = best_tp_fitness
			bw_reallocate_result[tp_id] = best_bw_reallocate
			best_Nchip_partition[tp_id] = copy.deepcopy(best_Nchip)

		# 2. calSP, 对多个TP Tile进行合并评估
		schedule_name, total_fitness = self.calSP(tp_sp_fitness, None)

		if debug_in_evaluation_tp_sp:
			print('schedule_name = ', schedule_name)
			print('fitness = ', total_fitness)
		
		return schedule_name, total_fitness, best_Nchip_partition

	# decodeCode: workload_code进行解码，解码成每个[tp_id, sp_id]内包含哪些workload的形式
	def decodeCode(self, code):
		sp_max_dict = {}
		for w_name, sp_tp_id in code.items():
			sp_id = sp_tp_id[1]
			tp_id = sp_tp_id[0]
			if tp_id not in sp_max_dict:
				sp_max_dict[tp_id] = 0
			if sp_id > sp_max_dict[tp_id]:
				sp_max_dict[tp_id] = sp_id
		
		tp_sp_space = {}
		for tp_id in range(len(sp_max_dict)):
			tp_sp_space[tp_id] = {}
			sp_max = sp_max_dict[tp_id]
			for sp_id in range(sp_max+1):
				tp_sp_space[tp_id][sp_id] = []

		for w_name, sp_tp_id in code.items():
			sp_id = sp_tp_id[1]
			tp_id = sp_tp_id[0]
			#if tp_id not in tp_sp_space:
			#	tp_sp_space[tp_id] = {}
			#if sp_id not in tp_sp_space[tp_id]:
			#	tp_sp_space[tp_id][sp_id] = []
			tp_sp_space[tp_id][sp_id].append(w_name)
		return tp_sp_space

	# getWorkloadScheduleCodeList_random: workload的调度，随机获得workload的tp_sp_id映射
	# --- workload_code_list: 任务调度方案列表
	# --- sp_num_max: chiplet拆分的最多块数
	def getWorkloadScheduleCodeList_random(self, tp_TH, sp_TH, TH):
		workload_code_list = []

		app_num = len(self.workload_dict)
		sp_num_max = min(app_num, sp_TH)

		app_id_dict = {}
		for app_id, app_name in enumerate(self.workload_dict):
			app_id_dict[app_name] = app_id

		max_iter = 100
		iter_num = 0
		while iter_num < max_iter:
			iter_num += 1
			sp_min = 0
			tp_max = tp_TH - 1
			sp_max = sp_num_max-1
			tp_min = 0

			workload_code = {}

			# tp_set & sp_set_dcit : 为了消除无任务的Tile，便于做到方案的唯一性
			tp_set = [0 for _ in range(int(tp_TH))]
			sp_set_dict = {}

			# 1. 获得模型在每个时间点下的sp id，以模型为部署粒度
			# sp_id_dict: key(tp_id), value(sp_list), 获得每个tp point下的网络模型的sp id
			sp_id_dict = {}
			for tp_id in range(tp_TH):
				sp_set_dict[tp_id] = [0 for _ in range(int(sp_num_max))]
				sp_list = []			# sp_list: key(app_id), value(sp_id)
				# --- 随机获取
				for _ in range(app_num):
					sp_id = random.randint(sp_min, sp_max)
					sp_list.append(sp_id)
				# --- 转换整理：按顺序排布、去除空的SP Tile
				sp_list.sort()
				sp_set = set(sp_list)
				sp_change_dict = {}		# sp_change_dict: key(sp_id_pre), value(sp_id_new), 转换sp id，用于去除空的SP Tile
				for index, sp_id in enumerate(sp_set):
					sp_change_dict[sp_id] = index
				for i, sp_id in enumerate(sp_list):
					sp_list[i] = sp_change_dict[sp_id]
				sp_id_dict[tp_id] = sp_list
			
			# 2. 对网络内的任务进行时间上的调度
			for app_name, workload in self.workload_dict.items():
				# 获得网络内每个任务的tp id
				workload_num = len(workload)
				tp_list = []				# tp_list: key(workload_id), value(tp_id)
				for _ in range(workload_num):
					tp_id = random.randint(tp_min, tp_max)
					tp_list.append(tp_id)
				tp_list.sort()
				
				# 整理记录workload位于的Tile id
				# --- workload_code: key(workload_name), value([tp_id, sp_id])
				for w_id, w_name in enumerate(workload):
					workload_name = app_name + w_name
					tp_id = tp_list[w_id]
					sp_id = sp_id_dict[tp_id][app_id_dict[app_name]]
					workload_code[workload_name] = [tp_id, sp_id]
					tp_set[tp_id] = 1
					sp_set_dict[tp_id][sp_id] = 1
			
			# 3. 消除空Tile，对workload的Tile id进行对应的转换
			tp_change_id_list = []
			sp_change_id_dict = {}
			tp_real = 0
			for tp_id, tp_use in enumerate(tp_set):
				tp_change_id_list.append(tp_real)
				tp_real += tp_use
				
				sp_change_id_dict[tp_id] = []
				sp_set = sp_set_dict[tp_id]
				sp_real = 0
				for sp_id in sp_set:
					sp_change_id_dict[tp_id].append(sp_real)
					sp_real += sp_id
			for workload_name, tp_sp_id in workload_code.items():
				tp_id = tp_change_id_list[tp_sp_id[0]]
				sp_id = sp_change_id_dict[tp_sp_id[0]][tp_sp_id[1]]
				workload_code[workload_name][0], workload_code[workload_name][1] = tp_id, sp_id
			
			# 4. 若方案唯一则记录
			if workload_code not in workload_code_list:
				iter_num = 0 
				workload_code_list.append(workload_code)
			if len(workload_code_list) > TH:
				break
		return workload_code_list, sp_num_max

	# evoluation_temporal_spatial: 探索函数，主运行函数
	def evoluation_temporal_spatial(self):
		self.initialize()

		# 1. workload schedule：任务调度方案获取 
		# ---目前方法：随机
		workload_code_list, max_sp_num = self.getWorkloadScheduleCodeList_random(self.space_tp_TH, self.space_sp_TH, self.workload_schedule_TH)
		
		self.chiplet_partition_dict = self.getChipletPartionDict(max_sp_num)

		if debug_in_evoluation_temporal_spatial_simple:
			print("DEBUG IN evoluation_temporal_spatial()------------------------------")
			print("---chiplet_partition_dict: ", self.chiplet_partition_dict)
			print("---Start---------------------------------------------")
		
		# 2. 迭代进化：遍历所有的workload调度的方案
		# --- 进行硬件资源分配和评估
		for workload_code in workload_code_list:
			if debug_in_evoluation_temporal_spatial_simple:
				print("-----Start a new workload mapping----------")
				print("-----workload_code = ", workload_code)

			# 2.1: Workload Schedule Method Decode
			# --- Decode and Workload Merge
			tp_sp_space = self.decodeCode(workload_code)
			tp_sp_idealParam = self.calIdealParam(tp_sp_space)
			merge_tp_sp_space = self.mergeWorkload(tp_sp_space)

			if debug_in_evoluation_temporal_spatial_simple:
				print("------tp_sp_space = ", tp_sp_space)
				print("-----merge_tp_sp_space = ", merge_tp_sp_space)
				print("-----merge_workload_dict = ", self.merge_workload_dict)
			
			tp_sp_space = copy.deepcopy(merge_tp_sp_space)

			# 2.2 : Evaluation
			# --- Chiplet Allocate + BW Reallocate + Method Evaluation
			schedule_name, fitness, Nchip_partition = self.evaluation_tp_sp(tp_sp_space, tp_sp_idealParam)

			# 2.3 : 迭代比较
			ob_id = Optimization_Objective_index[self.Optimization_Objective]
			if self.fitness_best == None or fitness[ob_id] < self.fitness_best[ob_id]:
				self.fitness_best = fitness
				self.schedule_best = schedule_name
				self.tp_sp_space_best = tp_sp_space
				self.Nchip_partition_best = Nchip_partition
			
			if debug_in_record_fitness_iter:
				print("--------------------------------------------------------")
				print("now_schedule_name: ", schedule_name)
				print("now_fitess: ", fitness)
				print("best_schedule_name: ", self.schedule_best)
				print("best_fitess: ", self.fitness_best)
			
			self.fitness_record.append(fitness)
			self.schedule_record.append(schedule_name)
			self.tp_sp_space_record.append(tp_sp_space)

			if debug_in_evoluation_temporal_spatial:
				print("-----best_fitess: ", self.fitness_best)
				print("-----best_schedule_name: ", self.schedule_best)
				print("-----best_Nchip_partition: ", self.Nchip_partition_best)

def plot(nn_name_list, architecture):
	id = 1
	row = len(nn_name_list)
	plt.figure("Fitness per chiplet num")
	for nn_name in nn_name_list:
		plt.subplot(row, 1, id)
		id += 1
		file = "{}/SE_result/{}_{}.txt".format(cur_dir, architecture, nn_name)
		f = open(file)
		lines = f.readlines()
		x = []
		y = []
		for line in lines:
			if line.startswith("chiplet"):
				line_items = line.replace("\n","").split("\t")
				chiplet_num = int(line_items[1])
				fitness = float(line_items[3])
				x.append(chiplet_num)
				y.append(fitness)
		if id > row:
			plt.xlabel("Chiplet Num", fontsize = 10)
		plt.ylabel("Fitness", fontsize = 10)
		plt.bar(x, y, width=0.5,color='rosybrown')
		plt.plot(x,y,color='brown')
		plt.tick_params(labelsize=8)
		for i in range(len(x)):
			plt.scatter(x[i],y[i],s=8,color='brown')
			#xy = (x[i], round(y[i]))
			#plt.annotate("%s" % round(y[i]), xy=xy, xytext=(-20, 10), textcoords='offset points')
		plt.title(nn_name, fontsize = 12, color='brown')
	plt.tight_layout(pad=1.1)
	plt.savefig(cur_dir + "/SE_result/fitness_change_per_Nchiplet_line.png", bbox_inches = 'tight')

# loadWorkloadDict: 加载任务列表
def loadWorkloadDict(workload_file, app_name_list):
	f = open(workload_file)
	lines = f.readlines()
	workload_dict = {}
	for line in lines:
		if line != "":
			line_item = line.split(": ")
			app_name = line_item[0]
			if app_name in app_name_list:
				workload_dict[app_name] = {}
				workload_line = line_item[1]
				workload_line = workload_line.replace("{", "")
				workload_line = workload_line.replace("}", "")
				workload_line = workload_line.replace("\n", "")
				workload_items = workload_line.split(";\t")
				for workload in workload_items:
					if workload != "":
						workload_list = workload.split(":")
						workload_name, layer_list = workload_list[0], workload_list[1]
						workload_name = workload_name.replace(app_name, "")
						layer_list = layer_list.replace("[","").replace("]","")
						layer_list = layer_list.split(",")
						start_layer = int(layer_list[0])
						end_layer = int(layer_list[-1])
						workload_dict[app_name][workload_name] = [start_layer, end_layer]

	return workload_dict

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--architecture', type=str, default="ours", help='architecture')
	parser.add_argument('--nn_list', type=str, default="resnet50", help='neural network, using + as split')
	parser.add_argument('--chiplet_num', type=int, default=16, help='NN model name')
	parser.add_argument('--Optimization_Objective', type=str, default="latency", help='Optimization Objective: edp, energy, latency')
	parser.add_argument('--alg', type=str, default='GA', help='use layer fuse or not')
	parser.add_argument('--encode_type', type=str, default='index', help='encode type')
	parser.add_argument('--BW_Reallocator_tag', type=int, default=1, help='use BW Reallocator or not')
	parser.add_argument('--tp_TH', type=int, default=10, help='tp max num')
	parser.add_argument('--sp_TH', type=int, default=10, help='sp max num')
	
	opt = parser.parse_args()
	architecture = opt.architecture
	chiplet_num = opt.chiplet_num
	Optimization_Objective = opt.Optimization_Objective
	nn_list = opt.nn_list
	nn_list.replace("\n", "")
	nn_name_list = nn_list.split("+")

	# 读取任务列表  
	# workload_dict = {"resnet18":{"":[1,17]}, "resnet50":{"w1":[1,21], "w2":[22,50]}, "VGG16":{"":[1,13]}}
	abs_path = os.path.dirname(os.path.abspath(__file__))
	SE_abs_path = os.path.join(abs_path, "../nnparser_SE_hetero_iodie")
	result_outdir = os.path.join(abs_path, "SE_result")
	result_outdir = os.path.join(result_outdir, architecture) # + "_" + opt.alg + "_" + opt.encode_type)
	workload_file = os.path.join(result_outdir, "workload_list.txt")
	workload_dict = loadWorkloadDict(workload_file, nn_name_list)

	# DSE
	MNN_Engine = multi_network_DSE(architecture, chiplet_num, workload_dict, Optimization_Objective, tp_TH=opt.tp_TH, sp_TH=opt.sp_TH, BW_tag=opt.BW_Reallocator_tag)
	start_time = datetime.datetime.now()
	MNN_Engine.evoluation_temporal_spatial()

	# 控制台输出
	print("Sim END----------------------------------------------------------")
	print("best schedule name : ", MNN_Engine.schedule_best)
	print("best Nchip Partition : ", MNN_Engine.Nchip_partition_best)
	print("best fitness : ", MNN_Engine.fitness_best)
	print("")

	# 文本输出
	file_path = cur_dir + "/multi_nn_result/explore_result_total.txt"
	if os.path.exists(file_path) == False:
		result_out_file = open(cur_dir + "/multi_nn_result/explore_result_total.txt", 'w')
	else:
		result_out_file = open(cur_dir + "/multi_nn_result/explore_result_total.txt", 'a')
	end_time = datetime.datetime.now()
	sim_time = end_time - start_time

	print("{:-^120}".format(" SIM TIME "), file = result_out_file)
	print("start_time = {}".format(start_time.strftime('%Y-%m-%d %H:%M:%S')), file = result_out_file)
	print("end_time = {}".format(end_time.strftime('%Y-%m-%d %H:%M:%S')), file = result_out_file)
	print("sim_time = {}".format(sim_time), file = result_out_file)

	print("{:-^120}".format(" SETTING "), file = result_out_file)
	print("Chiplet Allocate Add", file = result_out_file)
	print("nn_list = {}".format(nn_name_list), file = result_out_file)
	print("workload = {}".format(workload_dict), file = result_out_file)
	print("chiplet_num = {}".format(chiplet_num), file = result_out_file)
	print("architecture = {}".format(architecture), file = result_out_file)
	print("alg = {}".format(opt.alg), file = result_out_file)
	print("encode_type = {}".format(opt.encode_type), file = result_out_file)
	print("BW_Reallocator_tag = {} (0: without BWR;  1: with BWR)".format(opt.BW_Reallocator_tag), file = result_out_file)
	print("tp_TH = {}".format(opt.tp_TH), file = result_out_file)
	print("sp_TH = {}".format(opt.sp_TH), file = result_out_file)

	print("{:-^120}".format(" RESULT "), file = result_out_file)
	print("total_sample_num = {}".format(MNN_Engine.sample_num), file = result_out_file)
	print("schedule name = {}".format(MNN_Engine.schedule_best), file = result_out_file)
	print("{:-<100}".format("schedule space result "), file = result_out_file)
	for tp_id, sp_space in MNN_Engine.tp_sp_space_best.items():
		line = "\ttp_id({})\t".format(tp_id)
		for sp_id in sp_space:
			sp_item = "sp_id({}): ".format(sp_id)
			for workload_name in sp_space[sp_id]:
				sp_item += workload_name + "+"
			sp_item = sp_item[:-1]
			sp_item += "; " + str(MNN_Engine.Nchip_partition_best[tp_id][sp_id])
			line += "{:30}\t\t".format(sp_item)
		print(line, file = result_out_file)
	print("{:-<100}".format(""), file = result_out_file)
	print("fitness_result: edp({}), latency({}), energy({})".format(MNN_Engine.fitness_best[0], MNN_Engine.fitness_best[1], MNN_Engine.fitness_best[2]), file = result_out_file)
	print("{:-^120}".format(" END "), file = result_out_file)
	print("", file = result_out_file)
	print("", file = result_out_file)
	print("", file = result_out_file)
	result_out_file.close()

	# 文本输出：迭代样本完整记录
	sample_record = False
	if sample_record:
		result_out_file = open(cur_dir + "/multi_nn_result/explore_result_sample_record.txt", 'w')
		print("{:-^120}".format(" SIM TIME "), file = result_out_file)
		print("start_time = {}".format(start_time), file = result_out_file)
		print("end_time = {}".format(end_time), file = result_out_file)
		print("sim_time = {}".format(sim_time), file = result_out_file)
		print("", file = result_out_file)

		print("{:-^120}".format(" SETTING "), file = result_out_file)
		print("nn_list = {}".format(nn_name_list), file = result_out_file)
		print("workload = {}".format(workload_dict), file = result_out_file)
		print("chiplet_num = {}".format(chiplet_num), file = result_out_file)
		print("Optimization_Objective = {}".format(Optimization_Objective), file = result_out_file)
		print("BW_Reallocator_tag = {} (0: without BWR;  1: with BWR)".format(opt.BW_Reallocator_tag), file = result_out_file)
		print("", file = result_out_file)

		print("{:-^120}".format(" RESULT "), file = result_out_file)
		print("total_sample_num = {}".format(MNN_Engine.sample_num), file = result_out_file)
		print("schedule name = {}".format(MNN_Engine.schedule_best), file = result_out_file)
		print("{:-<100}".format("schedule space result "), file = result_out_file)
		for tp_id, sp_space in MNN_Engine.tp_sp_space_best.items():
			line = "\ttp_id({})\t".format(tp_id)
			for sp_id in sp_space:
				sp_item = "sp_id({}): ".format(sp_id)
				for workload_name in sp_space[sp_id]:
					sp_item += workload_name + "+"
				sp_item = sp_item[:-1]
				sp_item += "; " + str(MNN_Engine.Nchip_partition_best[tp_id][sp_id])
				line += "{:30}\t\t".format(sp_item)
			print(line, file = result_out_file)
		print("{:-<100}".format(""), file = result_out_file)
		print("fitness_result: edp({}), latency({}), energy({})".format(MNN_Engine.fitness_best[0], MNN_Engine.fitness_best[1], MNN_Engine.fitness_best[2]), file = result_out_file)
		print("", file = result_out_file)

		print("{:-^120}".format(" SAMPLE RECORD "), file = result_out_file)
		print("", file = result_out_file)
		print("{:-^120}".format(" END "), file = result_out_file)

		id = 0
		for sample in MNN_Engine.schedule_record:
			fitness = str(MNN_Engine.fitness_record[id])
			Nchip = str(MNN_Engine.Nchip_par_record[id])
			id += 1
			print("sample={:100}\t, ---Nchip_par:{:60}\t, ---fitness:\t{:15}\t".format(str(sample), Nchip, str(fitness)), file = result_out_file)

	#MNN_Engine.plot()
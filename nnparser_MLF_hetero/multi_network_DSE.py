from linecache import lazycache
import os
import copy
import random
from matplotlib import pyplot as plt
import argparse
import numpy as np
PE_Frequency = 1000 * 1000 * 1000

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
	for line in lines:
		if line.startswith("#") or line.startswith("*"):
			pass
		else:
			line = line.replace("\n","")
			line_item = line.split(" ")
			layer_name = line_item[0]
			layer_id = int(layer_name.replace("layer",""))
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
	nn_f.close()
	return nnParam_dict

def getChipletPartition(chiplet_num, nn_ratio):
	# 获得chiplet数目的可能拆分
	max_TH = 100
	par_num = 0
	unchange_TH = 20
	unchange_times = 0
	nn_num = len(nn_ratio)
	chiplet_partiton_list = []
	while (1):
		Nchip_list = []
		Nchip_rest = chiplet_num - nn_num
		for i in range(nn_num-1):
			Nchip = random.randint(0, Nchip_rest)
			Nchip_list.append(Nchip+1)
			Nchip_rest -= Nchip
		Nchip_list.append(Nchip_rest+1)
		Nchip_list.sort()
		if Nchip_list not in chiplet_partiton_list:
			chiplet_partiton_list.append(Nchip_list)
			unchange_times = 0
			par_num += 1
		else:
			unchange_times += 1
		if unchange_times == unchange_TH or par_num == max_TH:
			break
	
	# 根据NN的理论的统计比较，分配chiplet数目
	nn_ratio_order = sorted(nn_ratio.items(), key = lambda x: x[1])
	nn_chiplet_num_list = []
	for par_list in chiplet_partiton_list:
		nn_chiplet_num = {}
		for i in range(len(par_list)):
			nn_name = nn_ratio_order[i][0]
			Nchip = par_list[i]
			nn_chiplet_num[nn_name] = Nchip
		nn_chiplet_num_list.append(nn_chiplet_num)
	return nn_chiplet_num_list

def getTemporalCode(nn_num, TH = 20):
	max_iter = 100
	iter_num = 0
	temporal_code_list = []
	while iter_num < max_iter:
		iter_num += 1
		t_code = []
		t_min = 0
		t_max = 0
		for i in range(nn_num):
			t_num = random.randint(t_min, t_max)
			t_code.append(t_num)
			if t_num == t_max:
				t_max += 1
		
		if t_code not in temporal_code_list:
			temporal_code_list.append(t_code)
			iter_num = 0
		
		if len(temporal_code_list) > TH:
			break
	return temporal_code_list

def getChipletPartionDict(chiplet_num, max_tp_num):
	# 获得chiplet数目的可能拆分
	par_per_tp_TH = 10
	par_num = 0
	unchange_TH = 50
	unchange_times = 0

	chiplet_partition_dict = {}
	for i in range(max_tp_num):
		tp_num = i + 1
		par_num = 0
		unchange_times = 0
		chiplet_partition_dict[tp_num] = []
		while(1):
			chiplet_partition = []
			Nchip_rest = chiplet_num - tp_num
			for i in range(tp_num-1):
				Nchip = random.randint(0, Nchip_rest)
				chiplet_partition.append(Nchip+1)
				Nchip_rest -= Nchip
			chiplet_partition.append(Nchip_rest+1)
			chiplet_partition.sort()
			if chiplet_partition not in chiplet_partition_dict[tp_num]:
				chiplet_partition_dict[tp_num].append(chiplet_partition)
				unchange_times = 0
				par_num += 1
			else:
				unchange_times += 1
				
			if unchange_times == unchange_TH or par_num == par_per_tp_TH:
				break
	return chiplet_partition_dict

def getChipletPartition_sp_tp(chiplet_partition_dict, sp_tp_idealParam):
	partition_TH = 200
	unchange_TH = 40

	# 根据TP内NN的理论的统计idealParam比较，分配chiplet数目
	tp_Nchip_dict = {}
	sp_tp_max_num = 1
	for sp_id, tp_idealParam in sp_tp_idealParam.items():
		tp_num = len(tp_idealParam)
		tp_idealParam_order = sorted(tp_idealParam.items(), key = lambda x: x[1])
		tp_Nchip_dict[sp_id] = []
		for chip_num_list in chiplet_partition_dict[tp_num]:
			tp_Nchip = {}
			for i in range(tp_num):
				chip_num = chip_num_list[i]
				tp_id = tp_idealParam_order[i][0]
				tp_Nchip[tp_id] = chip_num
			tp_Nchip_dict[sp_id].append(tp_Nchip)
		sp_tp_max_num *= len(tp_Nchip_dict[sp_id])

	partition_num = 0
	unchange_num = 0
	partition_list = []
	while (1):
		partition = []
		for sp_id in tp_Nchip_dict:
			tp_num_max = len(tp_Nchip_dict[sp_id])
			id = random.randint(0,tp_num_max-1)
			partition.append(id)
		if partition not in partition_list:
			partition_list.append(partition)
			partition_num += 1
			unchange_num = 0
		else:
			unchange_num += 1
		
		if unchange_num >= unchange_TH or partition_num >= partition_TH or partition_num >= sp_tp_max_num:
			break
	
	sp_tp_Nchip_list = []
	for partition in partition_list:
		sp_tp_Nchip = {}
		for sp_id in tp_Nchip_dict:
			id = partition[sp_id]
			sp_tp_Nchip[sp_id] = tp_Nchip_dict[sp_id][id]
		sp_tp_Nchip_list.append(sp_tp_Nchip)
	
	return sp_tp_Nchip_list

class multi_network_DSE:
	def __init__(self, chiplet_num, workload_dict, Optimization_Objective, debug_tag = 0):
		self.chiplet_num = chiplet_num
		self.workload_dict = workload_dict
		self.workload_list = []
		self.debug_tag = debug_tag
		self.Optimization_Objective = Optimization_Objective

		self.ideal_param_dict = None
		self.ideal_param_dict_workload = None

		self.workload_fitness_dict = {}
		self.workload_BWNeeded_dict = {}

		self.fitness_best = None
		self.fitness_record = []
		self.schedule_best = None
		self.schedule_record = []
		self.Nchip_partition_best = None
		self.Nchip_par_record = []
		self.sp_tp_space_best = None
		self.sp_tp_space_record = []
	
	# 获得理论的各任务的参数量信息
	def getIdealParam(self):
		self.ideal_param_dict = {}
		self.ideal_param_dict_workload = {}
		for nn_name, workload in self.workload_dict.items():
			self.ideal_param_dict[nn_name] = copy.deepcopy(getNNParam(nn_name))
			for w_name, [start_id, end_id] in workload.items():
				workload_name = nn_name + w_name
				self.workload_list.append(workload_name)
				param = 0
				for layer_id in range(start_id, end_id+1):
					param += self.ideal_param_dict[nn_name][layer_id]
				self.ideal_param_dict_workload[workload_name] = param

	def calTP(self, fitness_dict, id):
		if id == None:
			tp_name = "("
		else:
			tp_name = "tp" + str(id) + "("
		latency_tp = 0
		energy_tp = 0
		for name, fitness in fitness_dict.items():
			tp_name += name + "_"
			latency_tp += fitness[1]
			energy_tp += fitness[2]
		tp_name = tp_name.strip("_")
		tp_name += ")"

		edp_tp = latency_tp * energy_tp / PE_Frequency
		fitness_list = [edp_tp, latency_tp, energy_tp]
		return tp_name, fitness_list
	
	def BW_Reallocate(self, tp_workload_fitness, tp_Nchip):
		tp_workload_fitness_bw = {}
		bw_reallocate_result = {}
		debug = False
		if len(tp_workload_fitness.keys()) == 1:
			for tp_id, workload_fitness in tp_workload_fitness.items():
				tp_workload_fitness_bw[tp_id] = {}
				bw_reallocate_result[tp_id] = {}
				for workload, fitness in workload_fitness.items():
					tp_workload_fitness_bw[tp_id][workload] = fitness
		else:
			"""
			tp_workload_fitness =  
			{
				0: {
					'resnet18': [1793044.2722481424, 514782.6666666666, 3483109258.2400002], 
					'resnet50': [9630241.645624578, 1326631.466666666, 7259168719.872001]
				}, 
				1: {'resnet50same': [15876227.827040443, 1926204.2666666668, 8242234793.984004]}
			}
			tp_Nchip =  {1: 6, 0: 10}
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

			for tp_id, workload_fitness in tp_workload_fitness.items():
				tp_workload_fitness_bw[tp_id] = {}
				bw_reallocate_result[tp_id] = {}
				Nchip = tp_Nchip[tp_id]
				if debug:
					print('workload_fitness.keys() = ', workload_fitness.keys())
					print('Nchip = ', Nchip)
				delay_min_dict[tp_id] = {}
				degrad_ratio_dict[tp_id] = {}
				L2_to_DRAM_min_dict[tp_id] = {}
				dram_to_L2_min_dict[tp_id] = {}
				eva_nn_chiplet_num_dict[tp_id] = Nchip

				for workload, fitness in workload_fitness.items():
					tp_workload_fitness_bw[tp_id][workload] = fitness
					workload_BW_list = self.workload_BWNeeded_dict[workload][Nchip]
					if debug:
						print('workload_BW_list = ', workload_BW_list)
					
					for layer_id, BW_list in workload_BW_list.items():
						# latency = BW_list[0]
						# NoC_NR = BW_list[1]
						# L2_to_DRAM_NR = BW_list[2]
						# DRAM_to_L2_NR = BW_list[3]
						delay_min_dict[tp_id][workload + '_' + str(layer_id)] = BW_list[0]
						degrad_ratio_dict[tp_id][workload + '_' + str(layer_id)] = max(BW_list[1], BW_list[2], BW_list[3])
						L2_to_DRAM_min_dict[tp_id][workload + '_' + str(layer_id)] = BW_list[2]
						dram_to_L2_min_dict[tp_id][workload + '_' + str(layer_id)] = BW_list[3]

			if debug:
				print('delay_min_dict = ', delay_min_dict)
				print('degrad_ratio_dict = ', degrad_ratio_dict)
				print('dram_to_L2_min_dict = ', dram_to_L2_min_dict)
				print('L2_to_DRAM_min_dict = ', L2_to_DRAM_min_dict)
			
			for nn_name in tp_workload_fitness.keys():
				total_cycle_dict[nn_name] = {}
				cycle = 0
				for layer, item in delay_min_dict[nn_name].items():
					cycle += item
					total_cycle_dict[nn_name][layer] = int(cycle)
					total_cycle_list.append(int(cycle))
			# total_cycle = max(total_cycle_dict.values())
			total_cycle_list.sort()
			if debug:
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
				if debug:
					print(tick, state)
				degrad_ratio_list = []
				for nn_name in state.keys():
					degrad_ratio_list.append(degrad_ratio_dict[nn_name][state[nn_name]])

				print(degrad_ratio_list, sum(np.array(degrad_ratio_list) == 1))

				######################## update ########################
				if sum(np.array(degrad_ratio_list) <= 1) >= 1 and sum(np.array(degrad_ratio_list) <= 1) < len(degrad_ratio_list):
					nn_name = nn_name_list[np.argmin(degrad_ratio_list)]
					improve_ratio = (1 - max(dram_to_L2_min_dict[nn_name][state[nn_name]], L2_to_DRAM_min_dict[nn_name][state[nn_name]])) / (len(degrad_ratio_list) - 1) * (eva_nn_chiplet_num_dict[nn_name] % 4)
					nn_name_list.remove(nn_name)

					print('improve_ratio = ', improve_ratio)
					print('tick - old_tick = ', tick - old_tick)
					for nn in nn_name_list:
						if debug:
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
					if debug:
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
			for tp_id, workload_fitness in tp_workload_fitness.items():
				for workload, fitness in workload_fitness.items():
					workload_BW_list = self.workload_BWNeeded_dict[workload][Nchip]
					layer_id = list(workload_BW_list.keys())[-1]
					latency_tp = total_cycle_dict[tp_id][workload + '_' + str(layer_id)]
					energy_tp = tp_workload_fitness_bw[tp_id][workload][2]
					edp_tp = latency_tp * energy_tp / PE_Frequency
					tp_workload_fitness_bw[tp_id][workload] = [edp_tp, latency_tp, energy_tp]
		return tp_workload_fitness_bw, bw_reallocate_result

	def calSP(self, fitness_dict, id):
		sp_name = "sp" + str(id) + "("
		latency_sp = 0
		energy_sp = 0
		for name, fitness in fitness_dict.items():
			sp_name += name + "_"
			if fitness[1] > latency_sp:
				latency_sp = fitness[1]
			energy_sp += fitness[2]
		sp_name = sp_name.strip("_")
		sp_name += ")"

		edp_sp = latency_sp * energy_sp / PE_Frequency
		fitness_list = [edp_sp, latency_sp, energy_sp]
		return sp_name, fitness_list

	# 评估函数
	def evaluation_sp_tp(self, sp_tp_space, sp_tp_Nchip):
		sp_tp_workload_fitness = {}

		for sp_id in sp_tp_space:
			sp_tp_workload_fitness[sp_id] = {}
			for tp_id in sp_tp_space[sp_id]:
				sp_tp_workload_fitness[sp_id][tp_id] = {}
				Nchip = sp_tp_Nchip[sp_id][tp_id]
				for workload in sp_tp_space[sp_id][tp_id]:
					fitness = self.workload_fitness_dict[workload][Nchip]
					sp_tp_workload_fitness[sp_id][tp_id][workload] = fitness
		
		print('sp_tp_workload_fitness = ', sp_tp_workload_fitness)
		sp_tp_fitness = {}
		bw_reallocate_result = {}
		for sp_id in sp_tp_workload_fitness:
			sp_tp_fitness[sp_id] = {}

			tp_workload_fitness = sp_tp_workload_fitness[sp_id]
			tp_Nchip = sp_tp_Nchip[sp_id]
			print('tp_workload_fitness = ', tp_workload_fitness)
			print('tp_Nchip = ', tp_Nchip)
			tp_workload_fitness_bw , bw_reallocate_result[sp_id] = self.BW_Reallocate(tp_workload_fitness, tp_Nchip)
			print('tp_workload_fitness_bw = ', tp_workload_fitness_bw)
			print('bw_reallocate_result = ', bw_reallocate_result)
			for tp_id, workload_fitness in tp_workload_fitness_bw.items():
				tp_name, fitness = self.calTP(workload_fitness, tp_id)
				sp_tp_fitness[sp_id][tp_name] = fitness
		
		sp_fitness = {}
		for sp_id, tp_fitness in sp_tp_fitness.items():
			sp_name, fitness = self.calSP(tp_fitness, sp_id)
			sp_fitness[sp_name] = fitness
		
		schedule_name, fitness = self.calTP(sp_fitness, None)
		print('schedule_name = ', schedule_name)
		print('fitness = ', fitness)
		return schedule_name, fitness
		
	# 获得每个Workload的性能参数
	def setTotalWorkloadFitness(self):
		for workload in self.workload_list:
			file = cur_dir + "/SE_result/GA_index/ours_" + workload + ".txt"
			f = open(file)
			lines = f.readlines()
			self.workload_fitness_dict[workload] = {}
			for line in lines:
				if line.startswith("chiplet"):
					line_item = line.replace("\n","").split("\t")
					chiplet_num = int(line_item[1])
					edp = float(line_item[3])
					latency = float(line_item[5])
					energy = float(line_item[7])
					self.workload_fitness_dict[workload][chiplet_num] = [edp, latency, energy]
			
			self.workload_BWNeeded_dict[workload] = {}
			BW_dir = cur_dir + "/SE_result/GA_index/BW_result/" + workload + "/"
			for i in range(self.chiplet_num):
				chiplet_num = i + 1
				self.workload_BWNeeded_dict[workload][chiplet_num] = {}
				BW_file = BW_dir + "chiplet_num_" + str(chiplet_num) + ".txt"
				BW_f = open(BW_file)
				lines = BW_f.readlines()
				for line in lines:
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
						self.workload_BWNeeded_dict[workload][chiplet_num][layer_id] = BW_list

	# 初始化操作
	def initialize(self):
		self.getIdealParam()
		self.setTotalWorkloadFitness()
		self.best_fitness = None
		self.fitness_record = []
		self.sample_record = []
		self.distance_record = []

	# 获得workload的sp_tp_id映射
	# workload_dict = {resnet18:{“w1”:[1,7],“w2”:[8,16]}, vgg16:{“w1”:[1,16]}}
	def getMultiNNDSECode(self, sp_TH=10, tp_TH=10, TH=1000):
		workload_code_list = []
		iter_num = 0
		max_iter = 1000
		sp_TH=10
		tp_TH=10
		tp_num_max = 0
		while iter_num < max_iter:
			iter_num += 1
			workload_code = {}
			tp_max_list = [0]
			tp_min = 0
			sp_max = 0
			sp_min = 0
			for nn_name, workload in self.workload_dict.items():
				sp_pre = sp_min
				for w_name in workload:
					workload_name = nn_name + w_name
					sp_id = random.randint(sp_pre, sp_max)
					tp_max = tp_max_list[sp_id]
					tp_id = random.randint(tp_min, tp_max)
					if tp_id == tp_max and tp_max < tp_TH:
						tp_max_list[sp_id] += 1
					if sp_id == sp_max and sp_max < sp_TH:
						sp_max += 1
						tp_max_list.append(0)
					sp_pre = sp_id
					workload_code[workload_name] = [sp_id, tp_id]
			if workload_code not in workload_code_list:
				iter_num = 0 
				workload_code_list.append(workload_code)
			if len(workload_code_list) > TH:
				break
			tp_max_list.sort()
			tp_num_max_1 = tp_max_list[-1]
			if tp_num_max_1 > tp_num_max:
				tp_num_max = tp_num_max_1
		return workload_code_list, tp_num_max

	# workload_code进行解码，解码成每个[sp_id, tp_id]内包含哪些workload的形式
	def decodeCode(self, code):
		sp_tp_space = {}
		for w_name, tp_sp_id in code.items():
			tp_id = tp_sp_id[1]
			sp_id = tp_sp_id[0]
			if sp_id not in sp_tp_space:
				sp_tp_space[sp_id] = {}
			if tp_id not in sp_tp_space[sp_id]:
				sp_tp_space[sp_id][tp_id] = []
			sp_tp_space[sp_id][tp_id].append(w_name)
		return sp_tp_space

	# 计算每个[sp_id, tp_id]内任务的理论参数量
	def calIdealParam(self, sp_tp_space):
		sp_tp_idealParam = {}
		for sp_id, tp_space  in sp_tp_space.items():
			sp_tp_idealParam[sp_id] = {}
			for tp_id, workload_list in tp_space.items():
				sp_tp_idealParam[sp_id][tp_id] = 0
				for workload in workload_list:
					sp_tp_idealParam[sp_id][tp_id] += self.ideal_param_dict_workload[workload]
		return sp_tp_idealParam

	def evoluation_temporal_spatial(self):
		self.initialize()
		workload_code_list, max_tp_num = self.getMultiNNDSECode(self.workload_dict)
		chiplet_partition_dict = getChipletPartionDict(self.chiplet_num, max_tp_num)

		print("Start----------------------")
		for workload_code in workload_code_list:
			print("Start a new workload mapping----------")
			print("workload_code = ", workload_code)

			sp_tp_space = self.decodeCode(workload_code)

			sp_tp_idealParam = self.calIdealParam(sp_tp_space)
			sp_tp_Nchip_list = getChipletPartition_sp_tp(chiplet_partition_dict, sp_tp_idealParam)

			for sp_tp_Nchip in sp_tp_Nchip_list:
				print("Start a new chiplet partition----------")
				print("sp_tp_space = ", sp_tp_space)
				print("sp_tp_Nchip = ", sp_tp_Nchip)
				schedule_name, fitness = self.evaluation_sp_tp(sp_tp_space, sp_tp_Nchip)
				if self.fitness_best == None or fitness < self.fitness_best:
					self.fitness_best = fitness
					self.schedule_best = schedule_name
					self.Nchip_partition_best = sp_tp_Nchip
					self.sp_tp_space_best = sp_tp_space
				
				print("best_fitess: ", self.fitness_best)
				print("best_schedule_name: ", self.schedule_best)
				
				self.fitness_record.append(fitness)
				self.schedule_record.append(schedule_name)
				self.Nchip_par_record.append(sp_tp_Nchip)
				self.sp_tp_space_record.append(sp_tp_space)


def plot(nn_name_list):
	id = 1
	row = len(nn_name_list)
	plt.figure("Fitness per chiplet num")
	for nn_name in nn_name_list:
		plt.subplot(row, 1, id)
		id += 1
		file = cur_dir + "/SE_result/GA_index/ours_" + nn_name + ".txt"
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
	plt.savefig(cur_dir + "/SE_result/GA_index/fitness_change_per_Nchiplet_line.png", bbox_inches = 'tight')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--nn_list', type=str, default="resnet50", help='neural network, using + as split')
	parser.add_argument('--chiplet_num', type=int, default=16, help='NN model name')
	parser.add_argument('--Optimization_Objective', type=str, default="latency", help='Optimization Objective: edp, energy, latency')
	opt = parser.parse_args()
	chiplet_num = opt.chiplet_num
	Optimization_Objective = opt.Optimization_Objective
	nn_list = opt.nn_list
	nn_list.replace("\n", "")
	nn_name_list = nn_list.split("+")

	workload_dict = {"resnet18":{"":[1,17]}, "resnet18same":{"":[1,17]}, "resnet50":{"":[1,50]}, "resnet50same":{"":[1,50]}}
	MNN_Engine = multi_network_DSE(chiplet_num, workload_dict, Optimization_Objective)
	MNN_Engine.evoluation_temporal_spatial()

	# 控制台输出
	print("Sim END---------------")
	print("best schedule name : ", MNN_Engine.schedule_best)
	print("best Nchip Partition : ", MNN_Engine.Nchip_partition_best)
	print("best fitness : ", MNN_Engine.fitness_best)
	print("")

	# 文本输出
	result_out_file = open(cur_dir + "/multi_nn_result/explore_result.txt", 'w')
	print("{:-^120}".format(" SETTING "), file = result_out_file)
	print("nn_list = {}".format(nn_name_list), file = result_out_file)
	print("chiplet_num = {}".format(chiplet_num), file = result_out_file)
	print("Optimization_Objective = {}".format(Optimization_Objective), file = result_out_file)
	print("", file = result_out_file)

	print("{:-^120}".format(" RESULT "), file = result_out_file)
	print("schedule name = {}".format(MNN_Engine.schedule_best), file = result_out_file)
	print("{:-<100}".format("schedule space result "), file = result_out_file)
	for sp_id, tp_space in MNN_Engine.sp_tp_space_best.items():
		line = "\tsp_id({})\t".format(sp_id)
		for tp_id in tp_space:
			tp_item = "tp_id({}): ".format(tp_id)
			for workload_name in tp_space[tp_id]:
				tp_item += workload_name + "+"
			tp_item = tp_item[:-1]
			tp_item += "; " + str(MNN_Engine.Nchip_partition_best[sp_id][tp_id])
			line += "{:30}\t\t".format(tp_item)
		print(line, file = result_out_file)
	print("{:-<100}".format(""), file = result_out_file)
	print("fitness_result: edp({}), latency({}), energy({})".format(MNN_Engine.fitness_best[0], MNN_Engine.fitness_best[1], MNN_Engine.fitness_best[2]), file = result_out_file)
	print("", file = result_out_file)

	print("{:-^120}".format(" SAMPLE RECORD "), file = result_out_file)
	id = 0
	for sample in MNN_Engine.schedule_record:
		fitness = str(MNN_Engine.fitness_record[id])
		Nchip = str(MNN_Engine.Nchip_par_record[id])
		id += 1
		print("sample={:100}\t, ---Nchip_par:{:60}\t, ---fitness:\t{:15}\t".format(str(sample), Nchip, str(fitness)), file = result_out_file)

	#MNN_Engine.plot()
from linecache import lazycache
import math
import os
import copy
import random
from matplotlib import pyplot as plt
import argparse

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
	nn_evaluation = 0
	lines = nn_f.readlines()
	for line in lines:
		if line.startswith("#") or line.startswith("*"):
			pass
		else:
			line = line.replace("\n","")
			line_item = line.split(" ")
			layer_name = line_item[0]
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
			nn_evaluation += layer_computation_num * layer_param_num * layer_param_num / 10000000000000000
			layer_computation_num_dict[layer_name] = layer_computation_num
			layer_param_num_dict[layer_name] = layer_param_num
			print("{}: computation num = {}; param num = {}; evaluation = {}".format(layer_name, layer_computation_num, layer_param_num, layer_computation_num * layer_param_num * layer_param_num / 10000000))
	nn_f.close()
	return nn_evaluation

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
		
class multi_network_DSE:
	def __init__(self, chiplet_num, nn_list, debug_tag = 0):
		self.chiplet_num = chiplet_num
		self.nn_list = nn_list
		self.debug_tag = debug_tag

		self.ideal_param_dict = None
		self.nn_chiplet_num_dict = None
		self.ideal_fitness_dict = None
		self.best_fitness = None

		self.fitness_record = []
		self.distance_record = []

		self.sim_fitness_dict = None
		self.sim_ideal_ratio = None
		self.nn_chiplet_fitness_dict = {}	# [nn_name][n_chiplet] = fitness
	
	def getIdealParam(self):
		self.ideal_param_dict = {}
		for nn_name in self.nn_list:
			nn_ideal_fitness = getNNParam(nn_name)
			self.ideal_param_dict[nn_name] = nn_ideal_fitness

	def getIdealFitness(self):
		self.ideal_fitness_dict = {}
		for nn_name in self.nn_list:
			self.ideal_fitness_dict[nn_name] = self.ideal_param_dict[nn_name] / self.nn_chiplet_num_dict[nn_name]

	def getInitialMethod(self):
		evaluation_total = sum(self.ideal_param_dict.values())
		evaluation_ordered_dict = sorted(self.ideal_param_dict.items(), key = lambda x: x[1])
		self.nn_chiplet_num_dict = {}
		left_chiplet_num = self.chiplet_num
		last_nn_name = None
		for (nn_name, nn_evaluation) in evaluation_ordered_dict:
			c_num = int(self.chiplet_num * nn_evaluation / evaluation_total)
			if c_num > left_chiplet_num:
				c_num = left_chiplet_num
			elif c_num == 0:
				c_num = 1
			left_chiplet_num -= c_num
			self.nn_chiplet_num_dict[nn_name] = c_num
			last_nn_name = nn_name
		assert(left_chiplet_num >= 0)
		self.nn_chiplet_num_dict[last_nn_name] += left_chiplet_num

		self.getIdealFitness()
		
		if self.debug_tag == 1:
			print("Debug in getInitialMethod()---------")
			print("---total_chiplet_num : ", self.chiplet_num)
			print("---evaluation_per_nn : ", self.ideal_param_dict)
			print("---nn_chiplet_num : ", self.nn_chiplet_num_dict)

	def getSimIdealRatio(self):
		self.sim_ideal_ratio = {}
		for nn_name in self.nn_list:
			self.sim_ideal_ratio[nn_name] = self.sim_fitness_dict[nn_name] / self.ideal_fitness_dict[nn_name]
	
	def evaluation(self, eva_nn_chiplet_num_dict):
		# evaluation nn latency
		fitness_dict = {}
		for nn_name, n_chiplet in eva_nn_chiplet_num_dict.items():
			fitness = self.nn_chiplet_fitness_dict[nn_name][n_chiplet]
			fitness_dict[nn_name] = fitness
		return fitness_dict
	
	def setTotalNNFitness(self):
		for nn_name in self.nn_list:
			file = cur_dir + "/SE_result/GA_index/ours_" + nn_name + ".txt"
			f = open(file)
			lines = f.readlines()
			self.nn_chiplet_fitness_dict[nn_name] = {}
			for line in lines:
				if line.startswith("chiplet"):
					line_item = line.replace("\n","").split("\t")
					chiplet_num = int(line_item[1])
					fitness = float(line_item[3])
					self.nn_chiplet_fitness_dict[nn_name][chiplet_num] = fitness
	
	def getFitness(self, fitness_dict):
		fitness_total = 0
		for nn_name, fitness in fitness_dict.items():
			fitness_total += fitness
		
		nn_name_list = list(fitness_dict.keys())
		nn_name_first = nn_name_list[0]
		nn_name_last = nn_name_list[-1]
		distance = fitness_dict[nn_name_last] - fitness_dict[nn_name_first]
		
		return fitness_total, distance

	def plot(self):
		x = list(range(len(self.fitness_record)))

		plt.figure("latency")
		plt.plot(x,self.fitness_record)
		for i in range(len(x)):
			plt.scatter(x[i],self.fitness_record[i],s=10)
			xy = (x[i], round(self.fitness_record[i]))
			plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(-70, 10), textcoords='offset points')
		plt.title("latency change line")
		plt.savefig(result_plot+"/latency_plot.png", bbox_inches = 'tight')

		plt.figure("distance")
		plt.plot(x,self.distance_record)
		for i in range(len(x)):
			plt.scatter(x[i],self.distance_record[i],s=10)
			xy = (x[i], round(self.distance_record[i]))
			plt.annotate("(%s,%s)" % xy, xy=xy, xytext=(-70, 10), textcoords='offset points')
		plt.title("distance change line")
		plt.savefig(result_plot+"/distance_plot.png", bbox_inches = 'tight')

		plt.figure("result edp")
		x = []
		y = []
		for (nn_name, sim_fitness) in self.sim_fitness_dict.items():
			x.append(nn_name)
			y.append(sim_fitness)
		plt.xlabel("nn_name")
		plt.ylabel("EDP")
		plt.bar(x, y, width=0.5)
		for i in range(len(x)):
			xy = (x[i], round(y[i]))
			plt.annotate("%s" % round(y[i]), xy=xy, xytext=(-20, 2), textcoords='offset points')
		plt.title("EDP per NN")
		plt.savefig(result_plot+"/EDP_result.png", bbox_inches = 'tight')

	def evoluation_someMethod(self, TH=10):
		# --- 初始化
		self.setTotalNNFitness()
		self.getIdealParam()
		self.getInitialMethod()
		self.sim_fitness_dict = self.evaluation(self.nn_chiplet_num_dict)
		fitness_total, distance = self.getFitness(self.sim_fitness_dict)
		self.getSimIdealRatio()
		self.fitness_record.append(fitness_total)
		self.distance_record.append(distance)

		# --- 迭代进化
		for i in range(TH):
			print("start iter {} ---------------".format(i))
			sim_ideal_ratio_order_dict = sorted(self.sim_ideal_ratio.items(), key = lambda x: x[1])
			nn_fast = sim_ideal_ratio_order_dict[0][0]
			nn_slow = sim_ideal_ratio_order_dict[-1][0]
			chiplet_num_fast = self.nn_chiplet_num_dict[nn_fast] - 1
			chiplet_num_slow = self.nn_chiplet_num_dict[nn_slow] + 1
			fast_index = 0
			stop_signal = 0
			print("sim_ideal_ratio_order_dict=",sim_ideal_ratio_order_dict)
			print("nn_fast={} , nn_slow={}".format(nn_fast, nn_slow))
			while chiplet_num_fast == 0:
				fast_index += 1
				if fast_index == len(self.nn_chiplet_fitness_dict) - 1:
					stop_signal = 1
					break
				else:
					nn_fast = sim_ideal_ratio_order_dict[fast_index][0]
					chiplet_num_fast = self.nn_chiplet_num_dict[nn_fast] - 1
			if stop_signal == 1:
				break
			change_nn_chiplet_num_dict = {nn_fast: chiplet_num_fast , nn_slow: chiplet_num_slow}
			change_nn_fitness_dict = self.evaluation(change_nn_chiplet_num_dict)
			change_fitness = change_nn_fitness_dict[nn_fast] + change_nn_fitness_dict[nn_slow] - self.sim_fitness_dict[nn_fast] - self.sim_fitness_dict[nn_slow]
			change_fitness_slow = change_nn_fitness_dict[nn_slow] - self.sim_fitness_dict[nn_slow]
			if change_fitness < 0:
				self.nn_chiplet_num_dict[nn_fast] = change_nn_chiplet_num_dict[nn_fast]
				self.nn_chiplet_num_dict[nn_slow] = change_nn_chiplet_num_dict[nn_slow]
				self.sim_fitness_dict[nn_fast] = change_nn_fitness_dict[nn_fast]
				self.sim_fitness_dict[nn_slow] = change_nn_fitness_dict[nn_slow]
				self.sim_ideal_ratio[nn_fast] = self.sim_fitness_dict[nn_fast] / self.ideal_fitness_dict[nn_fast]
				self.sim_ideal_ratio[nn_slow] = self.sim_fitness_dict[nn_fast] / self.ideal_fitness_dict[nn_slow]
			elif change_fitness_slow < 0:
				self.sim_ideal_ratio[nn_fast] *= 2
			else:
				self.sim_ideal_ratio[nn_slow] /= 2
			print("nn_chiplet_num: ", self.nn_chiplet_num_dict)
			print("fitness_record: ", self.fitness_record)
			print("fitness change = {}".format(change_fitness))

	def evoluation_random(self):
		# --- 初始化
		self.setTotalNNFitness()
		self.getIdealParam()
		nn_chiplet_num_list = getChipletPartition(self.chiplet_num, self.ideal_param_dict)
		self.best_fitness = None
		iter = 0
		for nn_chiplet_num in nn_chiplet_num_list:
			iter += 1
			print("start iter {} ---------------".format(iter))
			
			fitness_dict = self.evaluation(nn_chiplet_num)
			fitness_total, distance = self.getFitness(fitness_dict)
			if self.best_fitness == None or fitness_total < self.best_fitness:
				self.best_fitness = fitness_total
				self.nn_chiplet_num_dict = nn_chiplet_num
				self.sim_fitness_dict = copy.deepcopy(fitness_dict)
			self.fitness_record.append(fitness_total)
			self.distance_record.append(distance)

			print("nn_chiplet_num: ", nn_chiplet_num)
			print("fitness = {}".format(fitness_total))
			print("fitness best = {}".format(self.best_fitness))

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
	parser.add_argument('--nn_list', type=str, default="resnet50", help='hardware architecture type (ours, nnbaton, simba)')	# simba , nnbaton
	parser.add_argument('--chiplet_num', type=int, default=16, help='NN model name')
	opt = parser.parse_args()
	chiplet_num = opt.chiplet_num
	nn_list = opt.nn_list
	nn_list.replace("\n", "")
	nn_name_list = nn_list.split("+")
	plot(nn_name_list)
	exit()
	print("nn_name_list: ", nn_name_list)
	MNN_Engine = multi_network_DSE(chiplet_num, nn_name_list)
	MNN_Engine.evoluation_random()
	print("Sim END---------------")
	print("nn_chiplet_num: ", MNN_Engine.nn_chiplet_num_dict)
	print("fitness_record: ", MNN_Engine.fitness_record)
	print("")
	MNN_Engine.plot()
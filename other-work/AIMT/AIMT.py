import yaml
import math

layer_type = ["CONV", "FC", "D_CONV"]
dim_type = ["P", "Q", "C", "K", "R", "S", "X", "Y", "stride", "padding"]

class Layer:
	def __init__(self, l_name, l_type):
		self.name = l_name
		self.type = l_type
		self.P = None
		self.Q = None
		self.C = None
		self.K = None
		self.R = None
		self.S = None
		self.X = None
		self.Y = None
		self.stride = None
		self.padding = None
	
	def setLayerDim(self, dim_dict):
		assert("P" in dim_dict)
		assert("C" in dim_dict)
		assert("K" in dim_dict)
		assert("R" in dim_dict)
		assert("stride" in dim_dict)
		assert("padding" in dim_dict)

		self.P = dim_dict["P"]
		self.C = dim_dict["C"]
		self.K = dim_dict["K"]
		self.R = dim_dict["R"]
		self.stride = dim_dict["stride"]
		self.padding = dim_dict["padding"]
		
		if "Q" not in dim_dict:
			self.Q = dim_dict["P"]
		else:
			self.Q = dim_dict["Q"]
		if "S" not in dim_dict:
			self.S = dim_dict["R"]
		else:
			self.S = dim_dict["S"]
		
		if "X" not in dim_dict:
			self.X = self.P * self.stride + self.R - self.stride - 2*self.padding
		else:
			self.X = dim_dict["X"]
		
		if "Y" not in dim_dict:
			if "X" in dim_dict:
				self.Y = dim_dict["X"]
			else:
				self.Y = self.Q * self.stride + self.S - self.stride - 2*self.padding
		else:
			self.Y = dim_dict["Y"]

	#def addPreLayer(self, name):
	#	self.pre_layer_num += 1
	
	#def setPostLayerName(self, name):
	#	self.post_layer_name.append(name)

class NeuralNetwork:
	def __init__(self, nn_name, batch=1):
		self.name = nn_name
		self.layer_dict = {}
		self.batch = batch

	def addLayer(self, l_name, l_type, dim_dict):
		layer_item = Layer(l_name, l_type)
		layer_item.setLayerDim(dim_dict)
		self.layer_dict[l_name] = layer_item
	
	#def setLayerDependence(self, dependence_set):
	#	for [src, dst] in dependence_set:
	#		self.layer_dict[src].setPostLayerName(dst)
	#		self.layer_dict[dst].addPreLayer()

class AIMT_schedule_table:
	def __init__(self):
		self.schedule_table = {}
		self.MB_status = {}
		self.CB_status = {}
	
	def addWorkloadName(self, nn_name, layer_name):
		if nn_name not in self.schedule_table:
			self.schedule_table[nn_name] = {}
		self.schedule_table[nn_name][layer_name] = {\
				"MB":{"indegree":0, "iters":0, "cycles":0}, \
				"CB":{"indegree":0, "iters":0, "cycles":0}, \
				"post_layer":[]}
	
	def addDependence(self, nn_name, dependence):
		for [l_src, l_dst] in dependence:
			self.schedule_table[nn_name][l_src]["post_layer"].append(l_dst)
			self.schedule_table[nn_name][l_dst]["MB"]["indegree"] += 1
			self.schedule_table[nn_name][l_dst]["CB"]["indegree"] += 1
	
	def addLayerEstimation(self, nn_name, layer_name, layer_estimation):
		[CB_cycle, MB_cycle, iters_num] = layer_estimation
		self.schedule_table[nn_name][layer_name]["MB"]["cycles"] = MB_cycle
		self.schedule_table[nn_name][layer_name]["MB"]["iters"] = iters_num
		self.schedule_table[nn_name][layer_name]["CB"]["cycles"] = CB_cycle
		self.schedule_table[nn_name][layer_name]["CB"]["iters"] = iters_num

	def finishMB(self, nn_name, layer_name):
		self.MB_status["{}\t{}".format(nn_name, layer_name)][1] -= 1
		self.schedule_table[nn_name][layer_name]["MB"]["iters"] -= 1
		if self.schedule_table[nn_name][layer_name]["MB"]["iters"] == 0:
			self.MB_status["{}\t{}".format(nn_name, layer_name)][2] = 1 # 运行结束了
			post_layer = self.schedule_table[nn_name][layer_name]["post_layer"]
			for post_name in post_layer:
				self.schedule_table[nn_name][post_name]["MB"]["indegree"] -= 1
				if (self.schedule_table[nn_name][post_name]["MB"]["indegree"]) == 0:
					self.MB_status["{}\t{}".format(nn_name, post_name)][0] = 1
			return 1
		return 0
	
	def finishCB(self, nn_name, layer_name):
		self.CB_status["{}\t{}".format(nn_name, layer_name)][1] -= 1
		self.schedule_table[nn_name][layer_name]["CB"]["iters"] -= 1
		if self.schedule_table[nn_name][layer_name]["CB"]["iters"] == 0:
			self.CB_status["{}\t{}".format(nn_name, layer_name)][2] = 1
			post_layer = self.schedule_table[nn_name][layer_name]["post_layer"]
			for post_name in post_layer:
				self.schedule_table[nn_name][post_name]["CB"]["indegree"] -= 1
				if (self.schedule_table[nn_name][post_name]["CB"]["indegree"]) == 0:
					self.CB_status["{}\t{}".format(nn_name, post_name)][0] = 1
			return 1
		return 0

	def statusInitialize(self):
		for nn_name, layer_dict in self.schedule_table.items():
			for layer_name, layer in layer_dict.items():
				workload_name = "{}\t{}".format(nn_name, layer_name)
				if (layer["MB"]["indegree"] == 0):
					self.MB_status[workload_name] = [1, layer["MB"]["iters"], 0]
				else:
					self.MB_status[workload_name] = [0, layer["MB"]["iters"], 0]
				
				if (layer["CB"]["indegree"] == 0):
					self.CB_status[workload_name] = [1, layer["CB"]["iters"], 0]
				else:
					self.CB_status[workload_name] = [0, layer["CB"]["iters"], 0]

# AIMT_DSE					(hardware param)
# --- 1. addNN				(nn)
# --- 2. addDependence		(dependence)
# --- 3. nnEstimation
# --- 4. schedule algorithm
class AIMT_DSE:
	def __init__(self, PE_dim_x, PE_dim_y, PE_array_num, BW, frequence, WL2_size):
		# -- architecture
		self.PE_dim_x = PE_dim_x
		self.PE_dim_y = PE_dim_y
		self.PE_array_num = PE_array_num
		self.BW = BW						# unit: GB/s
		self.frequence = frequence			# unit: GHz
		self.WL2_cycle = WL2_size / BW		# unit: ns
		# -- neural network
		self.nn_dict = {}
		# -- AIMT schedule table
		self.AIMT_ST = AIMT_schedule_table()
		# -- Schduling and Mapping
		# self.MB_scheduled_Q = []
		self.CB_scheduled_Q = []
		self.CB_candidate_Q = []
		self.MB_candidate_Q = []

		self.MB_C = 0
		self.CB_C = 0
		self.MB_idle_C = 0
		self.CB_idle_C = 0
		self.RM_C = self.WL2_cycle
		self.threshold = 20 # using for AVL_CB
		self.AVL_CB = 0

		self.MB_Time = 0
		self.CB_Time = 0
		self.Time = 0
		self.MB_timestamp_record = []
		self.CB_timestamp_record = []
		self.finish = 0

		self.total_iters_num = 0
	
	def addNN(self, nn, nn_name):
		self.nn_dict[nn_name] = nn
		for layer_name in nn.layer_dict:
			#print("nn_name: ", nn_name)
			self.AIMT_ST.addWorkloadName(nn_name, layer_name)
	
	def addDependence(self, nn_name, dependence):
		self.AIMT_ST.addDependence(nn_name, dependence)

	def layerEstimation(self, layer, batch):
		# --- PE Array Untilization
		util_dim_x = min(layer.K, self.PE_dim_x)
		util_dim_y = min(layer.C* layer.R * layer.S, self.PE_dim_y)
		# -- iters_num
		if layer.type == "CONV":
			iters_num = math.ceil(layer.K / self.PE_dim_x) * math.ceil(layer.C * layer.R * layer.S / self.PE_dim_y)
		elif layer.type == "FC":
			iters_num = math.ceil(layer.K / self.PE_dim_x / self.PE_array_num) * math.ceil(layer.C * layer.R * layer.S / self.PE_dim_y)
		
		# --- CB_cycle
		filling_time = max(util_dim_x, util_dim_y)
		CB_cycle = (math.ceil(layer.P * layer.Q / self.PE_array_num) * batch + filling_time) / self.frequence

		# --- MB_cycle
		if layer.type == "CONV":
			MB_cycle = util_dim_x * util_dim_y / self.BW
		elif layer.type == "FC":
			MB_cycle = util_dim_x * util_dim_y * self.PE_array_num / self.BW

		estimation_result = [CB_cycle, MB_cycle, iters_num]

		self.total_iters_num += iters_num

		#print("filling_time: ", filling_time)
		#print("self.PE_array_num: ", self.PE_array_num)
		#print(estimation_result)
		#exit()
		
		return estimation_result

	def nnEstimation(self, nn_name):
		nn = self.nn_dict[nn_name]
		layer_dict = nn.layer_dict
		batch = nn.batch
		nn_estimation = []
		for layer_name, layer_item in layer_dict.items():
			layer_estimation = self.layerEstimation(layer_item, batch)
			nn_estimation.append(layer_estimation)
			self.AIMT_ST.addLayerEstimation(nn_name, layer_name, layer_estimation)
		#self.estimation_dict[nn_name] = nn_estimation
		#print(self.AIMT_ST.schedule_table)
		#exit()

	def multiNNEstimation(self):
		for nn_name in self.nn_dict:
			self.nnEstimation(nn_name)
	
	def scheduleInitialize(self):
		# CB_candidate_Q, MB_candidate_Q
		self.AIMT_ST.statusInitialize()
		for w_name, status in self.AIMT_ST.MB_status.items():
			ready = status[0]
			iters = status[1]
			[nn_name, layer_name] = w_name.split("\t")
			cycles = self.AIMT_ST.schedule_table[nn_name][layer_name]["MB"]["cycles"]
			if ready:
				self.AIMT_ST.MB_status[w_name][2] = 1
				self.MB_candidate_Q.append([w_name, iters, cycles])
			
		for w_name, status in self.AIMT_ST.CB_status.items():
			ready = status[0]
			iters = status[1]
			[nn_name, layer_name] = w_name.split("\t")
			cycles = self.AIMT_ST.schedule_table[nn_name][layer_name]["CB"]["cycles"]
			if ready:
				self.AIMT_ST.CB_status[w_name][2] = 1
				self.CB_candidate_Q.append([w_name, iters, cycles])

	def scheduleMBFinish(self):
		# step1 : 选择合适的MB进行load
		target = None	# A2[2]
		target_id = None
		target_CB_cycles = None
		for MB_id, MB_list in enumerate(self.MB_candidate_Q):	# A2[3]
			MB_name = MB_list[0]
			MB_cycles = MB_list[2]
			name_list = MB_name.split("\t")
			CB_cycles = self.AIMT_ST.schedule_table[name_list[0]][name_list[1]]["CB"]["cycles"]
			if MB_cycles < self.RM_C:	# A2[4]
				if self.AVL_CB < self.threshold:	# A2[5]
					if CB_cycles > MB_cycles:	# A2[6]
						target = MB_name		# A2[7]
						target_CB_cycles = CB_cycles
						target_id = MB_id
						break
				else:	# A2[8]
					target = MB_name	# A2[9]
					target_CB_cycles = CB_cycles
					target_id = MB_id
					break
		
		if target == None:	# A2[10]
			pass		# A2[11,12]
		else:
			# step2 : MB load的处理
			iters, MB_cycles = self.MB_candidate_Q[target_id][1], self.MB_candidate_Q[target_id][2]
			# A2[13]
			if iters == 1:
				_ = self.MB_candidate_Q.pop(target_id)
			else:
				self.MB_candidate_Q[target_id][1] -= 1
			# self.MB_scheduled_Q.append([target, MB_cycles])
			self.MB_C += MB_cycles	# A2[15]
			self.RM_C -= MB_cycles
			self.AVL_CB = max(self.AVL_CB-MB_cycles, 0) + target_CB_cycles # A2[16]

		# step3 : 选择合适的CB
		for CB_id, CB_list in enumerate(self.CB_candidate_Q):	# A2[17]
			CB_name = CB_list[0]
			if self.CB_C <= self.MB_C:							# A2[18] 计算资源空闲
				name_list = CB_name.split("\t")
				MB_iters = self.AIMT_ST.schedule_table[name_list[0]][name_list[1]]["MB"]["iters"]
				CB_iters = CB_list[1]
				# A2[20]
				if CB_iters > MB_iters:						# 同层CB与MB间依赖性，表示CB对应的权重已经加载完毕
					if CB_iters == 1:						# CB全部加载结束
						self.CB_candidate_Q.pop(CB_id)	
					else:									# CB还有一些子块
						self.CB_candidate_Q[CB_id][1] -= 1
					self.CB_C += CB_list[2]					# A2[21]
					self.CB_scheduled_Q.append([CB_name,CB_list[2]])	# A2[22]
			else:
				break
		
		return target

	def scheduleCBFinish(self):
		cycle = self.MB_C - self.CB_Time
		finish_CB_index = []
		#print("before_CB_s", self.CB_scheduled_Q)
		#print("self.MB_C: ", self.MB_C)
		#print("self.CB_time: ", self.CB_Time)
		#print("self.self.RM_C: ", self.RM_C)
		for s_id, target in enumerate(self.CB_scheduled_Q):
			CB_cycle = target[1]
			if CB_cycle > cycle:
				break
			cycle -= CB_cycle
			self.CB_Time += CB_cycle
			finish_CB_index.append(s_id)

			CB_name_list = target[0].split("\t")
			self.AIMT_ST.finishCB(CB_name_list[0], CB_name_list[1])
			self.CB_timestamp_record.append([target, self.CB_Time-CB_cycle, self.CB_Time])
			
			MB_cycle = self.AIMT_ST.schedule_table[CB_name_list[0]][CB_name_list[1]]["MB"]["cycles"]
			self.RM_C += MB_cycle
		
		print("len self.CB_scheduled_Q : ", len(self.CB_scheduled_Q))
		print("self.CB_scheduled_Q : ", self.CB_scheduled_Q)

		for s_id in finish_CB_index:
			self.CB_scheduled_Q.pop(0)
		
		#print("after_CB_s", self.CB_scheduled_Q)
		
	def scheduleTimeUpdate(self, MB_name):
		if MB_name == None:
			self.MB_C = self.CB_C
		else:
			name_list = MB_name.split("\t")
			self.AIMT_ST.finishMB(name_list[0], name_list[1])
			print("MB_finish")
			MB_cycles = self.AIMT_ST.schedule_table[name_list[0]][name_list[1]]["MB"]["cycles"]
			self.MB_timestamp_record.append([MB_name, self.MB_C - MB_cycles, self.MB_C])
		self.Time = self.MB_C
		self.CB_C = max(self.Time, self.CB_C)
		self.scheduleCBFinish()
		if (len(self.CB_scheduled_Q) == 0):
			self.CB_Time = self.MB_C
	
	def scheduleCandidateUpdate(self):
		# MB Candidate Update
		for workload_name, status_list in self.AIMT_ST.MB_status.items():
			if status_list[0] == 1 and status_list[2] == 0:
				[nn_name, layer_name] = workload_name.split("\t")
				cycles = self.AIMT_ST.schedule_table[nn_name][layer_name]["MB"]["cycles"]
				self.AIMT_ST.MB_status[workload_name][2] = 1
				self.MB_candidate_Q.append([workload_name, status_list[1], cycles])
			
			if status_list[2] == 0:
				self.finish = 0
		
		# CB Candidate Update
		for workload_name, status_list in self.AIMT_ST.CB_status.items():
			if status_list[0] == 1 and status_list[2] == 0:
				[nn_name, layer_name] = workload_name.split("\t")
				cycles = self.AIMT_ST.schedule_table[nn_name][layer_name]["CB"]["cycles"]
				self.AIMT_ST.CB_status[workload_name][2] = 1
				self.CB_candidate_Q.append([workload_name, status_list[1], cycles])

	def schedule(self):
		self.scheduleInitialize()
		while self.finish == 0:
			MB_name = self.scheduleMBFinish()
			self.scheduleTimeUpdate(MB_name)
			self.scheduleCandidateUpdate()

			if len(self.CB_timestamp_record) == self.total_iters_num:
				self.finish = 1
		print("{}\t{}\t{}".format(self.Time,self.MB_C, self.CB_C))
	
	def schedule_ours(self):
		self.scheduleInitialize()
		while self.finish == 0:
			MB_name = self.scheduleMBFinish()
			self.scheduleTimeUpdate(MB_name)
			self.scheduleCandidateUpdate()

			if len(self.CB_timestamp_record) == self.total_iters_num:
				self.finish = 1

#class ST_DSE:
#	def __init__(self, PE_dim_x, PE_dim_y, PE_array_num, BW, frequence, WL2_size):
		
def model_load(nn_name):
	nn_f = open("{}.txt".format(nn_name), "r")
	lines = nn_f.readlines()
	nn_dict = {}
	for line in lines:
		if line.startswith("#") or line.startswith("*"):
			pass
		else:
			line_item = line.split(" ")
			layer_name = line_item[0]
			
			P = int(line_item[8])
			Q = int(line_item[9])
			C = int(line_item[3])
			K = int(line_item[7])
			R = int(line_item[4])
			S = int(line_item[4])
			stride = int(line_item[5])
			padding = int(line_item[6])
			X = int(line_item[1])
			Y = int(line_item[2])
			nn_dict[layer_name] = {"P":P,"Q":Q,"C":C,"K":K,"R":R,"S":S,"X":X,"Y":Y,"stride":stride, "padding":padding}
			print(str(layer_name) + " : " + str(nn_dict[layer_name]))
	nn_f.close()
	return nn_dict

def getSTResult(CB_record):
	CB_list = []
	CB_iters_num = []
	total_iters_num = {}
	w_name_pre = None
	for CB_result_sample in CB_record:
		w_name = CB_result_sample[0][0]
		print(w_name)
		if w_name == w_name_pre:
			CB_iters_num[-1] += 1
		elif w_name != w_name_pre:
			CB_list.append(w_name)
			CB_iters_num.append(1)
		w_name_pre = w_name

		if w_name not in total_iters_num:
			total_iters_num[w_name] = 1
		else:
			total_iters_num[w_name] += 1
	
	for id, w_name in enumerate(CB_list):
		total_iters = total_iters_num[w_name]
		CB_iters_num[id] /= total_iters

	'''
	file_o = "AIMT_o.txt"
	f_o = open(file_o, 'w')
	print("---------- {} ------------".format("total iter_num"), file=f_o)
	for w_name, iters in total_iters_num.items():
		print("{}\t\t{}".format(w_name, iters), file=f_o)
	print("", file=f_o)
	print("---------- {} ------------".format("CB_list"), file=f_o)
	for id, w_name in enumerate(CB_list):
		iters = CB_iters_num[id]
		print("{}\t\t{}".format(w_name, iters), file=f_o)
	f_o.close()
	'''
	return CB_list, CB_iters_num

def AIMT_run():
	# -- hardware config
	config_file = "AIMT.yaml"
	c_f = open(config_file)
	config = c_f.read()
	config_data = yaml.load(config)
	print(config_data)
	PE_dim_x = config_data["PE_dim_x"]
	PE_dim_y = config_data["PE_dim_y"]
	PE_array_num = config_data["PE_array_num"]
	BW = config_data["BW"]
	frequence = config_data["frequence"]
	WL2_size = config_data["WL2_size"]
	AIMT = AIMT_DSE(PE_dim_x, PE_dim_y, PE_array_num, BW, frequence, WL2_size)

	# -- model load
	nn_list = ["resnet50", "VGG16"]
	nn_fc_start = [49, 13]
	layer_num = {"resnet50":50, "VGG16":16}

	dependence_dict = {}
	for nn_name in nn_list:
		dependence_dict[nn_name] = []
		for src in range(layer_num[nn_name]-1):
			dst = src+1
			l_src = "layer" + str(src+1)
			l_dst = "layer" + str(dst+1)
			dependence_dict[nn_name].append([l_src, l_dst])

	for nn_id, nn_name in enumerate(nn_list):
		layer_dict = model_load(nn_name)
		nn = NeuralNetwork(nn_name)
		fc_start = nn_fc_start[nn_id]
		l_id = 0
		for layer_name, dim_dict in layer_dict.items():
			if l_id < fc_start:
				nn.addLayer(layer_name, "CONV", dim_dict)
			else:
				nn.addLayer(layer_name, "FC", dim_dict)
			l_id += 1
		AIMT.addNN(nn, nn_name)
		AIMT.addDependence(nn_name, dependence_dict[nn_name])
	
	AIMT.multiNNEstimation()
	AIMT.schedule()

	CB_record = AIMT.CB_timestamp_record
	CB_list, CB_iters_num = getSTResult(CB_record)

	return CB_list, CB_iters_num

if __name__ == '__main__':
	AIMT_run()
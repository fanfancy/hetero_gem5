import math
import os
import sys
import random

layer_list = {"resnet18":[1,2,2,2,2,6,7,7,7,10,11,11,11,14,15,15,15],"resnet50":[1,2,3,4,5,3,4,5,3,4,11,12,13,14,15,13,14,15,13,14,15,13,23,24,25,26,27,25,26,27,25,26,27,25,26,27,25,26,27,25,41,42,43,44,45,43,44,45,43,50],"VGG16":[1,2,3,4,5,6,6,8,9,9,11,11,11],"alexnet":[1,2,3,4,5,6,7,8],"lenet":[1,2,3,4,5]}

ratio = {}
#ratio = {"layer1":[12,16,6],"layer2":[8,16,16],"layer3":[1,4,16],"layer4":[1,4,16],"layer5":[1,4,10]}


struct_name = str(sys.argv[1])
if struct_name == "ours":
	struct_name = "ours_ours_"
elif struct_name == "simba_ours":
	struct_name = "simba_ours_"
app_name = str(sys.argv[2])

file_p = struct_name + str(app_name) + "_P_stable.txt"
file_K = struct_name + str(app_name) + "_K_stable.txt"
file_PK = struct_name + str(app_name) + "_PK_stable.txt"
f_p = open(file_p)
f_pk = open(file_PK)
f_k = open(file_K)

lines_p = f_p.readlines()
line_p = lines_p[0]
line_e_p = lines_p[1]
line_d_p = lines_p[2]
line_par_p = lines_p[3]
lines_pk = f_pk.readlines()
line_pk = lines_pk[0]
line_e_pk = lines_pk[1]
line_d_pk = lines_pk[2]
line_par_pk = lines_pk[3]
lines_k = f_k.readlines()
line_k = lines_k[0]
line_e_k = lines_k[1]
line_d_k = lines_k[2]
line_par_k = lines_k[3]

def getLayerParam(app_name):
	layer_num = 0
	layer_dict = {}
	layer_id_list = []
	f = open("./../../nn_input_noc_nop/" + app_name + ".txt")

	print("network model ----- " + app_name + " -------------")

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
			print("layer" + str(layer_num) + " : " + str(layer_dict[layer_id]))
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
			print("layer=",str(layer_id),"  par_real=", util)
		item_num += 1
	return result_dict

def lineParse(line, dim = 0, ratio_list = {}, chiplet=16):
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
				#result_dict[layer_id] = float(line_item) * ratio_list[layer_id][dim] / chiplet
		item_num += 1
	return result_dict

layer_param_list = getLayerParam(app_name)
scale_p = getScaleRatio(line_par_p, layer_param_list)
scale_pk = getScaleRatio(line_par_pk, layer_param_list)
scale_k = getScaleRatio(line_par_k, layer_param_list)

result_p = lineParse(line_p, dim=0, ratio_list=scale_p)
result_pk = lineParse(line_pk, dim=1, ratio_list=scale_pk)
result_k = lineParse(line_k, dim=2, ratio_list=scale_k)

result_e_p = lineParse(line_e_p, dim=0, ratio_list=scale_p)
result_e_pk = lineParse(line_e_pk, dim=1, ratio_list=scale_pk)
result_e_k = lineParse(line_e_k, dim=2, ratio_list=scale_k)

result_d_p = lineParse(line_d_p)
result_d_pk = lineParse(line_d_pk)
result_d_k = lineParse(line_d_k)

edp_list = {}
energy_list = {}
delay_list = {}
for layer_id in result_p:
	edp_list[layer_id] = "\t" + str(result_p[layer_id]) + "\t" + str(result_pk[layer_id]) + "\t" + str(result_k[layer_id])
	delay_list[layer_id] = "\t" + str(result_d_p[layer_id]) + "\t" + str(result_d_pk[layer_id]) + "\t" + str(result_d_k[layer_id])
	energy_list[layer_id] = "\t" + str(result_e_p[layer_id]) + "\t" + str(result_e_pk[layer_id]) + "\t" + str(result_e_k[layer_id])


fout = open("./../intra_layer_edp_parse/"+struct_name + app_name + "_intra_layer.txt", 'w')
fout_e = open("./../intra_layer_edp_parse/"+struct_name + app_name + "_intra_layer_energy.txt", 'w')
fout_d = open("./../intra_layer_edp_parse/"+struct_name + app_name + "_intra_layer_delay.txt", 'w')
line1 = app_name + "\t" + "P" + "\t" + "PK" + "\t" + "K"
print(line1, file = fout)
print(line1, file = fout_e)
print(line1, file = fout_d)

layer_num = 1
for index in layer_list[app_name]:
	layer_id = "layer" + str(index)
	layer_cur = "layer" + str(layer_num)
	
	line_edp = layer_cur + edp_list[layer_id]
	line_d = layer_cur + delay_list[layer_id]
	line_e = layer_cur + energy_list[layer_id]
	print(line_edp, file=fout)
	print(line_d, file=fout_d)
	print(line_e, file=fout_e)

	layer_num += 1

print(result_k)
print(result_p)
print(result_pk)
fout.close()

print(result_e_k)
print(result_e_p)
print(result_e_pk)
fout_e.close()

print(result_d_k)
print(result_d_p)
print(result_d_pk)
fout_d.close()

f_p.close()
f_k.close()
f_pk.close()

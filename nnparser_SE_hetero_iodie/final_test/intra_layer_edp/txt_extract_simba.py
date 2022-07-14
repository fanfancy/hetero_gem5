import math
import os
import sys
import random

app_name = str(sys.argv[1])

layer_list = {"resnet18":[1,2,2,2,2,6,7,7,7,10,11,11,11,14,15,15,15],"resnet50":[1,2,3,4,5,3,4,5,3,4,11,12,13,14,15,13,14,15,13,14,15,13,23,24,25,26,27,25,26,27,25,26,27,25,26,27,25,26,27,25,41,42,43,44,45,43,44,45,43,50],"VGG16":[1,2,3,4,5,6,6,8,9,9,11,11,11],"alexnet":[1,2,3,4,5,6,7,8],"lenet":[1,2,3,4,5]}
file_K = "ours_simba_" + str(app_name) + "_K_stable.txt"
file_C = "ours_simba_" + str(app_name) + "_C_stable.txt"
file_KC = "ours_simba_" + str(app_name) + "_KC_stable.txt"
f_k = open(file_K)
f_c = open(file_C)
f_kc = open(file_KC)

lines_k = f_k.readlines()
line_k = lines_k[0]
line_e_k = lines_k[1]
line_d_k = lines_k[2]
line_code_k = lines_k[3]
lines_c = f_c.readlines()
line_c = lines_c[0]
line_e_c = lines_c[1]
line_d_c = lines_c[2]
line_code_c = lines_c[3]
lines_kc = f_kc.readlines()
line_kc = lines_kc[0]
line_e_kc = lines_kc[1]
line_d_kc = lines_kc[2]
line_code_kc = lines_kc[3]

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

def lineParse(line, scale = {}):
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
			if scale == {}:
				result_dict[layer_id] = float(line_item) * 1
			else:
				result_dict[layer_id] = float(line_item) * scale[layer_id]
		item_num += 1
	return result_dict

def codeParse(line, para_t, layer_param, chiplets = 16, K0 = 16):
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
			K_num = K0
			C_num = K0
			for index in range(4):
				par = int(par_code[index])
				par_num = int(par_num_code[index])
				if par == 3:
					K_num *= par_num
				elif par == 2:
					C_num *= par_num
			result = 0
			real_num = {"K":1, "C":1}
			if para_t == "K":
				result = layer_param[layer_id]["K"] / K_num
				if result > 1:
					result = 1
				real_num["K"] = math.ceil(chiplets * result)
			elif para_t == "C":
				result = layer_param[layer_id]["C"] / C_num
				if result > 1:
					result = 1
				real_num["C"] = math.ceil(chiplets * result)
			elif para_t == "KC":
				result_k = layer_param[layer_id]["K"] / K_num
				result_c = layer_param[layer_id]["C"] / C_num
				if result_c > 1:
					result_c = 1
				if result_k > 1:
					result_k = 1
				result = result_k * result_c
				real_num["C"] = math.ceil((chiplets**0.5) * result_c)
				real_num["K"] = math.ceil((chiplets**0.5) * result_k)
			
			result_dict[layer_id] = result
			par_real[layer_id] = str(real_num["K"]) + ","+ str(real_num["C"])
		item_num += 1
	return result_dict, par_real

layer_param_list = getLayerParam(app_name)
para_result_k, par_real_k = codeParse(line_code_k, "K", layer_param_list)
para_result_c, par_real_c = codeParse(line_code_c, "C", layer_param_list)
para_result_kc, par_real_kc = codeParse(line_code_kc, "KC", layer_param_list)

print("scale k : ", para_result_k)
print("scale c : ", para_result_c)
print("scale kc : ", para_result_kc)

result_k = lineParse(line_k, para_result_k)
result_c = lineParse(line_c, para_result_c)
result_kc = lineParse(line_kc, para_result_kc)

result_e_k = lineParse(line_e_k, para_result_k)
result_e_c = lineParse(line_e_c, para_result_c)
result_e_kc = lineParse(line_e_kc, para_result_kc)

result_d_k = lineParse(line_d_k)
result_d_c = lineParse(line_d_c)
result_d_kc = lineParse(line_d_kc)

edp_list = {}
energy_list = {}
delay_list = {}
par_real_list = {}
for layer_id in result_k:
	edp_list[layer_id] = "\t" + str(result_k[layer_id]) + "\t" + str(result_c[layer_id]) + "\t" + str(result_kc[layer_id])
	delay_list[layer_id] = "\t" + str(result_d_k[layer_id]) + "\t" + str(result_d_c[layer_id]) + "\t" + str(result_d_kc[layer_id])
	energy_list[layer_id] = "\t" + str(result_e_k[layer_id]) + "\t" + str(result_e_c[layer_id]) + "\t" + str(result_e_kc[layer_id])
	par_real_list[layer_id] = "\t" + str(par_real_k[layer_id]) + "\t" + str(par_real_c[layer_id]) + "\t" + str(par_real_kc[layer_id])

fout = open("ours_simba_" + app_name + "_intra_layer.txt", 'w')
fout_e = open("ours_simba_" + app_name + "_intra_layer_energy.txt", 'w')
fout_d = open("ours_simba_" + app_name + "_intra_layer_delay.txt", 'w')
fout_par = open("ours_simba_" + app_name + "_intra_layer_parallel.txt", 'w')
line1 = app_name + "\t" + "K" + "\t" + "C" + "\t" + "KC"
print(line1, file = fout_d)
print(line1, file = fout)
print(line1, file = fout_e)
print(line1, file = fout_par)

layer_num = 1
for index in layer_list[app_name]:
	layer_id = "layer" + str(index)
	layer_cur = "layer" + str(layer_num)
	
	line_edp = layer_cur + edp_list[layer_id]
	line_d = layer_cur + delay_list[layer_id]
	line_e = layer_cur + energy_list[layer_id]
	line_par = layer_cur + par_real_list[layer_id]
	print(line_edp, file=fout)
	print(line_d, file=fout_d)
	print(line_e, file=fout_e)
	print(line_par, file=fout_par)

	layer_num += 1


print(result_k)
print(result_c)
print(result_kc)
fout.close()

print(result_e_k)
print(result_e_c)
print(result_e_kc)
fout_e.close()

print(result_d_k)
print(result_d_c)
print(result_d_kc)
fout_d.close()

f_k.close()
f_kc.close()
f_c.close()

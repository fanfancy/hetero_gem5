import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum

from numpy.core.fromnumeric import mean
from mesh import *
import re


class layerType(Enum):
	CONV = 0
	POOL = 1
	FC = 2

class layerModel:
	def __init__(self,layer_type, i_H, i_W, i_ch, w_size, stride, padding, o_ch):
		self.layer_type = layer_type
		self.i_H = i_H
		self.i_W = i_W
		self.i_ch = i_ch
		self.w_size = w_size
		self.stride = stride
		self.padding = padding
		self.o_ch = o_ch
		self.pool_flag = 0
		self.pool_w_size = 0
		self.pool_stride = 0
		self.pool_padding = 0

		self.o_W_0 = int((i_W + 2*padding - w_size)/stride) + 1
		self.o_H_0 = int((i_H + 2*padding - w_size)/stride) + 1
		self.o_W = self.o_W_0
		self.o_H = self.o_H_0

	def setPoolParam(self, pool_w_size, pool_stride, pool_padding):
		self.pool_flag = 1
		self.pool_w_size = pool_w_size
		self.pool_stride = pool_stride
		self.pool_padding = pool_padding
		self.o_W = int((self.o_W_0 + 2 * pool_padding - pool_w_size)/pool_stride) + 1
		self.o_H = int((self.o_H_0 + 2 * pool_padding - pool_w_size)/pool_stride) + 1

	# 层内总的乘法数目
	def calComputationNum(self):
		if self.pool_flag == 1:
			c_num = self.w_size * self.w_size * self.i_ch * self.o_H_0 * self.o_W_0 * self.o_ch + self.pool_w_size * self.pool_w_size * self.o_W * self.o_H * self.o_ch
		else:
			c_num = self.w_size * self.w_size * self.i_ch * self.o_H * self.o_W * self.o_ch
		# 区分池化与卷积、全连接
		#if self.layer_type.value != 1:
			#c_num = self.o_H * self.o_W *self.o_ch * self.w_size * self.w_size * self.i_ch
		#else:
			# 当是池化时，假设最大值或者均值池化，都是输入特征值乘上一个权重（0、1、0.25……）
		#	c_num = self.i_H * self.i_W * self.i_ch
		return c_num
	
	# 层的输入神经元的数目
	def getInputNeuNum(self):
		i_num = self.i_H * self.i_W * self.i_ch
		return i_num
	
	# 层的每个输出神经元所需的计算量（乘法数目）
	def getNeuComputeNum(self):
		if self.pool_flag == 0:
			c_num = self.w_size * self.w_size * self.i_ch
		else:
			c_num = math.ceil(self.calComputationNum() / self.getLayerNeuNum())
		#if self.layer_type.value != 1:
		#	c_num = self.w_size * self.w_size * self.i_ch
		#else:
		#	c_num = self.w_size * self.w_size
		return c_num

	# 层的(输出)神经元数目
	def getLayerNeuNum(self):
		o_num = self.o_H * self.o_W * self.o_ch
		return o_num


#配置文件处理
def DNN_input(file_name , DNN):
	layer_num = 0
	layer_list = {}
	f = open("./nn_input/" + file_name + ".txt")
	lines = f.readlines()
	pool_addr = {}
	line_num = 0
	for line in lines:
		if line.startswith("#"):
			pass
		else:
			line_item = line.split(" ")
			if line_item[1] == "POOL":
				pool_addr[line_num] = int(line_item[5])
			line_num += 1

	for line in lines:
		if line.startswith("#"):
			pass
		else:
			line_item = line.split(" ")
			if line_item[1] == "CONV":
				layer_type = layerType.CONV
				layer_list[layer_num] = layerModel(layer_type,int(line_item[2]),int(line_item[3]), int(line_item[4]), int(line_item[5]), int(line_item[6]), int(line_item[7]), int(line_item[8]))
				DNN.addLayer(layer_list[layer_num])
			elif line_item[1] == "POOL":
				pool_w_size = int(line_item[5])
				pool_stride = int(line_item[6])
				pool_padding = int(line_item[7])
				DNN.layer_list[DNN.layer_num - 1].setPoolParam(pool_w_size, pool_stride, pool_padding)
			else:
				layer_type = layerType.FC
				layer_list[layer_num] = layerModel(layer_type,int(line_item[2]),int(line_item[3]), int(line_item[4]), int(line_item[5]), int(line_item[6]), int(line_item[7]), int(line_item[8]))
				DNN.addLayer(layer_list[layer_num])
			layer_num += 1
	for layer_id in layer_list:
		print("###layer id "+str(layer_id)+" ###")
		layer = layer_list[layer_id]
		print("\t input_H/W = ", layer.i_H,"\t,  input_ch = ",layer.i_ch,";\toutput_H/W = ",layer.o_H,"\t,  output_ch = ",layer.o_ch)
		print("\t total_computation_num = ",layer.calComputationNum())
		print("\t output neuron_num = ",layer.getLayerNeuNum())
		print("\t pool_size = ",layer.pool_w_size)
	
	DNN.setLayerInfo()
	f.close()

def mapping_group_size( group_size, neuron_layer):
	layer_node_num = {}
	for i in neuron_layer:
		layer_node_num[i] = math.ceil(neuron_layer[i] / group_size)
	return layer_node_num

class DNNModel:
	def __init__(self, name):
		self.name = name
		self.layer_num = 0
		self.layer_list = {}
		self.layer_neuron_out = {}
		self.layer_compute_num = {}
		self.layer_neu_compute = {}

	def addLayer(self, layer):
		self.layer_list[self.layer_num] = layer
		self.layer_num += 1

	def getTotalComputeNum(self):
		c_num = 0
		for i in range(0,self.layer_num):
			c_num += self.layer_list[i].calComputationNum()
		return c_num
	
	def setLayerInfo(self):
		for i in range(0, self.layer_num):
			self.layer_neuron_out[i] = self.layer_list[i].getLayerNeuNum()
			self.layer_compute_num[i] = self.layer_list[i].calComputationNum()
			self.layer_neu_compute[i] = self.layer_list[i].getNeuComputeNum()

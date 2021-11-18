import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from single_engine_predict_granularity import *
from mesh_hetero import *
from matplotlib import pyplot as plt
from config import *
import openpyxl
import subprocess
from concurrent.futures import ThreadPoolExecutor
from randomTest_noc_nop_granularity_model import run_randomTest_granu_model
import multiprocessing
container = []

model_list = ["lenet", "alexnet", "VGG16" , "resnet_18", "resnet_50"]
layer_dict = {}
layer_times = {}


def task(model,layer_name):
    print (" ################# running",model,layer_name )
    run_randomTest_granu_model(model,layer_name)
    return 1


pool = multiprocessing.Pool(processes = 24)
for model in model_list:

    f = open("./nn_input_pqck/" + model + ".txt")
    lines = f.readlines()
    line_num = 0
    for line in lines:
        if line.startswith("#"):
            pass
        else:
            line_item = line.split(" ")
            real_layer_name = line_item[0]
            conv_layer_label = line_item[-1]
            conv_layer_label = conv_layer_label.replace("\n", "")
            
            if "conv_layer" in conv_layer_label:
                item = conv_layer_label.split("*")
                conv_layer_id = item[0]
                if len(item)>1:
                    conv_layer_time = int(item[1])
                else:
                    conv_layer_time = 1
                layer_dict[conv_layer_id] = real_layer_name
                # print (conv_layer_id,conv_layer_time)
                layer_times[real_layer_name] = conv_layer_time

    for item in layer_dict:
        layer_name = layer_dict[item]
        pool.apply_async(task, (model,layer_name,))
        
pool.close()
pool.join() 
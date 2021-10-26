import math
import os
import sys
import random
import numpy as np
import copy
from enum import Enum
from numpy.core.fromnumeric import mean
import re
np.set_printoptions(threshold=sys.maxsize)
from DNN import *
from run_configs import *
if if_hetero == 0:
    from mesh import *
else:
    from mesh_hetero import *

# 遗传算法
class GA_all:
    def __init__(self, dnn, core_node_num):
        self.dnn = dnn
        self.core_node_num = core_node_num
        self.layer_num = dnn.layer_num

        self.num_generation = 200
        self.num_generation_left = 40
        self.iter_max = 150
        self.ratio = 0.5
        self.pm = 0.3
    
    def check_layer_has_node(self, individual):
        individual_repair = individual.copy()
        for layer_id in range (0,self.layer_num):
            cal_map = individual_repair[0:self.core_node_num].tolist()
            if not (layer_id in cal_map):
                # print ("layer",layer_id,"has no layer")
                if (-1 in cal_map):
                    index_no_layer = cal_map.index(-1)
                    individual_repair[index_no_layer] = layer_id
                else:
                    result = max(set(cal_map), key=cal_map.count)
                    index_many_layer = cal_map.index(result)
                    individual_repair[index_many_layer] = layer_id
        return individual_repair

    def check_map(self, individual):
        # print ("individual",individual)
        individual_repair = individual.copy()
        for node_id in range (0,self.core_node_num):
            if not (node_id in individual[self.core_node_num:]):
                phy_map = individual_repair[self.core_node_num:].tolist()
                result = max(set(phy_map), key=phy_map.count)
                index_many_layer = phy_map.index(result) + self.core_node_num
                individual_repair[index_many_layer] = node_id
                # print ("individual_repair", individual_repair)
        return individual_repair

    def check_layer_has_node_only (self, individual):
        for layer_id in range (0,self.layer_num):
            if not (layer_id in individual[:self.core_node_num]):
                return False
        return True

    def check_map_only(self, individual):
        for node_id in range (0,self.core_node_num):
            if not (node_id in individual[self.core_node_num:]):
               return False
        return True

    def createFirstGeneration(self):
        generation = np.zeros((self.num_generation, self.core_node_num*2))
        for ind_id in range (self.num_generation):
            if_check_map = False
            generation[ind_id][0: self.layer_num] = np.arange(0,self.layer_num)
            generation[ind_id][self.layer_num :self.core_node_num] = np.random.randint(-1,self.layer_num,size=self.core_node_num - self.layer_num)
            generation[ind_id] = self.check_layer_has_node(generation[ind_id])
            the_map = np.arange(self.core_node_num)
            np.random.shuffle(the_map)
            generation[ind_id][self.core_node_num:] = the_map
        return generation
    
    def calF(self,indv):
        # 计算compute时间
        F_cur=F.copy()
        node_num_in_layer = {}
        layer_comp_cycles_1node = {}
        layer_send_pkt_num = {}
        for layer in range (self.layer_num):
            node_num_in_layer[layer] = sum(indv[0:self.core_node_num] == layer)
            layer_comp_cycles_1node[layer] = math.ceil (self.dnn.layer_neu_compute[layer] * self.dnn.layer_neuron_out[layer] / node_num_in_layer[layer] / PE_ability)  ## layer_neu_compute = 每层的每个输出神经元所需的计算量
            layer_send_pkt_num[layer] = math.ceil(self.dnn.layer_neuron_out[layer] / node_num_in_layer[layer] /(neu_per_flit*flit_per_pkt)) 
            
        max_comp_cycle = max(layer_comp_cycles_1node.values())

        if (debug):
            print ("indv",indv)
            print ("node_num_in_layer",node_num_in_layer)
            print ("layer_comp_cycles_1node",layer_comp_cycles_1node)
            print ("layer_send_pkt_num",layer_send_pkt_num)
            print ("max_comp_cycle",max_comp_cycle)

        for possible_src in range (self.core_node_num):
            for possible_dst in range (self.core_node_num):
                src_layer_id = indv[possible_src]
                dst_layer_id =  indv[possible_dst]

                src_phy_id = int(indv[possible_src + self.core_node_num])
                dst_phy_id = int(indv[possible_dst + self.core_node_num])
                
                if src_layer_id != -1 and src_layer_id + 1 ==  dst_layer_id:  # 这两个节点之间具有通信需求
                    bw_needed =  layer_send_pkt_num[src_layer_id] / max_comp_cycle * flit_per_pkt
                    for link in route_table[(src_phy_id + 1000,dst_phy_id + 1000)]:
                        F_cur[link] += bw_needed / bw_scales [link]
        if(debug):
            ratio = 1
            for item in F_cur:
                if F_cur[item]>1:
                    print (item, F_cur[item])
                    ratio = ratio * (F_cur[item])
            print ("F_cur max",max(F_cur.values()))
            print ("F_cur sum",sum(F_cur.values()))
            print ("F_cur len",len(F_cur))
            print ("ratio=",ratio)
        return F_cur,max_comp_cycle

    def calFitness(self,gen):
        fitness = np.zeros(self.num_generation)
        fitness_1 = np.zeros(self.num_generation)
        all_max_F_cur = np.zeros(self.num_generation)
        for ind_id in range(0, self.num_generation):
            # 先检查
            F_cur_ind,max_comp_cycle = self.calF(gen[ind_id])
            if (max(F_cur_ind.values()) < 1):
                degrade_ratio = 1
            else:
                degrade_ratio = max(F_cur_ind.values()) + congestion_ratio
            fitness[ind_id] = degrade_ratio * max_comp_cycle
            all_max_F_cur[ind_id] = max(F_cur_ind.values())
            fitness_1[ind_id] = 1/fitness[ind_id]
        return fitness,fitness_1,all_max_F_cur

    def calProbability(self, list, num):
        total = list.sum(axis = 0)
        probability = np.zeros(num)
        for i in range(0, num):
            probability[i] = list[i] / total
        return probability

    def calSelectTable(self, select_probability):
        probability_list = {}
        probability_pre = 0
        for i in range(0, self.num_generation):
            probability_list[i] = probability_pre + select_probability[i]
            probability_pre += select_probability[i]
        return probability_list

    def selectGeneration(self, select_probability, select_table):
        num_ran1 = random.random()
        flag = 0
        time1 = 0
        while flag == 0:
            if num_ran1 < select_table[time1]:
                flag = 1
                break
            time1 += 1
        num_ran2 = random.random()
        flag = 0
        time2 = 0
        while flag == 0:
            if num_ran2 < select_table[time2]:
                flag = 1
                break
            time2 += 1
        if select_probability[time1] > select_probability[time2]:
            return time1
        else:
            return time2

    def cross(self, father1,father2):
        # 交叉
        index = random.randint(1, self.core_node_num*2 - 1)
        gen1 = np.zeros(self.core_node_num*2)
        gen2 = np.zeros(self.core_node_num*2)
        for i in range(0, self.core_node_num*2):
            if i < index:
                gen1[i] = father1[i]
                gen2[i] = father2[i]
            else:
                gen1[i] = father2[i]
                gen2[i] = father1[i]
        
        # todocheck
        gen1 = self.check_layer_has_node(gen1)
        gen1 = self.check_map(gen1)
        gen2 = self.check_layer_has_node(gen2)
        gen2 = self.check_map(gen2)

        rand1 = random.random()
        rand2 = random.random()
        
        assert(self.check_layer_has_node_only(gen1))
        assert(self.check_layer_has_node_only(gen2))
        assert(self.check_map_only(gen1))
        assert(self.check_map_only(gen2))

        # 变异
        if rand1 < self.pm:
            index = random.randint(0, self.core_node_num*2 - 1)
            if index <= self.core_node_num-1:
                value = random.randint(-1, self.layer_num-1)
                gen1[index] = value
                gen1 = self.check_layer_has_node(gen1)
            else:
                index2 = random.randint(0, self.core_node_num - 1) + self.core_node_num
                temp = gen1[index]
                gen1[index] = gen1[index2]
                gen1[index2] = temp
    
        if rand2 < self.pm:
            index = random.randint(0, self.core_node_num*2 - 1)
            if index <= self.core_node_num-1:
                value = random.randint(-1, self.layer_num-1)
                gen2[index] = value
                gen2 = self.check_layer_has_node(gen2)
            else:
                index2 = random.randint(0, self.core_node_num - 1) + self.core_node_num
                temp = gen2[index]
                gen2[index] = gen2[index2]
                gen2[index2] = temp

        assert(self.check_layer_has_node_only(gen1))
        assert(self.check_layer_has_node_only(gen2))
        assert(self.check_map_only(gen1))
        assert(self.check_map_only(gen2))
        return gen1, gen2

    def createNextGeneration(self, generation, select_probability, select_table):
        generation_next = np.zeros((self.num_generation, self.core_node_num *2 ), dtype = int)
        times = int((self.num_generation - self.num_generation_left)/2)
        for i in range(0, times):
            # 选择
            gerneration_id_1 = self.selectGeneration(select_probability, select_table)
            gerneration_id_2 = self.selectGeneration(select_probability, select_table)
            father1 = generation[gerneration_id_1]
            father2 = generation[gerneration_id_2]
            # 交叉 + 变异
            gen1, gen2 = self.cross(father1,father2)
            
            gen_id = i * 2
            for l_id in range(0, self.core_node_num *2 ):
                generation_next[gen_id][l_id] = gen1[l_id]
                generation_next[gen_id + 1][l_id] = gen2[l_id]
            
        select_gen_list = np.argsort(-select_probability)
        for i in range(0, self.num_generation_left):
            gen_id = self.num_generation - self.num_generation_left + i
            select_gen_id = select_gen_list[i]
            for l_id in range(0, self.core_node_num *2 ):
                generation_next[gen_id][l_id] = generation[select_gen_id][l_id]
        return generation_next


    def GA(self):
        generation = self.createFirstGeneration()
        for iter_id in range (self.iter_max):
            fitness_gen, fitness_1_gen,all_max_F_cur = self.calFitness(generation)
            print ("iter=",iter_id, np.average(fitness_gen), np.min(fitness_gen))
            select_probability = self.calProbability(fitness_1_gen, self.num_generation)
            select_table = self.calSelectTable(select_probability)
            generation = self.createNextGeneration(generation, select_probability, select_table)

        # 最终结果 
        fitness_gen, fitness_1_gen,all_max_F_cur = self.calFitness(generation)
        best_sort = np.argsort(fitness_gen)
        best_id = best_sort[0]
        final_best_id = best_sort[0] # 在所有fitness相同用的结果里寻找f最小的结果
        best_fitness = fitness_gen[best_id]
        for id in range (len(fitness_gen)):
            best_F_cur_value = 1000
            if fitness_gen[id] == best_fitness:
                if all_max_F_cur[id] < best_F_cur_value:
                    final_best_id = id
                    best_F_cur_value = all_max_F_cur[id] 

        generation_best = generation[final_best_id]
        
        # check最终结果
        best_F_cur,max_comp_cycle = self.calF(generation_best)
        if (max(best_F_cur.values()) < 1):
            degrade_ratio = 1
        else:
            degrade_ratio = max(best_F_cur.values()) + congestion_ratio
        best_fitness = max_comp_cycle * degrade_ratio

        print ("generation_best",generation_best)
        print ("best_F_cur",best_F_cur)
        print ("max(best_F_cur.values()",max(best_F_cur.values()))
        print ("best_fitness",best_fitness)
        return(generation_best)


def create_task_list(folder_name, max_file_num, scale_size, node_num, compute_node, send_num_node, send_dst_node, wait_node ,map):
    for i in range(0,max_file_num):
        f = open(folder_name + "/" + str(i) + ".txt",'w')
        f.close()

    for node_id in range(0, node_num): # cal_node_id
        
        f = open(folder_name + "/" + str(node_id) + ".txt",'w')
        
        send_line = {}
        wait_line = ""
        cal_line = ""
        wait_cmd_line = {}
        # wait指令
        if node_id in wait_node:
            wait_line = "wait "+str(wait_node[node_id])
            #print ("wait "+str(wait_node[node]), file = f)
        wait_cmd_num = 0
        for wait_id in range (node_num):
            if map[wait_id]!= -1:
                wait_cmd_line[wait_cmd_num] = "wait_cmd " + str(wait_id)
                wait_cmd_num += 1
        # send指令
        dst_num = 0
        if node_id in send_dst_node and (send_dst_node[node_id]!=[]): # 本层有send任务
            send_pkt_num = math.ceil(send_num_node[node_id] / scale_size)
            pkt_sent = 0
            while pkt_sent <  send_pkt_num:
                for i in range(0, len(send_dst_node[node_id])):
                    id = (i + node_id) % len(send_dst_node[node_id])
                    dst_id = send_dst_node[node_id][id]
                    send_line[dst_num] = "send "+str(dst_id)+" "+str(1)
                    dst_num += 1
                pkt_sent += 1
        
        if dst_num > 0:
            for dst_i in send_line:
                print(send_line[dst_i], file = f)
        
        # cal指令
        if node_id in compute_node:
            cal_cycle = math.ceil(compute_node[node_id] / scale_size)
            cal_line = "cal "+str(cal_cycle)
            print (cal_line, file = f)
        
        if wait_line != "":
            print (wait_line, file = f)
        
        
        print("finish", file = f)
        f.close()


def generate_info (dnn, map):
    node_num_in_layer = {}
    layer_comp_cycles_1node = {}
    layer_send_pkt_num = {}
    layer_wait_pkt_num = {}
    
    for layer in range (dnn.layer_num):
        node_num_in_layer[layer] = sum(map[0:CORE_NUM] == layer)
        layer_comp_cycles_1node[layer] = math.ceil (dnn.layer_neu_compute[layer] * dnn.layer_neuron_out[layer] / node_num_in_layer[layer] / PE_ability)  ## layer_neu_compute = 每层的每个输出神经元所需的计算量
        layer_send_pkt_num[layer] = math.ceil(dnn.layer_neuron_out[layer] / node_num_in_layer[layer] /(neu_per_flit*flit_per_pkt))
    for layer in range (1, dnn.layer_num):
        layer_wait_pkt_num[layer] = layer_send_pkt_num[layer-1] *  node_num_in_layer[layer-1]

    print ("node_num_in_layer" ,node_num_in_layer )
    print ("layer_comp_cycles_1node" ,layer_comp_cycles_1node )
    print ("layer_send_pkt_num" , layer_send_pkt_num)
    print ("layer_wait_pkt_num" ,layer_wait_pkt_num )
    print ("map1:" , map[:CORE_NUM])
    print ("map2:" , map[CORE_NUM:])
    print ("len(map)" , len(map))

    phy_node_wait_pkt = {}
    phy_node_send_pkt = {}
    phy_node_send_dst = {}
    phy_node_cal_cycles = {}

    for cal_id in range (0, CORE_NUM):
        layer_id = int(map[cal_id])
        phy_id = int(map[cal_id+CORE_NUM])
        if layer_id != -1:
            phy_node_cal_cycles[phy_id] = layer_comp_cycles_1node[layer_id]
            if layer_id != dnn.layer_num:
                phy_node_send_pkt[phy_id]   = layer_send_pkt_num[layer_id]
            if layer_id != 0:
                phy_node_wait_pkt[phy_id]   = layer_wait_pkt_num[layer_id]
    
    for cal_id_src in range (0, CORE_NUM):
        phy_node_send_dst[cal_id_src] = []

    for cal_id_src in range (0, CORE_NUM):
        for cal_id_dst in range (0, CORE_NUM):
            layer_id_src = int (map[cal_id_src] )
            layer_id_dst = int (map[cal_id_dst] )
            phy_id_src = int(map[cal_id_src + CORE_NUM])
            phy_id_dst = int(map[cal_id_dst + CORE_NUM])

            if (layer_id_src != -1) and ( (layer_id_src + 1) == layer_id_dst):
                phy_node_send_dst[phy_id_src].append(phy_id_dst)
    
    print ("\nphy_node_wait_pkt", phy_node_wait_pkt)
    print ("\nphy_node_send_pkt", phy_node_send_pkt)
    print ("\nphy_node_send_dst", phy_node_send_dst)
    print ("\nphy_node_cal_cycles", phy_node_cal_cycles)
    return phy_node_wait_pkt,phy_node_send_pkt,phy_node_send_dst,phy_node_cal_cycles

def neuron_group(dnn,node_num):
    map_best = np.zeros((node_num*2))
    map_best = map_best -1 
    max_node_num_cur = node_num
    print (max_node_num_cur)
    print ("node_num",node_num)

    while (1): 
        layer_node_num = {}
        neuron_sum = sum(dnn.layer_neuron_out.values())
        for i in dnn.layer_neuron_out :
            layer_node_num[i] = math.ceil(dnn.layer_neuron_out[i] / neuron_sum * max_node_num_cur) 
            print ("layer_node_num[i]",i,layer_node_num[i])
        if sum(layer_node_num.values()) <= max_node_num:
            break
        else:
            max_node_num_cur -= 1 

    cal_id = 0
    for item in layer_node_num:
        map_best[cal_id] = -1
        node_sum_onelayer =  layer_node_num[item]
        while node_sum_onelayer != 0:
            map_best[cal_id] = item
            node_sum_onelayer -= 1
            cal_id += 1
    print ("dnn.layer_neuron_out",dnn.layer_neuron_out)
    print ("layer_node_num",layer_node_num)
    print ("map_best",map_best[0:CORE_NUM])

    return map_best

def cmp_group(dnn,node_num):
    map_best = np.zeros((node_num*2))
    map_best = map_best -1 
    max_node_num_cur = node_num
    print (max_node_num_cur)


    while (1): 
        layer_node_num = {}
        neuron_sum = sum(dnn.layer_compute_num.values())
        print ("computation sum =",neuron_sum)
        for i in dnn.layer_compute_num :
            layer_node_num[i] = math.ceil(dnn.layer_compute_num[i] / neuron_sum * max_node_num_cur) 
        if sum(layer_node_num.values()) <= max_node_num:
            break
        else:
            max_node_num_cur -= 1 

    cal_id = 0
    for item in layer_node_num:
        map_best[cal_id] = -1
        node_sum_onelayer =  layer_node_num[item]
        while node_sum_onelayer != 0:
            map_best[cal_id] = item
            node_sum_onelayer -= 1
            cal_id += 1
    print ("dnn.layer_neu_compute",dnn.layer_neu_compute)
    print ("layer_node_num",layer_node_num)
    print ("map_best",map_best[0:CORE_NUM])

    return map_best

        
def calF_general(dnn,indv):
    F_cur=F.copy()
    node_num_in_layer = {}
    layer_comp_cycles_1node = {}
    layer_send_pkt_num = {}
    for layer in range (dnn.layer_num):
        node_num_in_layer[layer] = sum(indv[0:CORE_NUM] == layer)
        layer_comp_cycles_1node[layer] = math.ceil (dnn.layer_neu_compute[layer] * dnn.layer_neuron_out[layer] / node_num_in_layer[layer] / PE_ability)  ## layer_neu_compute = 每层的每个输出神经元所需的计算量
        layer_send_pkt_num[layer] = math.ceil(dnn.layer_neuron_out[layer] / node_num_in_layer[layer] /(neu_per_flit*flit_per_pkt)) 
        
    max_comp_cycle = max(layer_comp_cycles_1node.values())
    if (debug):
        print ("indv",indv)
        print ("node_num_in_layer",node_num_in_layer)
        print ("layer_comp_cycles_1node",layer_comp_cycles_1node)
        print ("layer_send_pkt_num",layer_send_pkt_num)
        print ("max_comp_cycle",max_comp_cycle)
    for possible_src in range (CORE_NUM):
        for possible_dst in range (CORE_NUM):
            src_layer_id = indv[possible_src]
            dst_layer_id =  indv[possible_dst]
            src_phy_id = int(indv[possible_src + CORE_NUM])
            dst_phy_id = int(indv[possible_dst + CORE_NUM])
            
            if src_layer_id != -1 and src_layer_id + 1 ==  dst_layer_id:  # 这两个节点之间具有通信需求
                bw_needed =  layer_send_pkt_num[src_layer_id] / max_comp_cycle * flit_per_pkt
                for link in route_table[(src_phy_id + 1000,dst_phy_id + 1000)]:
                    F_cur[link] += bw_needed / bw_scales[link]
    if(debug):
        ratio = 1
        for item in F_cur:
            if F_cur[item]>1:
                print (item, F_cur[item])
                ratio = ratio * (F_cur[item])
        print ("F_cur max",max(F_cur.values()))
        print ("F_cur sum",sum(F_cur.values()))
        print ("F_cur len",len(F_cur))
        print ("ratio=",ratio)
        sys.exit()
    return F_cur,max_comp_cycle


# 建立DNN模型
print ("task = ",task)
print ("method = ", method)
print ("NoC_w = ", NoC_w)
print ("no multicast")
DNN1 = DNNModel("DNN1")
DNN_input(task, DNN1)
result_list = []
throughput_list =[]
max_f_list = []

for PE_ability in PE_ability_list:
    print ("\n\n ########PE_ability=",PE_ability,"########")
    if method=="GA_RF":
        ga = GA_all(DNN1,CORE_NUM)
        genbest = ga.GA()
    elif method=="neuron_group_seq":
        genbest = neuron_group(DNN1,CORE_NUM)
        genbest[CORE_NUM:] = np.arange(0,CORE_NUM)
        print ("genbest=",genbest[:CORE_NUM],"genbest=",genbest[CORE_NUM:])
    elif method=="neuron_group_rdm":
        genbest = neuron_group(DNN1,CORE_NUM)
        map_list =  np.arange(0,CORE_NUM)
        np.random.shuffle(map_list)
        genbest[CORE_NUM:] = map_list
        print ("genbest=",genbest[:CORE_NUM],"genbest=",genbest[CORE_NUM:])
    elif method == "comp_group_seq":
        genbest = cmp_group(DNN1,CORE_NUM)
        genbest[CORE_NUM:] = np.arange(0,CORE_NUM)
        print ("genbest=",genbest[:CORE_NUM],"genbest=",genbest[CORE_NUM:])
    elif method == "comp_group_rdm":
        genbest = cmp_group(DNN1,CORE_NUM)
        map_list =  np.arange(0,CORE_NUM)
        np.random.shuffle(map_list)
        genbest[CORE_NUM:] = map_list
        print ("genbest=",genbest[:CORE_NUM],"genbest=",genbest[CORE_NUM:])

    phy_node_wait_pkt,phy_node_send_pkt,phy_node_send_dst,phy_node_cal_cycles = generate_info(DNN1,genbest)
    #生成任务文件

    scale_size = 1
    output_folder_name = task+'_'+method+'_'+str(NoC_w)+'_'+str(PE_ability)
    os.system('mkdir ./task/'+output_folder_name)
    if (if_create_task == 1):
        create_task_list( "./task/"+output_folder_name, CORE_NUM, scale_size, CORE_NUM, phy_node_cal_cycles, phy_node_send_pkt, phy_node_send_dst, phy_node_wait_pkt,genbest)


    # 输出最终结果
    best_F_cur,max_comp_cycle = calF_general(DNN1, genbest)
    if (max(best_F_cur.values()) < 1):
        degrade_ratio = 1
    else:
        degrade_ratio = max(best_F_cur.values()) + congestion_ratio
    best_fitness = max_comp_cycle * degrade_ratio

    print ("generation_best",genbest)
    print ("best_F_cur",best_F_cur)
    print ("max(best_F_cur.values()",max(best_F_cur.values()))
    print ("best_fitness",best_fitness)
    result_list.append(best_fitness)
    throughput_list.append(1/best_fitness*clock_freq)
    max_f_list.append(max(best_F_cur.values()))

print ("PE_ability_list",PE_ability_list)
print ("result_list", result_list)
print ("max_f_list", max_f_list)
print ("throughput_list:")
for throughput in throughput_list:
    print (throughput)
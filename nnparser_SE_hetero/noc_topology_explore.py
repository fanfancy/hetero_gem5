import numpy as np
import copy
from single_engine_predict_noc_nop import simple_multicast
from mesh_hetero import *
from matplotlib import pyplot as plt

degrade_ratio_list = []
excel_datas = []

def setmappingSet(height, lenth, set1, set2):
    num = height * lenth
    assert(num == set1*set2)
    list1 = {}
    list2 = {}
    node_list = []
    ID = 0
    for i in range(num):
        if i % lenth == 0:
            ID += 1
        node_list.append(ID)
        ID += 1
    print(node_list)
    for i in range(num):
        set1_id = i // set2
        if set1_id not in list1:
            list1[set1_id] = []
        list1[set1_id].append(node_list[i])

    for i in range(num):
        set2_id = i // set1
        if set2_id not in list2:
            list2[set2_id] = []
        list2[set2_id].append(list1[i % set1][set2_id])
    return list1, list2

def cal_degrade_ratio(HW_param, NoC_param, topology, set_act, set_wgt, act_bw, wgt_bw, if_multicast, debug = False):
    route_table = NoC_param["route_table"]
    bw_scales = NoC_param["bw_scales"]
    F = NoC_param["F"]

    CoreNum = HW_param["PE"][0] * HW_param["PE"][1]
    PE_lenth = HW_param["PE"][1]
    PE_height = HW_param["PE"][0]

    # memory node id
    # ol2_node = PE_height * (PE_lenth+1)
    ol2_node = 0
    if PE_height == 2:
        al2_node = ol2_node + PE_lenth + 1
        wl2_node = ol2_node + PE_lenth + 1
    else:
        assert(PE_height > 1)
        al2_node = ol2_node + PE_lenth + 1
        wl2_node = ol2_node + (PE_lenth + 1) * 2

    F_cur=F.copy()
    
    set_out = {0:[1,2,3,4, 6,7,8,9, 11,12,13,14, 16,17,18,19]}

    # 对act构建通信需求
    for act_transfer in set_act.values():
        if if_multicast == False:
            for dst in act_transfer:
                # print('dst = ', dst)
                for link in route_table[(al2_node + 1000, dst + 1000)]:
                    if debug:
                        print(al2_node + 1000, dst + 1000)
                        print(link)

                    F_cur[link] += ( act_bw / bw_scales[link] )
        else :
            link_set = simple_multicast(al2_node + 1000, [dst + 1000 for dst in act_transfer], route_table) 
            for link in link_set:
                if debug:
                    print(al2_node + 1000, act_transfer)
                    print(link)

                F_cur[link] += ( act_bw / bw_scales[link] )

    # 对wgt构建通信需求
    for wgt_transfer in set_wgt.values():
        if if_multicast == False:
            for dst in wgt_transfer:
                for link in route_table[(wl2_node + 1000, dst + 1000)]:
                    if debug:
                        print(wl2_node + 1000, dst + 1000)
                        print(link)

                    F_cur[link] += ( wgt_bw / bw_scales[link] )
        
        else:
            link_set = simple_multicast(wl2_node + 1000, [dst + 1000 for dst in wgt_transfer], route_table) 
            for link in link_set:
                if debug:
                    print(wl2_node + 1000, wgt_transfer)
                    print(link)

                F_cur[link] += ( wgt_bw / bw_scales[link] )
    
    # 对out构建通信需求
    out_bw = 1
    for out_transfer in set_out.values():
        for dst in out_transfer:					#写output不存在多播可能
            for link in route_table[(dst + 1000, ol2_node + 1000)]:
                if debug:
                    print(dst + 1000, ol2_node + 1000)
                    print(link)

                F_cur[link] += ( out_bw / bw_scales[link] )

    F_cur[(0, 1000)] = 0
    F_cur[(1005, 5)] = 0
    F_cur[(1010, 10)] = 0

    if (max(F_cur.values()) < 1):
            degrade_ratio = 1
    else:
        degrade_ratio = max(F_cur.values()) 
        
    return degrade_ratio, F_cur

def check_topology():
    topology = 'Ring'
    HW_param = {"Chiplet":[1, 1], "PE":[4, 4], "intra_PE":{"C":8,"K":8}}
    act_wgt_group = [2, 8]
    Sample_Num = 20
    
    NoC_w = HW_param["PE"][1] + 1
    NOC_NODE_NUM = NoC_w * HW_param["PE"][0]
    NoP_w = HW_param["Chiplet"][1]
    NOP_SIZE = NoP_w * HW_param["Chiplet"][0]
    
    TOPO_param = {"NoC_w":NoC_w, "NOC_NODE_NUM": NOC_NODE_NUM, "NoP_w": NoP_w, "NOP_SIZE": NOP_SIZE,"nop_scale_ratio": nop_bandwidth/noc_bandwidth}

    if_multicast = True
    debug = True
    set_act, set_wgt = setmappingSet(4, 4, act_wgt_group[0], act_wgt_group[1])
    if debug:
        print('set_act = ', set_act)
        print('set_wgt = ', set_wgt)

    act_bw_array = np.linspace(2, 16, Sample_Num)
    wgt_bw_array = np.linspace(2, 16, Sample_Num)
    degrade_ratio_array = np.zeros((Sample_Num, Sample_Num))
    X, Y = np.meshgrid(act_bw_array, wgt_bw_array)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('act_bw')
    ax.set_ylabel('wgt_bw')
    ax.set_zlabel('degrade_ratio')
    
    plt.title('act_wgt_group = %s, topology = %s' % (str(act_wgt_group), topology))
    NoC_param, all_sim_node_num = construct_noc_nop_topo(TOPO_param["NOC_NODE_NUM"],TOPO_param["NoC_w"], TOPO_param["NOP_SIZE"],TOPO_param["NoP_w"], TOPO_param["nop_scale_ratio"], topology = topology)

    for j, act_bw in enumerate(act_bw_array):
        for k, wgt_bw in enumerate(wgt_bw_array):
            print('###### act_bw = %f, wgt_bw = %f ######' % (act_bw, wgt_bw))
            degrade_ratio, F_cur = cal_degrade_ratio(HW_param, NoC_param, topology, set_act, set_wgt, act_bw, wgt_bw, if_multicast, debug = debug)

            if debug:
                print('degrade_ratio = ', degrade_ratio)
                print("F_cur = ", F_cur)
            else:
                print('degrade_ratio = ', degrade_ratio)
            degrade_ratio_array[k, j] = degrade_ratio
    
    # Plot a basic wireframe.
    ax.plot_wireframe(X, Y, degrade_ratio_array)
    if debug:
        print(X)
        print(Y)
        print(degrade_ratio_array)

    plt.show()

def noc_topology_explore():
    HW_param = {"Chiplet":[1, 1], "PE":[4, 4], "intra_PE":{"C":8,"K":8}}
    Topology_list = ['Mesh', 'Torus', 'Routerless', 'Ring']
    color_list = ['b', 'r', 'y', 'g']
    # act_wgt_group = [[1, 16], [2, 8], [4, 4], [8, 2], [16, 1]]
    act_wgt_group = [2, 8]
    Sample_Num = 20
    
    NoC_w = HW_param["PE"][1] + 1
    NOC_NODE_NUM = NoC_w * HW_param["PE"][0]
    NoP_w = HW_param["Chiplet"][1]
    NOP_SIZE = NoP_w * HW_param["Chiplet"][0]
    
    TOPO_param = {"NoC_w":NoC_w, "NOC_NODE_NUM": NOC_NODE_NUM, "NoP_w": NoP_w, "NOP_SIZE": NOP_SIZE,"nop_scale_ratio": nop_bandwidth/noc_bandwidth}

    if_multicast = True
    debug = False
    set_act, set_wgt = setmappingSet(4, 4, act_wgt_group[0], act_wgt_group[1])
    if debug:
        print('set_act = ', set_act)
        print('set_wgt = ', set_wgt)

    act_bw_array = np.linspace(2, 16, Sample_Num)
    wgt_bw_array = np.linspace(2, 16, Sample_Num)
    degrade_ratio_array = np.zeros((Sample_Num, Sample_Num))
    X, Y = np.meshgrid(act_bw_array, wgt_bw_array)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('act_bw')
    ax.set_ylabel('wgt_bw')
    ax.set_zlabel('degrade_ratio')
    plt.title('act_wgt_group = ' + str(act_wgt_group))

    # --- 生成noc-nop结构图
    for i, topology in enumerate(Topology_list):
        NoC_param, all_sim_node_num = construct_noc_nop_topo(TOPO_param["NOC_NODE_NUM"],TOPO_param["NoC_w"], TOPO_param["NOP_SIZE"],TOPO_param["NoP_w"], TOPO_param["nop_scale_ratio"], topology = topology)

        print('topology = ', topology)

        for j, act_bw in enumerate(act_bw_array):
            for k, wgt_bw in enumerate(wgt_bw_array):
                print('###### act_bw = %f, wgt_bw = %f ######' % (act_bw, wgt_bw))
                degrade_ratio, F_cur = cal_degrade_ratio(HW_param, NoC_param, topology, set_act, set_wgt, act_bw, wgt_bw, if_multicast, debug = debug)

                if debug:
                    print('degrade_ratio = ', degrade_ratio)
                    print("F_cur = ", F_cur)
                else:
                    print('degrade_ratio = ', degrade_ratio)
                degrade_ratio_array[k, j] = degrade_ratio
        
        # Plot wireframe.
        ax.plot_wireframe(X, Y, degrade_ratio_array, color = color_list[i])
        if debug:
            print(degrade_ratio_array)

    plt.legend([t for t in Topology_list], loc = 'best')
    plt.show()


if __name__ == '__main__':
    noc_topology_explore()
    # check_topology()

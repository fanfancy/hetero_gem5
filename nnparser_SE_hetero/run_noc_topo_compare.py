from mesh_hetero import construct_noc_nop_topo
from config import *
from randomTest_intralayer import getLayerParam, randomTest, randomTest_NoC_ours
from GaEncode import GaEncode

def randomTest_NoC_Mesh(app_name, chiplet_parallel = "All", core_parallel = "All", dataflow = "ours"):
        # --- 硬件参数
    HW_param = {"Chiplet":[4,4],"PE":[4,4],"intra_PE":{"C":16,"K":16}}       	# from granularity exploration
    # memory_param = {"OL1":1.5,"OL2":1.5*16,"AL1":800/1024,"AL2":64,"WL1":18,"WL2":18*16} 	from nnbaton
    memory_param = {"OL1":8 ,"OL2":128,"AL1":16,"AL2":256,"WL1":64,"WL2":1024}		# from granularity exploration
    NoC_w = HW_param["PE"][1] + 1
    NOC_NODE_NUM = NoC_w * HW_param["PE"][0]
    NoP_w = HW_param["Chiplet"][1] + 1
    NOP_SIZE = NoP_w * HW_param["Chiplet"][0]
    TOPO_param = {"NoC_w":NoC_w, "NOC_NODE_NUM": NOC_NODE_NUM, "NoP_w": NoP_w, "NOP_SIZE": NOP_SIZE,"nop_scale_ratio": nop_bandwidth/noc_bandwidth}
    
    # --- 生成noc-nop结构图
    NoC_param, all_sim_node_num = construct_noc_nop_topo(TOPO_param["NOC_NODE_NUM"],TOPO_param["NoC_w"], TOPO_param["NOP_SIZE"],TOPO_param["NoP_w"], TOPO_param["nop_scale_ratio"], topology = 'Mesh')
    debug = 1
    if_multicast = 1

    # --- 神经网络参数
    layer_dict = getLayerParam(app_name)

    edp_res_min_dict = {}
    energy_min_dict = {}
    delay_min_dict = {}
    code_min_dict = {}

    for layer_name in layer_dict:
        # ---输出文件
        filename = './final_test/intra_layer_edp_per_layer/Mesh_'+dataflow+"_"+app_name+"_"+layer_name+"_"+chiplet_parallel+'.xls'
        network_param = layer_dict[layer_name]
        GATest = GaEncode(network_param, HW_param, debug, chiplet_parallel = chiplet_parallel, core_parallel = core_parallel, flag=dataflow)

        iterTime = 10000 	# run 1w random mapping exploration

        random_test_iter = 1

        for i in range(random_test_iter):
            edp_res_min, energy_min, delay_min, code_min = randomTest(GATest, iterTime, HW_param, memory_param, NoC_param, all_sim_node_num, if_multicast, filename)
            edp_res_min_dict[layer_name] = edp_res_min
            energy_min_dict[layer_name] = energy_min
            delay_min_dict[layer_name] = delay_min
            code_min_dict[layer_name] = code_min
    file_1 = "./final_test/intra_layer_edp/Mesh_"+dataflow+"_"+ app_name + "_" + chiplet_parallel + ".txt"
    f = open(file_1,'w')
    print(edp_res_min_dict, file=f)
    print(energy_min_dict, file=f)
    print(delay_min_dict, file=f)
    print(code_min_dict, file = f)
    f.close()

def randomTest_NoC_CMesh(app_name, chiplet_parallel = "All", core_parallel = "All", dataflow = "ours"):
        # --- 硬件参数
    HW_param = {"Chiplet":[4,4],"PE":[4,4],"intra_PE":{"C":16,"K":16}}       	# from granularity exploration
    # memory_param = {"OL1":1.5,"OL2":1.5*16,"AL1":800/1024,"AL2":64,"WL1":18,"WL2":18*16} 	from nnbaton
    memory_param = {"OL1":8 ,"OL2":128,"AL1":16,"AL2":256,"WL1":64,"WL2":1024}		# from granularity exploration
    NoC_w = HW_param["PE"][1] + 1
    NOC_NODE_NUM = NoC_w * HW_param["PE"][0]
    NoP_w = HW_param["Chiplet"][1] + 1
    NOP_SIZE = NoP_w * HW_param["Chiplet"][0]
    TOPO_param = {"NoC_w":NoC_w, "NOC_NODE_NUM": NOC_NODE_NUM, "NoP_w": NoP_w, "NOP_SIZE": NOP_SIZE,"nop_scale_ratio": nop_bandwidth/noc_bandwidth}
    
    # --- 生成noc-nop结构图
    NoC_param, all_sim_node_num = construct_noc_nop_topo(TOPO_param["NOC_NODE_NUM"],TOPO_param["NoC_w"], TOPO_param["NOP_SIZE"],TOPO_param["NoP_w"], TOPO_param["nop_scale_ratio"], topology = 'CMesh')
    debug = 0
    if_multicast = 1

    # --- 神经网络参数
    layer_dict = getLayerParam(app_name)

    edp_res_min_dict = {}
    energy_min_dict = {}
    delay_min_dict = {}
    code_min_dict = {}

    for layer_name in layer_dict:
        # ---输出文件
        filename = './final_test/intra_layer_edp_per_layer/CMesh_'+dataflow+"_"+app_name+"_"+layer_name+"_"+chiplet_parallel+'.xls'
        network_param = layer_dict[layer_name]
        GATest = GaEncode(network_param, HW_param, debug, chiplet_parallel = chiplet_parallel, core_parallel = core_parallel, flag=dataflow)

        iterTime = 10000 	# run 1w random mapping exploration

        random_test_iter = 1

        for i in range(random_test_iter):
            edp_res_min, energy_min, delay_min, code_min = randomTest(GATest, iterTime, HW_param, memory_param, NoC_param, all_sim_node_num, if_multicast, filename)
            edp_res_min_dict[layer_name] = edp_res_min
            energy_min_dict[layer_name] = energy_min
            delay_min_dict[layer_name] = delay_min
            code_min_dict[layer_name] = code_min
    file_1 = "./final_test/intra_layer_edp/CMesh_"+dataflow+"_"+ app_name + "_" + chiplet_parallel + ".txt"
    f = open(file_1,'w')
    print(edp_res_min_dict, file=f)
    print(energy_min_dict, file=f)
    print(delay_min_dict, file=f)
    print(code_min_dict, file = f)
    f.close()

if __name__ == '__main__':
    app_name = 'VGG16'
    struct_name = 'ours'
    dataflow = 'ours'
    # randomTest_NoC_ours(app_name, "P_stable", "All", dataflow)
    # randomTest_NoC_ours(app_name, "PK_stable", "All", dataflow)
    # randomTest_NoC_ours(app_name, "K_stable", "All", dataflow)
    # randomTest_NoC_Mesh(app_name, "P_stable", "All", dataflow)
    # randomTest_NoC_Mesh(app_name, "PK_stable", "All", dataflow)
    # randomTest_NoC_Mesh(app_name, "K_stable", "All", dataflow)
    randomTest_NoC_CMesh(app_name, "P_stable", "All", dataflow)
    # randomTest_NoC_CMesh(app_name, "PK_stable", "All", dataflow)
    # randomTest_NoC_CMesh(app_name, "K_stable", "All", dataflow)


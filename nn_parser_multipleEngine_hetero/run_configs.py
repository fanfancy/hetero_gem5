if_hetero = 1
if_create_task = 1
method ="GA_RF"  # "comp_group_seq"  "comp_group_rdm" "GA_RF"   "neuron_group_seq"  "neuron_group_rdm" 
task = "lenet"          # "lenet"    "VGG16"            "resnet_50"  "alexnet"
# nocrouter的总数= NOC_NODE_NUM *NOP_SIZE
NoC_w = 2
NoP_w = 2

PE_ability_list = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]
# PE_ability_list = [50,100,150,200,250,300,350,400,450,500,550,600]
# PE_ability_list = [310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600]
# PE_ability_list = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240]



clock_freq = 100*1000000
#硬件参数
group_size = 100

neu_per_flit = 2
flit_per_pkt = 5
NOC_NODE_NUM = NoC_w*NoC_w
NOP_SIZE = NoP_w*NoP_w
CORE_NUM = NOC_NODE_NUM*NOP_SIZE
nop_scale_ratio = 0.5

# 计算、发送任务的缩放比例

# 算法一的节点范围
max_node_num = NOC_NODE_NUM*NOP_SIZE
min_node_num = 4

debug = 0
congestion_ratio = 0

#### single engine 
NoC_w = 5
NOC_NODE_NUM = 20
NoP_w = 3
NOP_SIZE = 6
CORE_NUM = NOC_NODE_NUM*NOP_SIZE
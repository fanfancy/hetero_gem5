nop_bandwidth 	= 100				#Gb/s
noc_bandwidth 	= 68				#Gb/s
act_wgt_width  	= 8 				# bit
output_width	= 24
PE_freq 		= 1					# Ghz
noc_link_width = noc_bandwidth / PE_freq # bit
neu_per_flit_act_wgt = int ( noc_link_width / act_wgt_width ) 
neu_per_flit_output = int ( noc_link_width  / output_width )
flit_per_pkt = 5

DRAM_energy_ratio = 8.75    	# PJ/bit
def SRAM_energy(size):
	return 0.016452 * size + 0.283548
DIE2DIE_energy_ratio = 1.17
NOC_energy_ratio = 0
MAC_energy_ratio = 0.024    	# 8bit MAC
 
chiplet_parallel = "All" 		# "Pq" "All" "Channel" "Hybrid"
core_parallel = "All"			# "Pq" "All" "Channel" "Hybrid" 

## applications
vgg16_conv1 = {"P":224,"Q":224,"C":3,"K":64,"R":3,"S":3}
vgg16_conv12 =  {"P":16,"Q":16,"C":512,"K":512,"R":3,"S":3} # 原本是14
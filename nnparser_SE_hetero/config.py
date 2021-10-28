neuron_width  = 8 				# bit
noc_link_width = 32
neu_per_flit = noc_link_width / neuron_width
flit_per_pkt = 5

DRAM_energy_ratio = 8.75    	# PJ/bit
def SRAM_energy(size):
	return 0.016452 * size + 0.283548
DIE2DIE_energy_ratio = 1.17
MAC_energy_ratio = 0.024    	# 8bit MAC
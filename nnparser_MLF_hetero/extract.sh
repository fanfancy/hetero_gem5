python3 workload_partition.py \
	--app_name_line resnet18+resnet50+VGG16 \
	--chiplet_num_min_TH 1 \
	--chiplet_num_max_TH 16 \
	--fuse_flag 0 \
	--alg_list GA+GA+GA \
	--architecture ours \
	--encode_type_list index+index+index \
	--workload_par_objective iact_num \
	--workload_num_TH 4
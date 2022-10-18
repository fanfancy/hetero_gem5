python layer_fuse.py \
	--app_name_line resnet18+resnet50+VGG16+vit \
	--chiplet_num_min_TH 1 \
	--chiplet_num_max_TH 36 \
	--fuse_flag 1 \
	--alg_list GA+GA+GA+GA \
	--architecture simba \
	--encode_type_list index+index+index+num \
	--workload_par_objective iact_num \
	--workload_num_TH 4